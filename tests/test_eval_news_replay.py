"""Tests for the news-filter eval harness. No network, no LLM.

The load-bearing property is the A/B identity: `veto=None` must reproduce the
existing backtest exactly, or every number the harness prints is meaningless.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from trader.backtest.trend import backtest_trend
from trader.config import NewsParams, TrendParams
from trader.eval.cache import ScoreCache
from trader.eval.news_replay import probe_decision_bars, replay, session_close_utc
from trader.news import Sentiment


def _bars(closes: list[float], start="2024-01-01") -> pd.DataFrame:
    # Narrow bars (±0.2%): a Donchian breakout needs the close to clear the prior
    # window's *high*, so a wide intrabar range would swallow the trend entirely.
    idx = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.002 for c in closes],
         "low": [c * 0.998 for c in closes], "close": closes,
         "volume": [1_000_000] * len(closes)},
        index=idx,
    )


def _trending(n=200) -> pd.DataFrame:
    """Long slow uptrend with a drawdown at the end — produces entries and an exit."""
    closes = [100 + i * 0.8 for i in range(n - 30)] + [
        100 + (n - 30) * 0.8 - i * 3 for i in range(30)
    ]
    return _bars(closes)


TP = TrendParams(entry_channel=20, exit_channel=10, trend_ma=30)


def _run(bars, **kw):
    return backtest_trend("TEST", bars, TP.entry_channel, TP.exit_channel,
                          TP.trend_ma, TP.use_regime_filter, TP.cat_stop_pct, **kw)


# --- the A/B identity ----------------------------------------------------------
def test_veto_none_is_identical_to_original_backtest():
    bars = _trending()
    assert _run(bars) == _run(bars, veto=None)


def test_never_blocking_veto_is_identical_to_no_veto():
    """A veto that always allows must not perturb the simulation — this is what
    makes the filtered arm comparable to the baseline."""
    bars = _trending()
    assert _run(bars, veto=lambda s, ts: False) == _run(bars)


def test_always_blocking_veto_produces_no_trades():
    assert _run(_trending(), veto=lambda s, ts: True) == []


def test_veto_is_consulted_at_the_decision_bar_not_the_execution_bar():
    """The callback must fire on the bar whose close produced the signal, one bar
    before the fill — otherwise the news window would include the fill day."""
    bars = _trending()
    seen: list[pd.Timestamp] = []

    def veto(sym, ts):
        seen.append(pd.Timestamp(ts))
        return False

    trades = _run(bars, veto=veto)
    assert trades and seen
    entry = pd.Timestamp(trades[0].entry_time)
    decisions_before_entry = [t for t in seen if t < entry]
    assert decisions_before_entry, "veto never ran before the first entry"
    # The decision that caused the entry is exactly the preceding bar.
    assert bars.index[bars.index.get_loc(entry) - 1] == decisions_before_entry[-1]


def test_vetoed_entry_is_retried_while_the_trend_leg_holds():
    """Live, a block is re-evaluated next tick. The sim must mirror that, else it
    would overstate the cost of a veto by abandoning the whole leg."""
    bars = _trending()
    calls: list[pd.Timestamp] = []

    def veto_once(sym, ts):
        calls.append(pd.Timestamp(ts))
        return len(calls) == 1  # block only the first opportunity

    trades = _run(bars, veto=veto_once)
    baseline = _run(bars)
    assert trades, "a single veto should delay the entry, not cancel the leg"
    assert pd.Timestamp(trades[0].entry_time) > pd.Timestamp(baseline[0].entry_time)


# --- the probe superset property -----------------------------------------------
@pytest.mark.parametrize("policy", [
    lambda i: False,           # never block
    lambda i: True,            # always block
    lambda i: i % 2 == 0,      # alternating
    lambda i: i < 3,           # block early, then allow
])
def test_probe_covers_every_bar_any_policy_consults(policy):
    """The harness scores the probe's bars and nothing else, so a policy that
    consults a bar the probe missed would silently fail open there and understate
    the filter's effect."""
    bars = _trending()
    probe = {ts for _, ts in probe_decision_bars({"TEST": bars}, TP, block=True)}

    n = 0
    consulted: set[pd.Timestamp] = set()

    def veto(sym, ts):
        nonlocal n
        consulted.add(pd.Timestamp(ts))
        n += 1
        return policy(n - 1)

    _run(bars, veto=veto)
    assert consulted, "policy never got consulted — test is vacuous"
    assert consulted <= probe, f"probe missed {sorted(consulted - probe)}"


def test_never_block_probe_is_the_lower_bound_on_scoring():
    """`--plan` quotes this number, so it must match what a passive filter costs."""
    bars = _trending()
    plan = {ts for _, ts in probe_decision_bars({"TEST": bars}, TP, block=False)}
    seen: set[pd.Timestamp] = set()
    _run(bars, veto=lambda s, ts: seen.add(pd.Timestamp(ts)) or False)
    assert seen == plan


# --- no-lookahead window -------------------------------------------------------
def test_session_close_maps_bar_date_to_the_close_instant():
    ts = pd.Timestamp("2024-03-05", tz="UTC")
    assert session_close_utc(ts, 21) == pd.Timestamp("2024-03-05T21:00:00Z")


def test_session_close_handles_naive_timestamps():
    assert session_close_utc(pd.Timestamp("2024-03-05"), 21) == \
        pd.Timestamp("2024-03-05T21:00:00Z")


async def test_news_window_never_extends_past_the_decision(tmp_path):
    """The whole no-lookahead guarantee in one assertion."""
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(symbol, start, end):
        windows.append((start, end))
        return ["a headline"]

    async def scorer(symbol, headlines):
        return Sentiment(label="neutral", confidence=0.1)

    bars = {"TEST": _trending()}
    await replay(bars, trend_params=TP, news_params=NewsParams(lookback_hours=48),
                 should_block=lambda s: (False, s.label), fetch_window=fetch,
                 cache=ScoreCache(tmp_path / "c.json"), scorer=scorer)

    assert windows
    for start, end in windows:
        assert end - start == timedelta(hours=48)
        # `end` is a session close; the bar it belongs to is that same day.
        assert end.hour == 21


# --- attribution ---------------------------------------------------------------
async def test_attribution_prices_a_veto_against_the_trade_it_killed(tmp_path):
    bars = {"TEST": _trending()}
    baseline = _run(bars["TEST"])
    first_entry = pd.Timestamp(baseline[0].entry_time)
    decision = bars["TEST"].index[bars["TEST"].index.get_loc(first_entry) - 1]

    def fetch(symbol, start, end):
        return ["bad news"]

    async def scorer(symbol, headlines):
        return Sentiment(label="bearish", confidence=0.9)

    # Block only the decision that opened the baseline's first trade.
    def should_block(sent):
        return (True, "bearish") if sent.label == "bearish" else (False, sent.label)

    res = await replay(bars, trend_params=TP, news_params=NewsParams(),
                       should_block=should_block, fetch_window=fetch,
                       cache=ScoreCache(tmp_path / "c.json"), scorer=scorer)

    assert res.filtered.n_trades == 0, "blocking everything should leave no trades"
    priced = [v for v in res.vetoes if v.decision_ts == decision]
    assert priced, "the veto at the baseline's entry decision was not recorded"
    assert priced[0].counterfactual_ret == pytest.approx(baseline[0].ret_pct)


async def test_block_precision_counts_dodged_losers(tmp_path):
    bars = {"TEST": _trending()}

    def fetch(symbol, start, end):
        return ["news"]

    async def scorer(symbol, headlines):
        return Sentiment(label="bearish", confidence=0.9)

    res = await replay(bars, trend_params=TP, news_params=NewsParams(),
                       should_block=lambda s: (True, "bearish"), fetch_window=fetch,
                       cache=ScoreCache(tmp_path / "c.json"), scorer=scorer)

    assert res.matched, "no veto could be priced"
    assert len(res.killed_winners) + len(res.avoided_losers) == len(res.matched)
    assert 0.0 <= res.block_precision <= 1.0


async def test_daily_retries_collapse_into_one_blocked_opportunity(tmp_path):
    """Blocking an entry re-asks every following day. Counting those as separate
    blocks would inflate the headline number and dilute block precision, so only
    the first veto of each run is a 'window'."""
    bars = {"TEST": _trending()}

    def fetch(symbol, start, end):
        return ["news"]

    async def scorer(symbol, headlines):
        return Sentiment(label="bearish", confidence=0.9)

    res = await replay(bars, trend_params=TP, news_params=NewsParams(),
                       should_block=lambda s: (True, "bearish"), fetch_window=fetch,
                       cache=ScoreCache(tmp_path / "c.json"), scorer=scorer)

    assert len(res.vetoes) > len(res.windows), "no retries were generated"
    assert len(res.windows) >= 1
    # Every window start is a real gap in the veto sequence.
    index = bars["TEST"].index
    starts = [v.decision_ts for v in res.vetoes if v.first_of_window]
    all_ts = sorted(v.decision_ts for v in res.vetoes)
    for ts in starts:
        pos = index.get_loc(ts)
        assert pos == 0 or index[pos - 1] not in all_ts
    # Only a window start can be priced against a baseline trade.
    assert all(v.first_of_window for v in res.matched)


async def test_a_one_day_block_is_marked_as_delayed_not_cancelled(tmp_path):
    """Distinguishing 'delayed' from 'cancelled' is what stops the report reading
    a blocked winner's return as a realized cost."""
    bars = {"TEST": _trending()}
    n = 0

    def fetch(symbol, start, end):
        return ["news"]

    async def scorer(symbol, headlines):
        nonlocal n
        n += 1
        return Sentiment(label="bearish" if n == 1 else "bullish", confidence=0.9)

    res = await replay(bars, trend_params=TP, news_params=NewsParams(),
                       should_block=lambda s: (s.label == "bearish", "bearish"),
                       fetch_window=fetch, cache=ScoreCache(tmp_path / "c.json"),
                       scorer=scorer)

    assert len(res.windows) == 1
    blocked = res.windows[0]
    assert blocked.counterfactual_ret is not None, "block should be priceable"
    assert blocked.reentered is True, "the next day's entry should count as re-entry"
    assert res.filtered.n_trades == res.baseline.n_trades


async def test_many_symbols_do_not_deadlock_the_thread_pool(tmp_path):
    """Simulations run in threads and block on scoring; scoring fetches headlines
    in threads too. Sharing one pool deadlocks once every worker is a simulation
    waiting on a fetch that cannot get a thread — hence the dedicated executor.

    Guarded by a timeout so a regression fails the suite instead of hanging it.
    """
    import time

    def fetch(symbol, start, end):
        time.sleep(0.01)  # forces the fetch onto a worker thread
        return ["news"]

    async def scorer(symbol, headlines):
        return Sentiment(label="neutral", confidence=0.1)

    bars = {f"SYM{i}": _trending() for i in range(40)}
    res = await asyncio.wait_for(
        replay(bars, trend_params=TP, news_params=NewsParams(),
               should_block=lambda s: (False, s.label), fetch_window=fetch,
               cache=ScoreCache(tmp_path / "c.json"), scorer=scorer, concurrency=4),
        timeout=60,
    )
    assert res.filtered.n_trades == res.baseline.n_trades


async def test_fetch_failure_fails_open_like_production(tmp_path):
    """An unreachable news API must not silently look like 'no bearish news'
    *and* must not block — the live filter fails open, so the eval must too."""
    def fetch(symbol, start, end):
        raise RuntimeError("alpaca down")

    async def scorer(symbol, headlines):  # pragma: no cover — must never run
        raise AssertionError("scorer called despite fetch failure")

    bars = {"TEST": _trending()}
    res = await replay(bars, trend_params=TP, news_params=NewsParams(),
                       should_block=lambda s: (s.error is None and s.label == "bearish",
                                               "x"),
                       fetch_window=fetch, cache=ScoreCache(tmp_path / "c.json"),
                       scorer=scorer)
    assert res.n_errors > 0
    assert res.vetoes == []
    assert res.filtered.n_trades == res.baseline.n_trades


# --- score cache ---------------------------------------------------------------
def test_cache_roundtrip(tmp_path: Path):
    c = ScoreCache(tmp_path / "s.json")
    s = Sentiment(label="bearish", confidence=0.8, headlines=["h"], model="m",
                  prompt_version="v1")
    c.put("NVDA", "2024-01-02", 48, "v1", "m", s)
    c.save()

    reloaded = ScoreCache(tmp_path / "s.json")
    got = reloaded.get("NVDA", "2024-01-02", 48, "v1", "m")
    assert got is not None and got.label == "bearish" and got.headlines == ["h"]


def test_cache_misses_on_prompt_or_model_change(tmp_path: Path):
    """A prompt change must invalidate cached verdicts — otherwise a regression
    hides behind stale scores."""
    c = ScoreCache(tmp_path / "s.json")
    c.put("NVDA", "2024-01-02", 48, "v1", "m", Sentiment(label="bearish"))
    assert c.get("NVDA", "2024-01-02", 48, "v2", "m") is None   # prompt changed
    assert c.get("NVDA", "2024-01-02", 48, "v1", "m2") is None  # model changed
    assert c.get("NVDA", "2024-01-02", 24, "v1", "m") is None   # window changed


def test_corrupt_cache_is_ignored_not_fatal(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    assert len(ScoreCache(p)) == 0
