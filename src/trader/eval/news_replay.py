"""Counterfactual eval: does the Claude news risk-filter earn its veto?

The filter (`trader.news`) can only *block* a trend entry, never initiate one. That
asymmetry sets the metric: a false positive kills a trade drawn from a distribution
with +5.8% expectancy, while a false negative merely returns us to the unfiltered
baseline. So the question is not "is the classifier accurate" but "what did the
blocks cost".

Method — two arms over identical history, through the same simulator:
  A. baseline : `backtest_trend(..., veto=None)`
  B. filtered : the same call with a veto backed by point-in-time headlines scored
                by the live prompt, through the live `should_block` thresholds.
Diff the metrics, then attribute each veto to the baseline trade it prevented.

No-lookahead: headlines are windowed to `[close - lookback, close)` where `close` is
the decision bar's session close (see `session_close_utc`). Nothing published after
the decision can reach the model.

Known contamination — state it in any writeup: the model has training knowledge of
what happened after these dates, which biases the filter toward looking *good*. Read
a poor result as a lower bound on how poor it is; do not read a good result as proof.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from ..backtest.engine import Trade
from ..backtest.metrics import Metrics, compute_metrics
from ..backtest.trend import backtest_trend
from ..news import PROMPT_VERSION, Sentiment, claude_score
from .cache import ScoreCache

log = logging.getLogger("trader.eval")

# US equity session close in UTC. Daily bars are stamped at the session *date*, but
# the decision is taken at the close, so this is what bounds the news window. During
# DST this is 20:00Z; 21:00 year-round is the conservative choice for a veto (a
# slightly wider window can only add headlines the agent genuinely had).
DEFAULT_CLOSE_UTC_HOUR = 21


def session_close_utc(ts, close_hour: int = DEFAULT_CLOSE_UTC_HOUR) -> pd.Timestamp:
    """Map a daily bar's timestamp to the moment its close was observed."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize() + timedelta(hours=close_hour)


@dataclass
class VetoEvent:
    symbol: str
    decision_ts: pd.Timestamp
    label: str
    confidence: float
    earnings_imminent: bool
    reason: str
    n_headlines: int
    # Return of the baseline trade this veto prevented. None when the veto is a
    # retry of an already-blocked entry, which has no baseline counterpart.
    counterfactual_ret: float | None = None
    # True for the veto that first blocked an entry opportunity, False for the
    # daily retries that follow it while the trend leg stays alive. One blocked
    # opportunity is the meaningful unit; retries would otherwise inflate counts.
    first_of_window: bool = False
    # Whether the filtered arm got into this leg anyway, later. Usually true — a
    # veto delays an entry rather than cancelling it — which is why the sum of
    # `counterfactual_ret` is NOT the cost of the filter. Only the metric delta is.
    reentered: bool | None = None


@dataclass
class ReplayResult:
    baseline: Metrics
    filtered: Metrics
    baseline_trades: list[Trade]
    filtered_trades: list[Trade]
    vetoes: list[VetoEvent]
    n_consulted: int = 0
    n_scored: int = 0        # LLM calls actually made (cache hits excluded)
    n_cached: int = 0
    n_errors: int = 0        # fetch or scoring failures — these failed open
    n_no_headlines: int = 0

    @property
    def windows(self) -> list[VetoEvent]:
        """Distinct blocked entry opportunities, not per-day retries."""
        return [v for v in self.vetoes if v.first_of_window]

    @property
    def matched(self) -> list[VetoEvent]:
        """Blocks we can price — those that killed an identifiable baseline trade."""
        return [v for v in self.vetoes if v.counterfactual_ret is not None]

    @property
    def killed_winners(self) -> list[VetoEvent]:
        return [v for v in self.matched if v.counterfactual_ret > 0]

    @property
    def avoided_losers(self) -> list[VetoEvent]:
        return [v for v in self.matched if v.counterfactual_ret <= 0]

    @property
    def block_precision(self) -> float | None:
        """Share of priced blocks that dodged a losing trade. Given the asymmetry
        (a block can only cost return), this is the number the filter lives by."""
        if not self.matched:
            return None
        return len(self.avoided_losers) / len(self.matched)


# --- enumerating decision bars -------------------------------------------------


class _Probe:
    """Veto callback that records decision bars without scoring anything.

    With `block=False` it records what a never-blocking filter is asked — the exact
    lower bound on scoring cost, and a tight estimate for a filter that fires rarely.
    With `block=True` it records the superset any policy could ask about: blocking
    keeps the sim flat, and no veto is consulted while holding a position.
    """

    def __init__(self, block: bool = False):
        self.block = block
        self.bars: list[tuple[str, pd.Timestamp]] = []

    def __call__(self, symbol: str, decision_ts) -> bool:
        self.bars.append((symbol, pd.Timestamp(decision_ts)))
        return self.block


def probe_decision_bars(bars_by_symbol: dict[str, pd.DataFrame], trend_params,
                        slippage_pct: float = 0.0005,
                        block: bool = False) -> list[tuple[str, pd.Timestamp]]:
    """Decision bars the news filter would be consulted on. Offline and free —
    this is what `--plan` counts."""
    tp = trend_params
    probe = _Probe(block=block)
    for sym, bars in bars_by_symbol.items():
        backtest_trend(sym, bars, tp.entry_channel, tp.exit_channel, tp.trend_ma,
                       tp.use_regime_filter, tp.cat_stop_pct, slippage_pct, veto=probe)
    return probe.bars


# --- scoring -------------------------------------------------------------------


class _Scorer:
    """Point-in-time sentiment for one (symbol, decision bar), scored at most once.

    Deduped by an in-flight task table: the simulator asks for the same day from
    several places, and a retry loop asks for consecutive days across symbols
    concurrently. Every path is cached on disk as well, so re-runs are free.
    """

    def __init__(self, *, fetch_window, scorer, cache: ScoreCache, lookback_hours: int,
                 model: str, close_hour: int, concurrency: int):
        self._fetch = fetch_window
        self._score = scorer
        self._cache = cache
        self._lookback = lookback_hours
        self._model = model
        self._close_hour = close_hour
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        self.n_scored = 0
        self.n_cached = 0
        self.n_errors = 0
        self.n_no_headlines = 0

    async def get(self, symbol: str, ts: pd.Timestamp) -> Sentiment:
        key = (symbol, str(ts.date()))
        task = self._tasks.get(key)
        if task is None:
            task = asyncio.ensure_future(self._compute(symbol, ts, key))
            self._tasks[key] = task
        return await task  # a Task may be awaited by any number of callers

    async def _compute(self, symbol: str, ts: pd.Timestamp,
                       key: tuple[str, str]) -> Sentiment:
        cached = self._cache.get(symbol, key[1], self._lookback, PROMPT_VERSION,
                                 self._model)
        if cached is not None:
            self.n_cached += 1
            return cached

        async with self._sem:
            end = session_close_utc(ts, self._close_hour)
            start = end - timedelta(hours=self._lookback)
            headlines: list[str] = []
            try:
                headlines = await asyncio.to_thread(self._fetch, symbol, start, end)
            except Exception as exc:  # noqa: BLE001 — fail open, as live does
                log.warning("headline fetch failed %s %s: %s", symbol, key[1], exc)
                sent = Sentiment(error=str(exc))
            else:
                if headlines:
                    try:
                        sent = await self._score(symbol, headlines)
                        self.n_scored += 1
                    except Exception as exc:  # noqa: BLE001
                        log.warning("scoring failed %s %s: %s", symbol, key[1], exc)
                        sent = Sentiment(error=str(exc))
                else:
                    sent = Sentiment(label="neutral", rationale="no recent news")
                    self.n_no_headlines += 1

        if sent.error:
            self.n_errors += 1
        sent.headlines = headlines
        sent.model = sent.model or self._model
        sent.prompt_version = sent.prompt_version or PROMPT_VERSION
        self._cache.put(symbol, key[1], self._lookback, PROMPT_VERSION, self._model, sent)
        return sent


class _LazyVeto:
    """Sync veto callable for the simulator, backed by async scoring.

    `backtest_trend` is a tight synchronous loop, so the filtered arm runs in a
    worker thread and each veto hands its scoring request back to the event loop.
    That keeps scoring *exact* — only bars the simulation actually reaches are
    scored — while still running symbols concurrently. Scoring every bar a veto
    could conceivably touch (the `block=True` probe) would cost an order of
    magnitude more LLM calls, most of them for bars the run never visits.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, scorer: _Scorer, should_block):
        self._loop = loop
        self._scorer = scorer
        self._should_block = should_block
        self._lock = threading.Lock()
        self.vetoes: list[VetoEvent] = []
        self.n_consulted = 0

    def __call__(self, symbol: str, decision_ts) -> bool:
        ts = pd.Timestamp(decision_ts)
        fut = asyncio.run_coroutine_threadsafe(self._scorer.get(symbol, ts), self._loop)
        sent = fut.result()
        blocked, reason = self._should_block(sent)
        with self._lock:
            self.n_consulted += 1
            if blocked:
                self.vetoes.append(VetoEvent(
                    symbol=symbol, decision_ts=ts, label=sent.label,
                    confidence=sent.confidence,
                    earnings_imminent=sent.earnings_imminent, reason=reason,
                    n_headlines=len(sent.headlines),
                ))
        return blocked


# --- attribution ---------------------------------------------------------------


def _next_bar(index: pd.Index, ts) -> pd.Timestamp | None:
    """The bar after `ts` — where a decision taken at `ts`'s close gets executed."""
    pos = index.searchsorted(pd.Timestamp(ts), side="right")
    return index[pos] if pos < len(index) else None


def _attribute(vetoes: list[VetoEvent], baseline_trades: list[Trade],
               filtered_trades: list[Trade],
               bars_by_symbol: dict[str, pd.DataFrame]) -> None:
    """Mark window starts, price each block against the trade it prevented, and
    record whether the filtered arm entered that leg anyway.

    A decision at bar i executes at bar i+1, so a veto at `decision_ts` maps to the
    baseline trade whose `entry_time` is the next bar. Retry vetoes fall on later
    bars and won't match — the baseline already entered on the original breakout.
    """
    by_entry: dict[tuple[str, pd.Timestamp], Trade] = {
        (t.symbol, pd.Timestamp(t.entry_time)): t for t in baseline_trades
    }
    filtered_by_symbol: dict[str, list[Trade]] = {}
    for t in filtered_trades:
        filtered_by_symbol.setdefault(t.symbol, []).append(t)
    by_symbol: dict[str, list[VetoEvent]] = {}
    for v in vetoes:
        by_symbol.setdefault(v.symbol, []).append(v)

    for symbol, events in by_symbol.items():
        bars = bars_by_symbol.get(symbol)
        if bars is None:
            continue
        events.sort(key=lambda e: e.decision_ts)
        prev_ts = None
        for v in events:
            pos = bars.index.get_loc(v.decision_ts)
            # A retry lands on the bar right after the previous block; anything
            # else starts a new blocked opportunity.
            v.first_of_window = prev_ts is None or bars.index[pos - 1] != prev_ts
            prev_ts = v.decision_ts

            exec_ts = _next_bar(bars.index, v.decision_ts)
            if exec_ts is None:
                continue
            trade = by_entry.get((symbol, pd.Timestamp(exec_ts)))
            if trade is None:
                continue
            v.counterfactual_ret = trade.ret_pct
            # Did the filter merely delay this entry? True whenever the filtered arm
            # opened a position on this symbol before the blocked trade would have
            # closed — i.e. it caught most of the same move, a few days late.
            v.reentered = any(
                pd.Timestamp(f.entry_time) > v.decision_ts
                and pd.Timestamp(f.entry_time) < pd.Timestamp(trade.exit_time)
                for f in filtered_by_symbol.get(symbol, [])
            )


# --- the run -------------------------------------------------------------------


async def replay(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    trend_params,
    news_params,
    should_block,
    fetch_window,
    cache: ScoreCache,
    scorer=None,
    model: str = "",
    slippage_pct: float = 0.0005,
    position_fraction: float = 0.10,
    close_hour: int = DEFAULT_CLOSE_UTC_HOUR,
    concurrency: int = 4,
    progress=None,
) -> ReplayResult:
    """Run both arms and return the comparison.

    `should_block` / `fetch_window` are passed in (normally the live `NewsSentiment`
    methods) so the eval exercises the *production* veto rule rather than a
    reimplementation that could silently drift from it.
    """
    tp = trend_params
    scorer = scorer or (lambda sym, hl: claude_score(sym, hl, model))

    baseline_trades: list[Trade] = []
    for sym, bars in bars_by_symbol.items():
        baseline_trades.extend(backtest_trend(
            sym, bars, tp.entry_channel, tp.exit_channel, tp.trend_ma,
            tp.use_regime_filter, tp.cat_stop_pct, slippage_pct,
        ))

    if progress:
        estimate = len({(s, str(ts.date()))
                        for s, ts in probe_decision_bars(bars_by_symbol, tp, slippage_pct)})
        progress(estimate)

    score = _Scorer(fetch_window=fetch_window, scorer=scorer, cache=cache,
                    lookback_hours=news_params.lookback_hours, model=model,
                    close_hour=close_hour, concurrency=concurrency)
    loop = asyncio.get_running_loop()
    veto = _LazyVeto(loop, score, should_block)

    # Each symbol's simulation runs in a worker thread; the vetoes inside them call
    # back into this loop to score. Symbols therefore overlap, bounded by the
    # scorer's semaphore rather than by the number of symbols.
    #
    # The simulations get their OWN executor rather than `asyncio.to_thread`'s
    # shared default. They spend their lives blocked on scoring futures, and the
    # scorer fetches headlines via `to_thread` — sharing one pool deadlocks as soon
    # as every worker is a sim waiting on a fetch that cannot get a thread.
    pool = ThreadPoolExecutor(max_workers=max(1, min(len(bars_by_symbol), 16)),
                              thread_name_prefix="trend-sim")
    try:
        per_symbol = await asyncio.gather(*(
            loop.run_in_executor(
                pool, backtest_trend, sym, bars, tp.entry_channel, tp.exit_channel,
                tp.trend_ma, tp.use_regime_filter, tp.cat_stop_pct, slippage_pct, veto,
            )
            for sym, bars in bars_by_symbol.items()
        ))
    finally:
        pool.shutdown(wait=True)
    filtered_trades = [t for trades in per_symbol for t in trades]
    cache.save()

    _attribute(veto.vetoes, baseline_trades, filtered_trades, bars_by_symbol)

    return ReplayResult(
        baseline=compute_metrics(baseline_trades, position_fraction),
        filtered=compute_metrics(filtered_trades, position_fraction),
        baseline_trades=baseline_trades,
        filtered_trades=filtered_trades,
        vetoes=veto.vetoes,
        n_consulted=veto.n_consulted,
        n_scored=score.n_scored,
        n_cached=score.n_cached,
        n_errors=score.n_errors,
        n_no_headlines=score.n_no_headlines,
    )
