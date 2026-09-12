"""Eval runner: measure what the news risk-filter's vetoes actually cost.

Usage:
    # what would this cost to run? (no LLM calls, no network beyond bars)
    PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
      python -m trader.eval.run --days 1500 --plan

    # the real thing
    PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
      python -m trader.eval.run --days 1500 --out eval-news.json

Verdicts are cached on disk (`--cache`), so a re-run after changing the report or
the thresholds costs nothing. Changing the prompt or model invalidates the cache by
design — that is the regression signal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from ..config import load_config
from ..market_data import MarketData
from ..news import PROMPT_VERSION, NewsSentiment
from .cache import ScoreCache
from .news_replay import DEFAULT_CLOSE_UTC_HOUR, probe_decision_bars, replay

log = logging.getLogger("trader.eval")


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _pf(x: float) -> str:
    return "inf" if x == float("inf") else f"{x:.2f}"


def _report(res, args) -> None:
    b, f = res.baseline, res.filtered

    print(f"\n{'':<22}{'baseline':>12}{'filtered':>12}{'delta':>12}")
    print("-" * 58)
    rows = [
        ("trades", f"{b.n_trades}", f"{f.n_trades}", f"{f.n_trades - b.n_trades:+d}"),
        ("win rate", f"{b.win_rate*100:.0f}%", f"{f.win_rate*100:.0f}%",
         f"{(f.win_rate - b.win_rate)*100:+.0f}pp"),
        ("expectancy/trade", _pct(b.expectancy), _pct(f.expectancy),
         _pct(f.expectancy - b.expectancy)),
        ("profit factor", _pf(b.profit_factor), _pf(f.profit_factor), ""),
        ("total return", _pct(b.total_return), _pct(f.total_return),
         _pct(f.total_return - b.total_return)),
        ("max drawdown", f"{b.max_drawdown*100:.0f}%", f"{f.max_drawdown*100:.0f}%",
         f"{(f.max_drawdown - b.max_drawdown)*100:+.0f}pp"),
        ("Sharpe (ann)", f"{b.sharpe:.2f}", f"{f.sharpe:.2f}", f"{f.sharpe - b.sharpe:+.2f}"),
    ]
    for label, bv, fv, dv in rows:
        print(f"{label:<22}{bv:>12}{fv:>12}{dv:>12}")

    rate = f" ({len(res.windows)/res.n_consulted*100:.0f}%)" if res.n_consulted else ""
    print(f"\nfilter activity: {res.n_consulted} entry decisions consulted, "
          f"{len(res.windows)} entries blocked{rate} "
          f"({len(res.vetoes)} veto-days incl. daily retries)")
    print(f"  LLM calls: {res.n_scored} | from cache: {res.n_cached} "
          f"| no headlines: {res.n_no_headlines} | errors (failed open): {res.n_errors}")

    # --- what the blocks landed on ----------------------------------------------
    matched, killed, avoided = res.matched, res.killed_winners, res.avoided_losers
    print(f"\nwhat the blocks landed on ({len(matched)} of {len(res.windows)} priceable):")
    if matched:
        delayed = [v for v in matched if v.reentered]
        print(f"  on a losing trade : {len(avoided):>3}  <- the filter working")
        print(f"  on a winning trade: {len(killed):>3}")
        print(f"  block precision   : {res.block_precision*100:.0f}% "
              f"(vs ~{(1 - res.baseline.win_rate)*100:.0f}% for blocking at random)")
        print(f"  re-entered later  : {len(delayed):>3} of {len(matched)} — a veto "
              f"usually delays an entry rather than cancelling it")
    else:
        print("  none — no block lined up with a baseline trade.")

    if killed:
        print("\n  biggest winners blocked (return of the trade, had it run):")
        for v in sorted(killed, key=lambda x: -x.counterfactual_ret)[:5]:
            tag = "re-entered later" if v.reentered else "leg abandoned"
            print(f"    {v.symbol:<6} {str(v.decision_ts.date()):<12} "
                  f"{_pct(v.counterfactual_ret):>9}  {tag:<17} {v.reason[:44]}")
        print("\n  These are NOT a realized cost — most were re-entered. The filter's"
              "\n  actual price is the expectancy/return delta in the table above.")

    # --- verdict ----------------------------------------------------------------
    print()
    if not res.vetoes:
        print("VERDICT: the filter never fired. It is neither helping nor hurting; "
              "either the thresholds are unreachable or the news window is empty.")
    elif f.expectancy < b.expectancy:
        print(f"VERDICT: the filter DESTROYS edge — expectancy "
              f"{_pct(b.expectancy)} -> {_pct(f.expectancy)}. It is vetoing trades "
              f"from a positive-expectancy distribution. Consider disabling it or "
              f"raising bearish_confidence_min.")
    elif f.max_drawdown < b.max_drawdown and f.expectancy >= b.expectancy:
        print("VERDICT: the filter earns its veto — same-or-better expectancy at "
              "lower drawdown.")
    else:
        print("VERDICT: the filter is roughly neutral on expectancy. Judge it on "
              "drawdown and on whether the block precision above beats a coin flip.")

    print("\nCAVEAT: the model has training knowledge of what happened after these "
          "dates, which biases the filter toward looking good. Read a poor result as "
          "a lower bound on how poor it is; do not read a good result as proof.\n")


def _to_json(res, args, cfg) -> dict:
    return {
        "config": {
            "days": args.days, "symbols": len(cfg.watchlist.symbols),
            "model": cfg.settings.trader_model, "prompt_version": PROMPT_VERSION,
            "lookback_hours": cfg.strategy.news.lookback_hours,
            "bearish_confidence_min": cfg.strategy.news.bearish_confidence_min,
            "block_on_imminent_earnings": cfg.strategy.news.block_on_imminent_earnings,
        },
        "baseline": res.baseline.__dict__,
        "filtered": res.filtered.__dict__,
        "activity": {
            "consulted": res.n_consulted, "blocked_entries": len(res.windows),
            "veto_days": len(res.vetoes), "llm_calls": res.n_scored,
            "cached": res.n_cached, "no_headlines": res.n_no_headlines,
            "errors": res.n_errors,
        },
        "attribution": {
            "priceable": len(res.matched),
            "killed_winners": len(res.killed_winners),
            "avoided_losers": len(res.avoided_losers),
            "block_precision": res.block_precision,
        },
        "vetoes": [
            {"symbol": v.symbol, "date": str(v.decision_ts.date()), "label": v.label,
             "confidence": v.confidence, "earnings_imminent": v.earnings_imminent,
             "reason": v.reason, "n_headlines": v.n_headlines,
             "counterfactual_ret": v.counterfactual_ret, "reentered": v.reentered}
            for v in res.windows  # retries omitted; one row per blocked entry
        ],
    }


async def _main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=1500, help="history window")
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--symbols", default="", help="comma-separated subset of the watchlist")
    ap.add_argument("--cache", type=Path, default=Path(".eval-cache/news-scores.json"))
    ap.add_argument("--out", type=Path, default=None, help="write the result as JSON")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel scorers (the Agent SDK spawns a subprocess each)")
    ap.add_argument("--close-utc-hour", type=int, default=DEFAULT_CLOSE_UTC_HOUR,
                    help="session close in UTC — bounds the no-lookahead news window")
    ap.add_argument("--plan", action="store_true",
                    help="report how many decisions need scoring, then exit")
    args = ap.parse_args()

    cfg = load_config()
    md = MarketData(cfg)
    tp, np_ = cfg.strategy.trend, cfg.strategy.news

    wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    symbols = [s for s in cfg.watchlist.symbols if not wanted or s.symbol in wanted]

    print(f"\n=== NEWS FILTER EVAL: {cfg.settings.agent_name} | {args.days}d "
          f"| {len(symbols)} symbols | model {cfg.settings.trader_model} "
          f"| prompt {PROMPT_VERSION} ===")
    print(f"veto rule: bearish>={np_.bearish_confidence_min} "
          f"| earnings={'block' if np_.block_on_imminent_earnings else 'allow'} "
          f"| lookback {np_.lookback_hours}h ending at {args.close_utc_hour:02d}:00Z\n")

    bars_by_symbol = {}
    for symcfg in symbols:
        try:
            df = md.get_history(symcfg, args.days, timeframe="1Day")
        except Exception as exc:  # noqa: BLE001
            print(f"  {symcfg.symbol:<8} data error: {exc}")
            continue
        if len(df) >= tp.trend_ma + 2:
            bars_by_symbol[symcfg.symbol] = df
        else:
            print(f"  {symcfg.symbol:<8} insufficient bars ({len(df)})")

    if not bars_by_symbol:
        print("no usable history — aborting.")
        return

    # `NewsSentiment` is built without a cache: the eval uses its own point-in-time
    # disk cache, and only the (cache-free) `fetch_window` / `should_block` methods
    # are used here. Reusing them means the eval tests the production veto rule.
    ns = NewsSentiment(cfg, cache=None)
    score_cache = ScoreCache(args.cache)

    if args.plan:
        slip = args.slippage_bps / 10000.0
        low = {(s, str(ts.date()))
               for s, ts in probe_decision_bars(bars_by_symbol, tp, slip, block=False)}
        high = {(s, str(ts.date()))
                for s, ts in probe_decision_bars(bars_by_symbol, tp, slip, block=True)}
        uncached = sum(
            1 for s, day in low
            if score_cache.get(s, day, np_.lookback_hours, PROMPT_VERSION,
                               cfg.settings.trader_model) is None
        )
        print(f"PLAN: {len(low)} decision-days if the filter never blocks "
              f"({uncached} not yet cached; {len(score_cache)} verdicts on disk).")
        print(f"      Each block adds daily retries, up to {len(high)} in the "
              f"pathological case where everything is blocked.\n")
        return

    def progress(n: int) -> None:
        print(f"  up to ~{n} decision-days to score (cached ones are free)...")

    res = await replay(
        bars_by_symbol,
        trend_params=tp, news_params=np_,
        should_block=ns.should_block, fetch_window=ns.fetch_window,
        cache=score_cache, model=cfg.settings.trader_model,
        slippage_pct=args.slippage_bps / 10000.0,
        position_fraction=cfg.risk.max_position_pct,
        close_hour=args.close_utc_hour, concurrency=args.concurrency,
        progress=progress,
    )
    score_cache.save()
    _report(res, args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_to_json(res, args, cfg), indent=2, default=str),
                            encoding="utf-8")
        print(f"wrote {args.out}\n")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
