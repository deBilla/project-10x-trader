"""Backtest runner: fetch historical bars for the configured profile, replay the
strategy, and print an edge report (per-symbol + portfolio, vs buy-and-hold).

Usage:
    python -m trader.backtest.run [--days 60] [--slippage-bps 5]
The profile is selected by TRADER_CONFIG_DIR (default config/, e.g. config/equity).
"""

from __future__ import annotations

import argparse
import logging

from ..config import load_config
from ..market_data import MarketData
from .engine import Trade, backtest_symbol
from .metrics import compute_metrics
from .trend import backtest_trend

log = logging.getLogger("trader.backtest")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["meanrev", "trend"], default="meanrev",
                    help="meanrev = intraday mean-reversion/breakout; trend = daily trend-following")
    ap.add_argument("--days", type=int, default=0, help="history window (days; 0 = mode default)")
    ap.add_argument("--slippage-bps", type=float, default=5.0, help="per-side slippage")
    ap.add_argument("--entry-channel", type=int, default=50)
    ap.add_argument("--exit-channel", type=int, default=20)
    ap.add_argument("--trend-ma", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    md = MarketData(cfg)
    slip = args.slippage_bps / 10000.0
    pos_frac = cfg.risk.max_position_pct
    trend = args.mode == "trend"
    days = args.days or (750 if trend else 60)

    print(f"\n=== BACKTEST [{args.mode}]: {cfg.settings.agent_name} profile "
          f"| {days}d | {len(cfg.watchlist.symbols)} symbols "
          f"| slippage {args.slippage_bps}bps/side | sizing {pos_frac:.0%} ===")
    if trend:
        print(f"strategy: DAILY Donchian trend — entry={args.entry_channel}d high, "
              f"exit={args.exit_channel}d low, regime filter SMA{args.trend_ma}, long-only, "
              f"ride winners (no fixed target)\n")
    else:
        print(f"strategy: entry_mode={cfg.strategy.entry_mode} trend_filter="
              f"{cfg.strategy.enforce_trend_filter} vol_req={cfg.strategy.require_volume_confirmation} "
              f"long_only={cfg.risk.long_only} stop/target={cfg.strategy.bracket.stop_loss_pct:.0%}"
              f"/{cfg.strategy.bracket.take_profit_pct:.0%}\n")

    hdr = f"{'symbol':<8}{'trades':>7}{'win%':>7}{'expR':>9}{'PF':>6}{'totRet':>9}{'maxDD':>8}{'B&H':>9}"
    print(hdr); print("-" * len(hdr))

    all_trades: list[Trade] = []
    bh_returns = []
    for sym in cfg.watchlist.symbols:
        try:
            df = md.get_history(sym, days, timeframe="1Day" if trend else None)
        except Exception as exc:  # noqa: BLE001
            print(f"{sym.symbol:<8}  data error: {exc}")
            continue
        if len(df) < 50:
            print(f"{sym.symbol:<8}  insufficient bars ({len(df)})")
            continue
        if trend:
            trades = backtest_trend(sym.symbol, df, args.entry_channel,
                                    args.exit_channel, args.trend_ma, True, 0.20, slip)
        else:
            trades = backtest_symbol(sym.symbol, df, cfg.strategy, cfg.risk, slip)
        all_trades.extend(trades)
        m = compute_metrics(trades, pos_frac)
        bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        bh_returns.append(bh)
        pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        print(f"{sym.symbol:<8}{m.n_trades:>7}{m.win_rate*100:>6.0f}%"
              f"{_fmt_pct(m.expectancy):>9}{pf:>6}{_fmt_pct(m.total_return):>9}"
              f"{m.max_drawdown*100:>7.0f}%{_fmt_pct(bh):>9}")

    print("-" * len(hdr))
    port = compute_metrics(all_trades, pos_frac)
    bh_avg = sum(bh_returns) / len(bh_returns) if bh_returns else 0.0
    pf = "inf" if port.profit_factor == float("inf") else f"{port.profit_factor:.2f}"
    print(f"{'ALL':<8}{port.n_trades:>7}{port.win_rate*100:>6.0f}%"
          f"{_fmt_pct(port.expectancy):>9}{pf:>6}{_fmt_pct(port.total_return):>9}"
          f"{port.max_drawdown*100:>7.0f}%{_fmt_pct(bh_avg):>9}")

    print(f"\nPortfolio edge: expectancy/trade {_fmt_pct(port.expectancy)} | "
          f"win {port.win_rate*100:.0f}% | profit factor {pf} | "
          f"Sharpe(ann) {port.sharpe:.2f} | avg hold {port.avg_bars_held:.0f} bars")
    verdict = ("POSITIVE edge (after modeled costs)" if port.expectancy > 0
               else "NO edge — negative expectancy after costs")
    print(f"VERDICT: {verdict}. Compare totRet vs B&H before trusting it.\n")


if __name__ == "__main__":
    main()
