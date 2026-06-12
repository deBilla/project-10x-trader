"""Walk-forward bar simulation. No lookahead: a signal computed at bar i's close is
acted on at bar i+1's open; stops/targets are checked intrabar on the high/low.
Costs are modeled as a round-turn fraction (slippage both sides; commission 0 for
Alpaca). Produces a list of closed trades with net % returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import RiskConfig, StrategyConfig
from ..indicators import compute_snapshot
from .signals import signal

_WINDOW = 120  # trailing bars handed to compute_snapshot (bounds per-bar cost)


@dataclass
class Trade:
    symbol: str
    direction: int      # +1 long, -1 short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    ret_pct: float      # net of costs, direction-adjusted
    reason: str         # stop | target | signal | eod
    bars_held: int


def backtest_symbol(
    symbol: str,
    bars: pd.DataFrame,
    strategy: StrategyConfig,
    risk: RiskConfig,
    slippage_pct: float = 0.0005,
) -> list[Trade]:
    """Simulate the strategy on one symbol's OHLCV history (oldest -> newest)."""
    bars = bars.sort_index()
    n = len(bars)
    sl = strategy.bracket.stop_loss_pct
    tp = strategy.bracket.take_profit_pct
    cost = 2 * slippage_pct  # round-turn
    long_only = risk.long_only

    # Minimum bars before indicators are valid.
    warmup = max(
        strategy.indicators.macd_slow + strategy.indicators.macd_signal,
        strategy.breakout.channel_period + 1,
        strategy.indicators.bbands_period,
        strategy.indicators.ema_slow,
        strategy.indicators.rsi_period + 1,
    ) + 2
    if n <= warmup + 2:
        return []

    # Pass 1: signal at each bar's close (bounded trailing window).
    sigs = [0] * n
    for i in range(warmup, n):
        window = bars.iloc[max(0, i - _WINDOW): i + 1]
        try:
            snap = compute_snapshot(symbol, window, strategy.indicators,
                                    strategy.thresholds, strategy.breakout)
        except ValueError:
            continue
        sigs[i] = signal(snap, strategy)

    # Pass 2: simulate entries/exits.
    trades: list[Trade] = []
    pos = None  # dict: direction, entry, entry_i
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    idx = bars.index

    for i in range(warmup + 1, n):
        if pos is not None and i > pos["entry_i"]:
            d, e = pos["direction"], pos["entry"]
            stop = e * (1 - sl) if d == 1 else e * (1 + sl)
            tgt = e * (1 + tp) if d == 1 else e * (1 - tp)
            exit_price = reason = None
            # Signal-flip exit happens at the open (chronologically first).
            if sigs[i - 1] != d:
                exit_price, reason = o[i], "signal"
            elif d == 1:
                if low[i] <= stop: exit_price, reason = stop, "stop"
                elif h[i] >= tgt: exit_price, reason = tgt, "target"
            else:
                if h[i] >= stop: exit_price, reason = stop, "stop"
                elif low[i] <= tgt: exit_price, reason = tgt, "target"
            if exit_price is not None:
                gross = (exit_price - e) / e * d
                trades.append(Trade(symbol, d, idx[pos["entry_i"]], idx[i], e,
                                    float(exit_price), gross - cost, reason,
                                    i - pos["entry_i"]))
                pos = None

        if pos is None:
            s = sigs[i - 1]
            if s == 1 or (s == -1 and not long_only):
                pos = {"direction": s, "entry": float(o[i]), "entry_i": i}

    # Close any open position at the final bar (mark-to-market).
    if pos is not None:
        d, e = pos["direction"], pos["entry"]
        gross = (c[-1] - e) / e * d
        trades.append(Trade(symbol, d, idx[pos["entry_i"]], idx[-1], e, float(c[-1]),
                            gross - cost, "eod", n - 1 - pos["entry_i"]))
    return trades
