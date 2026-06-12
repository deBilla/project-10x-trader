"""Daily trend-following backtest (classic Donchian/Turtle-style).

Rationale: retail can't win the intraday/speed/order-flow game, but trend-following
on daily bars is a publicly documented, durable edge that needs no speed or order-flow
data. Rules:
  - ENTRY (long): close breaks above the prior `entry_channel`-day high
    AND (optional regime filter) close is above its `trend_ma`-day SMA.
  - EXIT: close breaks below the prior `exit_channel`-day low (trailing), OR a wide
    catastrophic stop is hit intrabar.
  - LONG ONLY. Winners are ridden (no fixed take-profit) — the whole point.
No lookahead: decisions use the bar's close, executed at the next bar's open.
"""

from __future__ import annotations

import pandas as pd

from .engine import Trade


def desired_long(
    bars: pd.DataFrame,
    entry_channel: int = 50,
    exit_channel: int = 20,
    trend_ma: int = 100,
    use_regime_filter: bool = True,
) -> bool:
    """Live signal: should we currently hold a long? True if the most recent
    breakout above the prior `entry_channel` high is more recent than the most
    recent break below the prior `exit_channel` low, and price is above its SMA.
    A faithful, replay-free read of the same trend state the backtest simulates.
    """
    bars = bars.sort_index()
    if len(bars) < max(entry_channel, exit_channel, trend_ma) + 2:
        return False
    prior_high = bars["high"].rolling(entry_channel).max().shift(1)
    prior_low = bars["low"].rolling(exit_channel).min().shift(1)
    close = bars["close"]
    entries = close > prior_high
    exits = close < prior_low

    last_entry = entries[entries].index.max() if entries.any() else None
    last_exit = exits[exits].index.max() if exits.any() else None
    if last_entry is None:
        return False
    in_trend = last_exit is None or last_entry > last_exit
    if not in_trend:
        return False
    if use_regime_filter:
        sma = close.rolling(trend_ma).mean().iloc[-1]
        if close.iloc[-1] <= sma:
            return False
    return True


def backtest_trend(
    symbol: str,
    bars: pd.DataFrame,
    entry_channel: int = 50,
    exit_channel: int = 20,
    trend_ma: int = 100,
    use_regime_filter: bool = True,
    cat_stop_pct: float = 0.20,
    slippage_pct: float = 0.0005,
) -> list[Trade]:
    bars = bars.sort_index()
    n = len(bars)
    start = max(entry_channel, exit_channel, trend_ma if use_regime_filter else 0) + 1
    if n <= start + 2:
        return []

    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    open_ = bars["open"].to_numpy()
    idx = bars.index

    # Prior-window channels (shifted: exclude the current bar) + regime SMA.
    prior_high = bars["high"].rolling(entry_channel).max().shift(1).to_numpy()
    prior_low = bars["low"].rolling(exit_channel).min().shift(1).to_numpy()
    sma = bars["close"].rolling(trend_ma).mean().to_numpy()

    trades: list[Trade] = []
    pos = None
    pending = None  # "enter" | "exit"

    for i in range(start, n):
        # Execute the action decided on the prior close, at this bar's open.
        if pending == "enter" and pos is None:
            entry = open_[i] * (1 + slippage_pct)
            pos = {"entry": entry, "entry_i": i}
        elif pending == "exit" and pos is not None:
            ex = open_[i] * (1 - slippage_pct)
            e = pos["entry"]
            trades.append(Trade(symbol, 1, idx[pos["entry_i"]], idx[i], e, ex,
                                (ex - e) / e, "trend_exit", i - pos["entry_i"]))
            pos = None
        pending = None

        # Intrabar catastrophic stop.
        if pos is not None and i > pos["entry_i"]:
            stop = pos["entry"] * (1 - cat_stop_pct)
            if low[i] <= stop:
                ex = stop * (1 - slippage_pct)
                e = pos["entry"]
                trades.append(Trade(symbol, 1, idx[pos["entry_i"]], idx[i], e, ex,
                                    (ex - e) / e, "cat_stop", i - pos["entry_i"]))
                pos = None

        # Decide for the next bar.
        if pos is None:
            breakout = close[i] > prior_high[i]
            regime_ok = (not use_regime_filter) or (close[i] > sma[i])
            if breakout and regime_ok:
                pending = "enter"
        else:
            if close[i] < prior_low[i]:
                pending = "exit"

    if pos is not None:
        e = pos["entry"]
        trades.append(Trade(symbol, 1, idx[pos["entry_i"]], idx[-1], e, float(close[-1]),
                            (close[-1] - e) / e, "eod", n - 1 - pos["entry_i"]))
    return trades
