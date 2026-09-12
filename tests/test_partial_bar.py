"""The live engine must evaluate the channel signal on completed sessions only.

Regression test for the divergence found in production: with the session open,
Alpaca's last 1Day bar is today's partial bar, so `close.iloc[-1]` was the live
intraday price. An intraday dip under the exit channel produced a `trend_break`
exit the backtester never takes, and the next tick re-entered once price
recovered — live traded ~2.75x the backtested rate, with same-day sell/buy pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.backtest.trend import desired_long
from trader.trend_trader import completed_bars


def _uptrend(n: int = 200, last_close: float | None = None) -> pd.DataFrame:
    """A clean rising series; optionally override the final close."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100.0, 300.0, n)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )
    if last_close is not None:
        df.iloc[-1, df.columns.get_loc("close")] = last_close
        df.iloc[-1, df.columns.get_loc("low")] = min(last_close, df["low"].iloc[-1])
    return df


def test_drops_todays_bar_while_session_open():
    df = _uptrend()
    df.index = df.index + (pd.Timestamp.now(tz="UTC").normalize() - df.index[-1])
    out = completed_bars(df, market_open=True)
    assert len(out) == len(df) - 1
    assert out.index[-1] < pd.Timestamp.now(tz="UTC").normalize()


def test_keeps_every_bar_once_session_closed():
    df = _uptrend()
    df.index = df.index + (pd.Timestamp.now(tz="UTC").normalize() - df.index[-1])
    assert len(completed_bars(df, market_open=False)) == len(df)


def test_keeps_bars_when_last_bar_is_historical():
    df = _uptrend()  # ends 2025, nowhere near today
    assert len(completed_bars(df, market_open=True)) == len(df)


def test_empty_frame_is_passed_through():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert completed_bars(empty, market_open=True).empty


def test_intraday_dip_no_longer_flips_the_signal():
    """The actual production bug, reproduced then fixed.

    A held uptrend whose live price has dipped below the prior 20-day low. Read
    raw, the signal says exit. Read on completed bars, it correctly stays long.
    """
    df = _uptrend()
    df.index = df.index + (pd.Timestamp.now(tz="UTC").normalize() - df.index[-1])
    prior_low = df["low"].iloc[-21:-1].min()

    dipped = df.copy()
    dipped.iloc[-1, dipped.columns.get_loc("close")] = prior_low * 0.98

    # the bug: partial bar drives the channel read
    assert desired_long(dipped, 50, 20, 100, True) is False
    # the fix: completed sessions only
    assert desired_long(completed_bars(dipped, market_open=True), 50, 20, 100, True) is True


def test_signal_is_identical_to_backtest_on_closed_data():
    """Live and simulated must agree bar-for-bar once the session is closed —
    that equivalence is the whole point of sharing `desired_long`."""
    df = _uptrend()
    for cut in range(150, len(df)):
        window = df.iloc[:cut]
        assert desired_long(window, 50, 20, 100, True) == desired_long(
            completed_bars(window, market_open=False), 50, 20, 100, True
        )
