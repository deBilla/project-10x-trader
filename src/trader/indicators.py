"""Deterministic technical-indicator computation.

LLMs are unreliable at arithmetic over raw OHLCV, so indicators are computed here
in code and only the resulting numbers are handed to Claude. Implemented with plain
pandas/numpy (standard formulas) for reproducibility and stable golden tests.

Input: a DataFrame indexed by timestamp with columns
    open, high, low, close, volume
ordered oldest -> newest.
Output: a flat numeric snapshot dict for one symbol.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BreakoutParams, IndicatorParams, Thresholds


def _wilder_rsi(close: pd.Series, period: int) -> float:
    """Classic Wilder RSI of the most recent bar."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(close: pd.Series, period: int) -> float:
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def _bollinger(close: pd.Series, period: int, std_mult: float) -> tuple[float, float, float]:
    window = close.tail(period)
    mid = float(window.mean())
    sd = float(window.std(ddof=0))  # population std
    return mid - std_mult * sd, mid, mid + std_mult * sd


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[float, float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def _session_vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average price over the provided window (typical price)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    denom = float(vol.sum())
    if denom == 0:
        return float(df["close"].iloc[-1])
    return float((typical * vol).sum() / denom)


def compute_snapshot(
    symbol: str,
    df: pd.DataFrame,
    params: IndicatorParams,
    thresholds: Thresholds,
    breakout: BreakoutParams | None = None,
) -> dict:
    """Compute the per-symbol indicator snapshot fed to Claude.

    Raises ValueError if there are too few bars for the longest indicator window.
    """
    channel = breakout.channel_period if breakout else 0
    required = max(
        params.rsi_period + 1,
        params.bbands_period,
        params.ema_slow,
        params.macd_slow + params.macd_signal,
        params.volume_avg_period,
        channel + 1,
    )
    if len(df) < required:
        raise ValueError(
            f"{symbol}: need >= {required} bars, got {len(df)}"
        )

    df = df.sort_index()
    close = df["close"].astype(float)
    price = float(close.iloc[-1])

    rsi = _wilder_rsi(close, params.rsi_period)
    bb_lower, bb_mid, bb_upper = _bollinger(close, params.bbands_period, params.bbands_std)
    ema_fast = _ema(close, params.ema_fast)
    ema_slow = _ema(close, params.ema_slow)
    macd_line, macd_signal, macd_hist = _macd(
        close, params.macd_fast, params.macd_slow, params.macd_signal
    )
    vwap = _session_vwap(df)

    avg_vol = float(df["volume"].tail(params.volume_avg_period).mean())
    cur_vol = float(df["volume"].iloc[-1])
    vol_ratio = (cur_vol / avg_vol) if avg_vol else 0.0

    # Position of price within the Bollinger band: 0 = lower band, 1 = upper band.
    band_width = bb_upper - bb_lower
    bb_pct = (price - bb_lower) / band_width if band_width else 0.5

    # Donchian breakout: is the current price clearing the prior N-bar range?
    breakout_fields: dict = {}
    if breakout is not None:
        prior = df.iloc[-(channel + 1):-1]  # exclude the current bar
        ch_high = float(prior["high"].max())
        ch_low = float(prior["low"].min())
        breakout_fields = {
            "channel_high": round(ch_high, 4),
            "channel_low": round(ch_low, 4),
            "breaking_high": price > ch_high,   # bullish breakout
            "breaking_low": price < ch_low,     # bearish breakdown
        }

    return {
        "symbol": symbol,
        "price": round(price, 4),
        "rsi": round(rsi, 2),
        "rsi_oversold": thresholds.rsi_oversold,
        "rsi_overbought": thresholds.rsi_overbought,
        "bb_lower": round(bb_lower, 4),
        "bb_mid": round(bb_mid, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_pct": round(bb_pct, 3),
        "ema_fast": round(ema_fast, 4),
        "ema_slow": round(ema_slow, 4),
        "ema_trend": "up" if ema_fast >= ema_slow else "down",
        "vwap": round(vwap, 4),
        "price_vs_vwap": "above" if price >= vwap else "below",
        "macd": round(macd_line, 4),
        "macd_signal": round(macd_signal, 4),
        "macd_hist": round(macd_hist, 4),
        "volume": round(cur_vol, 2),
        "avg_volume": round(avg_vol, 2),
        "volume_ratio": round(vol_ratio, 3),
        "volume_surge": vol_ratio >= thresholds.volume_surge_mult,
        **breakout_fields,
    }


def bars_to_dataframe(bars: list[dict]) -> pd.DataFrame:
    """Normalize a list of Alpaca bar dicts into the OHLCV DataFrame shape.

    Accepts bars keyed either by full names (open/high/low/close/volume) or
    Alpaca's short keys (o/h/l/c/v) with a timestamp under t/timestamp.
    """
    rows = []
    index = []
    for b in bars:
        ts = b.get("timestamp") or b.get("t")
        index.append(pd.to_datetime(ts))
        rows.append(
            {
                "open": float(b.get("open", b.get("o", math.nan))),
                "high": float(b.get("high", b.get("h", math.nan))),
                "low": float(b.get("low", b.get("l", math.nan))),
                "close": float(b.get("close", b.get("c", math.nan))),
                "volume": float(b.get("volume", b.get("v", 0.0))),
            }
        )
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    return df.sort_index()
