"""Deterministic encoding of the strategy rules the system prompt gives Claude.

This is the *testable hypothesis*: the same mean-reversion + breakout logic, trend
filter, volume gate, and entry mode — expressed as code so it can be replayed over
history. The live LLM is instructed to follow exactly these rules, so backtesting
them measures the strategy's edge (the LLM may deviate, but the rules are the bet).

`signal(snapshot, strategy)` returns +1 (long), -1 (short), or 0 (flat).
"""

from __future__ import annotations

from ..config import StrategyConfig

# Bollinger position thresholds for "at the band" (bb_pct: 0=lower band, 1=upper).
_BB_LOW = 0.20
_BB_HIGH = 0.80


def signal(s: dict, strategy: StrategyConfig) -> int:
    rsi = s["rsi"]
    bo = strategy.breakout
    longs: list[str] = []
    shorts: list[str] = []

    # --- Strategy A: mean reversion (RSI extreme + Bollinger band touch) ---
    if rsi <= s["rsi_oversold"] and s["bb_pct"] <= _BB_LOW:
        longs.append("meanrev")
    if rsi >= s["rsi_overbought"] and s["bb_pct"] >= _BB_HIGH:
        shorts.append("meanrev")

    # --- Strategy B: momentum / breakout (Donchian break + trend + RSI band) ---
    if bo.enabled:
        if (
            s.get("breaking_high")
            and s["ema_trend"] == "up"
            and s["price_vs_vwap"] == "above"
            and bo.min_rsi_long <= rsi <= bo.max_rsi_long
        ):
            longs.append("breakout")
        if (
            s.get("breaking_low")
            and s["ema_trend"] == "down"
            and s["price_vs_vwap"] == "below"
            and bo.min_rsi_short <= rsi <= bo.max_rsi_short
        ):
            shorts.append("breakout")

    # --- Trend filter (applies to mean-reversion only; breakout self-aligns) ---
    if strategy.enforce_trend_filter:
        strong_down = s["ema_trend"] == "down" and s["price_vs_vwap"] == "below"
        strong_up = s["ema_trend"] == "up" and s["price_vs_vwap"] == "above"
        if strong_down:
            longs = [x for x in longs if x != "meanrev"]
        if strong_up:
            shorts = [x for x in shorts if x != "meanrev"]

    # --- Volume confirmation (shared) ---
    if strategy.require_volume_confirmation and not s.get("volume_surge"):
        longs, shorts = [], []

    # --- Confluence mode additionally needs MACD agreement ---
    if strategy.entry_mode == "confluence":
        if s["macd_hist"] <= 0:
            longs = []
        if s["macd_hist"] >= 0:
            shorts = []

    if longs and not shorts:
        return 1
    if shorts and not longs:
        return -1
    return 0
