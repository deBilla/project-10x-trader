"""System prompt (strategy rules + contract) and per-tick user message builder."""

from __future__ import annotations

import json

from .config import AppConfig
from .market_data import AccountSnapshot

SYSTEM_TEMPLATE = """\
You are a disciplined intraday trading agent operating a PAPER account on Alpaca.
You trade only the symbols in the provided snapshot. You act once per tick.

You have TWO complementary strategies. Each tick, for each symbol, check if EITHER
fires. They are mutually exclusive — never long and short the same symbol.

STRATEGY A — MEAN REVERSION (best in ranging markets):
- Primary trigger is RSI({rsi_period}):
    * RSI <= {rsi_oversold} => long bias (oversold).
    * RSI >= {rsi_overbought} => short bias (overbought).
- Confirm with Bollinger Bands: bb_pct near 0 (at/below lower band) supports a
  long; near 1 (at/above upper band) supports a short.
{trend_filter}
STRATEGY B — MOMENTUM / BREAKOUT (best in trending markets — trade WITH the trend):
{breakout_block}
{entry_mode_block}SHARED RULES (apply to BOTH strategies):
{volume_rule}
- MACD is a secondary confirmation only (macd_hist sign agreeing with your bias).
- When there is genuinely no usable signal, HOLD.

EXECUTION CONTRACT:
{execution_contract}
- Position sizing: propose a notional you believe is reasonable, but know that a
  hard risk gate caps any single position at {max_position_pct:.0%} of account
  equity and will DENY oversized or non-whitelisted orders. If a call is denied,
  do not retry the same violating order.
- Never place an order for a symbol not present in the snapshot.

Think step by step about each symbol, state your decision and the indicators that
justify it, then issue at most one order per symbol this tick.
"""


_TREND_FILTER_BLOCK = """\
- Trend filter (avoid fighting strong trends):
    * Take LONGS only when NOT in a strong downtrend
      (prefer price >= vwap OR ema_trend == "up").
    * Take SHORTS only when NOT in a strong uptrend
      (prefer price <= vwap OR ema_trend == "down").
"""

# Bracketed entries (equities) vs long-only no-bracket (spot crypto).
_CONTRACT_BRACKET = """\
- To enter, call place_stock_order (equities) or place_crypto_order (crypto) as a
  BRACKET: always include a stop_loss (~{stop_loss_pct:.1%} from entry) and a
  take_profit (~{take_profit_pct:.1%} from entry, ~2:1 reward:risk).
- To exit an existing position, call close_position.
- The risk gate DENIES un-bracketed entries."""

_CONTRACT_LONG_ONLY = """\
- LONG ONLY, no brackets. Enter with the matching tool — place_stock_order for
  equities, place_crypto_order for crypto — side="buy", time_in_force "day". For a
  small account use a NOTIONAL dollar amount (fractional shares), not whole-share
  qty. Do NOT attach a bracket; sell/short entries are DENIED by the risk gate.
- You do NOT manage stops: a deterministic software stop/target (entry
  -{stop_loss_pct:.1%} / +{take_profit_pct:.1%}) runs outside you each tick and
  auto-closes positions. You may also close early via close_position on reversal."""

_BREAKOUT_TEMPLATE = """\
- LONG breakout: breaking_high is true (price clears the prior {channel}-bar high)
  AND ema_trend == "up" AND price >= vwap, with RSI between {min_rsi_long} and
  {max_rsi_long} (momentum, not a blow-off top). macd_hist > 0 confirms.
- SHORT breakdown: breaking_low is true (price breaks the prior {channel}-bar low)
  AND ema_trend == "down" AND price <= vwap, with RSI between {min_rsi_short} and
  {max_rsi_short}. macd_hist < 0 confirms.
- Do NOT chase: skip if price has already run far beyond the channel level."""

_BREAKOUT_DISABLED = "- (disabled)"


_ENTRY_MODE_ACTIVE = """\
ENTRY MODE — ACTIVE: a SINGLE qualifying primary signal from EITHER strategy is
enough to enter (you do NOT need both strategies or full confirmation). Lean
toward taking trades when a clear signal is present; you may run several positions
across the watchlist. The hard risk gate still caps size and count.

"""


def _volume_rule(st) -> str:
    if st.require_volume_confirmation:
        return (
            "- Require volume confirmation: only act when volume_surge is true "
            f"(current volume >= {st.thresholds.volume_surge_mult}x its average)."
        )
    return "- Volume surge is a plus, not a requirement — you may act without it."


def build_system_prompt(cfg: AppConfig) -> str:
    st = cfg.strategy
    bo = st.breakout
    breakout_block = (
        _BREAKOUT_TEMPLATE.format(
            channel=bo.channel_period,
            min_rsi_long=bo.min_rsi_long, max_rsi_long=bo.max_rsi_long,
            min_rsi_short=bo.min_rsi_short, max_rsi_short=bo.max_rsi_short,
        )
        if bo.enabled
        else _BREAKOUT_DISABLED
    )
    contract_tmpl = _CONTRACT_LONG_ONLY if cfg.risk.long_only else _CONTRACT_BRACKET
    execution_contract = contract_tmpl.format(
        stop_loss_pct=st.bracket.stop_loss_pct,
        take_profit_pct=st.bracket.take_profit_pct,
    )
    return SYSTEM_TEMPLATE.format(
        rsi_period=st.indicators.rsi_period,
        rsi_oversold=st.thresholds.rsi_oversold,
        rsi_overbought=st.thresholds.rsi_overbought,
        max_position_pct=cfg.risk.max_position_pct,
        trend_filter=_TREND_FILTER_BLOCK if st.enforce_trend_filter else "",
        breakout_block=breakout_block,
        execution_contract=execution_contract,
        entry_mode_block=_ENTRY_MODE_ACTIVE if st.entry_mode == "single_signal" else "",
        volume_rule=_volume_rule(st),
    )


def build_user_message(
    account: AccountSnapshot,
    snapshots: list[dict],
) -> str:
    """The per-tick payload: account state + the numeric indicator snapshot."""
    payload = {
        "account": {
            "equity": round(account.equity, 2),
            "cash": round(account.cash, 2),
            "buying_power": round(account.buying_power, 2),
            "open_positions": account.open_position_symbols,
        },
        "market": snapshots,
        "instruction": (
            "Evaluate each symbol against the strategy. Place bracket orders for "
            "valid setups, close positions that should exit, otherwise hold."
        ),
    }
    return "MARKET SNAPSHOT:\n" + json.dumps(payload, indent=2)
