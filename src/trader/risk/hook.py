"""In-process risk gate implemented as a Claude Agent SDK PreToolUse hook.

Every order tool call (``place_*`` / ``close_*``) passes through ``risk_hook``
before it reaches the Alpaca MCP server. The hook can DENY the call, which stops
it from ever hitting the Alpaca API. This enforces hard limits the LLM cannot
override: symbol whitelist, max position size, max open positions, mandatory
bracket stop-loss, and a daily-drawdown pause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..config import RiskConfig, WatchlistConfig


@dataclass
class RiskContext:
    """Fresh per-tick state the hook needs to make decisions."""

    equity: float
    open_position_symbols: list[str]
    latest_prices: dict[str, float]
    paused: bool = False  # True after a daily-drawdown breach


@dataclass
class RiskDecision:
    """Recorded outcome of a single gate evaluation (for journaling)."""

    tool_name: str
    decision: str  # "allow" | "deny"
    reason: str
    tool_input: dict = field(default_factory=dict)


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _allow() -> dict:
    # An empty dict lets the SDK fall through to normal permission handling.
    return {}


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_symbol(ti: dict) -> str | None:
    return ti.get("symbol") or ti.get("symbol_or_symbols") or ti.get("ticker")


def _extract_stop_loss(ti: dict) -> float | None:
    """Find a bracket stop-loss price across the param shapes Alpaca/MCP may use.

    The official alpaca-mcp `place_stock_order` uses flat bracket keys
    (`stop_loss_stop_price`, `stop_loss_limit_price`); older/nested shapes use
    `stop_loss` (scalar or {stop_price/price}). We accept all of them.
    """
    for key in (
        "stop_loss_stop_price",
        "stop_loss_price",
        "stop_loss_limit_price",
        "stop_loss",
    ):
        v = ti.get(key)
        if isinstance(v, dict):
            v = v.get("stop_price") or v.get("price")
        f = _as_float(v)
        if f is not None:
            return f
    return None


def _extract_notional(ti: dict, price: float | None) -> float | None:
    """Resolve order notional from explicit notional or quantity * price."""
    notional = _as_float(ti.get("notional"))
    if notional is not None:
        return notional
    qty = _as_float(ti.get("quantity") or ti.get("qty"))
    if qty is not None and price is not None:
        return qty * price
    return None


def evaluate_order(
    tool_name: str,
    tool_input: dict,
    ctx: RiskContext,
    risk: RiskConfig,
    whitelist: set[str],
) -> RiskDecision:
    """Pure decision logic (no SDK types) — directly unit-testable."""
    bare = tool_name.split("__")[-1]

    # Risk-reducing operations are always permitted.
    if bare in {"close_position", "close_all_positions", "cancel_all_orders"}:
        return RiskDecision(tool_name, "allow", "risk-reducing operation", tool_input)

    # From here on we are gating an ENTRY order (place_*).
    if ctx.paused:
        return RiskDecision(
            tool_name, "deny",
            "daily drawdown limit hit — trading paused until next session",
            tool_input,
        )

    # Long-only mode (e.g. spot crypto can't be shorted): block sell entries.
    # Position exits are handled via close_position, not a sell order.
    if risk.long_only and str(tool_input.get("side", "")).lower() == "sell":
        return RiskDecision(
            tool_name, "deny", "long-only mode: short/sell entries not allowed",
            tool_input,
        )

    symbol = _extract_symbol(tool_input)
    if symbol is None:
        return RiskDecision(tool_name, "deny", "order missing symbol", tool_input)
    if symbol not in whitelist:
        return RiskDecision(
            tool_name, "deny", f"symbol {symbol} not in whitelist", tool_input
        )

    price = ctx.latest_prices.get(symbol)

    # Mandatory bracket stop-loss.
    stop = _extract_stop_loss(tool_input)
    if risk.require_stop_loss and stop is None:
        return RiskDecision(
            tool_name, "deny", "entry order has no bracket stop_loss", tool_input
        )
    if stop is not None and price:
        dist = abs(price - stop) / price
        if dist > risk.max_stop_distance_pct:
            return RiskDecision(
                tool_name, "deny",
                f"stop distance {dist:.2%} exceeds max {risk.max_stop_distance_pct:.2%}",
                tool_input,
            )

    # Open-position cap (a new symbol would add a position).
    if symbol not in ctx.open_position_symbols:
        if len(ctx.open_position_symbols) >= risk.max_open_positions:
            return RiskDecision(
                tool_name, "deny",
                f"max open positions ({risk.max_open_positions}) reached",
                tool_input,
            )

    # Position-size cap.
    notional = _extract_notional(tool_input, price)
    if notional is None:
        return RiskDecision(
            tool_name, "deny",
            "cannot determine order notional (no notional/quantity or price)",
            tool_input,
        )
    cap = risk.max_position_pct * ctx.equity
    if notional > cap:
        return RiskDecision(
            tool_name, "deny",
            f"notional ${notional:,.2f} exceeds cap ${cap:,.2f} "
            f"({risk.max_position_pct:.0%} of ${ctx.equity:,.2f} equity)",
            tool_input,
        )

    return RiskDecision(tool_name, "allow", "within risk limits", tool_input)


def make_risk_hook(
    risk: RiskConfig,
    watchlist: WatchlistConfig,
    context_provider: Callable[[], RiskContext],
    on_decision: Callable[[RiskDecision], None] | None = None,
):
    """Build the async PreToolUse hook closure registered in ClaudeAgentOptions.

    ``context_provider`` returns the live RiskContext for the current tick.
    ``on_decision`` (optional) records each decision for journaling.
    """
    whitelist = watchlist.whitelist

    async def risk_hook(input_data, tool_use_id, context):  # SDK hook signature
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {}) or {}

        decision = evaluate_order(
            tool_name, tool_input, context_provider(), risk, whitelist
        )
        if on_decision is not None:
            on_decision(decision)

        if decision.decision == "deny":
            return _deny(decision.reason)
        return _allow()

    return risk_hook
