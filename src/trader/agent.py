"""The ReAct tick: build context -> query Claude -> stream + journal the result.

One ``run_tick`` is a full Reasoning+Acting cycle for the whole watchlist. The
risk hook (in-process) gates any order Claude attempts mid-stream.
"""

from __future__ import annotations

import json
import logging
import re

from claude_agent_sdk import ClaudeSDKClient

from .config import AppConfig
from .indicators import compute_snapshot
from .market_data import MarketData
from .mcp_client import build_options
from .persistence.cache import Cache
from .persistence.journal import Journal
from .prompt import build_system_prompt, build_user_message
from .risk.drawdown import DrawdownMonitor
from .risk.hook import RiskContext, RiskDecision, make_risk_hook

log = logging.getLogger("trader.agent")

_ID_RE = re.compile(r'"(?:order_)?id"\s*:\s*"([^"]+)"')


def _block_text(block) -> str:
    return getattr(block, "text", "") or ""


def _extract_order_ids(text: str) -> list[str]:
    return _ID_RE.findall(text or "")


class TradingAgent:
    def __init__(
        self,
        cfg: AppConfig,
        market: MarketData,
        cache: Cache,
        journal: Journal,
        drawdown: DrawdownMonitor,
    ):
        self._cfg = cfg
        self._market = market
        self._cache = cache
        self._journal = journal
        self._drawdown = drawdown
        self._system_prompt = build_system_prompt(cfg)

    def _gather(self) -> tuple[dict, list[dict], dict[str, float]]:
        """Fetch account + per-symbol indicator snapshots (deterministic)."""
        account = self._market.get_account()
        snapshots: list[dict] = []
        prices: dict[str, float] = {}
        for sym in self._cfg.watchlist.symbols:
            try:
                df = self._market.get_bars(sym)
                snap = compute_snapshot(
                    sym.symbol, df,
                    self._cfg.strategy.indicators,
                    self._cfg.strategy.thresholds,
                    self._cfg.strategy.breakout,
                )
                snapshots.append(snap)
                prices[sym.symbol] = snap["price"]
            except ValueError as exc:
                log.warning("skipping %s: %s", sym.symbol, exc)
        account_dict = {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "open_positions": account.open_position_symbols,
        }
        return account_dict, snapshots, prices

    def _enforce_software_stops(self, prices: dict[str, float]) -> list[dict]:
        """Deterministic stop/target for long-only positions (e.g. spot crypto,
        which can't carry a bracket). Closes any owned, whitelisted position whose
        price has crossed entry ± configured pct. Runs before the LLM, so the hard
        exit never depends on the model. Returns the exits taken (for journaling).
        """
        if not self._cfg.risk.long_only:
            return []
        stop_pct = self._cfg.strategy.bracket.stop_loss_pct
        tp_pct = self._cfg.strategy.bracket.take_profit_pct
        whitelist = self._cfg.watchlist.whitelist
        exits: list[dict] = []
        for sym, pos in self._market.get_positions_detail().items():
            if sym not in whitelist:
                continue
            entry = pos["avg_entry_price"]
            price = prices.get(sym) or pos.get("current_price")
            if not entry or not price:
                continue
            reason = None
            if price <= entry * (1 - stop_pct):
                reason = f"software STOP: {price:.2f} <= entry {entry:.2f} -{stop_pct:.1%}"
            elif price >= entry * (1 + tp_pct):
                reason = f"software TARGET: {price:.2f} >= entry {entry:.2f} +{tp_pct:.1%}"
            if reason:
                try:
                    self._market.close_position(sym)
                    log.info("[%s] auto-exit %s — %s", self._cfg.settings.agent_name, sym, reason)
                    exits.append({"symbol": sym, "reason": reason, "entry": entry, "price": price})
                except Exception as exc:  # noqa: BLE001
                    log.warning("auto-exit failed for %s: %s", sym, exc)
        return exits

    async def run_tick(self) -> str:
        """Execute one full ReAct cycle; returns the journal doc id."""
        # Reconcile: cancel our own stale unfilled orders (older than one interval)
        # so they don't lock buying power or stack when a venue fills slowly.
        try:
            stale = self._market.cancel_stale_orders(
                list(self._cfg.watchlist.whitelist),
                older_than_seconds=self._cfg.strategy.tick_interval_minutes * 60,
            )
            if stale:
                log.info("[%s] cancelled %d stale unfilled order(s)",
                         self._cfg.settings.agent_name, stale)
        except Exception:  # noqa: BLE001 — never let reconciliation break the tick
            log.warning("order reconciliation failed", exc_info=True)

        account_dict, snapshots, prices = self._gather()
        equity = account_dict["equity"]

        # Deterministic stop/target exits (long-only/crypto) before anything else.
        auto_exits = self._enforce_software_stops(prices)

        # Drawdown check next — may liquidate + latch pause before any new trade.
        dd = self._drawdown.check(equity)
        if dd.breached:
            log.warning(
                "DRAWDOWN BREACH: equity=%.2f anchor=%.2f (%.2f%%) — paused",
                dd.equity, dd.anchor, dd.drawdown_pct * 100,
            )

        self._cache.set_snapshot(snapshots)
        self._cache.set_latest_prices(prices)

        # Per-tick risk context + decision sink.
        risk_decisions: list[RiskDecision] = []
        ctx = RiskContext(
            equity=equity,
            open_position_symbols=account_dict["open_positions"],
            latest_prices=prices,
            paused=dd.paused,
        )
        risk_hook = make_risk_hook(
            self._cfg.risk,
            self._cfg.watchlist,
            context_provider=lambda: ctx,
            on_decision=risk_decisions.append,
        )

        options = build_options(self._cfg, self._system_prompt, risk_hook)

        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        order_ids: list[str] = []
        error: str | None = None

        # If paused, we still run the loop so the journal captures the state, but
        # the hook will deny any entry orders.
        user_message = build_user_message(
            account=_AccountView(account_dict), snapshots=snapshots
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_message)
                async for msg in client.receive_response():
                    for block in getattr(msg, "content", []) or []:
                        kind = type(block).__name__
                        if kind == "TextBlock":
                            reasoning_parts.append(_block_text(block))
                        elif kind == "ToolUseBlock":
                            tool_calls.append(
                                {
                                    "name": getattr(block, "name", ""),
                                    "input": getattr(block, "input", {}),
                                }
                            )
                        elif kind == "ToolResultBlock":
                            content = getattr(block, "content", "")
                            text = (
                                content if isinstance(content, str) else json.dumps(content, default=str)
                            )
                            tool_results.append({"content": text})
                            order_ids.extend(_extract_order_ids(text))
        except Exception as exc:  # noqa: BLE001 — journal the failure, keep the loop alive
            error = f"{type(exc).__name__}: {exc}"
            log.exception("tick failed")

        doc_id = self._journal.record_tick(
            agent=self._cfg.settings.agent_name,
            snapshot=snapshots,
            account=account_dict,
            reasoning="\n".join(p for p in reasoning_parts if p),
            tool_calls=tool_calls,
            tool_results=tool_results,
            risk_decisions=[d.__dict__ for d in risk_decisions],
            order_ids=sorted(set(order_ids)),
            auto_exits=auto_exits,
            error=error,
        )
        log.info(
            "[%s] tick journaled %s: %d tool_calls, %d denied, %d orders, %d auto-exits%s",
            self._cfg.settings.agent_name,
            doc_id,
            len(tool_calls),
            sum(1 for d in risk_decisions if d.decision == "deny"),
            len(set(order_ids)),
            len(auto_exits),
            " [PAUSED]" if dd.paused else "",
        )
        return doc_id


class _AccountView:
    """Adapter so build_user_message can read account attributes from the dict."""

    def __init__(self, d: dict):
        self.equity = d["equity"]
        self.cash = d["cash"]
        self.buying_power = d["buying_power"]
        self.open_position_symbols = d["open_positions"]
