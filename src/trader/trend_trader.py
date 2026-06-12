"""Deterministic daily trend-following executor — the live engine for the strategy
the backtester validated (positive expectancy, PF ~3, regime-tested).

No LLM in the decision: each cycle it computes the trend state per symbol from daily
bars and reconciles holdings — enter longs on confirmed uptrends, exit on trend break
or a catastrophic stop. Long-only, fractional (notional) sizing, so it fits a small
account. Same risk caps as the rest of the system (sizing %, max positions, daily
drawdown kill-switch).
"""

from __future__ import annotations

import logging

from .backtest.trend import desired_long
from .config import AppConfig
from .market_data import MarketData
from .persistence.journal import Journal
from .risk.drawdown import DrawdownMonitor

log = logging.getLogger("trader.trend")


class TrendTrader:
    def __init__(self, cfg: AppConfig, market: MarketData, journal: Journal,
                 drawdown: DrawdownMonitor, cache=None):
        self._cfg = cfg
        self._market = market
        self._journal = journal
        self._drawdown = drawdown
        self._news = None
        if cfg.strategy.news.enabled and cache is not None:
            from .news import NewsSentiment
            self._news = NewsSentiment(cfg, cache)

    async def run_tick(self) -> str:
        cfg = self._cfg
        tp = cfg.strategy.trend
        name = cfg.settings.agent_name
        whitelist = cfg.watchlist.whitelist

        acct = self._market.get_account()
        dd = self._drawdown.check(acct.equity)
        positions = self._market.get_positions_detail()
        open_syms = [s for s in positions if s in whitelist]

        actions: list[dict] = []
        states: list[dict] = []
        order_ids: list[str] = []
        news_log: list[dict] = []

        for symcfg in cfg.watchlist.symbols:
            sym = symcfg.symbol
            try:
                bars = self._market.get_history(symcfg, tp.history_days, timeframe="1Day")
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] history error %s: %s", name, sym, exc)
                continue
            if len(bars) < tp.trend_ma + 2:
                continue

            want_long = desired_long(bars, tp.entry_channel, tp.exit_channel,
                                     tp.trend_ma, tp.use_regime_filter) and not dd.paused
            holding = sym in positions
            states.append({"symbol": sym, "want_long": want_long, "holding": holding,
                           "price": round(float(bars["close"].iloc[-1]), 2)})

            try:
                if holding:
                    pos = positions[sym]
                    entry = pos["avg_entry_price"]
                    price = pos.get("current_price") or float(bars["close"].iloc[-1])
                    # Catastrophic stop overrides; otherwise exit on trend break.
                    if entry and price <= entry * (1 - tp.cat_stop_pct):
                        self._market.close_position(sym)
                        actions.append({"action": "SELL", "symbol": sym, "reason": "cat_stop"})
                    elif not want_long:
                        self._market.close_position(sym)
                        actions.append({"action": "SELL", "symbol": sym, "reason": "trend_break"})
                elif want_long and len(open_syms) < cfg.risk.max_open_positions:
                    # News risk-filter: veto the entry on strongly bearish news or
                    # imminent earnings (fail-open). News can only block, not initiate.
                    blocked = False
                    if self._news is not None:
                        sent = await self._news.assess(sym)
                        blocked, reason = self._news.should_block(sent)
                        news_log.append({"symbol": sym, "label": sent.label,
                                         "confidence": sent.confidence,
                                         "earnings_imminent": sent.earnings_imminent,
                                         "rationale": sent.rationale,
                                         "decision": "blocked" if blocked else "entered"})
                        if blocked:
                            log.info("[%s] BLOCKED %s entry — %s", name, sym, reason)
                            actions.append({"action": "BLOCKED", "symbol": sym, "reason": reason})
                    if not blocked:
                        notional = round(cfg.risk.max_position_pct * acct.equity, 2)
                        oid = self._market.submit_notional_buy(sym, notional)
                        order_ids.append(oid)
                        open_syms.append(sym)
                        actions.append({"action": "BUY", "symbol": sym, "notional": notional})
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] order failed %s: %s", name, sym, exc)
                actions.append({"action": "ERROR", "symbol": sym, "error": str(exc)})

        reasoning = (
            f"trend executor: {sum(1 for s in states if s['want_long'])}/{len(states)} "
            f"in uptrend; {len(actions)} actions."
            + (" [PAUSED: drawdown]" if dd.paused else "")
        )
        doc_id = self._journal.record_tick(
            agent=name,
            snapshot=states,
            account={"equity": acct.equity, "cash": acct.cash,
                     "buying_power": acct.buying_power, "open_positions": open_syms},
            reasoning=reasoning,
            tool_calls=actions,
            tool_results=[],
            risk_decisions=[],
            order_ids=order_ids,
            news=news_log,
            error=None,
        )
        log.info("[%s] trend tick %s: %d uptrend, %d actions%s", name, doc_id,
                 sum(1 for s in states if s["want_long"]), len(actions),
                 " [PAUSED]" if dd.paused else "")
        return doc_id
