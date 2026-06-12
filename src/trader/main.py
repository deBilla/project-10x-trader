"""Entrypoint: wire dependencies and run the agent (once or on a schedule)."""

from __future__ import annotations

import asyncio
import logging
import sys

from .agent import TradingAgent
from .config import load_config
from .market_data import MarketData
from .persistence.cache import Cache
from .persistence.journal import Journal
from .risk.drawdown import DrawdownMonitor
from .scheduler import TickScheduler

log = logging.getLogger("trader")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def _check_health(cache: Cache, journal: Journal) -> None:
    cache.ping()
    journal.ping()
    log.info("redis + mongo reachable")


async def _run() -> None:
    cfg = load_config()

    agent_name = cfg.settings.agent_name
    market = MarketData(cfg)
    cache = Cache(cfg.settings.redis_url, prefix=agent_name)
    journal = Journal(cfg.settings.mongo_uri, cfg.settings.mongo_db)
    _check_health(cache, journal)

    # Drawdown liquidation is scoped to THIS agent's own symbols only.
    own_symbols = list(cfg.watchlist.whitelist)
    drawdown = DrawdownMonitor(
        cfg.risk, cache, liquidate_fn=lambda: market.close_positions(own_symbols)
    )
    if cfg.strategy.mode == "trend":
        from .trend_trader import TrendTrader
        agent = TrendTrader(cfg, market, journal, drawdown, cache=cache)
        log.info("[%s] decision engine: deterministic TREND executor", agent_name)
    else:
        agent = TradingAgent(cfg, market, cache, journal, drawdown)
        log.info("[%s] decision engine: LLM", agent_name)

    # Phase-1 acceptance: prove MCP/Alpaca wiring by logging the paper balance.
    acct = market.get_account()
    log.info(
        "[%s] paper account: equity=$%.2f buying_power=$%.2f open=%s | watchlist=%s",
        agent_name, acct.equity, acct.buying_power, acct.open_position_symbols,
        own_symbols,
    )

    if cfg.settings.run_once:
        log.info("TRADER_RUN_ONCE=true — running a single tick")
        await agent.run_tick()
        return

    scheduler = TickScheduler(cfg, agent, market)
    await scheduler.start()


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        log.info("shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
