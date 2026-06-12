"""Tick scheduling and gating.

Fires the agent's ReAct cycle on a fixed interval, but only when there is
something tradeable: equities require an open US market; crypto trades 24/7, so a
tick still runs off-hours if the watchlist contains any crypto symbol.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .agent import TradingAgent
from .config import AppConfig
from .market_data import MarketData

log = logging.getLogger("trader.scheduler")


class TickScheduler:
    def __init__(self, cfg: AppConfig, agent: TradingAgent, market: MarketData):
        self._cfg = cfg
        self._agent = agent
        self._market = market
        self._has_crypto = any(
            s.asset_class == "crypto" for s in cfg.watchlist.symbols
        )
        self._running = False

    def _should_run(self) -> bool:
        if self._has_crypto:
            return True
        try:
            return self._market.is_market_open()
        except Exception:  # noqa: BLE001 — if clock check fails, skip this tick
            log.warning("market clock check failed; skipping tick")
            return False

    async def _tick(self) -> None:
        if self._running:
            log.info("previous tick still running; skipping")
            return
        if not self._should_run():
            log.info("market closed and no crypto in watchlist; skipping tick")
            return
        self._running = True
        try:
            await self._agent.run_tick()
        finally:
            self._running = False

    async def start(self) -> None:
        import asyncio

        scheduler = AsyncIOScheduler()
        # NOTE: do NOT pass next_run_time=None — that adds the job PAUSED and the
        # interval never fires. Omitting it makes the first scheduled run land one
        # interval from now; the explicit _tick() below covers the immediate run.
        scheduler.add_job(
            self._tick,
            IntervalTrigger(minutes=self._cfg.strategy.tick_interval_minutes),
            max_instances=1,        # never overlap ticks
            coalesce=True,          # collapse missed runs into one
        )
        scheduler.start()
        log.info(
            "scheduler started: every %d min",
            self._cfg.strategy.tick_interval_minutes,
        )
        # Run one tick immediately on boot, then let the interval take over.
        await self._tick()
        # Keep the event loop alive.
        stop = asyncio.Event()
        await stop.wait()
