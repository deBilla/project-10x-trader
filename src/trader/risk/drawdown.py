"""Daily-drawdown kill-switch.

Anchors the day's starting equity; if equity falls >= ``daily_drawdown_pct``
below it, force-liquidates all positions and sets a pause flag (in Redis) that
halts new trading until the next calendar trading day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from ..config import RiskConfig


@dataclass
class DrawdownStatus:
    anchor: float
    equity: float
    drawdown_pct: float
    breached: bool
    paused: bool


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DrawdownMonitor:
    """Stateless logic + Redis-backed daily state via the cache layer."""

    def __init__(self, risk: RiskConfig, cache, liquidate_fn: Callable[[], None]):
        self._risk = risk
        self._cache = cache
        self._liquidate = liquidate_fn

    def is_paused(self) -> bool:
        return self._cache.is_paused(_today_key())

    def check(self, equity: float) -> DrawdownStatus:
        """Evaluate drawdown for the current equity; trip the switch if breached."""
        day = _today_key()

        anchor = self._cache.get_daily_anchor(day)
        if anchor is None:
            # First observation of the day establishes the anchor.
            anchor = equity
            self._cache.set_daily_anchor(day, anchor)

        drawdown = (anchor - equity) / anchor if anchor else 0.0
        already_paused = self._cache.is_paused(day)
        breached = drawdown >= self._risk.daily_drawdown_pct

        if breached and not already_paused:
            # Trip once: liquidate and latch the pause for the rest of the day.
            self._liquidate()
            self._cache.set_paused(day)
            already_paused = True

        return DrawdownStatus(
            anchor=anchor,
            equity=equity,
            drawdown_pct=drawdown,
            breached=breached,
            paused=already_paused,
        )
