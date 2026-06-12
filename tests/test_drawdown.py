from trader.config import RiskConfig
from trader.risk.drawdown import DrawdownMonitor


class FakeCache:
    """Minimal in-memory stand-in for the Redis cache (date-keyed state)."""

    def __init__(self):
        self.anchors = {}
        self.paused = set()

    def get_daily_anchor(self, day):
        return self.anchors.get(day)

    def set_daily_anchor(self, day, equity):
        self.anchors[day] = equity

    def is_paused(self, day):
        return day in self.paused

    def set_paused(self, day):
        self.paused.add(day)


RISK = RiskConfig(daily_drawdown_pct=0.10)


def test_first_check_sets_anchor_no_breach():
    cache = FakeCache()
    calls = []
    mon = DrawdownMonitor(RISK, cache, liquidate_fn=lambda: calls.append(1))

    status = mon.check(10_000.0)
    assert status.anchor == 10_000.0
    assert status.breached is False
    assert status.paused is False
    assert calls == []


def test_breach_liquidates_once_and_latches_pause():
    cache = FakeCache()
    calls = []
    mon = DrawdownMonitor(RISK, cache, liquidate_fn=lambda: calls.append(1))

    mon.check(10_000.0)            # establish anchor
    status = mon.check(8_900.0)    # -11% -> breach
    assert status.breached is True
    assert status.paused is True
    assert calls == [1]

    # Subsequent checks stay paused but DO NOT liquidate again.
    status2 = mon.check(8_800.0)
    assert status2.paused is True
    assert calls == [1]
    assert mon.is_paused() is True


def test_small_drawdown_does_not_trip():
    cache = FakeCache()
    calls = []
    mon = DrawdownMonitor(RISK, cache, liquidate_fn=lambda: calls.append(1))

    mon.check(10_000.0)
    status = mon.check(9_500.0)    # -5%, below 10% threshold
    assert status.breached is False
    assert status.paused is False
    assert calls == []
