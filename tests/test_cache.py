"""Two agents sharing one Redis must not clobber each other's namespaced state."""

import fakeredis
import pytest

from trader.persistence import cache as cache_mod
from trader.persistence.cache import Cache


@pytest.fixture
def shared_redis(monkeypatch):
    server = fakeredis.FakeServer()

    def _from_url(url, **kw):
        return fakeredis.FakeStrictRedis(server=server, decode_responses=True)

    monkeypatch.setattr(cache_mod.redis.Redis, "from_url", staticmethod(_from_url))
    return server


def test_snapshots_isolated_by_prefix(shared_redis):
    eq = Cache("redis://x", prefix="equity")
    cr = Cache("redis://x", prefix="crypto")

    eq.set_snapshot([{"symbol": "SPY"}])
    cr.set_snapshot([{"symbol": "BTC/USD"}])

    assert eq.get_snapshot()[0]["symbol"] == "SPY"
    assert cr.get_snapshot()[0]["symbol"] == "BTC/USD"


def test_anchor_and_pause_isolated_by_prefix(shared_redis):
    eq = Cache("redis://x", prefix="equity")
    cr = Cache("redis://x", prefix="crypto")
    day = "2026-06-10"

    eq.set_daily_anchor(day, 100_000.0)
    eq.set_paused(day)

    # crypto agent sees none of the equity agent's drawdown state
    assert cr.get_daily_anchor(day) is None
    assert cr.is_paused(day) is False
    # equity agent sees its own
    assert eq.get_daily_anchor(day) == 100_000.0
    assert eq.is_paused(day) is True
