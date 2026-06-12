"""Redis-backed cache: latest snapshots, daily drawdown anchor, and pause flag.

Keeps fast-changing tick state out of MongoDB and survives daemon restarts within
a trading day (anchor/pause are keyed by date with a TTL).
"""

from __future__ import annotations

import json

import redis

_DAY_TTL_SECONDS = 60 * 60 * 36  # 36h: comfortably covers one trading day + buffer


class Cache:
    def __init__(self, redis_url: str, prefix: str = "default"):
        self._r = redis.Redis.from_url(redis_url, decode_responses=True)
        # Per-agent key namespace so multiple agents share one Redis without
        # clobbering each other's snapshots / drawdown anchors / pause flags.
        self._ns = f"trader:{prefix}"

    # --- snapshot cache -------------------------------------------------------
    def set_snapshot(self, snapshots: list[dict]) -> None:
        self._r.set(f"{self._ns}:snapshot:latest", json.dumps(snapshots))

    def get_snapshot(self) -> list[dict] | None:
        raw = self._r.get(f"{self._ns}:snapshot:latest")
        return json.loads(raw) if raw else None

    def set_latest_prices(self, prices: dict[str, float]) -> None:
        self._r.set(f"{self._ns}:prices:latest", json.dumps(prices))

    def get_latest_prices(self) -> dict[str, float]:
        raw = self._r.get(f"{self._ns}:prices:latest")
        return json.loads(raw) if raw else {}

    # --- daily drawdown state -------------------------------------------------
    def get_daily_anchor(self, day: str) -> float | None:
        raw = self._r.get(f"{self._ns}:anchor:{day}")
        return float(raw) if raw is not None else None

    def set_daily_anchor(self, day: str, equity: float) -> None:
        self._r.set(f"{self._ns}:anchor:{day}", equity, ex=_DAY_TTL_SECONDS)

    def is_paused(self, day: str) -> bool:
        return self._r.get(f"{self._ns}:paused:{day}") is not None

    def set_paused(self, day: str) -> None:
        self._r.set(f"{self._ns}:paused:{day}", "1", ex=_DAY_TTL_SECONDS)

    # --- news sentiment cache (per symbol, short TTL) -------------------------
    def get_news_sentiment(self, symbol: str) -> dict | None:
        raw = self._r.get(f"{self._ns}:news:{symbol}")
        return json.loads(raw) if raw else None

    def set_news_sentiment(self, symbol: str, sentiment: dict, ttl_seconds: int) -> None:
        self._r.set(f"{self._ns}:news:{symbol}", json.dumps(sentiment), ex=ttl_seconds)

    def ping(self) -> bool:
        return bool(self._r.ping())
