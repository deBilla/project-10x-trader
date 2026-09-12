"""On-disk cache of scored sentiments, keyed by everything that can change a verdict.

Scoring a multi-year backtest is the expensive part of the eval (one LLM call per
entry decision). Caching to disk makes re-runs free, so the harness can be iterated
on — reporting, attribution, thresholds — without re-paying for the model.

The key deliberately includes `prompt_version` and `model`: changing either must
produce a cache miss, otherwise a prompt regression would be invisible behind stale
verdicts. That is the entire point of the harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..news import Sentiment


def _key(symbol: str, day: str, lookback_hours: int, prompt_version: str, model: str) -> str:
    return f"{symbol}|{day}|{lookback_hours}h|{prompt_version}|{model}"


class ScoreCache:
    """JSON-file backed. Loaded once, flushed on `save()` — the harness scores in
    bulk, so per-write durability isn't worth the syscalls."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                # A corrupt cache must never fail the run — it's a speed optimization.
                print(f"warning: ignoring unreadable score cache {path}: {exc}")

    def get(self, symbol: str, day: str, lookback_hours: int,
            prompt_version: str, model: str) -> Sentiment | None:
        raw = self._data.get(_key(symbol, day, lookback_hours, prompt_version, model))
        return Sentiment(**raw) if raw is not None else None

    def put(self, symbol: str, day: str, lookback_hours: int,
            prompt_version: str, model: str, sent: Sentiment) -> None:
        self._data[_key(symbol, day, lookback_hours, prompt_version, model)] = asdict(sent)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=0, default=str), encoding="utf-8")
        tmp.replace(self._path)  # atomic: a killed run can't truncate the cache

    def __len__(self) -> int:
        return len(self._data)
