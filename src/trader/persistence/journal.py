"""MongoDB trade journal — the immutable record of every ReAct tick.

One document per tick captures the full context window: the numeric snapshot
given to Claude, Claude's reasoning text, every tool call + response, the risk
decisions, and the resulting Alpaca order ids. This is the primary artifact for
debugging *why* the agent traded.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pymongo import MongoClient


class Journal:
    def __init__(self, mongo_uri: str, db_name: str):
        self._client = MongoClient(mongo_uri)
        self._ticks = self._client[db_name]["ticks"]

    def record_tick(
        self,
        *,
        agent: str = "default",
        snapshot: list[dict],
        account: dict,
        reasoning: str,
        tool_calls: list[dict],
        tool_results: list[dict],
        risk_decisions: list[dict],
        order_ids: list[str],
        auto_exits: list[dict] | None = None,
        news: list[dict] | None = None,
        error: str | None = None,
    ) -> str:
        """Persist one tick; returns the inserted document id as a string."""
        doc = {
            "ts": datetime.now(timezone.utc),
            "agent": agent,
            "auto_exits": auto_exits or [],
            "news": news or [],
            "account": account,
            "snapshot": snapshot,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "risk_decisions": risk_decisions,
            "order_ids": order_ids,
            "error": error,
        }
        result = self._ticks.insert_one(doc)
        return str(result.inserted_id)

    def ping(self) -> bool:
        self._client.admin.command("ping")
        return True
