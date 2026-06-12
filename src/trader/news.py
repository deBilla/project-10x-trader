"""News risk-filter: fetch recent headlines (Alpaca/Benzinga) and have Claude score
sentiment, used to veto trend-breakout entries that are walking into bad news or an
imminent earnings gap. Claude can only *block* an entry, never initiate one.

Sentiment is cached per symbol (Redis, short TTL) and only computed for breakout
candidates, so LLM cost stays tiny. All failures fail OPEN (never block the
validated trend entry on an error).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import NewsParams

log = logging.getLogger("trader.news")

_SYSTEM = (
    "You are a terse financial news analyst. Given recent headlines for one ticker, "
    "judge the near-term (days) directional risk for a LONG position. Respond with "
    "ONLY a JSON object, no prose: "
    '{"label":"bullish|neutral|bearish","confidence":0.0-1.0,'
    '"earnings_imminent":true|false,"rationale":"<=20 words"}. '
    "earnings_imminent = the company reports earnings within ~3 trading days."
)


@dataclass
class Sentiment:
    label: str = "neutral"
    confidence: float = 0.0
    earnings_imminent: bool = False
    rationale: str = ""
    error: str | None = None

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"


def parse_sentiment(text: str) -> Sentiment:
    """Tolerant JSON parse of the model's reply (handles code fences / surrounding text)."""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return Sentiment(error="no json found")
        d = json.loads(text[start:end + 1])
        label = str(d.get("label", "neutral")).lower()
        if label not in ("bullish", "neutral", "bearish"):
            label = "neutral"
        return Sentiment(
            label=label,
            confidence=float(d.get("confidence", 0.0)),
            earnings_imminent=bool(d.get("earnings_imminent", False)),
            rationale=str(d.get("rationale", ""))[:200],
        )
    except Exception as exc:  # noqa: BLE001
        return Sentiment(error=f"parse: {exc}")


class NewsSentiment:
    def __init__(self, cfg, cache, scorer=None):
        """`scorer` is an async fn(symbol, headlines)->Sentiment; defaults to Claude.
        Injectable so tests can stub it without an LLM call."""
        self._cfg = cfg
        self._cache = cache
        self._params: NewsParams = cfg.strategy.news
        self._scorer = scorer or self._claude_score
        self._news_client = None

    def fetch_recent(self, symbol: str) -> list[str]:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        if self._news_client is None:
            s = self._cfg.settings
            self._news_client = NewsClient(s.alpaca_api_key, s.alpaca_secret_key)
        start = datetime.now(timezone.utc) - timedelta(hours=self._params.lookback_hours)
        res = self._news_client.get_news(
            NewsRequest(symbols=symbol, start=start, limit=15)
        )
        # Version-robust: newer alpaca-py exposes articles under .data['news'];
        # older versions had a .news attribute.
        arts = []
        if hasattr(res, "data") and isinstance(res.data, dict):
            arts = res.data.get("news", []) or []
        elif getattr(res, "news", None):
            arts = res.news
        out = []
        for a in arts:
            hl = a.get("headline") if isinstance(a, dict) else getattr(a, "headline", "")
            if hl:
                out.append(hl)
        return out

    async def _claude_score(self, symbol: str, headlines: list[str]) -> Sentiment:
        from claude_agent_sdk import ClaudeAgentOptions, query

        prompt = (
            f"Ticker: {symbol}\nRecent headlines (newest first):\n"
            + "\n".join(f"- {h}" for h in headlines[:15])
        )
        opts = ClaudeAgentOptions(system_prompt=_SYSTEM, max_turns=1,
                                  model=self._cfg.settings.trader_model)
        parts: list[str] = []
        async for msg in query(prompt=prompt, options=opts):
            for b in getattr(msg, "content", []) or []:
                if type(b).__name__ == "TextBlock":
                    parts.append(getattr(b, "text", ""))
        return parse_sentiment("".join(parts))

    async def assess(self, symbol: str) -> Sentiment:
        """Cached per-symbol sentiment. No headlines -> neutral. Errors fail open."""
        cached = self._cache.get_news_sentiment(symbol)
        if cached is not None:
            return Sentiment(**cached)
        try:
            headlines = self.fetch_recent(symbol)
            if not headlines:
                sent = Sentiment(label="neutral", rationale="no recent news")
            else:
                sent = await self._scorer(symbol, headlines)
        except Exception as exc:  # noqa: BLE001 — fail open
            log.warning("news assess failed for %s: %s", symbol, exc)
            sent = Sentiment(error=str(exc))
        self._cache.set_news_sentiment(
            symbol, sent.__dict__, self._params.cache_ttl_minutes * 60
        )
        return sent

    def should_block(self, sent: Sentiment) -> tuple[bool, str]:
        """Veto decision for a long entry. Fail-open on errors."""
        p = self._params
        if sent.error:
            return False, f"news error (fail-open): {sent.error}"
        if p.block_on_imminent_earnings and sent.earnings_imminent:
            return True, "earnings imminent (gap risk)"
        if p.block_on_bearish and sent.is_bearish and sent.confidence >= p.bearish_confidence_min:
            return True, f"bearish news (conf {sent.confidence:.2f}): {sent.rationale}"
        return False, sent.label
