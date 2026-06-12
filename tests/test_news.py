import pytest

from trader.config import load_config
from trader.news import NewsSentiment, Sentiment, parse_sentiment


# --- tolerant JSON parsing -----------------------------------------------------
def test_parse_clean_json():
    s = parse_sentiment('{"label":"bearish","confidence":0.8,"earnings_imminent":false,"rationale":"downgrade"}')
    assert s.label == "bearish" and s.confidence == 0.8 and s.is_bearish


def test_parse_with_code_fence_and_prose():
    s = parse_sentiment('Here:\n```json\n{"label":"BULLISH","confidence":0.7,"earnings_imminent":true,"rationale":"x"}\n```')
    assert s.label == "bullish" and s.earnings_imminent is True


def test_parse_garbage_is_error_neutral():
    s = parse_sentiment("no json here")
    assert s.error and s.label == "neutral"


def test_parse_unknown_label_falls_back_neutral():
    assert parse_sentiment('{"label":"spicy","confidence":0.9}').label == "neutral"


# --- should_block veto logic (stubbed scorer; no LLM) --------------------------
class _StubCache:
    def __init__(self): self.store = {}
    def get_news_sentiment(self, sym): return self.store.get(sym)
    def set_news_sentiment(self, sym, d, ttl): self.store[sym] = d


@pytest.fixture
def ns():
    import os
    os.environ["TRADER_CONFIG_DIR"] = "config/equity"
    load_config.cache_clear()
    cfg = load_config()
    return NewsSentiment(cfg, _StubCache())


def test_block_on_strong_bearish(ns):
    blocked, _ = ns.should_block(Sentiment(label="bearish", confidence=0.8))
    assert blocked is True


def test_no_block_on_weak_bearish(ns):
    # below bearish_confidence_min (0.6)
    blocked, _ = ns.should_block(Sentiment(label="bearish", confidence=0.4))
    assert blocked is False


def test_block_on_imminent_earnings(ns):
    blocked, reason = ns.should_block(Sentiment(label="neutral", earnings_imminent=True))
    assert blocked is True and "earnings" in reason


def test_no_block_on_bullish_or_neutral(ns):
    assert ns.should_block(Sentiment(label="bullish", confidence=0.9))[0] is False
    assert ns.should_block(Sentiment(label="neutral"))[0] is False


def test_error_fails_open(ns):
    blocked, reason = ns.should_block(Sentiment(error="boom"))
    assert blocked is False and "fail-open" in reason


# --- assess(): no headlines -> neutral (no LLM call), and caching --------------
async def test_assess_no_news_is_neutral_and_cached(ns, monkeypatch):
    monkeypatch.setattr(ns, "fetch_recent", lambda sym: [])
    s = await ns.assess("SPY")
    assert s.label == "neutral"
    assert ns._cache.get_news_sentiment("SPY") is not None  # cached


async def test_assess_uses_injected_scorer(monkeypatch):
    import os
    os.environ["TRADER_CONFIG_DIR"] = "config/equity"
    load_config.cache_clear()
    cfg = load_config()

    async def fake_scorer(sym, headlines):
        return Sentiment(label="bearish", confidence=0.9, rationale="stub")

    ns = NewsSentiment(cfg, _StubCache(), scorer=fake_scorer)
    monkeypatch.setattr(ns, "fetch_recent", lambda sym: ["Some headline"])
    s = await ns.assess("NVDA")
    assert s.label == "bearish" and s.confidence == 0.9
    assert ns.should_block(s)[0] is True
