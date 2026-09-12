"""Offline evaluation of the system's LLM components.

The trend engine is validated by `trader.backtest`; this package covers the piece
the backtester can't see — the Claude news risk-filter that holds veto power over
entries. See `news_replay` for the method and its caveats.
"""
