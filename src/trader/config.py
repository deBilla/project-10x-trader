"""Configuration loading: environment settings + YAML config files.

Environment variables (secrets, infra endpoints) load via pydantic-settings.
Strategy/risk/watchlist tuning lives in YAML so it can change without code edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-level settings sourced from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Alpaca
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    alpaca_paper_trade: bool = Field(default=True, alias="ALPACA_PAPER_TRADE")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    trader_model: str = Field(default="claude-opus-4-8", alias="TRADER_MODEL")

    # Alpaca MCP server wiring.
    # transport: "stdio" (SDK spawns the server as a subprocess) or
    #            "http" (connect to an already-running streamable-http server).
    alpaca_mcp_transport: str = Field(default="stdio", alias="ALPACA_MCP_TRANSPORT")
    # For stdio: the command + args used to launch the server.
    alpaca_mcp_command: str = Field(
        default="alpaca-mcp-server", alias="ALPACA_MCP_COMMAND"
    )
    alpaca_mcp_args: str = Field(
        default="", alias="ALPACA_MCP_ARGS"
    )  # comma-separated; empty = run server with stdio defaults
    # For http: the server URL.
    alpaca_mcp_url: str = Field(
        default="http://alpaca-mcp:8000/mcp", alias="ALPACA_MCP_URL"
    )

    # Persistence
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="trader", alias="MONGO_DB")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Config + runtime
    config_dir: Path = Field(default=Path("config"), alias="TRADER_CONFIG_DIR")
    # Identifies this agent instance; namespaces Redis state and tags journal docs.
    agent_name: str = Field(default="default", alias="TRADER_AGENT_NAME")
    run_once: bool = Field(default=False, alias="TRADER_RUN_ONCE")
    dry_run: bool = Field(default=False, alias="TRADER_DRY_RUN")


# --- YAML-backed config models -------------------------------------------------


class SymbolConfig(BaseModel):
    symbol: str
    asset_class: str  # "us_equity" | "crypto"
    bar_timeframe: str = "5Min"
    lookback_bars: int = 120


class WatchlistConfig(BaseModel):
    symbols: list[SymbolConfig]

    @property
    def whitelist(self) -> set[str]:
        return {s.symbol for s in self.symbols}

    def get(self, symbol: str) -> SymbolConfig | None:
        return next((s for s in self.symbols if s.symbol == symbol), None)


class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_open_positions: int = 3
    daily_drawdown_pct: float = 0.10
    require_stop_loss: bool = True
    max_stop_distance_pct: float = 0.05
    # Long-only mode (crypto): deny short entries. Exits go via close_position.
    long_only: bool = False


class IndicatorParams(BaseModel):
    rsi_period: int = 14
    bbands_period: int = 20
    bbands_std: float = 2.0
    ema_fast: int = 9
    ema_slow: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_avg_period: int = 20


class Thresholds(BaseModel):
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    volume_surge_mult: float = 1.2


class BracketHints(BaseModel):
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


class BreakoutParams(BaseModel):
    """Momentum/breakout path — lets the agent trade WITH a trend, not only fade it."""

    enabled: bool = True
    channel_period: int = 20   # Donchian lookback for the breakout high/low
    min_rsi_long: float = 50   # momentum longs want strength, not exhaustion
    max_rsi_long: float = 72   # but not a blow-off top
    min_rsi_short: float = 28
    max_rsi_short: float = 50


class NewsParams(BaseModel):
    """News risk-filter on trend entries (Claude-scored sentiment)."""

    enabled: bool = True
    lookback_hours: int = 48
    block_on_bearish: bool = True
    bearish_confidence_min: float = 0.6   # min confidence to veto a long
    block_on_imminent_earnings: bool = True
    cache_ttl_minutes: int = 120          # reuse a symbol's sentiment within this


class TrendParams(BaseModel):
    """Daily trend-following params (validated by the backtester)."""

    entry_channel: int = 50    # break above prior N-day high to enter
    exit_channel: int = 20     # break below prior N-day low to exit
    trend_ma: int = 100        # regime filter: only long above this SMA
    use_regime_filter: bool = True
    cat_stop_pct: float = 0.20  # catastrophic stop (rarely hit; channel exit leads)
    history_days: int = 400     # daily bars to pull for the signal


class StrategyConfig(BaseModel):
    tick_interval_minutes: int = 5
    # Live decision engine: "llm" (Claude reasons over indicators) or
    # "trend" (deterministic daily trend-following executor — the backtested edge).
    mode: str = "llm"
    trend: TrendParams = TrendParams()
    # When false, the trend filter is dropped from the prompt (allows mean-reversion
    # entries against the prevailing trend). Keep true in production.
    enforce_trend_filter: bool = True
    # "confluence" (strict: signals must align) vs "single_signal" (active: any one
    # primary trigger from either strategy is enough to enter).
    entry_mode: str = "confluence"
    # When false, a volume surge is preferred but not mandatory for entry.
    require_volume_confirmation: bool = True
    news: NewsParams = NewsParams()
    indicators: IndicatorParams = IndicatorParams()
    thresholds: Thresholds = Thresholds()
    bracket: BracketHints = BracketHints()
    breakout: BreakoutParams = BreakoutParams()


class AppConfig(BaseModel):
    """Aggregate of all config; what the rest of the app depends on."""

    settings: Settings
    watchlist: WatchlistConfig
    risk: RiskConfig
    strategy: StrategyConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load and cache the full application config."""
    settings = Settings()
    cfg_dir = settings.config_dir
    return AppConfig(
        settings=settings,
        watchlist=WatchlistConfig(**_load_yaml(cfg_dir / "watchlist.yaml")),
        risk=RiskConfig(**_load_yaml(cfg_dir / "risk.yaml")),
        strategy=StrategyConfig(**_load_yaml(cfg_dir / "strategy.yaml")),
    )
