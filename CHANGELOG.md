# Changelog

Evolution of Project 10X. This project pivoted hard based on evidence — the entries
below double as a decision log. Dates are development milestones, not releases.

The format is loosely [Keep a Changelog](https://keepachangelog.com/). Paper trading
throughout; no real capital.

---

## [0.1.0] — 2026-06-09 — MVP: LLM ReAct bot
### Added
- Core daemon: scheduled ReAct loop, official **Alpaca MCP server** (HTTP) for
  execution, deterministic indicators (RSI/Bollinger/EMA/VWAP/MACD) fed to Claude.
- **In-process risk hook** (`PreToolUse`): per-trade size cap, whitelist, mandatory
  bracket stop-loss, daily-drawdown kill-switch — the LLM cannot bypass it.
- Persistence: MongoDB trade journal + Redis cache. Docker Compose stack
  (agent + alpaca-mcp + redis + mongo). Initial strategy: mean-reversion.
### Notes
- Decision: precompute indicators in code (LLMs are unreliable at arithmetic);
  enforce risk in-process, not via a separate proxy.

## [0.2.0] — 2026-06-09/10 — Containerized & authenticated
### Added
- Full stack live; **Claude auth via subscription OAuth** (bundled CLI +
  `CLAUDE_CODE_OAUTH_TOKEN`), no API key.
- Dual strategy: added **momentum/breakout** path alongside mean-reversion;
  `enforce_trend_filter` toggle; live config mount.
### Fixed
- Risk hook silently denied **all** bracket orders — it didn't recognize the real
  `place_stock_order` stop key (`stop_loss_stop_price`). Would have blocked every
  trade forever. Found via the first real order; added regression test.
- Docker on colima: bypassed broken `osxkeychain` cred helper and dangling
  OrbStack Compose/buildx plugins.

## [0.3.0] — 2026-06-10 — Second agent (crypto)
### Added
- **Two-agent architecture** via profiles (`TRADER_AGENT_NAME` + `TRADER_CONFIG_DIR`):
  `agent-equity` and `agent-crypto`, sharing one account/MCP/Redis/Mongo.
- Per-agent Redis namespacing; scoped drawdown liquidation (each agent only closes
  its own symbols); journal `agent` tag.
- Crypto: **long-only + fractional + deterministic software stop** (Alpaca spot crypto
  has no brackets/shorting).
### Fixed
- **Scheduler never fired on interval** (`next_run_time=None` added the job *paused*).
  Agents only ticked on boot/restart. Verified the fix with an unattended interval tick.
### Changed
- Crypto agent **paused** (gated behind a `crypto` Compose profile): Alpaca **paper**
  does not fill crypto orders (confirmed: market/limit/notional/qty all sit unfilled).

## [0.4.0] — 2026-06-10/11 — Realistic small account
### Changed
- Switched to a fresh **$1,000** paper account; equity went **long-only + fractional
  (notional) + software stops** (fractional shares can't carry brackets).
- Added an **active profile** (12 → looser entries) on request.
### Fixed
- Order **reconciliation**: cancel own stale unfilled orders each tick so they can't
  lock buying power or stack.

## [0.5.0] — 2026-06-11 — Backtester + the big pivot
### Added
- **Backtesting harness** (`src/trader/backtest/`): no-lookahead bar simulation,
  intrabar stops/targets, modeled slippage; metrics (expectancy, profit factor,
  Sharpe, max drawdown, vs buy-and-hold). Mean-reversion and trend modes.
### Findings
- Intraday mean-reversion (active **and** strict): **negative expectancy**, PF ~0.4,
  active variant ≈ **−50%** — a guaranteed loser. Killed it.
- Daily **trend-following** (Donchian + SMA regime): expectancy **+7.7%/trade**,
  PF ~3, Sharpe ~1.8, regime-tested through 2022. **Real edge.**
### Fixed
- Backtest data corrupted by **unadjusted stock splits** (fake NVDA −78%); now fetch
  `adjustment=ALL`.
### Changed
- **Live engine pivoted** to a deterministic daily **TrendTrader** (`mode: trend`) —
  no LLM in the trade decision, so live matches the backtested edge.
- Caught & fixed a stale-Redis-drawdown-anchor false kill-switch after the account
  resize.

## [0.6.0] — 2026-06-11 — Diversification
### Changed
- Equity universe **12 → 30** diversified names (sector stocks + index/sector ETFs).
- Max positions **4 → 10**, size **20% → 10%**.
### Findings
- Re-backtest on 30 symbols: PF **2.82**, Sharpe **2.41**, max DD **9%**, 251 trades —
  edge held *including losers* (TSLA/UNH/V), easing survivorship concerns;
  diversification improved risk-adjusted return.

## [0.7.0] — 2026-06-12 — Options explored, then declined
### Findings
- Covered calls need **100-share lots** (~$30k+/position) — impossible at $1k.
- Small-account option *buying*: **no historical data to backtest**, ~6.5% bid/ask
  spreads, negative base rate. Declined — fails our "validate first" bar.

## [0.8.0] — 2026-06-12 — News risk-filter (LLM where it's useful)
### Added
- **`news.py`**: Alpaca/Benzinga headlines + **Claude sentiment** (JSON: label /
  confidence / earnings_imminent / rationale), Redis-cached, fails open.
- Wired a **veto** into the trend entry path — blocks longs on strongly bearish news
  or imminent earnings; can block, never initiate. Journaled under `news`.
### Notes
- The LLM finally used in its strength (news synthesis), atop the deterministic edge.
- News "alpha" isn't backtestable (lookahead) → shipped as a **risk filter evaluated
  forward** via the journal. Verified live: blocked an ORCL entry on an earnings-miss.
- alpaca-py ≥ 0.43 returns news under `res.data['news']` (older: `res.news`).
