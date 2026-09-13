# Project 10X — Alpaca Trading Agent

Autonomous **paper-trading** system on Alpaca. It started as an LLM-driven day-trading
bot and — through rigorous backtesting — evolved into a **deterministic daily
trend-following** engine with an **LLM news risk-filter**, a **backtester** for
validating ideas, and a **two-agent** (equities / crypto) architecture sharing one
stack.

> Paper trading only. No real capital. The design philosophy: **never trade an
> unvalidated edge** — measure first, deploy second.

---

## TL;DR of what was learned (and why the design is what it is)

This project's value is as much the findings as the code:

| Approach | Backtest result | Verdict |
| --- | --- | --- |
| Intraday mean-reversion (RSI/Bollinger), LLM-driven | expectancy **−0.08%/trade**, PF 0.45, **−50%** | ❌ Guaranteed loser — churn + costs |
| Daily **trend-following** (Donchian + SMA regime) | expectancy **+5.8%/trade**, PF **2.82**, Sharpe **2.41** | ✅ Real, regime-tested edge |
| Small-account options (buying) | unbacktestable + 6.5% spreads + negative base rate | ❌ Doesn't fit; skipped |
| News sentiment | not a backtestable *signal*; its cost as a veto is measurable ([eval harness](#eval-harness-does-the-news-filter-earn-its-veto)) | ⚠️ **Risk filter** only — must beat the ~58% base rate of blocking at random |

**Conclusion baked into the system:** use a *deterministic, validated* signal for the
trade decision, and the *LLM only where it has genuine edge* (synthesizing news/events).
Activity ≠ profit — the winning strategy trades **rarely**.

> Full evolution & decision log: [`CHANGELOG.md`](./CHANGELOG.md).

---

## Visual guide

Start with [Understand the trading agent, from the basics](https://debilla.github.io/project-10x-trader/agent-loop.html): nine Drawpro diagrams with step-by-step explanations of ticks, ReAct, tools, risk checks, and state. [Local page](docs/agent-loop.html) · [Diagram exports](docs/assets/agent-guide/).

## Architecture

```
                       ┌──────────────── agent (per profile) ────────────────┐
 daily bars (alpaca-py)│  TrendTrader.run_tick()  [deterministic]            │
 ─────────────────────►│   for each symbol: desired_long()? (Donchian+SMA)   │
                       │   reconcile holdings: enter / hold / exit           │
                       │   on ENTRY → NewsSentiment veto? ──► Claude (LLM)    │
                       │        (block bearish / earnings; never initiates)  │
                       │   risk caps: size %, max positions, daily drawdown   │
                       └───────────────┬───────────────────┬─────────────────┘
            orders (alpaca-py) ────────┘      journal ──────┴──► MongoDB
                                               cache / drawdown state ──► Redis

 Backtester (offline): same rules over history → expectancy / PF / Sharpe / drawdown
 Alpaca MCP server: available (HTTP) for the legacy LLM mode; trend mode trades direct
```

Two agents run the **same image** with different config + identity:
- **`agent-equity`** — 30 diversified large-caps + ETFs, daily trend-following, **active**.
- **`agent-crypto`** — BTC/ETH/SOL, long-only software-stops, **paused** (Alpaca paper
  doesn't fill crypto orders; gated behind a compose `crypto` profile).

---

## Components

| Path | Role |
| --- | --- |
| `src/trader/trend_trader.py` | **Live engine** — deterministic daily trend executor (current strategy) |
| `src/trader/backtest/` | Backtester: `signals.py`, `engine.py` (mean-rev), `trend.py` (trend + `desired_long`), `metrics.py`, `run.py` |
| `src/trader/eval/` | **Eval harness** for the LLM news filter: `news_replay.py` (counterfactual A/B), `cache.py`, `run.py` |
| `src/trader/news.py` | News fetch (Alpaca/Benzinga) + Claude sentiment scoring + veto logic |
| `src/trader/indicators.py` | Deterministic indicators (RSI/Bollinger/EMA/VWAP/MACD/Donchian) |
| `src/trader/market_data.py` | alpaca-py wrappers: bars, history, account, orders, liquidation |
| `src/trader/risk/` | `hook.py` (LLM-mode PreToolUse gate), `drawdown.py` (daily kill-switch) |
| `src/trader/agent.py` | Legacy **LLM** ReAct engine (mean-reversion; kept for reference/backtests) |
| `src/trader/persistence/` | `journal.py` (Mongo trade journal), `cache.py` (Redis, namespaced per agent) |
| `src/trader/scheduler.py`, `main.py` | Tick scheduling (market-hours aware) + dependency wiring |
| `config/equity/`, `config/crypto/` | Per-profile `watchlist.yaml` / `risk.yaml` / `strategy.yaml` |

---

## The strategy (live: trend-following)

Classic Donchian/Turtle-style, long-only, on **daily** bars:
- **Enter** when close breaks above the prior **50-day high** *and* is above the **100-day SMA** (regime filter).
- **Exit** when close breaks below the prior **20-day low** (trailing) or a **20% catastrophic stop**.
- **Ride winners** (no fixed take-profit), few positions, ~weeks-long holds.
- Up to **10 positions** at ~**10%** each (≈$100 on the $1k account), fractional shares.

All parameters live in `config/equity/strategy.yaml` (`mode: trend`, `trend:` block).

### News risk-filter
Before any breakout entry, `NewsSentiment` pulls recent headlines and Claude scores
them to JSON (`label / confidence / earnings_imminent / rationale`). The entry is
**vetoed** if news is strongly bearish (≥ `bearish_confidence_min`) or earnings is
imminent. It **can only block, never initiate** — the validated trend signal stays in
charge. Only breakout candidates are scored; results are Redis-cached; failures fail
**open**. Configured under `news:` in `strategy.yaml`.

---

## Backtester

Validate any idea over history **before** trusting it:

```bash
# trend-following over ~4 years across the configured universe
PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
  python -m trader.backtest.run --mode trend --days 1500

# intraday mean-reversion (the legacy/losing strategy), 60 days
PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
  python -m trader.backtest.run --mode meanrev --days 60
```

No-lookahead (signal at close → act next open), intrabar stop/target, modeled
slippage. Reports per-symbol + portfolio: win%, expectancy/trade, profit factor,
total return, max drawdown, Sharpe, vs buy-and-hold.

> **Note:** stock data is fetched **split/dividend-adjusted** (`adjustment=ALL`) —
> raw data shows splits as fake crashes and corrupts results.

---

## Eval harness (does the news filter earn its veto?)

The trend engine is backtested; the **LLM was not**. `trader.eval` closes that gap by
replaying history twice through the *same* simulator — once unfiltered, once with the
live prompt scoring point-in-time headlines and the live `should_block` thresholds:

```bash
# what would this cost? (offline, no LLM calls)
PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
  python -m trader.eval.run --days 1500 --plan

# the real thing; verdicts are cached to .eval-cache/ so re-runs are free
PYTHONPATH=src TRADER_CONFIG_DIR=config/equity \
  python -m trader.eval.run --days 1500 --out eval-news.json
```

Reports baseline-vs-filtered expectancy / PF / drawdown / Sharpe, plus **block
precision** — the share of blocks that landed on a trade that would have lost.

Why precision and not accuracy: the filter can only *block*. A false positive kills a
trade drawn from a **+5.8% expectancy** distribution; a false negative just returns
you to baseline. Blocking at random already scores ~58% (the base loss rate), so a
filter has to beat that to be worth anything.

**Reading the output.** The realized cost of the filter is the **expectancy/return
delta**, not the sum of the blocked trades' returns — a veto usually *delays* an entry
(the trend signal is re-evaluated the next day) rather than cancelling it, so most of
the move is recaptured. The report labels each block `re-entered later` vs
`leg abandoned` for exactly this reason.

**Contamination, stated plainly:** the model knows how these dates turned out. That
biases the filter toward looking good, so read a poor result as a *lower bound* on
how poor it is, and don't treat a good result as proof of an edge.

Cache keys include the prompt version and model, so **changing the prompt invalidates
every verdict** — that is the regression signal. Bump `PROMPT_VERSION` in
`src/trader/news.py` whenever `_SYSTEM` changes.

The live path is instrumented to feed this: each tick's journal `news` entries now
record the `headlines`, `model`, and `prompt_version` behind every verdict, so
production decisions can be re-scored against a future prompt.

---

## Risk controls

- **Position size cap** — max % of equity per position (`risk.yaml`).
- **Max open positions** — concurrency cap.
- **Daily drawdown kill-switch** (`risk/drawdown.py`) — if equity falls ≥ `daily_drawdown_pct` below the day's anchor, liquidate this agent's symbols and pause until next day.
- **Whitelist** — only configured symbols are tradeable.
- **Long-only / software stops** for fractional + crypto (no resting bracket possible).

---

## Configuration (per profile)

```
config/<profile>/watchlist.yaml   # symbols + asset class
config/<profile>/risk.yaml        # max_position_pct, max_open_positions,
                                  # daily_drawdown_pct, long_only, require_stop_loss
config/<profile>/strategy.yaml    # mode (trend|llm), tick interval, trend{}, news{}, ...
```
`config/` is volume-mounted into the containers, so most tuning is **just a restart**
(no rebuild). Code changes need an image rebuild.

---

## Running it

### Prerequisites
- Docker (with a working Compose/buildx — see *Operational notes*).
- A paper Alpaca account → `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in `.env`.
- **Claude auth via subscription** (no API key): `claude setup-token` →
  `CLAUDE_CODE_OAUTH_TOKEN=...` in `.env`. The Agent SDK ships a bundled CLI and uses
  this token; `ANTHROPIC_API_KEY` is blanked in compose so it can't override it.

### Start
```bash
cp .env.example .env          # fill in keys + CLAUDE_CODE_OAUTH_TOKEN
docker compose up -d --build  # starts agent-equity + alpaca-mcp + redis + mongo + mongo-express
# crypto agent is gated behind a profile:
docker compose --profile crypto up -d agent-crypto
```

### Observe
- **Trades / P&L:** Alpaca paper dashboard (app.alpaca.markets, Paper mode).
- **Decisions / news:** Mongo journal at **http://localhost:8081** (mongo-express, `admin`/`trader`) → `trader` DB → `ticks` collection. Each doc carries `agent`, `snapshot`, `reasoning`, `tool_calls` (actions incl. `BLOCKED`), `news`, `order_ids`.
- **Logs:** `docker compose logs -f agent-equity | grep "trend tick"`

---

## Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest pytest-asyncio fakeredis ...
pytest                    # 70 tests: indicators, risk gate, drawdown, backtest, news filter, eval harness
```
Tests need no network/Alpaca/Redis/Mongo (LLM and data calls are stubbed).

---

## Operational notes / gotchas

- **Auth:** subscription OAuth token, not an API key. A truncated `setup-token` fails
  with `401 Invalid bearer token` (a valid one is ~108 chars).
- **Docker on this host (colima):** Compose's `buildx bake` hangs and the
  `osxkeychain` cred helper is broken. Workaround: build images directly with
  `DOCKER_BUILDKIT=1 docker build ...` and run compose with
  `DOCKER_CONFIG=/tmp/dockercfg_trader` (a config copy with `credsStore` stripped).
- **Changing account balance mid-day** leaves a stale Redis drawdown anchor → false
  kill-switch. Clear `trader:<agent>:anchor:*` / `:paused:*` and restart.
- **alpaca-py ≥ 0.43** returns news under `res.data['news']` (older: `res.news`).
- **Alpaca paper crypto** does not fill orders reliably → crypto agent paused.

---

## Honest caveats

- The trend edge was measured on **mega-cap survivors over a bull-ish multi-year
  window** — real validation needs out-of-sample + a broader/delisted-inclusive
  universe + walk-forward.
- Fractional positions have **no resting stops** (managed on ticks during market
  hours) → **overnight/gap risk**.
- **News "alpha" is not a proven signal** — it's a risk filter. Its *cost* is now
  measurable offline (`trader.eval`, above), but the measurement is contaminated by
  the model's hindsight, so it bounds the damage rather than proving an edge.
- Beating buy-and-hold consistently is hard; the realistic win here is *similar return
  with much lower drawdown/exposure*, plus a framework to keep testing.

## Out of scope
Live/real capital; secondary exchanges; multi-leg/short options; macro
(Fed/CPI) calendar; news-driven entries; a remote dashboard.
