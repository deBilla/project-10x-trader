import pytest

from trader.config import RiskConfig
from trader.risk.hook import RiskContext, evaluate_order

RISK = RiskConfig(
    max_position_pct=0.05,
    max_open_positions=3,
    daily_drawdown_pct=0.10,
    require_stop_loss=True,
    max_stop_distance_pct=0.05,
)
WHITELIST = {"SPY", "QQQ", "BTC/USD"}

PLACE = "mcp__alpaca__place_stock_order"


def _ctx(equity=10_000.0, open_syms=None, price=100.0, paused=False):
    return RiskContext(
        equity=equity,
        open_position_symbols=open_syms or [],
        latest_prices={"SPY": price, "QQQ": price, "BTC/USD": price},
        paused=paused,
    )


def _order(**kw):
    base = {"symbol": "SPY", "side": "buy", "quantity": 1, "stop_loss": 98.0}
    base.update(kw)
    return base


def test_valid_order_allowed():
    d = evaluate_order(PLACE, _order(), _ctx(), RISK, WHITELIST)
    assert d.decision == "allow"


def test_oversize_notional_denied():
    # 100 shares * $100 = $10,000 == 100% of equity, cap is 5%.
    d = evaluate_order(PLACE, _order(quantity=100), _ctx(), RISK, WHITELIST)
    assert d.decision == "deny"
    assert "notional" in d.reason


def test_non_whitelisted_symbol_denied():
    d = evaluate_order(PLACE, _order(symbol="TSLA"), _ctx(), RISK, WHITELIST)
    assert d.decision == "deny"
    assert "whitelist" in d.reason


def test_missing_stop_loss_denied():
    o = _order()
    o.pop("stop_loss")
    d = evaluate_order(PLACE, o, _ctx(), RISK, WHITELIST)
    assert d.decision == "deny"
    assert "stop_loss" in d.reason


def test_stop_too_far_denied():
    # stop at 80 vs price 100 = 20% distance, max is 5%.
    d = evaluate_order(PLACE, _order(stop_loss=80.0), _ctx(), RISK, WHITELIST)
    assert d.decision == "deny"
    assert "stop distance" in d.reason


def test_max_open_positions_denied():
    ctx = _ctx(open_syms=["QQQ", "BTC/USD", "AAPL"])  # already 3 open
    d = evaluate_order(PLACE, _order(symbol="SPY"), ctx, RISK, WHITELIST)
    assert d.decision == "deny"
    assert "max open positions" in d.reason


def test_adding_to_existing_position_not_blocked_by_cap():
    # SPY already open; 3 positions but SPY is one of them -> not a new position.
    ctx = _ctx(open_syms=["SPY", "QQQ", "BTC/USD"])
    d = evaluate_order(PLACE, _order(symbol="SPY"), ctx, RISK, WHITELIST)
    assert d.decision == "allow"


def test_paused_denies_entries():
    d = evaluate_order(PLACE, _order(), _ctx(paused=True), RISK, WHITELIST)
    assert d.decision == "deny"
    assert "paused" in d.reason


def test_close_position_always_allowed_even_when_paused():
    d = evaluate_order(
        "mcp__alpaca__close_position", {"symbol": "SPY"}, _ctx(paused=True),
        RISK, WHITELIST,
    )
    assert d.decision == "allow"


def test_notional_param_used_directly():
    d = evaluate_order(
        PLACE, _order(quantity=None, notional=400.0), _ctx(), RISK, WHITELIST
    )
    assert d.decision == "allow"


def test_nested_stop_loss_dict_parsed():
    d = evaluate_order(
        PLACE, _order(stop_loss={"stop_price": 98.0}), _ctx(), RISK, WHITELIST
    )
    assert d.decision == "allow"


LONG_ONLY = RiskConfig(
    max_position_pct=0.05, max_open_positions=3, daily_drawdown_pct=0.10,
    require_stop_loss=False, long_only=True,
)
PLACE_CRYPTO = "mcp__alpaca__place_crypto_order"


def test_long_only_denies_sell_entry():
    o = {"symbol": "BTC/USD", "side": "sell", "qty": 0.001}
    ctx = RiskContext(equity=10_000.0, open_position_symbols=[],
                      latest_prices={"BTC/USD": 60000.0}, paused=False)
    d = evaluate_order(PLACE_CRYPTO, o, ctx, LONG_ONLY, {"BTC/USD"})
    assert d.decision == "deny"
    assert "long-only" in d.reason


def test_long_only_allows_crypto_buy_without_bracket():
    # No stop_loss, require_stop_loss=False -> allowed (software stop handles it).
    o = {"symbol": "BTC/USD", "side": "buy", "qty": 0.001}
    ctx = RiskContext(equity=10_000.0, open_position_symbols=[],
                      latest_prices={"BTC/USD": 60000.0}, paused=False)
    d = evaluate_order(PLACE_CRYPTO, o, ctx, LONG_ONLY, {"BTC/USD"})
    assert d.decision == "allow", d.reason


def test_real_mcp_bracket_keys_allowed():
    # Matches the actual alpaca-mcp place_stock_order bracket payload.
    o = {
        "symbol": "SPY", "side": "buy", "qty": "4", "type": "market",
        "time_in_force": "day", "order_class": "bracket",
        "take_profit_limit_price": "104.0", "stop_loss_stop_price": "98.0",
    }
    d = evaluate_order(PLACE, o, _ctx(), RISK, WHITELIST)
    assert d.decision == "allow", d.reason
