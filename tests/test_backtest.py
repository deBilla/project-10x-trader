import numpy as np
import pandas as pd

from trader.config import RiskConfig, StrategyConfig
from trader.backtest.engine import backtest_symbol
from trader.backtest.metrics import compute_metrics
from trader.backtest.signals import signal
from trader.backtest.trend import backtest_trend, desired_long


def _active_strategy():
    # active profile: single-signal, no trend filter, no volume gate, RSI 40/60
    s = StrategyConfig()
    s.enforce_trend_filter = False
    s.entry_mode = "single_signal"
    s.require_volume_confirmation = False
    s.thresholds.rsi_oversold = 40
    s.thresholds.rsi_overbought = 60
    return s


def _snap(**kw):
    base = dict(rsi=50, rsi_oversold=40, rsi_overbought=60, bb_pct=0.5,
                ema_trend="up", price_vs_vwap="above", macd_hist=0.1,
                volume_surge=True, breaking_high=False, breaking_low=False)
    base.update(kw)
    return base


def test_signal_oversold_long():
    assert signal(_snap(rsi=25, bb_pct=0.1), _active_strategy()) == 1


def test_signal_overbought_short():
    assert signal(_snap(rsi=75, bb_pct=0.9), _active_strategy()) == -1


def test_signal_neutral_flat():
    assert signal(_snap(rsi=50, bb_pct=0.5), _active_strategy()) == 0


def test_signal_breakout_long():
    s = _snap(rsi=60, breaking_high=True, ema_trend="up", price_vs_vwap="above")
    assert signal(s, _active_strategy()) == 1


def test_volume_gate_blocks_when_required():
    strat = _active_strategy()
    strat.require_volume_confirmation = True
    assert signal(_snap(rsi=25, bb_pct=0.1, volume_surge=False), strat) == 0


def test_trend_filter_suppresses_meanrev_long_in_downtrend():
    strat = _active_strategy()
    strat.enforce_trend_filter = True
    s = _snap(rsi=25, bb_pct=0.1, ema_trend="down", price_vs_vwap="below")
    assert signal(s, strat) == 0


def _bars(closes, vol=2000.0):
    n = len(closes)
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": closes,
        "high": [c + 0.3 for c in closes],
        "low": [c - 0.3 for c in closes],
        "close": closes,
        "volume": [vol] * n,
    }, index=idx)


def test_engine_takes_meanrev_trade_on_dip_and_recovery():
    # flat -> sharp dip (oversold, lower band) -> recovery to/above target
    closes = [100.0] * 120 + list(np.linspace(100, 90, 20)) + list(np.linspace(90, 101, 20))
    trades = backtest_symbol("TEST", _bars(closes), _active_strategy(), RiskConfig())
    assert len(trades) >= 1
    # at least one long was taken
    assert any(t.direction == 1 for t in trades)
    m = compute_metrics(trades, position_fraction=0.2)
    assert m.n_trades == len(trades)
    assert -1.0 < m.expectancy < 1.0  # sane range


def _daily_bars(closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1e6] * n,
    }, index=idx)


def test_trend_rides_uptrend_and_profits():
    # long base, then a sustained uptrend the breakout system should catch and ride
    closes = [100.0] * 120 + list(np.linspace(100, 180, 120))
    trades = backtest_trend("UP", _daily_bars(closes), entry_channel=50,
                            exit_channel=20, trend_ma=100)
    assert len(trades) >= 1
    assert sum(t.ret_pct for t in trades) > 0  # net profitable on a clean uptrend


def test_trend_no_entry_in_downtrend():
    # steady downtrend: regime filter (close < SMA) should block longs
    closes = list(np.linspace(200, 100, 240))
    trades = backtest_trend("DOWN", _daily_bars(closes), entry_channel=50,
                            exit_channel=20, trend_ma=100)
    assert trades == []


def test_desired_long_true_in_uptrend_false_in_downtrend():
    up = _daily_bars([100.0] * 120 + list(np.linspace(100, 180, 120)))
    down = _daily_bars(list(np.linspace(200, 100, 240)))
    assert desired_long(up, 50, 20, 100, True) is True
    assert desired_long(down, 50, 20, 100, True) is False
