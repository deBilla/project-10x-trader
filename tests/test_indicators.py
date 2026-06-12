import numpy as np
import pandas as pd
import pytest

from trader.config import BreakoutParams, IndicatorParams, Thresholds
from trader.indicators import bars_to_dataframe, compute_snapshot


def _df(closes, volumes=None):
    n = len(closes)
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="5min", tz="UTC")
    volumes = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


PARAMS = IndicatorParams()
THRESH = Thresholds()


def test_rsi_all_gains_is_100():
    closes = [100 + i for i in range(60)]  # monotonically increasing
    snap = compute_snapshot("SPY", _df(closes), PARAMS, THRESH)
    assert snap["rsi"] == 100.0
    assert snap["ema_trend"] == "up"


def test_rsi_all_losses_is_0():
    closes = [200 - i for i in range(60)]  # monotonically decreasing
    snap = compute_snapshot("SPY", _df(closes), PARAMS, THRESH)
    assert snap["rsi"] == 0.0
    assert snap["ema_trend"] == "down"


def test_bollinger_mid_equals_mean_of_window():
    closes = [100.0] * 40
    closes[-20:] = list(np.linspace(100, 120, 20))
    snap = compute_snapshot("SPY", _df(closes), PARAMS, THRESH)
    expected_mid = float(np.mean(closes[-PARAMS.bbands_period:]))
    assert snap["bb_mid"] == pytest.approx(expected_mid, abs=1e-4)
    # Bands are symmetric around the mid.
    assert snap["bb_upper"] - snap["bb_mid"] == pytest.approx(
        snap["bb_mid"] - snap["bb_lower"], abs=1e-4
    )


def test_volume_surge_flag():
    closes = [100 + (i % 3) for i in range(60)]
    vols = [1000.0] * 59 + [5000.0]  # last bar spikes
    snap = compute_snapshot("SPY", _df(closes, vols), PARAMS, THRESH)
    assert snap["volume_surge"] is True
    assert snap["volume_ratio"] > THRESH.volume_surge_mult


def test_breakout_flags_detect_new_high():
    # Flat at 100, then the final bar pops to a new high above the prior channel.
    closes = [100.0] * 59 + [105.0]
    snap = compute_snapshot("SPY", _df(closes), PARAMS, THRESH, BreakoutParams())
    assert snap["breaking_high"] is True
    assert snap["breaking_low"] is False
    assert snap["channel_high"] < snap["price"]


def test_breakout_fields_absent_when_not_requested():
    closes = [100 + (i % 4) for i in range(60)]
    snap = compute_snapshot("SPY", _df(closes), PARAMS, THRESH)  # no breakout arg
    assert "breaking_high" not in snap


def test_too_few_bars_raises():
    with pytest.raises(ValueError):
        compute_snapshot("SPY", _df([100, 101, 102]), PARAMS, THRESH)


def test_bars_to_dataframe_accepts_short_keys():
    bars = [
        {"t": "2026-01-01T09:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        {"t": "2026-01-01T09:35:00Z", "o": 1.5, "h": 2.5, "l": 1, "c": 2, "v": 20},
    ]
    df = bars_to_dataframe(bars)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 2.0
