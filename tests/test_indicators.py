"""Tests for the indicator maths in the data layer.

The indicators are the input to every decision the agent makes, so an error
here would propagate silently into every result in the evaluation. They are
checked against known reference values rather than against themselves.
"""

import pytest

from cryptomind import config
from cryptomind.data_layer import (
    build_snapshot,
    candle_price_provider,
    relative_strength_index,
    simple_moving_average,
)


# --- moving average ----------------------------------------------------------
def test_sma_averages_the_last_period_values_only():
    assert simple_moving_average([1, 2, 3, 10, 20, 30], 3) == pytest.approx(20.0)


def test_sma_returns_none_without_enough_data():
    assert simple_moving_average([1, 2], 5) is None


# --- RSI ---------------------------------------------------------------------
# The standard 15-close worked example for RSI(14). Fifteen closes give exactly
# fourteen price changes, so the first RSI value is the plain Wilder seed with
# no smoothing iterations — which makes it checkable entirely by hand:
#
#   gains  = 0.06 + 0.72 + 0.50 + 0.27 + 0.32 + 0.42 + 0.24 + 0.14 + 0.67 = 3.34
#   losses = 0.25 + 0.54 + 0.19 + 0.42                                    = 1.40
#   avg gain = 3.34/14 = 0.238571,  avg loss = 1.40/14 = 0.10
#   RS  = 2.385714
#   RSI = 100 - 100/(1 + RS) = 70.4641
#
# (Textbooks often quote 70.53 alongside this series; that figure comes from a
# slightly different set of closes. The arithmetic above is the reference here.)
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
]


def test_rsi_matches_the_hand_computed_reference():
    assert relative_strength_index(WILDER_CLOSES, period=14) == pytest.approx(70.4641, abs=0.001)


def test_rsi_is_100_when_price_only_rises():
    """No downward movement at all -> the formula's maximum."""
    assert relative_strength_index(list(range(1, 40)), period=14) == 100.0


def test_rsi_is_0_when_price_only_falls():
    assert relative_strength_index(list(range(40, 1, -1)), period=14) == pytest.approx(0.0)


def test_rsi_is_neutral_for_symmetric_movement():
    """Equal-sized ups and downs should sit near the 50 line."""
    closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(60)]
    assert relative_strength_index(closes, period=14) == pytest.approx(50.0, abs=5.0)


def test_rsi_returns_none_without_enough_data():
    assert relative_strength_index([1.0, 2.0, 3.0], period=14) is None


# --- snapshot ----------------------------------------------------------------
def _candles(closes):
    """Minimal OHLCV rows: only timestamp and close are used by the indicators."""
    return [[i * 3_600_000, c, c, c, c, 1.0] for i, c in enumerate(closes)]


def test_build_snapshot_rejects_too_little_history():
    with pytest.raises(ValueError):
        build_snapshot("BTC/USDT", _candles([100.0] * 5))


def test_build_snapshot_labels_a_rising_market_bullish():
    """A steady uptrend must put the fast MA above the slow one."""
    candles = _candles([100.0 + i for i in range(config.MIN_CANDLES + 5)])
    snap = build_snapshot("BTC/USDT", candles)

    assert snap.ma_gap_pct > 0
    assert snap.ma_signal == "bullish"
    assert snap.rsi_signal == "overbought"      # nothing but gains
    assert snap.price == candles[-1][4]


def test_build_snapshot_labels_a_falling_market_bearish():
    candles = _candles([200.0 - i for i in range(config.MIN_CANDLES + 5)])
    snap = build_snapshot("BTC/USDT", candles)

    assert snap.ma_gap_pct < 0
    assert snap.ma_signal == "bearish"
    assert snap.rsi_signal == "oversold"


# --- historical price provider -----------------------------------------------
def test_candle_price_provider_returns_the_price_at_the_requested_time():
    """Grading a past decision must read the price at that moment, not the last one."""
    candles = _candles([10.0, 20.0, 30.0, 40.0])
    provider = candle_price_provider(candles)

    assert provider("BTC/USDT", 0) == 10.0
    assert provider("BTC/USDT", 2 * 3600) == 30.0
    # Between candles -> the first one at or after the requested time.
    assert provider("BTC/USDT", 1.5 * 3600) == 30.0


def test_candle_price_provider_raises_past_the_end_of_the_series():
    provider = candle_price_provider(_candles([10.0, 20.0]))
    with pytest.raises(RuntimeError):
        provider("BTC/USDT", 99 * 3600)
