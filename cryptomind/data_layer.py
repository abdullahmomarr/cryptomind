"""Live data layer: real OHLCV from a public exchange + technical indicators.

We use ccxt against Binance's public endpoints (no API key needed). Indicators
(RSI and a fast/slow moving-average crossover) are implemented in pure Python so
they are easy to read in the report and easy to unit-test without numeric
dependencies.

The key object produced here is a *market snapshot*: a small dict describing the
market condition at one point in time. The whole memory loop revolves around
these snapshots (we store them, and we measure similarity between them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import config


# A single OHLCV candle as returned by ccxt: [timestamp_ms, open, high, low, close, volume]
Candle = Sequence[float]


# ---------------------------------------------------------------------------
# Indicator maths (pure Python, no numpy/pandas)
# ---------------------------------------------------------------------------
def simple_moving_average(values: Sequence[float], period: int) -> float | None:
    """Mean of the last `period` values, or None if there isn't enough data."""
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def relative_strength_index(closes: Sequence[float], period: int = config.RSI_PERIOD) -> float | None:
    """RSI using Wilder's smoothing.

    Returns a value in [0, 100], or None if there isn't enough data. RSI > 70 is
    conventionally "overbought", RSI < 30 "oversold".
    """
    if len(closes) < period + 1:
        return None

    # Seed the average gain/loss with the first `period` price changes.
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder-smooth across the remaining candles.
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0  # no downward movement -> maximally "strong"
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------
@dataclass
class MarketSnapshot:
    """A compact description of one market condition.

    The numeric *condition* fields (rsi, ma_gap_pct) are what the memory layer
    uses to measure similarity between situations.
    """

    pair: str
    timestamp_ms: int
    price: float
    rsi: float
    sma_fast: float
    sma_slow: float
    ma_gap_pct: float        # (fast - slow) / slow * 100; +ve = bullish crossover
    rsi_signal: str          # "overbought" | "oversold" | "neutral"
    ma_signal: str           # "bullish" | "bearish"

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "timestamp_ms": self.timestamp_ms,
            "price": round(self.price, 4),
            "rsi": round(self.rsi, 2),
            "sma_fast": round(self.sma_fast, 4),
            "sma_slow": round(self.sma_slow, 4),
            "ma_gap_pct": round(self.ma_gap_pct, 4),
            "rsi_signal": self.rsi_signal,
            "ma_signal": self.ma_signal,
        }


def build_snapshot(pair: str, candles: Sequence[Candle]) -> MarketSnapshot:
    """Compute indicators from a window of candles and return a snapshot.

    Uses the *last* candle in `candles` as "now". Raises ValueError if there
    aren't enough candles for the indicators to be valid.
    """
    if len(candles) < config.MIN_CANDLES:
        raise ValueError(
            f"Need at least {config.MIN_CANDLES} candles for indicators, got {len(candles)}"
        )

    closes = [c[4] for c in candles]
    last = candles[-1]

    sma_fast = simple_moving_average(closes, config.SMA_FAST)
    sma_slow = simple_moving_average(closes, config.SMA_SLOW)
    rsi = relative_strength_index(closes, config.RSI_PERIOD)

    # build_snapshot only runs with >= MIN_CANDLES, so these are never None.
    assert sma_fast is not None and sma_slow is not None and rsi is not None

    ma_gap_pct = (sma_fast - sma_slow) / sma_slow * 100.0

    if rsi >= config.RSI_OVERBOUGHT:
        rsi_signal = "overbought"
    elif rsi <= config.RSI_OVERSOLD:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    ma_signal = "bullish" if ma_gap_pct >= 0 else "bearish"

    return MarketSnapshot(
        pair=pair,
        timestamp_ms=int(last[0]),
        price=float(last[4]),
        rsi=rsi,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        ma_gap_pct=ma_gap_pct,
        rsi_signal=rsi_signal,
        ma_signal=ma_signal,
    )


# ---------------------------------------------------------------------------
# Exchange access (ccxt)
# ---------------------------------------------------------------------------
class MarketData:
    """Thin wrapper over a ccxt exchange for the data we need.

    Created lazily so that importing this module (e.g. during unit tests) never
    requires network access. Network errors are surfaced as RuntimeError with a
    readable message rather than leaking ccxt internals everywhere.
    """

    def __init__(self, exchange_id: str = config.EXCHANGE):
        import ccxt  # imported here so tests that don't touch the network stay light

        try:
            self._exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        except AttributeError as exc:  # pragma: no cover - misconfiguration only
            raise RuntimeError(f"Unknown exchange '{exchange_id}'") from exc

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = config.DEFAULT_TIMEFRAME,
        limit: int = 200,
        since_ms: int | None = None,
    ) -> list[Candle]:
        """Fetch OHLCV candles. `since_ms` pulls history from that point onward."""
        try:
            return self._exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as exc:  # ccxt raises a wide variety of network errors
            raise RuntimeError(f"Failed to fetch OHLCV for {pair}: {exc}") from exc

    def current_price(self, pair: str) -> float:
        """Return the latest traded price for a pair."""
        try:
            ticker = self._exchange.fetch_ticker(pair)
            return float(ticker["last"])
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch price for {pair}: {exc}") from exc

    def latest_snapshot(
        self, pair: str, timeframe: str = config.DEFAULT_TIMEFRAME
    ) -> MarketSnapshot:
        """Fetch recent candles and compute the current market snapshot."""
        candles = self.fetch_ohlcv(pair, timeframe=timeframe, limit=config.MIN_CANDLES + 50)
        return build_snapshot(pair, candles)
