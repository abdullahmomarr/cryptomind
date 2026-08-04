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

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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

    def price_at(self, pair: str, at_unix_seconds: float, timeframe: str = "1m") -> float:
        """Return the traded price at (or immediately after) a past moment.

        This is what `memory.verify_outcomes` grades against, so that a decision
        is always judged on the price its verification window actually points
        at, no matter how late the verification sweep runs. If the moment is
        still in the future we fall back to the current price, which is the best
        estimate available and only happens within one candle of the boundary.
        """
        since_ms = int(at_unix_seconds * 1000)
        candles = self.fetch_ohlcv(pair, timeframe=timeframe, limit=2, since_ms=since_ms)
        if not candles:
            # Nothing at/after that point yet -> the window has only just closed.
            return self.current_price(pair)
        return float(candles[0][4])

    def latest_snapshot(
        self, pair: str, timeframe: str = config.DEFAULT_TIMEFRAME
    ) -> MarketSnapshot:
        """Fetch recent candles and compute the current market snapshot."""
        candles = self.fetch_ohlcv(pair, timeframe=timeframe, limit=config.MIN_CANDLES + 50)
        return build_snapshot(pair, candles)


# ---------------------------------------------------------------------------
# Reproducible candle cache (for evaluation)
# ---------------------------------------------------------------------------
def _cache_path(pair: str, timeframe: str, limit: int, since_ms: int | None, cache_dir: Path) -> Path:
    """Deterministic filename for one (pair, timeframe, span) request."""
    safe_pair = pair.replace("/", "-")
    stamp = "latest" if since_ms is None else str(since_ms)
    return Path(cache_dir) / f"{config.EXCHANGE}_{safe_pair}_{timeframe}_{limit}_{stamp}.json"


def load_candles(
    pair: str,
    *,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    limit: int = 500,
    since_ms: int | None = None,
    cache_dir: Path = config.CACHE_DIR,
    refresh: bool = False,
    market: "MarketData | None" = None,
) -> list[Candle]:
    """Fetch candles, caching them on disk so evaluation runs are reproducible.

    An evaluation that re-downloads "the last 30 days" every time it runs cannot
    be reproduced — not by a marker, and not by us next week. Caching the exact
    candles a result was computed from fixes that, and lets the whole test suite
    and the sweeps run offline.
    """
    path = _cache_path(pair, timeframe, limit, since_ms, cache_dir)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    market = market or MarketData()
    candles = market.fetch_ohlcv(pair, timeframe=timeframe, limit=limit, since_ms=since_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candles), encoding="utf-8")
    return candles


def candle_price_provider(candles: Sequence[Candle]) -> "Callable[[str, float], float]":
    """Build a price provider that reads a fixed candle series.

    Lets `memory.verify_outcomes` be exercised against historical data — and in
    tests — with exactly the same code path it uses against the live exchange.
    Returns the close of the first candle at or after the requested time.
    """
    ordered = sorted(candles, key=lambda c: c[0])
    times = [c[0] for c in ordered]

    def provider(pair: str, at_unix_seconds: float) -> float:
        idx = bisect.bisect_left(times, int(at_unix_seconds * 1000))
        if idx >= len(ordered):
            raise RuntimeError(
                f"No candle at or after {at_unix_seconds} for {pair} in the loaded series"
            )
        return float(ordered[idx][4])

    return provider
