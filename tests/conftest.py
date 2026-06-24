"""Shared pytest fixtures: an in-memory database needs no network or files."""

import pytest

from cryptomind import memory


@pytest.fixture
def conn():
    """A fresh in-memory SQLite memory store for each test."""
    connection = memory.connect(":memory:")
    yield connection
    connection.close()


def make_snapshot(rsi: float, ma_gap_pct: float, price: float = 100.0) -> dict:
    """Minimal market snapshot dict for similarity/calibration tests."""
    return {
        "pair": "BTC/USDT",
        "timestamp_ms": 0,
        "price": price,
        "rsi": rsi,
        "sma_fast": price,
        "sma_slow": price,
        "ma_gap_pct": ma_gap_pct,
        "rsi_signal": "neutral",
        "ma_signal": "bullish" if ma_gap_pct >= 0 else "bearish",
    }
