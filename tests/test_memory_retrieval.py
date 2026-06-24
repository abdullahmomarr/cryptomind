"""Tests for similarity retrieval (snapshot_distance / retrieve_similar)."""

from cryptomind import memory
from tests.conftest import make_snapshot


def test_distance_zero_for_identical_snapshots():
    a = make_snapshot(50, 1.0)
    assert memory.snapshot_distance(a, a) == 0.0


def test_distance_grows_with_difference():
    base = make_snapshot(50, 0.0)
    near = make_snapshot(55, 0.0)
    far = make_snapshot(90, 0.0)
    assert memory.snapshot_distance(base, near) < memory.snapshot_distance(base, far)


def _seed(conn, rsi, ma_gap, action="BUY"):
    return memory.log_recommendation(
        conn, pair="BTC/USDT", action=action, confidence=50, raw_confidence=50,
        predicted_direction="x", reasoning="x", entry_price=100.0,
        market_snapshot=make_snapshot(rsi, ma_gap),
    )


def test_retrieve_similar_orders_by_closeness_and_respects_k(conn):
    # Three past decisions at different RSI levels.
    _seed(conn, rsi=20, ma_gap=0.0)   # far from query (RSI 52)
    near_id = _seed(conn, rsi=50, ma_gap=0.0)   # closest
    _seed(conn, rsi=80, ma_gap=0.0)   # far

    query = make_snapshot(52, 0.0)
    results = memory.retrieve_similar(conn, query, "BTC/USDT", k=2)

    assert len(results) == 2                      # k respected
    assert results[0]["id"] == near_id            # nearest first
    assert results[0]["distance"] <= results[1]["distance"]


def test_retrieve_similar_filters_by_pair(conn):
    _seed(conn, rsi=50, ma_gap=0.0)  # BTC
    memory.log_recommendation(
        conn, pair="ETH/USDT", action="BUY", confidence=50, raw_confidence=50,
        predicted_direction="x", reasoning="x", entry_price=10.0,
        market_snapshot=make_snapshot(50, 0.0),
    )
    results = memory.retrieve_similar(conn, make_snapshot(50, 0.0), "ETH/USDT", k=5)
    assert len(results) == 1
    assert results[0]["pair"] == "ETH/USDT"


def test_retrieve_similar_includes_outcome_when_present(conn):
    rec_id = _seed(conn, rsi=50, ma_gap=0.0)
    memory.record_outcome(conn, rec_id, entry_price=100.0, exit_price=103.0, action="BUY")

    results = memory.retrieve_similar(conn, make_snapshot(50, 0.0), "BTC/USDT", k=1)
    assert results[0]["outcome"] is not None
    assert results[0]["outcome"]["was_correct"] is True


def test_retrieve_similar_empty_memory_returns_empty(conn):
    assert memory.retrieve_similar(conn, make_snapshot(50, 0.0), "BTC/USDT", k=3) == []
