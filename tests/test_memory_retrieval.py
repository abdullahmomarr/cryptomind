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


# --- as_of: the no-leakage guarantee an offline evaluation depends on --------
def test_as_of_excludes_decisions_made_later(conn):
    """A replay must not retrieve decisions taken after the one being scored."""
    past = memory.log_recommendation(
        conn, pair="BTC/USDT", action="BUY", confidence=50, raw_confidence=50,
        predicted_direction="x", reasoning="x", entry_price=100.0,
        market_snapshot=make_snapshot(50, 0.0), timestamp=1_000.0,
    )
    memory.log_recommendation(   # taken later — invisible from t=2000
        conn, pair="BTC/USDT", action="SELL", confidence=50, raw_confidence=50,
        predicted_direction="x", reasoning="x", entry_price=100.0,
        market_snapshot=make_snapshot(50, 0.0), timestamp=9_000.0,
    )

    results = memory.retrieve_similar(
        conn, make_snapshot(50, 0.0), "BTC/USDT", k=5, as_of=2_000.0)

    assert [r["id"] for r in results] == [past]


def test_as_of_hides_outcomes_not_yet_graded(conn):
    """An outcome graded in the future must read as 'not yet known'.

    Otherwise a replay would calibrate on the result of a decision whose
    verification window had not closed yet at the moment being simulated.
    """
    rec_id = memory.log_recommendation(
        conn, pair="BTC/USDT", action="BUY", confidence=50, raw_confidence=50,
        predicted_direction="x", reasoning="x", entry_price=100.0,
        market_snapshot=make_snapshot(50, 0.0), timestamp=1_000.0,
    )
    # Graded at t=5000, i.e. after the moment we are simulating (t=2000).
    memory.record_outcome(
        conn, rec_id, entry_price=100.0, exit_price=105.0, action="BUY", checked_at=5_000.0)

    early = memory.retrieve_similar(
        conn, make_snapshot(50, 0.0), "BTC/USDT", k=5, as_of=2_000.0)
    assert early[0]["outcome"] is None          # not knowable yet

    later = memory.retrieve_similar(
        conn, make_snapshot(50, 0.0), "BTC/USDT", k=5, as_of=6_000.0)
    assert later[0]["outcome"]["was_correct"] is True   # knowable now


def test_without_as_of_everything_is_visible(conn):
    """Live calls pass no as_of: 'all of memory' and 'knowable now' coincide."""
    _seed(conn, rsi=50, ma_gap=0.0)
    _seed(conn, rsi=51, ma_gap=0.0)
    assert len(memory.retrieve_similar(conn, make_snapshot(50, 0.0), "BTC/USDT", k=5)) == 2


def test_similarity_fields_are_swappable(conn):
    """The distance metric is a parameter, so the evaluation can ablate it."""
    a, b = make_snapshot(50, 0.0), make_snapshot(90, 0.0)
    # Judged on RSI the two are far apart; judged on the MA gap alone they match.
    assert memory.snapshot_distance(a, b, fields=("rsi",)) > 0
    assert memory.snapshot_distance(a, b, fields=("ma_gap_pct",)) == 0.0
