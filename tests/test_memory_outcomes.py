"""Tests for outcome verification logic (grade_outcome / record_outcome / verify_outcomes)."""

import time

import pytest

from cryptomind import memory
from tests.conftest import make_snapshot


# --- grade_outcome: the core correctness rule --------------------------------
@pytest.mark.parametrize(
    "action,pct_change,expected",
    [
        ("BUY", 2.0, True),     # price up -> BUY correct
        ("BUY", -2.0, False),   # price down -> BUY wrong
        ("BUY", 0.1, False),    # inside band -> not enough of a rise
        ("SELL", -2.0, True),   # price down -> SELL correct
        ("SELL", 2.0, False),   # price up -> SELL wrong
        ("HOLD", 0.2, True),    # stayed within band -> HOLD correct
        ("HOLD", 3.0, False),   # big move -> HOLD wrong
        ("HOLD", -3.0, False),
    ],
)
def test_grade_outcome(action, pct_change, expected):
    assert memory.grade_outcome(action, pct_change, band=0.5) is expected


def test_grade_outcome_rejects_unknown_action():
    with pytest.raises(ValueError):
        memory.grade_outcome("MOON", 1.0)


# --- record_outcome computes the % change and stores correctness -------------
def test_record_outcome_computes_pct_and_correctness(conn):
    rec_id = memory.log_recommendation(
        conn, pair="BTC/USDT", action="BUY", confidence=60, raw_confidence=60,
        predicted_direction="up", reasoning="t", entry_price=100.0,
        market_snapshot=make_snapshot(25, 1.0),
    )
    outcome = memory.record_outcome(
        conn, rec_id, entry_price=100.0, exit_price=105.0, action="BUY",
    )
    assert outcome["actual_pct_change"] == pytest.approx(5.0)
    assert outcome["was_correct"] is True


# --- verify_outcomes only grades old, ungraded recommendations ---------------
def test_verify_outcomes_grades_only_old_pending(conn):
    now = time.time()
    # Old, ungraded -> should be graded.
    old_id = memory.log_recommendation(
        conn, pair="BTC/USDT", action="BUY", confidence=60, raw_confidence=60,
        predicted_direction="up", reasoning="t", entry_price=100.0,
        market_snapshot=make_snapshot(25, 1.0), timestamp=now - 10 * 3600,
    )
    # Recent -> should be left alone (not old enough).
    memory.log_recommendation(
        conn, pair="ETH/USDT", action="SELL", confidence=60, raw_confidence=60,
        predicted_direction="down", reasoning="t", entry_price=50.0,
        market_snapshot=make_snapshot(75, -1.0), timestamp=now,
    )

    # Fake price provider: BTC rose 10% -> BUY should be graded correct.
    prices = {"BTC/USDT": 110.0, "ETH/USDT": 49.0}
    results = memory.verify_outcomes(conn, window_hours=4, price_provider=lambda p: prices[p])

    assert len(results) == 1
    assert results[0]["recommendation_id"] == old_id
    assert results[0]["was_correct"] is True

    # Second sweep finds nothing new (already graded; the other is too recent).
    assert memory.verify_outcomes(conn, window_hours=4, price_provider=lambda p: prices[p]) == []
