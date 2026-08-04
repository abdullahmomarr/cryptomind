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
    provider = lambda pair, at: prices[pair]  # noqa: E731
    results = memory.verify_outcomes(conn, window_hours=4, price_provider=provider)

    assert len(results) == 1
    assert results[0]["recommendation_id"] == old_id
    assert results[0]["was_correct"] is True

    # Second sweep finds nothing new (already graded; the other is too recent).
    assert memory.verify_outcomes(conn, window_hours=4, price_provider=provider) == []


def test_verify_outcomes_grades_on_the_window_not_on_now(conn):
    """A late sweep must not stretch the horizon a decision is judged over.

    A call made 10 days ago with a 4h window is graded on the price 4h after it
    was made — not on today's price, which would silently judge it over 10 days
    and make it incomparable with every promptly-graded row in the same table.
    """
    now = time.time()
    made_at = now - 10 * 24 * 3600
    rec_id = memory.log_recommendation(
        conn, pair="BTC/USDT", action="BUY", confidence=60, raw_confidence=60,
        predicted_direction="up", reasoning="t", entry_price=100.0,
        market_snapshot=make_snapshot(25, 1.0), timestamp=made_at,
    )

    asked_for = []

    def provider(pair, at_ts):
        asked_for.append(at_ts)
        # Price 4h later was 101; price today is 300. Only the former is correct.
        return 101.0 if at_ts < now - 3600 else 300.0

    results = memory.verify_outcomes(conn, window_hours=4, price_provider=provider)

    assert len(results) == 1
    assert asked_for == [pytest.approx(made_at + 4 * 3600)]
    assert results[0]["exit_price"] == 101.0
    assert results[0]["actual_pct_change"] == pytest.approx(1.0)
    assert results[0]["horizon_hours"] == 4
    assert results[0]["recommendation_id"] == rec_id


def test_record_outcome_respects_a_custom_band(conn):
    """The HOLD band is a parameter the evaluation can sweep, not a constant."""
    rec_id = memory.log_recommendation(
        conn, pair="BTC/USDT", action="HOLD", confidence=50, raw_confidence=50,
        predicted_direction="flat", reasoning="t", entry_price=100.0,
        market_snapshot=make_snapshot(50, 0.0),
    )
    # A 1% move is outside the default 0.5% band but inside a 2% one.
    tight = memory.record_outcome(
        conn, rec_id, entry_price=100.0, exit_price=101.0, action="HOLD", band=0.5)
    assert tight["was_correct"] is False

    loose = memory.record_outcome(
        conn, rec_id, entry_price=100.0, exit_price=101.0, action="HOLD", band=2.0)
    assert loose["was_correct"] is True


def test_reset_clears_memory(conn):
    """Seeding twice must not be able to silently double the history."""
    for _ in range(3):
        rec_id = memory.log_recommendation(
            conn, pair="BTC/USDT", action="BUY", confidence=50, raw_confidence=50,
            predicted_direction="up", reasoning="t", entry_price=100.0,
            market_snapshot=make_snapshot(30, 1.0),
        )
        memory.record_outcome(conn, rec_id, entry_price=100.0, exit_price=102.0, action="BUY")

    assert memory.accuracy_stats(conn)["graded"] == 3
    assert memory.reset(conn) == 3
    assert memory.accuracy_stats(conn)["graded"] == 0
    assert memory.recent_recommendations(conn) == []
