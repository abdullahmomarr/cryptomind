"""Tests for confidence calibration (calibrate_confidence)."""

from cryptomind import memory


def _past(was_correct: bool | None):
    """A retrieved-decision dict shaped like retrieve_similar output."""
    outcome = None if was_correct is None else {
        "actual_pct_change": 1.0 if was_correct else -1.0,
        "was_correct": was_correct,
        "exit_price": 101.0,
    }
    return {"id": 1, "pair": "BTC/USDT", "action": "BUY", "confidence": 50,
            "snapshot": {}, "distance": 0.0, "outcome": outcome}


def test_no_history_keeps_confidence_unchanged():
    calibrated, explanation = memory.calibrate_confidence(80, [])
    assert calibrated == 80
    assert "unchanged" in explanation.lower()


def test_ungraded_history_is_ignored():
    # Past decisions exist but none have outcomes yet -> treated as no history.
    calibrated, _ = memory.calibrate_confidence(80, [_past(None), _past(None)])
    assert calibrated == 80


def test_poor_history_pulls_confidence_down():
    # Agent was wrong every time in similar setups -> calibrated should drop.
    past = [_past(False) for _ in range(3)]
    calibrated, explanation = memory.calibrate_confidence(80, past)
    assert calibrated < 80
    assert "0/3" in explanation or "0%" in explanation


def test_strong_history_pulls_low_confidence_up():
    # Agent was always right in similar setups -> calibrated should rise.
    past = [_past(True) for _ in range(3)]
    calibrated, _ = memory.calibrate_confidence(40, past)
    assert calibrated > 40


def test_calibrated_value_between_raw_and_hitrate():
    # Mixed 50% hit-rate with raw 90 -> result lands between 50 and 90.
    past = [_past(True), _past(False)]
    calibrated, _ = memory.calibrate_confidence(90, past)
    assert 50 <= calibrated <= 90


def test_more_samples_trust_history_more():
    # Same 0% hit-rate, but more samples should pull confidence lower
    # (history is trusted more as n grows).
    few = memory.calibrate_confidence(80, [_past(False)])[0]
    many = memory.calibrate_confidence(80, [_past(False) for _ in range(8)])[0]
    assert many < few
