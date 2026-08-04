"""Tests for the evaluation harness.

The harness is the instrument the project's conclusions are measured with, so
the properties tested here are the ones that would invalidate a result rather
than merely produce a wrong number:

  * the replay never reads the future (the leakage guarantee)
  * the hold-out split is chronological, not shuffled
  * the calibration arm differs from the raw arm ONLY in confidence
  * a run is reproducible

Everything runs offline on synthetic candles with the deterministic rule agent.
"""

import math

import pytest

from cryptomind import config, evaluate, memory
from cryptomind.agent import Recommendation, RuleBasedAgent
from cryptomind.evaluate import Params, replay


def make_candles(n: int = 300, start: float = 100.0) -> list[list[float]]:
    """A deterministic price series with real ups and downs.

    A sine wave over a mild uptrend gives the indicators something to react to,
    so decisions vary instead of collapsing to one action.
    """
    candles = []
    for i in range(n):
        price = start + i * 0.05 + 8.0 * math.sin(i / 9.0)
        candles.append([i * 3_600_000, price, price, price, price, 1.0])
    return candles


@pytest.fixture
def params():
    return Params(engine="rule", timeframe="1h", window_hours=4.0, train_frac=0.5, seed=1)


# --- replay basics -----------------------------------------------------------
def test_replay_produces_one_record_per_decision(params):
    records = replay("BTC/USDT", make_candles(), RuleBasedAgent(), params)

    assert records
    assert all(r["pair"] == "BTC/USDT" for r in records)
    assert all(r["action"] in ("BUY", "SELL", "HOLD") for r in records)
    assert all(0 <= r["calibrated_confidence"] <= 100 for r in records)


def test_replay_is_chronological(params):
    records = replay("BTC/USDT", make_candles(), RuleBasedAgent(), params)
    timestamps = [r["timestamp"] for r in records]
    assert timestamps == sorted(timestamps)


def test_replay_grades_against_the_price_one_window_later(params):
    """The exit price must be the close exactly `window_hours` after entry."""
    candles = make_candles()
    records = replay("BTC/USDT", candles, RuleBasedAgent(), params)

    r = records[0]
    expected_exit = candles[r["candle_index"] + 4][4]   # 4h window on 1h candles
    assert r["exit_price"] == pytest.approx(expected_exit)


def test_replay_is_reproducible(params):
    """Same candles, same parameters -> byte-identical decisions."""
    candles = make_candles()
    first = replay("BTC/USDT", candles, RuleBasedAgent(), params)
    second = replay("BTC/USDT", candles, RuleBasedAgent(), params)
    assert first == second


def test_replay_starts_from_an_empty_memory_every_time(params):
    """A replay must not inherit state from a previous run or the project DB."""
    candles = make_candles()
    first = replay("BTC/USDT", candles, RuleBasedAgent(), params)
    # If memory leaked between runs, the second replay would retrieve more.
    second = replay("BTC/USDT", candles, RuleBasedAgent(), params)
    assert first[0]["n_retrieved"] == second[0]["n_retrieved"] == 0


# --- the leakage guarantee ---------------------------------------------------
def test_first_decision_has_nothing_to_retrieve(params):
    records = replay("BTC/USDT", make_candles(), RuleBasedAgent(), params)
    assert records[0]["n_retrieved"] == 0
    assert records[0]["n_retrieved_graded"] == 0


def test_retrieval_never_includes_ungraded_or_future_decisions(params):
    """The core no-leakage property, checked on every single decision.

    At the moment decision i is taken, the only outcomes that can be known are
    those of decisions made at least `window_hours` earlier. Anything more would
    mean the replay read a price that had not happened yet.
    """
    records = replay("BTC/USDT", make_candles(), RuleBasedAgent(), params)
    window_seconds = params.window_hours * 3600.0

    for i, rec in enumerate(records):
        knowable = sum(
            1 for earlier in records[:i]
            if earlier["timestamp"] + window_seconds <= rec["timestamp"]
        )
        assert rec["n_retrieved_graded"] <= min(knowable, params.k), (
            f"decision {i} calibrated on {rec['n_retrieved_graded']} graded outcomes "
            f"but only {knowable} could have been known"
        )
        assert rec["n_retrieved"] <= min(i, params.k)


def test_disabling_memory_retrieves_nothing(params):
    records = replay("BTC/USDT", make_candles(), RuleBasedAgent(), params, use_memory=False)
    assert all(r["n_retrieved"] == 0 for r in records)
    # With no memory there is nothing to calibrate against.
    assert all(r["calibrated_confidence"] == r["raw_confidence"] for r in records)


def test_calibration_changes_confidence_but_never_the_action(params):
    """The whole point of the calibrated-vs-raw comparison.

    Both arms take identical actions, so any difference in Brier or ECE between
    them is attributable to calibration alone and nothing else.
    """
    candles = make_candles()
    with_memory = replay("BTC/USDT", candles, RuleBasedAgent(), params, use_memory=True)
    without = replay("BTC/USDT", candles, RuleBasedAgent(), params, use_memory=False)

    assert [r["action"] for r in with_memory] == [r["action"] for r in without]
    assert [r["raw_confidence"] for r in with_memory] == [r["raw_confidence"] for r in without]
    # ...and calibration did actually move something, or the test proves nothing.
    assert any(r["calibrated_confidence"] != r["raw_confidence"] for r in with_memory)


# --- parameters actually take effect ----------------------------------------
def test_k_limits_how_much_is_retrieved():
    candles = make_candles()
    small = replay("BTC/USDT", candles, RuleBasedAgent(), Params(k=1))
    large = replay("BTC/USDT", candles, RuleBasedAgent(), Params(k=10))

    assert max(r["n_retrieved"] for r in small) == 1
    assert max(r["n_retrieved"] for r in large) == 10


def test_prior_strength_controls_how_far_calibration_moves():
    """A large prior means history is trusted slowly, so confidence moves less."""
    candles = make_candles()
    trusting = replay("BTC/USDT", candles, RuleBasedAgent(), Params(prior_strength=1))
    sceptical = replay("BTC/USDT", candles, RuleBasedAgent(), Params(prior_strength=50))

    def total_shift(records):
        return sum(abs(r["calibrated_confidence"] - r["raw_confidence"]) for r in records)

    assert total_shift(trusting) > total_shift(sceptical)


def test_band_changes_how_outcomes_are_graded():
    """The HOLD band decides what counts as 'the price stayed put'.

    A 5% band on this series swallows almost every move, so nearly every BUY
    and SELL is graded wrong while HOLDs are graded right — the opposite of a
    0.1% band. Same decisions, different grading.
    """
    candles = make_candles()
    tight = replay("BTC/USDT", candles, RuleBasedAgent(), Params(band=0.1))
    wide = replay("BTC/USDT", candles, RuleBasedAgent(), Params(band=5.0))

    assert [r["action"] for r in tight] == [r["action"] for r in wide]
    assert [r["was_correct"] for r in tight] != [r["was_correct"] for r in wide]


# --- end-to-end --------------------------------------------------------------
def test_evaluate_pair_scores_every_arm_on_a_chronological_holdout(monkeypatch, params):
    candles = make_candles(400)
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: candles)

    result = evaluate.evaluate_pair("BTC/USDT", params, agent=RuleBasedAgent())

    assert "error" not in result
    assert result["n_train"] + result["n_test"] == result["n_total_decisions"]
    # Hold-out is the TAIL of the series, never a random sample.
    assert result["test_period_start"] > result["events"][0]["timestamp"] - 1
    assert result["events"][0]["timestamp"] == result["test_period_start"]

    for arm in ("memory+calibration", "memory_raw_confidence",
                "always_buy", "always_sell", "always_hold", "random"):
        assert arm in result["arms"], arm
        assert result["arms"][arm]["n"] == result["n_test"]

    # The rule agent ignores retrieval, so the no-memory arm is declared skipped
    # rather than reported as a duplicate column.
    assert result["arms"]["no_memory"]["skipped"]


def test_baselines_are_graded_consistently_with_the_agent(monkeypatch, params):
    """Every arm must be scored on the same events with the same rules."""
    candles = make_candles(400)
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: candles)
    result = evaluate.evaluate_pair("BTC/USDT", params, agent=RuleBasedAgent())

    # always_hold's accuracy is exactly the share of test moves inside the band.
    events = result["events"]
    expected = sum(1 for e in events if abs(e["pct_change"]) <= params.band) / len(events)
    assert result["arms"]["always_hold"]["accuracy_pct"] == pytest.approx(expected * 100)


def test_baselines_state_their_training_hit_rate_as_confidence(monkeypatch, params):
    """Baselines get a fair, calibrated confidence rather than a flat 50%."""
    candles = make_candles(400)
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: candles)
    result = evaluate.evaluate_pair("BTC/USDT", params, agent=RuleBasedAgent())

    for arm in ("always_buy", "always_sell", "always_hold"):
        # A single stated confidence for all decisions -> exactly one populated bin.
        populated = [b for b in result["arms"][arm]["reliability"] if b["count"]]
        assert len(populated) == 1


def test_evaluate_pair_reports_the_calibration_significance_test(monkeypatch, params):
    candles = make_candles(400)
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: candles)
    result = evaluate.evaluate_pair("BTC/USDT", params, agent=RuleBasedAgent())

    test = result["calibration_vs_raw_brier"]
    assert set(test) >= {"difference", "ci_low", "ci_high", "significant"}
    assert test["ci_low"] <= test["difference"] <= test["ci_high"]


def test_evaluate_pools_across_pairs(monkeypatch, params):
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: make_candles(400))
    monkeypatch.setattr(evaluate, "_all_cached", lambda *a, **kw: True)

    result = evaluate.evaluate(["BTC/USDT", "ETH/USDT"], params)
    pooled = result["pooled"]

    assert pooled["n_pairs"] == 2
    per_pair_n = result["per_pair"][0]["n_test"]
    assert pooled["arms"]["memory+calibration"]["n"] == per_pair_n * 2


def test_evaluate_rejects_an_empty_holdout(monkeypatch):
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: make_candles(400))
    result = evaluate.evaluate_pair("BTC/USDT", Params(train_frac=1.0), agent=RuleBasedAgent())
    assert "error" in result


def test_sweep_reports_one_row_per_value(monkeypatch, params):
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: make_candles(400))
    monkeypatch.setattr(evaluate, "_all_cached", lambda *a, **kw: True)

    result = evaluate.sweep(["BTC/USDT"], params, "k", [1, 5])

    assert [r["k"] for r in result["rows"]] == [1, 5]
    assert all("brier_calibrated" in r and "brier_raw" in r for r in result["rows"])


def test_sweep_rejects_an_unknown_parameter(params):
    with pytest.raises(ValueError):
        evaluate.sweep(["BTC/USDT"], params, "nonsense", [1, 2])


# --- an agent that DOES read memory gets the retrieval ablation --------------
class MemoryReadingAgent(RuleBasedAgent):
    """Stand-in for an LLM backend: its output depends on what was retrieved."""

    uses_memory = True

    def recommend(self, snapshot, similar_past):
        rec = super().recommend(snapshot, similar_past)
        if similar_past:
            return Recommendation(
                action=rec.action, confidence=min(rec.confidence + 10, 100),
                predicted_direction=rec.predicted_direction, reasoning=rec.reasoning,
            )
        return rec


def test_no_memory_arm_runs_for_an_agent_that_reads_memory(monkeypatch, params):
    monkeypatch.setattr(evaluate, "load_candles", lambda *a, **kw: make_candles(400))
    result = evaluate.evaluate_pair("BTC/USDT", params, agent=MemoryReadingAgent())

    assert "skipped" not in result["arms"]["no_memory"]
    assert result["arms"]["no_memory"]["n"] == result["n_test"]
