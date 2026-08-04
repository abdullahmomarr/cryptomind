"""Tests for the evaluation metrics.

These matter more than usual: every number in the Evaluation chapter comes out
of this module, so a silent error here would be an error in the report's
conclusions rather than a bug a user would notice.
"""

import pytest

from cryptomind import metrics


# --- accuracy ----------------------------------------------------------------
def test_accuracy_counts_correct_fraction():
    assert metrics.accuracy([True, True, False, False]) == 0.5
    assert metrics.accuracy([]) == 0.0


# --- Brier -------------------------------------------------------------------
def test_brier_perfect_forecaster_scores_zero():
    """100% confident and right, 0% confident and wrong -> no error at all."""
    assert metrics.brier_score([100, 0], [True, False]) == pytest.approx(0.0)


def test_brier_maximally_wrong_forecaster_scores_one():
    assert metrics.brier_score([100, 0], [False, True]) == pytest.approx(1.0)


def test_brier_of_a_constant_fifty_percent_is_the_reference_quarter():
    """The line every arm is compared against: shrugging scores 0.25."""
    assert metrics.brier_score([50] * 4, [True, False, True, False]) == pytest.approx(0.25)


def test_brier_punishes_overconfidence_more_than_hedging():
    """The reason Brier is the headline metric and accuracy is not.

    Both forecasters get the same call wrong; the confident one is penalised
    far harder, which is exactly the behaviour calibration is meant to induce.
    """
    confident_and_wrong = metrics.brier_score([95], [False])
    hedged_and_wrong = metrics.brier_score([55], [False])
    assert confident_and_wrong > hedged_and_wrong


# --- reliability / ECE -------------------------------------------------------
def test_reliability_bins_group_by_confidence():
    bins = metrics.reliability_bins([15, 15, 85, 85], [False, False, True, True], n_bins=10)
    populated = [b for b in bins if b["count"]]
    assert len(populated) == 2
    assert populated[0]["mean_confidence"] == 15
    assert populated[0]["accuracy"] == 0.0
    assert populated[1]["accuracy"] == 100.0


def test_reliability_last_bin_includes_full_confidence():
    """A 100% call must land in a bucket rather than vanishing off the end."""
    bins = metrics.reliability_bins([100], [True], n_bins=10)
    assert bins[-1]["count"] == 1


def test_ece_is_zero_for_a_perfectly_calibrated_forecaster():
    """Says 100% on four calls and is right on all four; says 0% and is wrong."""
    ece = metrics.expected_calibration_error([100, 100, 0, 0], [True, True, False, False])
    assert ece == pytest.approx(0.0)


def test_ece_measures_the_gap_in_percentage_points():
    """Always says 90%, is right half the time -> 40 points of overconfidence."""
    ece = metrics.expected_calibration_error([90] * 4, [True, False, True, False])
    assert ece == pytest.approx(40.0)


# --- discrimination ----------------------------------------------------------
def test_auc_is_one_when_confidence_perfectly_ranks_outcomes():
    assert metrics.discrimination_auc([90, 80, 20, 10],
                                      [True, True, False, False]) == pytest.approx(1.0)


def test_auc_is_zero_when_confidence_ranks_outcomes_backwards():
    assert metrics.discrimination_auc([10, 20, 80, 90],
                                      [True, True, False, False]) == pytest.approx(0.0)


def test_auc_of_a_constant_forecaster_is_a_coin_flip():
    """The whole reason AUC is reported alongside Brier.

    'Always BUY at 27%' is perfectly calibrated and scores a flattering Brier,
    but its confidence carries no information whatsoever — AUC says so.
    """
    assert metrics.discrimination_auc([27] * 6,
                                      [True, False, True, False, True, False]) == 0.5


def test_auc_is_none_when_every_call_went_the_same_way():
    assert metrics.discrimination_auc([50, 60], [True, True]) is None


def test_resolution_is_zero_for_a_constant_forecaster():
    """A forecaster that never varies cannot separate anything, by definition."""
    d = metrics.brier_decomposition([40] * 8, [True, False, True, False,
                                               True, False, True, False])
    assert d["resolution"] == pytest.approx(0.0)


def test_resolution_is_positive_for_a_discriminating_forecaster():
    d = metrics.brier_decomposition([95, 95, 5, 5], [True, True, False, False])
    assert d["resolution"] > 0


def test_murphy_decomposition_reconstructs_the_brier_score():
    """Brier = reliability - resolution + uncertainty, to binning precision."""
    confidences = [95, 85, 75, 65, 55, 45, 35, 25, 15, 5] * 4
    correct = [True, True, True, False, True, False, False, True, False, False] * 4

    d = metrics.brier_decomposition(confidences, correct, n_bins=10)
    rebuilt = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert rebuilt == pytest.approx(metrics.brier_score(confidences, correct), abs=1e-9)


# --- economic proxy ----------------------------------------------------------
def test_signed_returns_follow_the_action():
    assert metrics.signed_returns(["BUY", "SELL", "HOLD"], [2.0, 2.0, 2.0]) == [2.0, -2.0, 0.0]


def test_cumulative_return_compounds():
    # +10% then -10% leaves you down, not flat.
    assert metrics.cumulative_return_pct([10.0, -10.0]) == pytest.approx(-1.0)


# --- bootstrap ---------------------------------------------------------------
def test_bootstrap_finds_a_real_difference_significant():
    a = [0.0] * 200      # one arm always right
    b = [1.0] * 200      # the other always wrong
    result = metrics.paired_bootstrap(a, b, metrics.mean, n_resamples=200, seed=1)
    assert result["difference"] == pytest.approx(-1.0)
    assert result["significant"] is True


def test_bootstrap_reports_no_difference_as_not_significant():
    values = [0.2, 0.4, 0.6, 0.8] * 25
    result = metrics.paired_bootstrap(values, list(values), metrics.mean,
                                      n_resamples=200, seed=1)
    assert result["difference"] == pytest.approx(0.0)
    assert result["significant"] is False


def test_bootstrap_is_deterministic_for_a_given_seed():
    """Reported intervals must be reproducible by whoever marks the report."""
    a = [0.1, 0.9] * 50
    b = [0.5, 0.5] * 50
    first = metrics.paired_bootstrap(a, b, metrics.mean, n_resamples=100, seed=7)
    second = metrics.paired_bootstrap(a, b, metrics.mean, n_resamples=100, seed=7)
    assert first == second


def test_bootstrap_rejects_unpaired_samples():
    with pytest.raises(ValueError):
        metrics.paired_bootstrap([1.0], [1.0, 2.0], metrics.mean)


# --- scorecard ---------------------------------------------------------------
def test_summarise_reports_every_headline_metric():
    summary = metrics.summarise(
        actions=["BUY", "SELL", "HOLD"],
        confidences=[80, 60, 50],
        correct=[True, False, True],
        pct_changes=[1.0, 1.0, 0.1],
    )
    assert summary["n"] == 3
    assert summary["accuracy_pct"] == pytest.approx(200 / 3)
    assert summary["action_counts"] == {"BUY": 1, "SELL": 1, "HOLD": 1}
    assert 0 <= summary["brier"] <= 1
    assert len(summary["reliability"]) == 10
