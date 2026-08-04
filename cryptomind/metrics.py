"""Evaluation metrics.

CryptoMind's central claim is not "it predicts the market" — it is "it knows how
much to trust itself". That is a claim about *calibration*, and accuracy alone
cannot support it: an agent that is right 55% of the time while saying 90% every
time is badly calibrated even though its accuracy is respectable.

So the headline metrics here score the confidence, not just the action:

  * accuracy      -- how often the action was graded correct (the obvious one)
  * Brier score   -- mean squared error of confidence-as-a-probability (lower is
                     better). Rewards being confident when right AND unconfident
                     when wrong, so it is the metric calibration should move.
  * ECE           -- expected calibration error: the average gap between stated
                     confidence and observed accuracy, bucketed. Directly
                     answers "when it says 70%, is it right 70% of the time?"
  * reliability   -- the per-bucket data behind ECE, for the report's diagram.

Plus a paired bootstrap so a difference between two arms can be reported with a
confidence interval rather than asserted from a single number.

Everything is pure Python: no numpy needed, and every formula is visible for the
report.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def accuracy(correct: Sequence[bool]) -> float:
    """Fraction of decisions graded correct, in [0, 1]."""
    if not correct:
        return 0.0
    return sum(1 for c in correct if c) / len(correct)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error between stated confidence and the 0/1 outcome.

        brier = mean( (confidence/100 - was_correct)^2 )

    Range [0, 1], lower is better. A constant 50% forecaster scores 0.25, which
    is the natural reference line: anything above it is worse than shrugging.
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    total = 0.0
    for conf, was_correct in zip(confidences, correct):
        p = float(conf) / 100.0
        y = 1.0 if was_correct else 0.0
        total += (p - y) ** 2
    return total / n


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> list[dict]:
    """Bucket decisions by stated confidence and measure accuracy within each.

    This is the data behind a reliability diagram: for a perfectly calibrated
    agent, `mean_confidence` and `accuracy` match in every populated bucket, and
    the plotted line sits on the diagonal.
    """
    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins * 100.0
        hi = (i + 1) / n_bins * 100.0
        members = [
            (float(c), bool(y))
            for c, y in zip(confidences, correct)
            # Last bin is closed on the right so confidence == 100 is counted.
            if (lo <= float(c) < hi) or (i == n_bins - 1 and float(c) == 100.0)
        ]
        if members:
            confs = [m[0] for m in members]
            outcomes = [m[1] for m in members]
            bins.append({
                "lower": lo,
                "upper": hi,
                "count": len(members),
                "mean_confidence": sum(confs) / len(confs),
                "accuracy": accuracy(outcomes) * 100.0,
            })
        else:
            bins.append({
                "lower": lo, "upper": hi, "count": 0,
                "mean_confidence": None, "accuracy": None,
            })
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Weighted average gap between stated confidence and observed accuracy.

        ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|

    Reported in percentage points: an ECE of 12 means that, on average, the
    stated confidence is 12 points away from how often the agent was actually
    right at that confidence level.
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    total = 0.0
    for b in reliability_bins(confidences, correct, n_bins):
        if b["count"]:
            total += (b["count"] / n) * abs(b["accuracy"] - b["mean_confidence"])
    return total


# ---------------------------------------------------------------------------
# Discrimination — the check that stops a degenerate winner
# ---------------------------------------------------------------------------
# Brier and ECE on their own can be gamed: a forecaster that states the same
# confidence on every call, equal to its own base rate, is *perfectly calibrated*
# and scores well, while being completely useless — it cannot tell a good setup
# from a bad one. "Always BUY at 27%" is exactly that forecaster. So calibration
# metrics must be read alongside discrimination, which asks a different
# question: does higher confidence actually mark the calls that turned out
# right?
def discrimination_auc(confidences: Sequence[float], correct: Sequence[bool]) -> float | None:
    """Probability that a correct call was given higher confidence than a wrong one.

    Equivalent to the area under the ROC curve, computed by ranks (Mann-Whitney
    U) with ties shared at half credit. 0.5 means the confidence carries no
    information at all — which is what any constant forecaster scores, however
    beautifully calibrated it is. Returns None when every call went the same
    way, since there is nothing to separate.
    """
    n_pos = sum(1 for y in correct if y)
    n_neg = len(correct) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(confidences)), key=lambda i: float(confidences[i]))
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and \
                float(confidences[order[j + 1]]) == float(confidences[order[i]]):
            j += 1
        shared = (i + j) / 2.0 + 1.0  # average rank across the tied block, 1-based
        for t in range(i, j + 1):
            ranks[order[t]] = shared
        i = j + 1

    rank_sum_pos = sum(r for r, y in zip(ranks, correct) if y)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def brier_decomposition(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> dict:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

      * reliability  -- how far stated confidence sits from observed accuracy
                        within each bucket (lower is better; this is what
                        calibration improves)
      * resolution   -- how far the buckets' accuracies spread away from the
                        overall base rate (higher is better; this is the
                        forecaster's ability to separate cases). **Zero for any
                        constant forecaster.**
      * uncertainty  -- the base rate's own variance. A property of the problem,
                        not of the forecaster, and identical for every arm
                        scored on the same events.

    Reporting the three parts separately is what prevents "always BUY" from
    being read as the best model: it wins on reliability by refusing to say
    anything, and scores exactly 0 on resolution for the same reason.
    """
    n = len(correct)
    if n == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0}

    base_rate = sum(1 for y in correct if y) / n
    uncertainty = base_rate * (1 - base_rate)

    reliability = resolution = 0.0
    for b in reliability_bins(confidences, correct, n_bins):
        if not b["count"]:
            continue
        weight = b["count"] / n
        bucket_conf = b["mean_confidence"] / 100.0
        bucket_acc = b["accuracy"] / 100.0
        reliability += weight * (bucket_conf - bucket_acc) ** 2
        resolution += weight * (bucket_acc - base_rate) ** 2

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "base_rate": base_rate,
    }


# ---------------------------------------------------------------------------
# Economic view
# ---------------------------------------------------------------------------
def signed_returns(actions: Sequence[str], pct_changes: Sequence[float]) -> list[float]:
    """Return the price move as experienced by each decision.

    BUY earns the move, SELL earns its negation, HOLD earns nothing. This is a
    deliberately naive proxy — no position sizing, no fees, no slippage — and
    the report should say so. It is included because accuracy alone can look
    fine while the agent is right on tiny moves and wrong on large ones.
    """
    out = []
    for action, pct in zip(actions, pct_changes):
        a = action.upper()
        if a == "BUY":
            out.append(float(pct))
        elif a == "SELL":
            out.append(-float(pct))
        else:
            out.append(0.0)
    return out


def cumulative_return_pct(signed: Sequence[float]) -> float:
    """Compound a series of per-decision percentage moves into a total."""
    equity = 1.0
    for pct in signed:
        equity *= (1.0 + pct / 100.0)
    return (equity - 1.0) * 100.0


def equity_curve(signed: Sequence[float]) -> list[float]:
    """Running compounded equity (starting at 1.0) for plotting."""
    equity = 1.0
    curve = [equity]
    for pct in signed:
        equity *= (1.0 + pct / 100.0)
        curve.append(equity)
    return curve


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap a confidence interval for statistic(a) - statistic(b).

    Paired: both arms are resampled on the SAME indices, because they are two
    scorings of the same decisions on the same market events. Ignoring the
    pairing would inflate the variance and hide a real difference.

    Returns the observed difference, the interval, and whether it excludes zero.
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs two equal-length samples")
    n = len(a)
    observed = statistic(a) - statistic(b)
    if n == 0:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "significant": False, "n_resamples": 0}

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistic([a[i] for i in idx]) - statistic([b[i] for i in idx]))
    diffs.sort()

    lo = diffs[int(alpha / 2 * n_resamples)]
    hi = diffs[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return {
        "difference": observed,
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0) or (hi < 0),
        "n_resamples": n_resamples,
    }


def squared_errors(confidences: Sequence[float], correct: Sequence[bool]) -> list[float]:
    """Per-decision Brier contributions, so the bootstrap can resample them."""
    return [
        (float(c) / 100.0 - (1.0 if y else 0.0)) ** 2
        for c, y in zip(confidences, correct)
    ]


def mean(values: Sequence[float]) -> float:
    """Plain mean — the statistic passed to `paired_bootstrap` for Brier/accuracy."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Assembling a full scorecard
# ---------------------------------------------------------------------------
def summarise(
    actions: Sequence[str],
    confidences: Sequence[float],
    correct: Sequence[bool],
    pct_changes: Sequence[float],
    n_bins: int = 10,
) -> dict:
    """Score one arm of the evaluation across every metric above."""
    signed = signed_returns(actions, pct_changes)
    counts = {a: 0 for a in ("BUY", "SELL", "HOLD")}
    for a in actions:
        counts[a.upper()] = counts.get(a.upper(), 0) + 1

    decomposition = brier_decomposition(confidences, correct, n_bins)
    return {
        "n": len(actions),
        "accuracy_pct": accuracy(correct) * 100.0,
        "brier": brier_score(confidences, correct),
        "ece_pct": expected_calibration_error(confidences, correct, n_bins),
        "auc": discrimination_auc(confidences, correct),
        "brier_decomposition": decomposition,
        "mean_confidence": mean([float(c) for c in confidences]),
        "cumulative_return_pct": cumulative_return_pct(signed),
        "mean_signed_return_pct": mean(signed),
        "action_counts": counts,
        "reliability": reliability_bins(confidences, correct, n_bins),
    }
