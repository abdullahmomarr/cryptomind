"""Re-test the headline Brier improvement with a moving-block bootstrap.

The evaluation's paired bootstrap resamples individual decisions, which assumes
they are independent. They are not: decisions are made every hour but graded
over four hours, so consecutive outcomes share most of their price path, and
whether a call was right is strongly autocorrelated. An i.i.d. bootstrap then
understates the uncertainty and its interval is too narrow.

A moving-block bootstrap resamples runs of consecutive decisions instead, so the
dependence inside each block is preserved. Blocks are drawn within each pair and
never span two pairs. The script reports the calibrated-minus-raw Brier
difference with a 95% interval for several block lengths, and the lag-1
autocorrelation of correctness that motivates it.

Reads results/main-60d/events.csv, writes results/main-60d/block_bootstrap.json.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "results" / "main-60d"
BLOCK_LENGTHS = (1, 4, 24)   # 1 = the i.i.d. bootstrap, 4 = one horizon, 24 = one day
RESAMPLES = 2000
SEED = 0


def sq(conf: float, correct: bool) -> float:
    return (conf / 100.0 - (1.0 if correct else 0.0)) ** 2


def lag1(y: list[float]) -> float:
    m = sum(y) / len(y)
    var = sum((v - m) ** 2 for v in y)
    return sum((y[i] - m) * (y[i + 1] - m) for i in range(len(y) - 1)) / var


def main() -> None:
    with (RUN / "events.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_pair: dict[str, list[float]] = {}
    autocorr = {}
    for pair in dict.fromkeys(r["pair"] for r in rows):
        pr = [r for r in rows if r["pair"] == pair]
        pr.sort(key=lambda r: float(r["timestamp"]))
        by_pair[pair] = [
            sq(float(r["calibrated_confidence"]), r["was_correct"] == "True")
            - sq(float(r["raw_confidence"]), r["was_correct"] == "True")
            for r in pr
        ]
        autocorr[pair] = lag1([1.0 if r["was_correct"] == "True" else 0.0 for r in pr])

    total = sum(len(v) for v in by_pair.values())
    observed = sum(sum(v) for v in by_pair.values()) / total

    rng = random.Random(SEED)
    results = []
    for length in BLOCK_LENGTHS:
        diffs = []
        for _ in range(RESAMPLES):
            acc, n = 0.0, 0
            for series in by_pair.values():
                size = len(series)
                blocks = -(-size // length)          # ceiling division
                for _ in range(blocks):
                    start = rng.randrange(size - length + 1)
                    chunk = series[start:start + length]
                    acc += sum(chunk)
                    n += len(chunk)
            diffs.append(acc / n)
        diffs.sort()
        lo = diffs[int(0.025 * RESAMPLES)]
        hi = diffs[int(0.975 * RESAMPLES)]
        results.append({"block_length_hours": length, "ci_low": lo, "ci_high": hi,
                        "significant": hi < 0 or lo > 0})
        print(f"block {length:>2}h: difference {observed:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  {'significant' if (hi < 0 or lo > 0) else 'NOT significant'}")

    for pair, ac in autocorr.items():
        print(f"{pair}: lag-1 autocorrelation of correctness {ac:.2f}")

    out = {"difference": observed, "n": total, "n_resamples": RESAMPLES, "seed": SEED,
           "blocks": results, "lag1_autocorrelation": autocorr}
    (RUN / "block_bootstrap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {(RUN / 'block_bootstrap.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
