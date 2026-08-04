"""Turning evaluation output into things the report can use.

Three outputs, all written under `results/<run>/`:

  * summary.json  -- every metric for every arm (the numbers to quote)
  * events.csv    -- one row per hold-out decision (the raw data, so a marker
                     can recompute any figure themselves)
  * figures/*.png -- the plots the Evaluation chapter needs

Matplotlib is imported lazily and failure is non-fatal: the numbers are the
result, the pictures are a rendering of them, and a missing plotting library
should never cost you the run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ARM_LABELS = {
    "memory+calibration": "Memory + calibration (full system)",
    "memory_raw_confidence": "Memory, uncalibrated confidence",
    "no_memory": "No memory (no retrieval)",
    "always_buy": "Baseline: always BUY",
    "always_sell": "Baseline: always SELL",
    "always_hold": "Baseline: always HOLD",
    "random": "Baseline: random",
}


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------
def print_summary(result: dict) -> None:
    """Print the scorecard that goes straight into the Evaluation chapter."""
    params = result["params"]
    print("\n" + "=" * 78)
    print(f"  EVALUATION — engine={params['engine']}  pairs={', '.join(result['pairs'])}")
    print(f"  {params['timeframe']} candles, ~{params['days']}d, "
          f"{params['window_hours']}h grading horizon, "
          f"k={params['k']}, prior={params['prior_strength']}, band=±{params['band']}%")
    print(f"  Chronological hold-out: last {(1 - params['train_frac']) * 100:.0f}% of decisions")
    if params.get("label"):
        print(f"  Regime label: {params['label']}")
    print("=" * 78)

    pooled = result.get("pooled")
    if not pooled:
        print("  No results (every pair failed).")
        return

    header = (f"  {'Arm':<36}{'n':>6}{'Acc%':>7}{'Brier':>8}{'ECE%':>7}"
              f"{'AUC':>7}{'Resol':>8}{'Ret%':>8}")
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for name, s in pooled["arms"].items():
        label = ARM_LABELS.get(name, name)
        auc = "  n/a" if s.get("auc") is None else f"{s['auc']:.3f}"
        resolution = s.get("brier_decomposition", {}).get("resolution", 0.0)
        print(f"  {label:<36}{s['n']:>6}{s['accuracy_pct']:>7.1f}{s['brier']:>8.4f}"
              f"{s['ece_pct']:>7.1f}{auc:>7}{resolution:>8.4f}"
              f"{s['cumulative_return_pct']:>8.2f}")

    print(f"\n  Buy & hold over the same window: "
          f"{pooled['mean_buy_and_hold_return_pct']:+.2f}%  (mean across pairs)")
    print("  Brier: lower is better; always saying 50% scores 0.2500.")
    print("  AUC:   does higher confidence mark the calls that were right? 0.5 = no.")
    print("  Resol: resolution — ability to separate cases. A constant-confidence")
    print("         forecaster scores 0 however well calibrated it looks.")

    # Any arm skipped, and why.
    for p in result["per_pair"]:
        for name, s in p.get("arms", {}).items():
            if isinstance(s, dict) and s.get("skipped"):
                print(f"  Note — '{ARM_LABELS.get(name, name)}' not run: {s['skipped']}")
                break
        break

    _print_headline(pooled)


def _print_headline(pooled: dict) -> None:
    """State the project's central claim as a measured result, either way."""
    arms = pooled["arms"]
    if "memory+calibration" not in arms or "memory_raw_confidence" not in arms:
        return
    cal, raw = arms["memory+calibration"], arms["memory_raw_confidence"]
    test = pooled.get("calibration_vs_raw_brier")

    print("\n  " + "-" * 74)
    print("  DOES CALIBRATION MAKE THE CONFIDENCE MORE HONEST?")
    print(f"    Brier  {raw['brier']:.4f} (raw)  ->  {cal['brier']:.4f} (calibrated)   "
          f"{'better' if cal['brier'] < raw['brier'] else 'worse'}")
    print(f"    ECE    {raw['ece_pct']:.1f}%  ->  {cal['ece_pct']:.1f}%   "
          f"{'better' if cal['ece_pct'] < raw['ece_pct'] else 'worse'}")
    print(f"    Actions are identical in both arms, so any difference here is "
          f"calibration alone.")
    if test:
        verdict = "significant" if test["significant"] else "NOT significant"
        print(f"    Paired bootstrap on the Brier difference: {test['difference']:+.4f} "
              f"[95% CI {test['ci_low']:+.4f}, {test['ci_high']:+.4f}] — {verdict}")

    # The result above is about honesty, not skill. Say so, so the chapter can't
    # accidentally claim the stronger thing.
    best_acc = max(arms.items(), key=lambda kv: kv[1]["accuracy_pct"])
    if best_acc[0] != "memory+calibration":
        print(f"\n    But note: '{ARM_LABELS.get(best_acc[0], best_acc[0])}' is more accurate "
              f"({best_acc[1]['accuracy_pct']:.1f}% vs {cal['accuracy_pct']:.1f}%).")
        print("    Calibration makes the agent honest about its confidence; it does not")
        print("    make the underlying rule-based decisions good. These are separate claims.")
    print("  " + "-" * 74 + "\n")


def print_sweep(sweep_result: dict) -> None:
    """Print an ablation table."""
    param = sweep_result["parameter"]
    print("\n" + "=" * 78)
    print(f"  ABLATION — sweeping {param}")
    print("=" * 78)
    header = (f"  {param:>10}{'n':>7}{'Acc%':>8}{'Brier(cal)':>12}"
              f"{'Brier(raw)':>12}{'ECE(cal)%':>11}{'ECE(raw)%':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in sweep_result["rows"]:
        print(f"  {row[param]:>10}{row['n']:>7}{row['accuracy_pct']:>8.1f}"
              f"{row['brier_calibrated']:>12.4f}{row['brier_raw']:>12.4f}"
              f"{row['ece_calibrated_pct']:>11.1f}{row['ece_raw_pct']:>11.1f}")
    print()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
EVENT_COLUMNS = [
    "pair", "timestamp", "candle_index", "rsi", "ma_gap_pct", "entry_price",
    "exit_price", "pct_change", "action", "raw_confidence",
    "calibrated_confidence", "was_correct", "n_retrieved", "n_retrieved_graded",
]


def write_results(result: dict, out_dir: Path) -> Path:
    """Write summary.json + events.csv. Returns the directory used."""
    from .evaluate import _strip_rows

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(_strip_rows(result), indent=2, default=str), encoding="utf-8"
    )

    rows = [e for p in result["per_pair"] for e in p.get("events", [])]
    if rows:
        with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print(f"  Wrote {out_dir / 'summary.json'}")
    if rows:
        print(f"  Wrote {out_dir / 'events.csv'} ({len(rows)} hold-out decisions)")
    return out_dir


def write_sweep(sweep_result: dict, out_dir: Path) -> None:
    """Write an ablation table as CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sweep_result["rows"]
    if not rows:
        return
    path = out_dir / f"sweep_{sweep_result['parameter']}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def write_figures(result: dict, out_dir: Path) -> list[Path]:
    """Render the Evaluation chapter's figures. Returns the paths written."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed, works in CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figures; "
              "`pip install matplotlib` to generate them)")
        return []

    pooled = result.get("pooled")
    if not pooled:
        return []

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    written.append(_reliability_figure(plt, pooled, fig_dir))
    written.append(_arm_comparison_figure(plt, pooled, fig_dir))
    conf_fig = _confidence_shift_figure(plt, result, fig_dir)
    if conf_fig:
        written.append(conf_fig)

    for path in written:
        print(f"  Wrote {path}")
    return written


def _reliability_figure(plt, pooled: dict, fig_dir: Path) -> Path:
    """The key figure: stated confidence vs observed accuracy, calibrated vs raw."""
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Perfect calibration")

    styles = {
        "memory_raw_confidence": ("o--", "#DD8452"),
        "memory+calibration": ("s-", "#4C72B0"),
    }
    for arm, (style, colour) in styles.items():
        bins = pooled["arms"].get(arm, {}).get("reliability", [])
        populated = [b for b in bins if b["count"]]
        if not populated:
            continue
        xs = [b["mean_confidence"] for b in populated]
        ys = [b["accuracy"] for b in populated]
        counts = [b["count"] for b in populated]
        ax.plot(xs, ys, style, color=colour, label=ARM_LABELS.get(arm, arm), zorder=2)
        # Marker area tracks how many decisions each point summarises, so a
        # bucket holding four calls cannot look as authoritative as one holding
        # four hundred.
        biggest = max(counts)
        ax.scatter(xs, ys, s=[30 + 320 * (c / biggest) for c in counts],
                   color=colour, alpha=0.35, zorder=1)
        ax_hist.bar(xs, counts, width=7, color=colour, alpha=0.6,
                    label=ARM_LABELS.get(arm, arm))

    ax.set_ylabel("Observed accuracy (%)")
    ax.set_title("Reliability diagram — is the confidence honest?")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.03, "Below the diagonal = overconfident\nMarker size = decisions in bucket",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.85})

    ax_hist.set_xlabel("Stated confidence (%)")
    ax_hist.set_ylabel("Decisions")
    ax_hist.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "reliability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _arm_comparison_figure(plt, pooled: dict, fig_dir: Path) -> Path:
    """Accuracy and Brier side by side for every arm, including the baselines."""
    arms = pooled["arms"]
    names = list(arms)
    labels = [ARM_LABELS.get(n, n).replace("Baseline: ", "").replace(" (full system)", "")
              for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.barh(labels, [arms[n]["accuracy_pct"] for n in names], color="#4C72B0")
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Accuracy by arm")
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(labels, [arms[n]["brier"] for n in names], color="#DD8452")
    ax2.axvline(0.25, color="k", ls="--", lw=1, label="Always saying 50%")
    ax2.set_xlabel("Brier score (lower is better)")
    ax2.set_title("Confidence quality by arm")
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = fig_dir / "arm_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _confidence_shift_figure(plt, result: dict, fig_dir: Path):
    """How far calibration actually moved each call, and in which direction."""
    events = [e for p in result["per_pair"] for e in p.get("events", [])]
    if not events:
        return None

    shifts = [e["calibrated_confidence"] - e["raw_confidence"] for e in events]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.hist([e["raw_confidence"] for e in events], bins=20, alpha=0.6,
             label="Raw", color="#DD8452")
    ax1.hist([e["calibrated_confidence"] for e in events], bins=20, alpha=0.6,
             label="Calibrated", color="#4C72B0")
    ax1.set_xlabel("Confidence (%)")
    ax1.set_ylabel("Decisions")
    ax1.set_title("Confidence distribution before / after calibration")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(shifts, bins=25, color="#55A868")
    ax2.axvline(0, color="k", lw=1)
    ax2.set_xlabel("Calibrated − raw (percentage points)")
    ax2.set_ylabel("Decisions")
    ax2.set_title("How far memory moved each call")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = fig_dir / "confidence_shift.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_sweep_figure(sweep_result: dict, out_dir: Path):
    """Plot the ablation curve for a swept parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = sweep_result["rows"]
    if not rows:
        return None
    param = sweep_result["parameter"]
    xs = [r[param] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(xs, [r["brier_calibrated"] for r in rows], "s-", label="Calibrated")
    ax1.plot(xs, [r["brier_raw"] for r in rows], "o--", label="Raw")
    ax1.set_xlabel(param)
    ax1.set_ylabel("Brier score")
    ax1.set_title(f"Confidence quality vs {param}")

    ax2.plot(xs, [r["ece_calibrated_pct"] for r in rows], "s-", label="Calibrated")
    ax2.plot(xs, [r["ece_raw_pct"] for r in rows], "o--", label="Raw")
    ax2.set_xlabel(param)
    ax2.set_ylabel("ECE (percentage points)")
    ax2.set_title(f"Calibration error vs {param}")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"sweep_{param}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path
