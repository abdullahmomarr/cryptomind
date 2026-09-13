#!/usr/bin/env python3
"""CryptoMind command-line entrypoint.

Subcommands:

  seed     Backfill the memory with REAL HISTORICAL recommendations + outcomes.
           e.g.  python run.py seed
                 python run.py seed --engine llm --limit 20
  live     Run one full live decision loop for a pair (uses Claude by default).
           e.g.  python run.py live --pair BTC/USDT
                 python run.py live --pair ETH/USDT --engine rule
  verify   Grade pending LIVE recommendations against the price their
           verification window points at.
           e.g.  python run.py verify --window-hours 4
  evaluate Walk-forward evaluation on a chronological hold-out: scores the
           full system against its own ablations and against baselines, and
           writes the metrics, per-decision CSV and figures for the report.
           e.g.  python run.py evaluate
                 python run.py evaluate --sweep k --values 1 3 5 10

Run `python run.py <command> -h` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The demo output uses emoji + box-drawing characters. On Windows the default
# console encoding (cp1252) can't encode them, so force UTF-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # older Pythons / already-wrapped streams
        pass

from cryptomind import config
from cryptomind.loop import run_live, run_verify
from cryptomind.seed import run_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptomind",
        description="CryptoMind — single-agent crypto advisor with a self-learning memory loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- seed ----------------------------------------------------------------
    p_seed = sub.add_parser("seed", help="Backfill memory from real historical data.")
    p_seed.add_argument("--pairs", nargs="+", default=config.DEFAULT_PAIRS,
                        help="Trading pairs to seed (default: BTC/ETH/SOL).")
    p_seed.add_argument("--engine", choices=["rule", "groq", "gemini", "openrouter", "claude", "llm"], default="rule",
                        help="Recommendation engine for seeding (default: rule, free/offline).")
    p_seed.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME,
                        help="Candle timeframe, e.g. 1h, 15m, 1d (default: 1h).")
    p_seed.add_argument("--days", type=int, default=30,
                        help="Approximate days of history to replay (default: 30).")
    p_seed.add_argument("--limit", type=int, default=60,
                        help="Target number of decisions per pair (default: 60).")
    p_seed.add_argument("--window-hours", type=float, default=config.DEFAULT_WINDOW_HOURS,
                        help="Hours ahead used to grade each decision (default: 4).")
    p_seed.add_argument("--k", type=int, default=config.RETRIEVAL_K,
                        help="Similar past decisions retrieved per call (default: 3).")
    p_seed.add_argument("--prior", type=int, default=config.CALIBRATION_PRIOR_STRENGTH,
                        help="Calibration prior strength; higher = slower to trust history.")
    p_seed.add_argument("--band", type=float, default=config.HOLD_BAND_PCT,
                        help="HOLD band in %% used to grade outcomes (default: 0.5).")
    p_seed.add_argument("--reset", action="store_true",
                        help="Clear stored memory first. Seeding is not idempotent — "
                             "without this a second run doubles the history.")

    # --- live ----------------------------------------------------------------
    p_live = sub.add_parser("live", help="Run one full live decision loop for a pair.")
    p_live.add_argument("--pair", default="BTC/USDT", help="Trading pair (default: BTC/USDT).")
    p_live.add_argument("--engine", choices=["rule", "groq", "gemini", "openrouter", "claude", "llm"], default="rule",
                        help="Reasoning engine (default: rule; 'gemini' = free LLM, 'claude' = paid).")

    # --- verify --------------------------------------------------------------
    p_verify = sub.add_parser("verify", help="Grade pending live recommendations.")
    p_verify.add_argument("--window-hours", type=float, default=config.DEFAULT_WINDOW_HOURS,
                          help="Only grade recommendations older than this (default: 4).")

    # --- evaluate ------------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="Walk-forward evaluation with a chronological hold-out, ablations and baselines.",
    )
    p_eval.add_argument("--pairs", nargs="+", default=config.DEFAULT_PAIRS)
    p_eval.add_argument("--engine", choices=["rule", "groq", "gemini", "openrouter", "claude", "llm"], default="rule",
                        help="Reasoning engine under evaluation (default: rule).")
    p_eval.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p_eval.add_argument("--days", type=int, default=60,
                        help="Days of history to replay (default: 60).")
    p_eval.add_argument("--window-hours", type=float, default=config.DEFAULT_WINDOW_HOURS)
    p_eval.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    p_eval.add_argument("--prior", type=int, default=config.CALIBRATION_PRIOR_STRENGTH)
    p_eval.add_argument("--band", type=float, default=config.HOLD_BAND_PCT)
    p_eval.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of decisions used only to build memory (default: 0.5). "
                             "The remaining tail is the scored hold-out.")
    p_eval.add_argument("--max-decisions", type=int, default=0,
                        help="Cap decisions per pair (0 = every valid candle). Use a small "
                             "number when evaluating a paid LLM engine.")
    p_eval.add_argument("--bins", type=int, default=10,
                        help="Confidence buckets for the reliability diagram / ECE.")
    p_eval.add_argument("--seed", type=int, default=0,
                        help="Seed for the random baseline and the bootstrap.")
    p_eval.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                        help="Start of the window to replay. Use it to evaluate a specific "
                             "market regime (e.g. a bear quarter) instead of the recent past.")
    p_eval.add_argument("--label", default="",
                        help="Name for this run, e.g. 'bear-2025-q1'. Used in the output path.")
    p_eval.add_argument("--out", default=None,
                        help="Output directory (default: results/<label or auto>).")
    p_eval.add_argument("--refresh-cache", action="store_true",
                        help="Re-download candles instead of using the cached copy.")
    p_eval.add_argument("--no-figures", action="store_true",
                        help="Skip plot generation.")
    p_eval.add_argument("--sweep", default=None, metavar="PARAM",
                        help="Ablate one parameter instead of a single run "
                             "(k, prior_strength, band, window_hours).")
    p_eval.add_argument("--values", nargs="+", default=None,
                        help="Values for --sweep, e.g. --sweep k --values 1 3 5 10.")

    return parser


def _since_ms(date_str: str | None) -> int | None:
    """Convert a YYYY-MM-DD start date to the millisecond timestamp ccxt wants."""
    if not date_str:
        return None
    from datetime import datetime, timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def run_evaluate(args) -> int:
    """Drive the evaluation harness from parsed CLI arguments."""
    from cryptomind import report
    from cryptomind.evaluate import Params, evaluate, sweep

    params = Params(
        engine=args.engine, timeframe=args.timeframe, days=args.days,
        window_hours=args.window_hours, k=args.k, prior_strength=args.prior,
        band=args.band, train_frac=args.train_frac, n_bins=args.bins,
        seed=args.seed, max_decisions=args.max_decisions, label=args.label,
    )
    since_ms = _since_ms(args.since)

    run_name = args.label or f"{args.engine}_{args.timeframe}_{args.days}d"
    out_dir = Path(args.out) if args.out else config.RESULTS_DIR / run_name

    if args.sweep:
        if not args.values:
            print("❌ --sweep needs --values, e.g. --sweep k --values 1 3 5 10")
            return 2
        result = sweep(args.pairs, params, args.sweep, args.values, since_ms=since_ms)
        report.print_sweep(result)
        report.write_sweep(result, out_dir)
        if not args.no_figures:
            report.write_sweep_figure(result, out_dir)
        return 0

    print(f"\nEvaluating {args.engine} engine on {', '.join(args.pairs)} ...")
    result = evaluate(args.pairs, params, refresh_cache=args.refresh_cache, since_ms=since_ms)
    report.print_summary(result)
    report.write_results(result, out_dir)
    if not args.no_figures:
        report.write_figures(result, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "seed":
            run_seed(
                args.pairs, engine=args.engine, timeframe=args.timeframe,
                days=args.days, limit=args.limit, window_hours=args.window_hours,
                k=args.k, prior_strength=args.prior, band=args.band, reset=args.reset,
            )
        elif args.command == "live":
            run_live(args.pair, engine=args.engine)
        elif args.command == "verify":
            run_verify(window_hours=args.window_hours)
        elif args.command == "evaluate":
            return run_evaluate(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        # Friendly top-level error so a network/key problem doesn't dump a traceback.
        print(f"\n❌ {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
