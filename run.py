#!/usr/bin/env python3
"""CryptoMind command-line entrypoint.

Subcommands:

  seed     Backfill the memory with REAL HISTORICAL recommendations + outcomes.
           e.g.  python run.py seed
                 python run.py seed --engine llm --limit 20
  live     Run one full live decision loop for a pair (uses Claude by default).
           e.g.  python run.py live --pair BTC/USDT
                 python run.py live --pair ETH/USDT --engine rule
  verify   Grade pending LIVE recommendations older than the window against
           current prices.
           e.g.  python run.py verify --window-hours 4

Run `python run.py <command> -h` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys

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
    p_seed.add_argument("--engine", choices=["rule", "gemini", "claude", "llm"], default="rule",
                        help="Recommendation engine for seeding (default: rule, free/offline).")
    p_seed.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME,
                        help="Candle timeframe, e.g. 1h, 15m, 1d (default: 1h).")
    p_seed.add_argument("--days", type=int, default=30,
                        help="Approximate days of history to replay (default: 30).")
    p_seed.add_argument("--limit", type=int, default=60,
                        help="Target number of decisions per pair (default: 60).")
    p_seed.add_argument("--window-hours", type=float, default=config.DEFAULT_WINDOW_HOURS,
                        help="Hours ahead used to grade each decision (default: 4).")

    # --- live ----------------------------------------------------------------
    p_live = sub.add_parser("live", help="Run one full live decision loop for a pair.")
    p_live.add_argument("--pair", default="BTC/USDT", help="Trading pair (default: BTC/USDT).")
    p_live.add_argument("--engine", choices=["rule", "gemini", "claude", "llm"], default="rule",
                        help="Reasoning engine (default: rule; 'gemini' = free LLM, 'claude' = paid).")

    # --- verify --------------------------------------------------------------
    p_verify = sub.add_parser("verify", help="Grade pending live recommendations.")
    p_verify.add_argument("--window-hours", type=float, default=config.DEFAULT_WINDOW_HOURS,
                          help="Only grade recommendations older than this (default: 4).")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "seed":
            run_seed(
                args.pairs, engine=args.engine, timeframe=args.timeframe,
                days=args.days, limit=args.limit, window_hours=args.window_hours,
            )
        elif args.command == "live":
            run_live(args.pair, engine=args.engine)
        elif args.command == "verify":
            run_verify(window_hours=args.window_hours)
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
