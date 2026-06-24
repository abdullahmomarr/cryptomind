"""Central configuration for the CryptoMind prototype.

Everything tunable lives here so the rest of the code reads cleanly and the
report can point at a single place for "the parameters we chose".
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# The SQLite database that holds the agent's memory (recommendations + outcomes).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "cryptomind.db"

# --- Market data -------------------------------------------------------------
EXCHANGE = "binance"                       # public data, no API key required
DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAME = "1h"                    # candle size used everywhere

# --- Technical indicators ----------------------------------------------------
RSI_PERIOD = 14                            # classic RSI lookback
SMA_FAST = 10                              # fast moving average (crossover)
SMA_SLOW = 30                              # slow moving average (crossover)

# How many candles we need before indicators are valid (slow MA + RSI warmup).
MIN_CANDLES = SMA_SLOW + RSI_PERIOD + 1

# RSI interpretation thresholds.
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Outcome verification ----------------------------------------------------
# A recommendation is only "judged" once this many hours of real price action
# have passed since it was made.
DEFAULT_WINDOW_HOURS = 4

# HOLD is considered correct if the price stayed within +/- this band (percent).
# BUY is correct if change > +band, SELL is correct if change < -band.
HOLD_BAND_PCT = 0.5

# --- Confidence calibration --------------------------------------------------
# When blending the agent's raw confidence with its historical hit-rate, we
# trust history more as we accumulate more similar past samples. This constant
# controls how quickly we start trusting history (higher = slower to trust).
CALIBRATION_PRIOR_STRENGTH = 2

# --- Agent / LLM -------------------------------------------------------------
# Current Sonnet model id. Swappable; see cryptomind/agent.py for the interface
# that also allows pointing at a local Ollama model later.
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def get_anthropic_key() -> str | None:
    """Return the Anthropic API key from the environment, or None if unset."""
    return os.environ.get(ANTHROPIC_API_KEY_ENV)
