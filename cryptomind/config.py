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
# Public data, no API key required. Configurable via the CRYPTOMIND_EXCHANGE env
# var because some exchanges (notably Binance) are geo-blocked from cloud hosts.
# Locally "binance" works; when deploying (e.g. Streamlit Cloud) set
# CRYPTOMIND_EXCHANGE=kraken (or coinbase) so the data layer keeps working.
EXCHANGE = os.environ.get("CRYPTOMIND_EXCHANGE", "binance")
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
# The reasoning layer is swappable (see cryptomind/agent.py). Two LLM backends
# are supported out of the box, plus the always-available rule-based engine:
#   * Claude (Anthropic)            -> needs ANTHROPIC_API_KEY (paid)
#   * Gemini via OpenAI-compatible  -> needs GEMINI_API_KEY (free tier)
# The Gemini path uses an OpenAI-compatible client, so the same agent class also
# works with Groq / OpenRouter / a local Ollama server by changing these values.

# -- Claude (Anthropic) --
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# -- Gemini (Google AI Studio, free tier) via its OpenAI-compatible endpoint --
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def get_anthropic_key() -> str | None:
    """Return the Anthropic API key from the environment, or None if unset."""
    return os.environ.get(ANTHROPIC_API_KEY_ENV)


def get_gemini_key() -> str | None:
    """Return the Google/Gemini API key from the environment, or None if unset."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
