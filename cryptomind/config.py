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

# Downloaded candles are cached here so an evaluation run is reproducible: the
# same command re-run next week replays the same prices, not whatever the market
# has done since.
CACHE_DIR = PROJECT_ROOT / "data" / "candles"

# Evaluation output (metrics, per-decision CSVs, figures for the report).
RESULTS_DIR = PROJECT_ROOT / "results"

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

# --- Memory retrieval ---------------------------------------------------------
# How many similar past decisions are retrieved to calibrate a new call. This is
# a default, not a hard-coded constant: it is threaded through the loop and the
# seeder so the evaluation can sweep it.
RETRIEVAL_K = 3

# --- Confidence calibration --------------------------------------------------
# When blending the agent's raw confidence with its historical hit-rate, we
# trust history more as we accumulate more similar past samples. This constant
# controls how quickly we start trusting history (higher = slower to trust).
CALIBRATION_PRIOR_STRENGTH = 2

# --- Agent / LLM -------------------------------------------------------------
# The reasoning layer is swappable (see cryptomind/agent.py). Several LLM backends
# are supported out of the box, plus the always-available rule-based engine:
#   * Claude (Anthropic)                -> needs ANTHROPIC_API_KEY (paid)
#   * OpenRouter via OpenAI-compatible  -> needs OPENROUTER_API_KEY (free tier)
#   * Groq via OpenAI-compatible        -> needs GROQ_API_KEY (free tier)
# The OpenAI-compatible client means the same agent class also works with a local
# Ollama server by changing these values.

# -- Claude (Anthropic) --
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# -- Groq (free tier) via its OpenAI-compatible endpoint --
# Groq serves open-weight models at high speed with a usable free quota, which
# makes it the practical backend for evaluating the LLM agent over hundreds of
# historical decisions.
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# -- OpenRouter (free tier) via its OpenAI-compatible endpoint --
# OpenRouter fronts a rotating set of free models behind one OpenAI-compatible
# endpoint. The default is an INSTRUCTION-TUNED model (Gemma), deliberately not a
# reasoning model: reasoning models such as Nemotron intermittently "think out
# loud" before answering, so the reply is prose the JSON parser cannot recover.
# Gemma returns the JSON object directly. Override with OPENROUTER_MODEL (env or
# Streamlit Secrets) — e.g. drop the ':free' suffix to use the paid endpoint and
# escape the shared free-model daily cap.
OPENROUTER_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"
)

# --- LLM call handling --------------------------------------------------------
# Free tiers rate limit aggressively, and an evaluation makes hundreds of calls
# in a row, so a single 429 must not abort a run that has been going for twenty
# minutes.
LLM_MAX_RETRIES = 5
LLM_BACKOFF_SECONDS = 2.0

# Every LLM response is cached on disk, keyed by a hash of the exact prompt and
# model. Two reasons: an evaluation costs its API calls only once, and a run
# becomes reproducible, which an LLM evaluation otherwise is not.
LLM_CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"


def get_anthropic_key() -> str | None:
    """Return the Anthropic API key from the environment, or None if unset."""
    return os.environ.get(ANTHROPIC_API_KEY_ENV)


def get_groq_key() -> str | None:
    """Return the Groq API key from the environment, or None if unset."""
    return os.environ.get("GROQ_API_KEY")


def get_openrouter_key() -> str | None:
    """Return the OpenRouter API key from the environment, or None if unset."""
    return os.environ.get("OPENROUTER_API_KEY")
