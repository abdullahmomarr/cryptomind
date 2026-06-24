# CryptoMind

A single-agent AI advisor for crypto traders. Its distinctive feature is a
**self-learning memory loop**: every recommendation is logged, its outcome is
later verified against real price data, and similar past decisions are retrieved
to *calibrate the agent's confidence* on new calls.

This repository is a **feasibility prototype** for a BSc final-year project
(University of London, CM3020 Artificial Intelligence). It demonstrates that the
one core technical feature — the memory loop — actually runs and produces visible
output, and it is built to be evaluated honestly (including its limitations).

---

## What it does (the loop)

```
   live market data ──► indicators (RSI, MA crossover)
            │
            ▼
   retrieve similar past decisions ──┐
            │                        │  (SQLite memory)
            ▼                        │
   agent produces a raw call ◄───────┘
            │
            ▼
   calibrate confidence using historical hit-rate in similar setups
            │
            ▼
   log the recommendation ──► (later) verify outcome vs real price ──► memory grows
```

The calibration step is what makes it *self-learning*: if the agent has
historically been right only 1/3 of the time in similar market conditions, a
confident new call is pulled back toward that reality.

---

## Components

| File | Responsibility |
|------|----------------|
| `cryptomind/data_layer.py` | ccxt OHLCV fetch (Binance public data) + pure-Python RSI and MA-crossover indicators → a *market snapshot*. |
| `cryptomind/agent.py` | The reasoning layer behind a swappable `Agent` interface. `LLMAgent` = Claude; `RuleBasedAgent` = deterministic (free/offline, and the seam for a future Ollama backend). |
| `cryptomind/memory.py` | **The key feature.** SQLite with `recommendations` + `outcomes`, and the four loop functions: `log_recommendation`, `verify_outcomes`, `retrieve_similar`, `calibrate_confidence`. |
| `cryptomind/loop.py` | The full live decision loop with demo-friendly output. |
| `cryptomind/seed.py` | Backfills history from **real historical** prices and verifies each decision against the known subsequent price. |
| `cryptomind/display.py` | Terminal formatting for the demo. |
| `run.py` | CLI: `seed`, `live`, `verify`. |
| `tests/` | Unit tests for the memory functions. |

---

## Setup

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Only for the Claude LLM agent) provide your API key
#    The rule-based engine and all memory/data code need NO key.
cp .env.example .env      # then edit .env, OR just export it:
export ANTHROPIC_API_KEY="sk-ant-..."        # macOS / Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..."        # Windows PowerShell
```

> The exchange data uses Binance's **public** endpoints — no exchange API key needed.

---

## How to run

You can drive CryptoMind two ways — an **interactive web UI** (best for the demo
video) or the **command line**. Both use exactly the same core logic.

### Option A — Interactive web app (recommended for the demo)

```bash
streamlit run app.py
```

This opens a browser app with three tabs:

- **▶️ Live decision** — pick a pair, click *Run live analysis*, and watch the
  whole story render: market snapshot → retrieved similar past decisions (with
  their real ✅/❌ outcomes) → raw vs calibrated confidence with the reason →
  final recommendation.
- **🌱 Build memory (seed)** — backfill real historical decisions with sliders
  for pairs / days / decisions-per-pair.
- **📚 Memory browser** — browse everything the agent remembers, see the running
  hit-rate, and verify pending live calls.

The app is a thin front-end (`app.py`) over the same `cryptomind` modules — none
of the assessed core logic changes.

#### Deploying the web app (Streamlit Community Cloud)

Streamlit needs a persistent server, so it deploys on
[Streamlit Community Cloud](https://share.streamlit.io) (free) — **not** on
serverless hosts like Vercel.

1. Push this repo to GitHub.
2. On share.streamlit.io: **Create app** → select the repo → branch `main` →
   main file `app.py`.
3. **No API key is required** — the app runs on the rule-based engine by default.
   (Only add `ANTHROPIC_API_KEY` in *Secrets* if you want to use the Claude engine.)
4. **Set the exchange** so live data works from the cloud. Binance is geo-blocked
   on most cloud hosts, so add an environment variable / secret:
   `CRYPTOMIND_EXCHANGE = "kraken"` (or `coinbase`). Locally you can leave it
   unset to keep using Binance.

### Option B — Command line

### 1. Seed the memory with real historical decisions (do this first)

The memory loop is only interesting once there's history. This replays ~30 days
of **real** 1h candles per pair, makes a decision at each step, and grades it
against the **known** subsequent price. Rows are tagged `REAL-HISTORICAL`.

```bash
# Fast, free, reproducible (deterministic rule-based engine):
python run.py seed

# Or use the real Claude LLM for each historical decision (slower, costs API credits):
python run.py seed --engine llm --limit 20

# Options: --pairs BTC/USDT ETH/USDT  --timeframe 1h  --days 30  --limit 60  --window-hours 4
```

### 2. Run the live loop (this is the demo)

Pulls live data, retrieves the seeded memory, calibrates confidence, logs the call:

```bash
python run.py live --pair BTC/USDT             # Claude LLM agent (default)
python run.py live --pair ETH/USDT --engine rule   # no API key needed
```

You'll see: market snapshot → retrieved similar past decisions and their real
outcomes → raw vs calibrated confidence with the reason for the adjustment →
final recommendation.

### 3. Verify live recommendations later

After a few hours, grade the live calls you made against current prices:

```bash
python run.py verify --window-hours 4
```

### Run the tests

```bash
pytest -v
```

---

## Design notes (for the report)

- **Swappable LLM:** every backend implements `Agent.recommend(snapshot, similar_past)`.
  Pointing at a local model later means writing one `OllamaAgent` class — nothing
  else changes.
- **Similarity:** normalised Euclidean distance over `rsi` and `ma_gap_pct`
  (pair-agnostic momentum/trend descriptors). Simple and transparent; see
  `memory.snapshot_distance`.
- **Calibration:** blends raw confidence with historical hit-rate, trusting
  history more as the number of similar graded samples grows
  (`weight = n / (n + prior)`). See `memory.calibrate_confidence`.
- **Outcome grading:** BUY correct if price rose beyond a small band, SELL if it
  fell beyond it, HOLD if it stayed within it. See `memory.grade_outcome`.
- **Honesty:** seeded data uses real exchange prices and is explicitly tagged so
  it is never mistaken for fabricated history.

See the project's evaluation chapter for what's working, what's limited, and
what would be improved next.
