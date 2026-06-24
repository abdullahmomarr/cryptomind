"""Memory layer — THE KEY FEATURE: the self-learning loop.

A SQLite database stores every recommendation and, later, its verified outcome.
Four functions make the loop self-learning:

  * log_recommendation     -> remember a decision
  * verify_outcomes        -> grade old decisions against real subsequent prices
  * retrieve_similar       -> recall the k most similar past decisions
  * calibrate_confidence   -> adjust new confidence using historical hit-rate

The maths (similarity, calibration, outcome grading) is deliberately plain
Python so it is transparent in the report and unit-testable without a network
or an LLM.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

from . import config

# A price provider is any callable pair -> current price. Lets verify_outcomes
# work against live ccxt in production and against historical data in seeding,
# without the memory layer knowing the difference.
PriceProvider = Callable[[str], float]


# ---------------------------------------------------------------------------
# Schema / connection
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL NOT NULL,          -- unix seconds when decision was made
    pair                TEXT NOT NULL,
    action              TEXT NOT NULL,          -- BUY | SELL | HOLD
    confidence          INTEGER NOT NULL,       -- calibrated confidence that was acted on
    raw_confidence      INTEGER NOT NULL,       -- agent's confidence before calibration
    predicted_direction TEXT,
    reasoning           TEXT,
    entry_price         REAL NOT NULL,          -- price at decision time
    market_snapshot     TEXT NOT NULL,          -- JSON of the indicators at decision time
    source              TEXT NOT NULL DEFAULT 'live'  -- 'live' or 'REAL-HISTORICAL' (seeded)
);

CREATE TABLE IF NOT EXISTS outcomes (
    recommendation_id INTEGER PRIMARY KEY,      -- one outcome per recommendation
    checked_at        REAL NOT NULL,
    exit_price        REAL NOT NULL,
    actual_pct_change REAL NOT NULL,
    was_correct       INTEGER NOT NULL,         -- 0/1
    notes             TEXT,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
);
"""


def connect(db_path=config.DB_PATH) -> sqlite3.Connection:
    """Open (and initialise, if needed) the memory database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# 1. Logging a recommendation
# ---------------------------------------------------------------------------
def log_recommendation(
    conn: sqlite3.Connection,
    *,
    pair: str,
    action: str,
    confidence: int,
    raw_confidence: int,
    predicted_direction: str,
    reasoning: str,
    entry_price: float,
    market_snapshot: dict,
    timestamp: float | None = None,
    source: str = "live",
) -> int:
    """Persist a new recommendation and return its row id."""
    ts = timestamp if timestamp is not None else time.time()
    cur = conn.execute(
        """
        INSERT INTO recommendations
            (timestamp, pair, action, confidence, raw_confidence,
             predicted_direction, reasoning, entry_price, market_snapshot, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts, pair, action, int(confidence), int(raw_confidence),
            predicted_direction, reasoning, float(entry_price),
            json.dumps(market_snapshot), source,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# 2. Verifying outcomes
# ---------------------------------------------------------------------------
def grade_outcome(action: str, pct_change: float, band: float = config.HOLD_BAND_PCT) -> bool:
    """Decide whether a recommendation was correct given the % price change.

    * BUY  is correct if price rose more than +band
    * SELL is correct if price fell more than -band
    * HOLD is correct if price stayed within +/- band
    """
    action = action.upper()
    if action == "BUY":
        return pct_change > band
    if action == "SELL":
        return pct_change < -band
    if action == "HOLD":
        return abs(pct_change) <= band
    raise ValueError(f"Unknown action '{action}'")


def record_outcome(
    conn: sqlite3.Connection,
    recommendation_id: int,
    *,
    entry_price: float,
    exit_price: float,
    action: str,
    checked_at: float | None = None,
    notes: str = "",
) -> dict:
    """Compute and store the outcome for one recommendation. Returns the outcome."""
    pct_change = (exit_price - entry_price) / entry_price * 100.0
    was_correct = grade_outcome(action, pct_change)
    ts = checked_at if checked_at is not None else time.time()

    conn.execute(
        """
        INSERT OR REPLACE INTO outcomes
            (recommendation_id, checked_at, exit_price, actual_pct_change, was_correct, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (recommendation_id, ts, float(exit_price), pct_change, int(was_correct), notes),
    )
    conn.commit()
    return {
        "recommendation_id": recommendation_id,
        "exit_price": exit_price,
        "actual_pct_change": pct_change,
        "was_correct": was_correct,
        "notes": notes,
    }


def verify_outcomes(
    conn: sqlite3.Connection,
    window_hours: float,
    price_provider: PriceProvider,
) -> list[dict]:
    """Grade every recommendation older than `window_hours` that has no outcome.

    Fetches the current price for each (via the supplied price_provider),
    computes the % change since entry, decides correctness, and writes outcomes.
    Returns the list of outcomes just recorded.
    """
    cutoff = time.time() - window_hours * 3600.0
    pending = conn.execute(
        """
        SELECT r.id, r.pair, r.action, r.entry_price
        FROM recommendations r
        LEFT JOIN outcomes o ON o.recommendation_id = r.id
        WHERE o.recommendation_id IS NULL AND r.timestamp <= ?
        ORDER BY r.timestamp ASC
        """,
        (cutoff,),
    ).fetchall()

    results = []
    for row in pending:
        try:
            exit_price = price_provider(row["pair"])
        except Exception as exc:
            # Network hiccup on one pair shouldn't abort the whole sweep.
            print(f"  ! Skipped rec #{row['id']} ({row['pair']}): {exc}")
            continue
        outcome = record_outcome(
            conn,
            row["id"],
            entry_price=row["entry_price"],
            exit_price=exit_price,
            action=row["action"],
            notes=f"verified after >= {window_hours}h against live price",
        )
        results.append(outcome)
    return results


# ---------------------------------------------------------------------------
# 3. Retrieving similar past decisions
# ---------------------------------------------------------------------------
# Which snapshot fields define "market condition" for similarity. RSI and the
# MA gap are pair-agnostic momentum/trend descriptors, so they compare cleanly
# across time (price itself is not used — it isn't a "condition").
SIMILARITY_FIELDS = ("rsi", "ma_gap_pct")

# Rough scales used to normalise each field so neither dominates the distance.
FIELD_SCALES = {"rsi": 100.0, "ma_gap_pct": 5.0}


def snapshot_distance(a: dict, b: dict) -> float:
    """Normalised Euclidean distance between two snapshots over SIMILARITY_FIELDS."""
    total = 0.0
    for field in SIMILARITY_FIELDS:
        scale = FIELD_SCALES.get(field, 1.0)
        diff = (float(a.get(field, 0.0)) - float(b.get(field, 0.0))) / scale
        total += diff * diff
    return total ** 0.5


@dataclass
class SimilarDecision:
    """A retrieved past decision plus its outcome (if known) and similarity."""

    id: int
    pair: str
    action: str
    confidence: int
    snapshot: dict
    distance: float
    outcome: dict | None  # {actual_pct_change, was_correct, ...} or None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pair": self.pair,
            "action": self.action,
            "confidence": self.confidence,
            "snapshot": self.snapshot,
            "distance": self.distance,
            "outcome": self.outcome,
        }


def retrieve_similar(
    conn: sqlite3.Connection,
    current_market_snapshot: dict,
    pair: str,
    k: int = 3,
) -> list[dict]:
    """Return the k most similar past decisions for `pair`, each with its outcome.

    Similarity is numeric distance over the snapshot condition fields. Results
    are returned as plain dicts (see SimilarDecision.to_dict) ordered nearest
    first.
    """
    rows = conn.execute(
        """
        SELECT r.id, r.pair, r.action, r.confidence, r.market_snapshot,
               o.actual_pct_change, o.was_correct, o.exit_price
        FROM recommendations r
        LEFT JOIN outcomes o ON o.recommendation_id = r.id
        WHERE r.pair = ?
        """,
        (pair,),
    ).fetchall()

    scored: list[SimilarDecision] = []
    for row in rows:
        snap = json.loads(row["market_snapshot"])
        outcome = None
        if row["was_correct"] is not None:
            outcome = {
                "actual_pct_change": row["actual_pct_change"],
                "was_correct": bool(row["was_correct"]),
                "exit_price": row["exit_price"],
            }
        scored.append(
            SimilarDecision(
                id=row["id"],
                pair=row["pair"],
                action=row["action"],
                confidence=row["confidence"],
                snapshot=snap,
                distance=snapshot_distance(current_market_snapshot, snap),
                outcome=outcome,
            )
        )

    scored.sort(key=lambda d: d.distance)
    return [d.to_dict() for d in scored[:k]]


# ---------------------------------------------------------------------------
# 4. Calibrating confidence from historical accuracy
# ---------------------------------------------------------------------------
def calibrate_confidence(
    raw_confidence: int,
    similar_past: list[dict],
    prior_strength: int = config.CALIBRATION_PRIOR_STRENGTH,
) -> tuple[int, str]:
    """Nudge raw confidence toward the agent's historical hit-rate in similar setups.

    Only past decisions with a *known* outcome count. We blend the raw confidence
    with the historical hit-rate, trusting history more as the number of similar
    graded samples (n) grows:

        weight = n / (n + prior_strength)
        calibrated = (1 - weight) * raw + weight * hit_rate

    Returns (calibrated_confidence, human-readable explanation).
    """
    graded = [d for d in similar_past if d.get("outcome") is not None]
    n = len(graded)

    if n == 0:
        return raw_confidence, (
            "No graded similar history yet — keeping raw confidence unchanged."
        )

    correct = sum(1 for d in graded if d["outcome"]["was_correct"])
    hit_rate = correct / n  # 0..1
    hit_rate_pct = hit_rate * 100.0

    weight = n / (n + prior_strength)
    calibrated = (1 - weight) * raw_confidence + weight * hit_rate_pct
    calibrated_int = int(max(0, min(100, round(calibrated))))

    direction = "up" if calibrated_int > raw_confidence else (
        "down" if calibrated_int < raw_confidence else "unchanged"
    )
    explanation = (
        f"In {n} similar past setup(s) the agent was right {correct}/{n} "
        f"({hit_rate_pct:.0f}% hit-rate). Blending raw {raw_confidence} with that "
        f"history (weight {weight:.2f}) adjusts confidence {direction} to {calibrated_int}."
    )
    return calibrated_int, explanation


# ---------------------------------------------------------------------------
# Small reporting helpers (used by the loop / seeder for demo output)
# ---------------------------------------------------------------------------
def recent_recommendations(
    conn: sqlite3.Connection, pair: str | None = None, limit: int = 20
) -> list[dict]:
    """Return the most recent recommendations joined with their outcomes.

    Used by the web UI's "memory browser" table. Newest first.
    """
    where = "WHERE r.pair = ?" if pair else ""
    params: tuple = (pair, limit) if pair else (limit,)
    rows = conn.execute(
        f"""
        SELECT r.id, r.timestamp, r.pair, r.action, r.confidence, r.raw_confidence,
               r.entry_price, r.source, r.market_snapshot,
               o.actual_pct_change, o.was_correct
        FROM recommendations r
        LEFT JOIN outcomes o ON o.recommendation_id = r.id
        {where}
        ORDER BY r.timestamp DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    out = []
    for row in rows:
        snap = json.loads(row["market_snapshot"])
        out.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "pair": row["pair"],
            "action": row["action"],
            "confidence": row["confidence"],
            "raw_confidence": row["raw_confidence"],
            "entry_price": row["entry_price"],
            "source": row["source"],
            "rsi": snap.get("rsi"),
            "ma_gap_pct": snap.get("ma_gap_pct"),
            "actual_pct_change": row["actual_pct_change"],
            "was_correct": (None if row["was_correct"] is None else bool(row["was_correct"])),
        })
    return out


def accuracy_stats(conn: sqlite3.Connection, pair: str | None = None) -> dict:
    """Return overall graded count and hit-rate, optionally filtered by pair."""
    where = "WHERE r.pair = ?" if pair else ""
    params = (pair,) if pair else ()
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS graded,
               COALESCE(SUM(o.was_correct), 0) AS correct
        FROM outcomes o
        JOIN recommendations r ON r.id = o.recommendation_id
        {where}
        """,
        params,
    ).fetchone()
    graded = row["graded"] or 0
    correct = row["correct"] or 0
    return {
        "graded": graded,
        "correct": correct,
        "hit_rate_pct": (correct / graded * 100.0) if graded else 0.0,
    }
