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

# A price provider is any callable (pair, at_unix_seconds) -> price at that time.
# Lets verify_outcomes work against live ccxt in production and against cached
# historical candles in evaluation, without the memory layer knowing the
# difference. Taking the timestamp is what keeps the grading horizon fixed: a
# recommendation is always judged on the price `window_hours` after it was made,
# never on "whatever the price happens to be when we got round to verifying".
PriceProvider = Callable[[str, float], float]


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
    horizon_hours     REAL,                     -- hours between entry and exit price
    notes             TEXT,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
);
"""

# Columns added after the first release. SQLite's CREATE TABLE IF NOT EXISTS
# won't add them to a database that already exists, so we patch them in.
MIGRATIONS = {
    "outcomes": {"horizon_hours": "REAL"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any columns missing from an older database file."""
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def connect(db_path=config.DB_PATH) -> sqlite3.Connection:
    """Open (and initialise, if needed) the memory database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def reset(conn: sqlite3.Connection, pair: str | None = None) -> int:
    """Delete stored memory (all of it, or just one pair). Returns rows removed.

    Seeding is not idempotent — running it twice would double every historical
    decision and inflate the sample count `n` that calibration weights itself
    by. Callers that re-seed should reset first.
    """
    if pair:
        ids = [r["id"] for r in conn.execute("SELECT id FROM recommendations WHERE pair = ?", (pair,))]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM outcomes WHERE recommendation_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM recommendations WHERE id IN ({marks})", ids)
        removed = len(ids)
    else:
        removed = conn.execute("SELECT COUNT(*) AS n FROM recommendations").fetchone()["n"]
        conn.execute("DELETE FROM outcomes")
        conn.execute("DELETE FROM recommendations")
    conn.commit()
    return removed


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
    horizon_hours: float | None = None,
    band: float = config.HOLD_BAND_PCT,
    notes: str = "",
) -> dict:
    """Compute and store the outcome for one recommendation. Returns the outcome.

    `horizon_hours` records how far after the decision `exit_price` was taken.
    Storing it means an evaluation can insist on comparing only decisions that
    were graded over the same horizon.
    """
    pct_change = (exit_price - entry_price) / entry_price * 100.0
    was_correct = grade_outcome(action, pct_change, band=band)
    ts = checked_at if checked_at is not None else time.time()

    conn.execute(
        """
        INSERT OR REPLACE INTO outcomes
            (recommendation_id, checked_at, exit_price, actual_pct_change,
             was_correct, horizon_hours, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (recommendation_id, ts, float(exit_price), pct_change, int(was_correct),
         horizon_hours, notes),
    )
    conn.commit()
    return {
        "recommendation_id": recommendation_id,
        "exit_price": exit_price,
        "actual_pct_change": pct_change,
        "was_correct": was_correct,
        "horizon_hours": horizon_hours,
        "notes": notes,
    }


def verify_outcomes(
    conn: sqlite3.Connection,
    window_hours: float,
    price_provider: PriceProvider,
    band: float = config.HOLD_BAND_PCT,
) -> list[dict]:
    """Grade every recommendation older than `window_hours` that has no outcome.

    Each pending decision is graded on the price exactly `window_hours` after it
    was made — NOT on the price right now. Verifying late must not silently
    stretch the horizon, or a call left ungraded for a week would be judged on a
    week of price action while a promptly-graded one is judged on four hours,
    and the two would then sit in the same table feeding the same calibration.

    Returns the list of outcomes just recorded.
    """
    cutoff = time.time() - window_hours * 3600.0
    pending = conn.execute(
        """
        SELECT r.id, r.pair, r.action, r.entry_price, r.timestamp
        FROM recommendations r
        LEFT JOIN outcomes o ON o.recommendation_id = r.id
        WHERE o.recommendation_id IS NULL AND r.timestamp <= ?
        ORDER BY r.timestamp ASC
        """,
        (cutoff,),
    ).fetchall()

    results = []
    for row in pending:
        target_ts = row["timestamp"] + window_hours * 3600.0
        try:
            exit_price = price_provider(row["pair"], target_ts)
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
            checked_at=target_ts,
            horizon_hours=window_hours,
            band=band,
            notes=f"graded on the price {window_hours}h after the decision",
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


def snapshot_distance(
    a: dict,
    b: dict,
    fields: tuple[str, ...] = SIMILARITY_FIELDS,
    scales: dict[str, float] | None = None,
) -> float:
    """Normalised Euclidean distance between two snapshots over `fields`.

    `fields` and `scales` are parameters rather than fixed constants so the
    evaluation can ablate the similarity metric itself.
    """
    scales = FIELD_SCALES if scales is None else scales
    total = 0.0
    for field in fields:
        scale = scales.get(field, 1.0)
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
    k: int = config.RETRIEVAL_K,
    as_of: float | None = None,
    fields: tuple[str, ...] = SIMILARITY_FIELDS,
    scales: dict[str, float] | None = None,
) -> list[dict]:
    """Return the k most similar past decisions for `pair`, each with its outcome.

    Similarity is numeric distance over the snapshot condition fields. Results
    are returned as plain dicts (see SimilarDecision.to_dict) ordered nearest
    first.

    `as_of` (unix seconds) restricts retrieval to what the agent could actually
    have known at that moment: only decisions made before `as_of`, and only
    outcomes that had already been graded by then. Live calls leave it None
    (now is now), but any offline replay MUST pass it — without it the query
    returns every row in the table, including decisions from after the one
    being scored, and the evaluation quietly reads the future.
    """
    sql = """
        SELECT r.id, r.pair, r.action, r.confidence, r.market_snapshot,
               o.actual_pct_change, o.was_correct, o.exit_price
        FROM recommendations r
        LEFT JOIN outcomes o ON o.recommendation_id = r.id{outcome_clause}
        WHERE r.pair = ?{time_clause}
    """
    if as_of is None:
        rows = conn.execute(
            sql.format(outcome_clause="", time_clause=""), (pair,)
        ).fetchall()
    else:
        rows = conn.execute(
            sql.format(
                outcome_clause=" AND o.checked_at <= ?",
                time_clause=" AND r.timestamp < ?",
            ),
            (as_of, pair, as_of),
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
                distance=snapshot_distance(current_market_snapshot, snap, fields, scales),
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
