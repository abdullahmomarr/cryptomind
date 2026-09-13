"""Offline evaluation harness.

Answers the question the whole project rests on: **does calibrating confidence
against retrieved memory actually make the agent's confidence more honest?**

Method — walk-forward replay with a chronological hold-out:

  1. Load a fixed set of real candles (cached on disk, so a run is reproducible).
  2. Walk forward one decision at a time. At each step the agent sees only the
     candles up to that point, retrieves only decisions made before that point
     whose outcomes were already graded by then (`as_of`), decides, and is graded
     against the real price `window_hours` later.
  3. Split chronologically: the first `train_frac` of decisions exist only to
     build memory; **only the hold-out tail is scored**. No shuffling — shuffling
     a time series lets the model learn from its own future.

Arms compared on the identical hold-out events:

  * memory+calibration  -- the full system
  * memory, raw conf.   -- same actions, uncalibrated confidence (isolates what
                           calibration does, since it changes confidence only)
  * no memory           -- agent decides without retrieval (only differs for an
                           LLM backend; the rule agent ignores memory by design)
  * always BUY / SELL / HOLD, random  -- baselines
  * buy & hold          -- the market's own return over the same window

Baselines are given the confidence they *deserve*: each one's own hit-rate over
the training split. Handing them a flat 50% would make the agent's calibration
look good for free.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass

from . import config, memory, metrics
from .agent import Agent, get_agent
from .data_layer import Candle, MarketData, build_snapshot, load_candles

TIMEFRAME_HOURS = {"15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0}


@dataclass
class Params:
    """Everything the evaluation can vary. Sweeping any of these is an ablation."""

    engine: str = "rule"
    timeframe: str = config.DEFAULT_TIMEFRAME
    days: int = 60
    window_hours: float = config.DEFAULT_WINDOW_HOURS
    k: int = config.RETRIEVAL_K
    prior_strength: int = config.CALIBRATION_PRIOR_STRENGTH
    band: float = config.HOLD_BAND_PCT
    train_frac: float = 0.5
    n_bins: int = 10
    seed: int = 0
    max_decisions: int = 0          # 0 = every valid candle
    similarity_fields: tuple[str, ...] = memory.SIMILARITY_FIELDS
    label: str = ""                 # e.g. "bull-2024-q1", for regime runs

    def timeframe_hours(self) -> float:
        if self.timeframe not in TIMEFRAME_HOURS:
            raise ValueError(f"Unsupported timeframe '{self.timeframe}'")
        return TIMEFRAME_HOURS[self.timeframe]


# ---------------------------------------------------------------------------
# Walk-forward replay
# ---------------------------------------------------------------------------
def replay(
    pair: str,
    candles: list[Candle],
    agent: Agent,
    params: Params,
    *,
    use_memory: bool = True,
    skips: list[str] | None = None,
) -> list[dict]:
    """Replay one pair chronologically, returning one record per decision.

    Each record carries the market state, what the agent said (raw and
    calibrated), and how it actually turned out. The memory database is
    in-memory and private to this call, so a replay can never be polluted by a
    previous run and re-running is idempotent.

    If `skips` is given, the reason string for each decision dropped because the
    agent could not produce a parseable recommendation is appended to it, so the
    caller can report how many live-LLM calls were unusable.
    """
    tf_hours = params.timeframe_hours()
    window_candles = max(1, round(params.window_hours / tf_hours))

    first_i = config.MIN_CANDLES - 1
    last_i = len(candles) - 1 - window_candles
    if last_i < first_i:
        return []

    indices = list(range(first_i, last_i + 1))
    if params.max_decisions:
        step = max(1, len(indices) // params.max_decisions)
        indices = indices[::step][: params.max_decisions]

    conn = memory.connect(":memory:")
    records: list[dict] = []
    try:
        for i in indices:
            try:
                snapshot = build_snapshot(pair, candles[: i + 1])
            except ValueError:
                continue
            snap = snapshot.to_dict()

            decision_ts = candles[i][0] / 1000.0
            entry_price = float(candles[i][4])
            exit_candle = candles[i + window_candles]
            exit_price = float(exit_candle[4])
            exit_ts = exit_candle[0] / 1000.0

            # `as_of` is what keeps this honest: only decisions taken before now,
            # and only outcomes that had already been graded by now. Outcomes are
            # written immediately below (we know the future price), but the
            # checked_at filter hides each one until the moment it would really
            # have been available.
            if use_memory:
                similar = memory.retrieve_similar(
                    conn, snap, pair, k=params.k, as_of=decision_ts,
                    fields=tuple(params.similarity_fields),
                )
            else:
                similar = []

            # A live LLM occasionally returns a reply the parser cannot turn into
            # a decision (it narrates instead of emitting JSON, or a provider
            # errors on one call). Over hundreds of sequential calls this is
            # expected, so a single bad reply must skip only that decision — not
            # abort the whole pair. The rule-based agent never raises here, so
            # its runs are unaffected. Skips are counted and surfaced so the
            # evaluation stays honest about how many decisions were dropped.
            try:
                rec = agent.recommend(snapshot, similar)
            except (ValueError, RuntimeError) as exc:
                if skips is not None:
                    skips.append(str(exc))
                continue
            if use_memory:
                calibrated, explanation = memory.calibrate_confidence(
                    rec.confidence, similar, prior_strength=params.prior_strength
                )
            else:
                calibrated, explanation = rec.confidence, "memory disabled for this arm"

            rec_id = memory.log_recommendation(
                conn,
                pair=pair, action=rec.action, confidence=calibrated,
                raw_confidence=rec.confidence,
                predicted_direction=rec.predicted_direction, reasoning=rec.reasoning,
                entry_price=entry_price, market_snapshot=snap,
                timestamp=decision_ts, source="REAL-HISTORICAL",
            )
            outcome = memory.record_outcome(
                conn, rec_id,
                entry_price=entry_price, exit_price=exit_price, action=rec.action,
                checked_at=exit_ts, horizon_hours=params.window_hours, band=params.band,
                notes="evaluation replay",
            )

            records.append({
                "pair": pair,
                "candle_index": i,
                "timestamp": decision_ts,
                "rsi": snap["rsi"],
                "ma_gap_pct": snap["ma_gap_pct"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_change": outcome["actual_pct_change"],
                "action": rec.action,
                "raw_confidence": rec.confidence,
                "calibrated_confidence": calibrated,
                "was_correct": bool(outcome["was_correct"]),
                "n_retrieved": len(similar),
                "n_retrieved_graded": sum(1 for s in similar if s.get("outcome")),
                "calibration_note": explanation,
            })
    finally:
        conn.close()
    return records


# ---------------------------------------------------------------------------
# Baseline arms
# ---------------------------------------------------------------------------
def _fixed_action_arm(records: list[dict], action: str, band: float) -> list[dict]:
    """Score a constant-action strategy on the same events."""
    return [
        {**r, "action": action,
         "was_correct": memory.grade_outcome(action, r["pct_change"], band=band)}
        for r in records
    ]


def _random_arm(records: list[dict], band: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for r in records:
        action = rng.choice(["BUY", "SELL", "HOLD"])
        out.append({**r, "action": action,
                    "was_correct": memory.grade_outcome(action, r["pct_change"], band=band)})
    return out


def _apply_base_rate_confidence(train: list[dict], test: list[dict]) -> list[dict]:
    """Give a baseline the confidence its training-split hit-rate justifies.

    A baseline handed a flat 50% would be trivially beaten on calibration
    metrics, which would flatter the agent. Letting each baseline state its own
    historical accuracy makes it a genuinely calibrated opponent.
    """
    rate = metrics.accuracy([r["was_correct"] for r in train]) * 100.0 if train else 50.0
    return [{**r, "confidence": rate} for r in test]


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate_pair(
    pair: str,
    params: Params,
    *,
    market: MarketData | None = None,
    refresh_cache: bool = False,
    since_ms: int | None = None,
    agent: Agent | None = None,
) -> dict:
    """Run every arm for one pair and score them on the hold-out split."""
    tf_hours = params.timeframe_hours()
    needed = int(params.days * 24 / tf_hours) + config.MIN_CANDLES
    needed = min(needed, 1000)  # exchange page limit

    candles = load_candles(
        pair, timeframe=params.timeframe, limit=needed, since_ms=since_ms,
        refresh=refresh_cache, market=market,
    )
    agent = agent or get_agent(params.engine)

    skips: list[str] = []
    full = replay(pair, candles, agent, params, use_memory=True, skips=skips)
    if not full:
        # Distinguish "no candles" from "the LLM produced nothing parseable on any
        # call" — the latter is the failure the report must not hide.
        if skips:
            return {
                "pair": pair,
                "error": f"no usable decisions: all {len(skips)} attempts were "
                         f"unparseable (e.g. {skips[0][:120]})",
                "n_skipped": len(skips),
            }
        return {"pair": pair, "error": "not enough candles for a single decision"}

    split = int(len(full) * params.train_frac)
    train, test = full[:split], full[split:]
    if not test:
        return {"pair": pair, "error": "hold-out split is empty; lower --train-frac"}

    arms: dict[str, list[dict]] = {}

    # --- the system itself, and the ablation of its calibration step ---------
    arms["memory+calibration"] = [{**r, "confidence": r["calibrated_confidence"]} for r in test]
    arms["memory_raw_confidence"] = [{**r, "confidence": r["raw_confidence"]} for r in test]

    # --- retrieval ablation --------------------------------------------------
    if getattr(agent, "uses_memory", False):
        nomem_full = replay(pair, candles, agent, params, use_memory=False)
        nomem_test = nomem_full[split:]
        arms["no_memory"] = [{**r, "confidence": r["raw_confidence"]} for r in nomem_test]
    else:
        # The rule-based agent ignores `similar_past` entirely, so disabling
        # retrieval would reproduce memory_raw_confidence exactly. Saying so is
        # more honest than printing a duplicate column.
        arms["no_memory"] = None  # type: ignore[assignment]

    # --- baselines -----------------------------------------------------------
    for action in ("BUY", "SELL", "HOLD"):
        tr = _fixed_action_arm(train, action, params.band)
        te = _fixed_action_arm(test, action, params.band)
        arms[f"always_{action.lower()}"] = _apply_base_rate_confidence(tr, te)

    rnd_all = _random_arm(full, params.band, params.seed)
    arms["random"] = _apply_base_rate_confidence(rnd_all[:split], rnd_all[split:])

    # --- score every arm -----------------------------------------------------
    scored = {}
    for name, rows in arms.items():
        if rows is None:
            scored[name] = {"skipped": "identical to memory_raw_confidence "
                                       "(this agent does not read memory)"}
            continue
        scored[name] = metrics.summarise(
            [r["action"] for r in rows],
            [r["confidence"] for r in rows],
            [r["was_correct"] for r in rows],
            [r["pct_change"] for r in rows],
            n_bins=params.n_bins,
        )

    # --- did calibration significantly improve the Brier score? --------------
    cal = arms["memory+calibration"]
    raw = arms["memory_raw_confidence"]
    brier_test = metrics.paired_bootstrap(
        metrics.squared_errors([r["confidence"] for r in cal], [r["was_correct"] for r in cal]),
        metrics.squared_errors([r["confidence"] for r in raw], [r["was_correct"] for r in raw]),
        statistic=metrics.mean, seed=params.seed,
    )

    # --- market reference ----------------------------------------------------
    buy_hold = (test[-1]["exit_price"] - test[0]["entry_price"]) / test[0]["entry_price"] * 100.0

    return {
        "pair": pair,
        "n_total_decisions": len(full),
        "n_skipped": len(skips),  # decisions dropped: agent gave no parseable reply
        "n_train": len(train),
        "n_test": len(test),
        "test_period_start": test[0]["timestamp"],
        "test_period_end": test[-1]["timestamp"],
        "buy_and_hold_return_pct": buy_hold,
        "arms": scored,
        "calibration_vs_raw_brier": brier_test,
        "events": arms["memory+calibration"],
        # Every arm's per-decision rows, kept so pairs can be pooled exactly.
        # Stripped before the summary is written to JSON (see `_strip_rows`).
        "_arm_rows": {name: rows for name, rows in arms.items() if rows is not None},
    }


def evaluate(
    pairs: list[str],
    params: Params,
    *,
    refresh_cache: bool = False,
    since_ms: int | None = None,
) -> dict:
    """Evaluate several pairs and aggregate."""
    market = MarketData() if refresh_cache or not _all_cached(pairs, params, since_ms) else None
    agent = get_agent(params.engine)

    per_pair = []
    for pair in pairs:
        print(f"  Replaying {pair} ...")
        try:
            per_pair.append(evaluate_pair(
                pair, params, market=market, refresh_cache=refresh_cache,
                since_ms=since_ms, agent=agent,
            ))
        except Exception as exc:
            print(f"  ! {pair} failed: {exc}")
            per_pair.append({"pair": pair, "error": str(exc)})

    ok = [p for p in per_pair if "error" not in p]
    return {
        "params": {**asdict(params), "similarity_fields": list(params.similarity_fields)},
        "pairs": pairs,
        "per_pair": per_pair,
        "pooled": _pool(ok, params) if ok else None,
    }


def _all_cached(pairs: list[str], params: Params, since_ms: int | None) -> bool:
    """True when every pair's candles are already on disk (so we can stay offline)."""
    from .data_layer import _cache_path

    tf_hours = params.timeframe_hours()
    needed = min(int(params.days * 24 / tf_hours) + config.MIN_CANDLES, 1000)
    return all(
        _cache_path(p, params.timeframe, needed, since_ms, config.CACHE_DIR).exists()
        for p in pairs
    )


def _pool(results: list[dict], params: Params) -> dict:
    """Pool every pair's hold-out decisions into one combined scorecard.

    Per-pair samples are small; pooling is what gives the headline numbers
    enough decisions to say anything with a straight face.
    """
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        for name, rows in r.get("_arm_rows", {}).items():
            by_arm.setdefault(name, []).extend(rows)

    pooled = {
        name: metrics.summarise(
            [x["action"] for x in rows], [x["confidence"] for x in rows],
            [x["was_correct"] for x in rows], [x["pct_change"] for x in rows],
            n_bins=params.n_bins,
        )
        for name, rows in by_arm.items() if rows
    }

    # Pooled significance test for the headline claim.
    cal, raw = by_arm.get("memory+calibration", []), by_arm.get("memory_raw_confidence", [])
    brier_test = metrics.paired_bootstrap(
        metrics.squared_errors([r["confidence"] for r in cal], [r["was_correct"] for r in cal]),
        metrics.squared_errors([r["confidence"] for r in raw], [r["was_correct"] for r in raw]),
        statistic=metrics.mean, seed=params.seed,
    ) if cal and raw else None

    return {
        "arms": pooled,
        "n_pairs": len(results),
        "calibration_vs_raw_brier": brier_test,
        "mean_buy_and_hold_return_pct": statistics.fmean(
            [r["buy_and_hold_return_pct"] for r in results]
        ),
    }


def _strip_rows(result: dict) -> dict:
    """Remove the bulky per-decision rows before a summary is serialised."""
    clean = dict(result)
    clean["per_pair"] = [
        {k: v for k, v in p.items() if k not in ("_arm_rows", "events")}
        for p in result.get("per_pair", [])
    ]
    return clean


# ---------------------------------------------------------------------------
# Ablation sweeps
# ---------------------------------------------------------------------------
SWEEPABLE = {"k": int, "prior_strength": int, "band": float, "window_hours": float}


def sweep(
    pairs: list[str],
    params: Params,
    parameter: str,
    values: list,
    *,
    since_ms: int | None = None,
) -> dict:
    """Re-run the evaluation once per value of one parameter.

    Cheap with the rule engine and cached candles, and it turns "we chose k=3"
    into "we chose k=3 and here is what k=1, 5 and 10 did".
    """
    if parameter not in SWEEPABLE:
        raise ValueError(f"Cannot sweep '{parameter}'. Options: {sorted(SWEEPABLE)}")

    rows = []
    for value in values:
        cast = SWEEPABLE[parameter](value)
        variant = Params(**{**asdict(params), parameter: cast,
                            "similarity_fields": params.similarity_fields})
        print(f"\n  --- {parameter} = {cast} ---")
        result = evaluate(pairs, variant, since_ms=since_ms)
        pooled = result.get("pooled")
        if not pooled:
            continue
        cal = pooled["arms"]["memory+calibration"]
        raw = pooled["arms"]["memory_raw_confidence"]
        rows.append({
            parameter: cast,
            "n": cal["n"],
            "accuracy_pct": cal["accuracy_pct"],
            "brier_calibrated": cal["brier"],
            "brier_raw": raw["brier"],
            "ece_calibrated_pct": cal["ece_pct"],
            "ece_raw_pct": raw["ece_pct"],
        })
    return {"parameter": parameter, "values": values, "rows": rows}
