"""Seeding / simulation helper.

The memory loop only gets interesting once there is history. This backfills the
database using REAL HISTORICAL price data:

  * pull older OHLCV candles for each pair,
  * walk forward through them in chronological order,
  * at each step build the real indicators and ask the agent for a decision
    (retrieving earlier seeded decisions so calibration is genuinely learning
    over the simulated history),
  * immediately verify each decision against the KNOWN subsequent real price.

Every seeded row is tagged source='REAL-HISTORICAL' so nothing is mistaken for
fabricated data: the prices and outcomes are real, only the "live moment" is
replayed from history.
"""

from __future__ import annotations

from . import config, display, memory
from .agent import Agent, get_agent
from .data_layer import MarketData, build_snapshot

# Hours-per-candle for the timeframes we support, used to convert the
# verification window (hours) into a number of candles to look ahead.
TIMEFRAME_HOURS = {"15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0}


def _timeframe_hours(timeframe: str) -> float:
    if timeframe not in TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Known: {list(TIMEFRAME_HOURS)}")
    return TIMEFRAME_HOURS[timeframe]


def seed_pair(
    conn,
    pair: str,
    agent: Agent,
    market: MarketData,
    *,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    days: int = 30,
    limit: int = 60,
    window_hours: float = config.DEFAULT_WINDOW_HOURS,
) -> dict:
    """Backfill up to `limit` verified historical recommendations for one pair."""
    tf_hours = _timeframe_hours(timeframe)
    window_candles = max(1, round(window_hours / tf_hours))

    # Candles we need: history span + indicator warmup + room to peek ahead.
    needed = int(days * 24 / tf_hours) + config.MIN_CANDLES + window_candles
    needed = min(needed, 1000)  # Binance returns at most 1000 candles per call

    print(f"\n  Fetching {needed} real {timeframe} candles for {pair} ...")
    candles = market.fetch_ohlcv(pair, timeframe=timeframe, limit=needed)
    if len(candles) < config.MIN_CANDLES + window_candles + 1:
        print(f"  ! Not enough history returned for {pair}; skipping.")
        return {"pair": pair, "seeded": 0}

    # Decisions can be made from the first candle that has valid indicators, up
    # to the last candle that still has `window_candles` of future data to grade.
    first_i = config.MIN_CANDLES - 1
    last_i = len(candles) - 1 - window_candles
    decision_indices = list(range(first_i, last_i + 1))

    # Sample evenly so we produce ~`limit` decisions rather than hundreds.
    step = max(1, len(decision_indices) // limit)
    chosen = decision_indices[::step][:limit]

    seeded = 0
    for i in chosen:
        prefix = candles[: i + 1]
        try:
            snapshot = build_snapshot(pair, prefix)
        except ValueError:
            continue  # not enough warmup yet
        snap_dict = snapshot.to_dict()

        decision_ts = candles[i][0] / 1000.0          # ms -> unix seconds
        entry_price = float(candles[i][4])
        exit_candle = candles[i + window_candles]
        exit_price = float(exit_candle[4])
        exit_ts = exit_candle[0] / 1000.0

        # Learn over history: retrieve earlier seeded decisions, calibrate on them.
        similar = memory.retrieve_similar(conn, snap_dict, pair, k=3)
        rec = agent.recommend(snapshot, similar)
        calibrated, _ = memory.calibrate_confidence(rec.confidence, similar)

        rec_id = memory.log_recommendation(
            conn,
            pair=pair,
            action=rec.action,
            confidence=calibrated,
            raw_confidence=rec.confidence,
            predicted_direction=rec.predicted_direction,
            reasoning=rec.reasoning,
            entry_price=entry_price,
            market_snapshot=snap_dict,
            timestamp=decision_ts,
            source="REAL-HISTORICAL",
        )
        memory.record_outcome(
            conn,
            rec_id,
            entry_price=entry_price,
            exit_price=exit_price,
            action=rec.action,
            checked_at=exit_ts,
            notes=f"REAL-HISTORICAL: graded against known price {window_hours}h later",
        )
        seeded += 1

    stats = memory.accuracy_stats(conn, pair)
    print(
        f"  ✔ Seeded {seeded} REAL-HISTORICAL decisions for {pair}. "
        f"Accuracy so far: {stats['correct']}/{stats['graded']} ({stats['hit_rate_pct']:.0f}%)."
    )
    return {"pair": pair, "seeded": seeded, "accuracy": stats}


def run_seed(
    pairs: list[str],
    engine: str = "rule",
    *,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    days: int = 30,
    limit: int = 60,
    window_hours: float = config.DEFAULT_WINDOW_HOURS,
    db_path=config.DB_PATH,
) -> dict:
    """Seed history for several pairs and print a demo-friendly summary.

    Returns a summary dict (total seeded + overall accuracy + per-pair results)
    so callers such as the web UI can render the result natively.
    """
    print(display.header(f"Seeding memory from REAL HISTORICAL data (engine: {engine})"))
    print(
        f"  Pairs={pairs}  timeframe={timeframe}  days≈{days}  "
        f"target≈{limit}/pair  verify-window={window_hours}h"
    )
    if engine != "rule":
        print(
            f"  ⚠ LLM engine: this makes one API call per decision "
            f"(up to ~{limit * len(pairs)} calls). Use 'rule' for a fast, free seed."
        )

    market = MarketData()
    agent = get_agent(engine)
    conn = memory.connect(db_path)

    total = 0
    per_pair = []
    for pair in pairs:
        try:
            result = seed_pair(
                conn, pair, agent, market,
                timeframe=timeframe, days=days, limit=limit, window_hours=window_hours,
            )
            total += result["seeded"]
            per_pair.append(result)
        except Exception as exc:
            print(f"  ! Failed to seed {pair}: {exc}")
            per_pair.append({"pair": pair, "seeded": 0, "error": str(exc)})

    overall = memory.accuracy_stats(conn)
    print(display.section("📦 SEEDING COMPLETE"))
    print(f"  Total seeded decisions: {total}")
    print(
        f"  Overall historical accuracy: {overall['correct']}/{overall['graded']} "
        f"({overall['hit_rate_pct']:.0f}%)"
    )
    print("  All seeded rows are tagged source='REAL-HISTORICAL' (real prices, replayed history).")
    print("  Now run the live loop to see this memory calibrate a fresh recommendation.\n")
    conn.close()
    return {"total": total, "overall": overall, "per_pair": per_pair, "engine": engine}
