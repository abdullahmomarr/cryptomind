"""The full live loop.

For a chosen pair this:
  1. pulls live market data and computes indicators
  2. retrieves similar past decisions from memory
  3. feeds market state + retrieved history into the agent
  4. gets a raw recommendation
  5. calibrates the confidence using historical accuracy
  6. logs the recommendation
  7. prints a clean, demo-friendly summary of the whole story
"""

from __future__ import annotations

from . import config, display, memory
from .agent import Agent, get_agent
from .data_layer import MarketData


def analyze(
    pair: str,
    engine: str = "llm",
    *,
    db_path=config.DB_PATH,
    market: MarketData | None = None,
    agent: Agent | None = None,
) -> dict:
    """Run one full pass of the memory loop for `pair` WITHOUT printing.

    This is the shared code path used by both the CLI (`run_live`) and the
    Streamlit web UI, so the two never drift apart. Returns a structured result
    describing every stage of the loop.
    """
    market = market or MarketData()
    agent = agent or get_agent(engine)
    conn = memory.connect(db_path)
    try:
        # 1. Live data + indicators -------------------------------------------
        snapshot = market.latest_snapshot(pair)
        snap_dict = snapshot.to_dict()

        # 2. Retrieve similar past decisions ----------------------------------
        similar = memory.retrieve_similar(conn, snap_dict, pair, k=3)

        # 3 + 4. Agent produces a raw recommendation --------------------------
        rec = agent.recommend(snapshot, similar)

        # 5. Calibrate confidence from historical accuracy --------------------
        calibrated, explanation = memory.calibrate_confidence(rec.confidence, similar)

        # 6. Log the (calibrated) recommendation ------------------------------
        rec_id = memory.log_recommendation(
            conn,
            pair=pair,
            action=rec.action,
            confidence=calibrated,
            raw_confidence=rec.confidence,
            predicted_direction=rec.predicted_direction,
            reasoning=rec.reasoning,
            entry_price=snapshot.price,
            market_snapshot=snap_dict,
            source="live",
        )
        stats = memory.accuracy_stats(conn, pair)
    finally:
        conn.close()

    return {
        "pair": pair,
        "engine": engine,
        "snapshot": snap_dict,
        "similar": similar,
        "action": rec.action,
        "raw_confidence": rec.confidence,
        "calibrated_confidence": calibrated,
        "calibration_explanation": explanation,
        "predicted_direction": rec.predicted_direction,
        "reasoning": rec.reasoning,
        "recommendation_id": rec_id,
        "stats": stats,
    }


def run_live(
    pair: str,
    engine: str = "llm",
    *,
    db_path=config.DB_PATH,
    market: MarketData | None = None,
    agent: Agent | None = None,
) -> dict:
    """Run one full pass of the memory loop for `pair` and print the demo summary."""
    print(display.header(f"CryptoMind — live decision for {pair}  (engine: {engine})"))
    try:
        result = analyze(pair, engine, db_path=db_path, market=market, agent=agent)
    except Exception as exc:
        print(f"\n❌ Could not complete the live loop: {exc}")
        raise

    display.print_snapshot(result["snapshot"])
    display.print_retrieved(result["similar"])
    display.print_calibration(
        result["raw_confidence"], result["calibrated_confidence"],
        result["calibration_explanation"],
    )
    display.print_recommendation(
        result["action"], result["calibrated_confidence"],
        result["predicted_direction"], result["reasoning"],
    )

    stats, rec_id = result["stats"], result["recommendation_id"]
    print(display.section("📈 MEMORY STATE"))
    if stats["graded"]:
        print(
            f"  Logged this call as recommendation #{rec_id}. "
            f"Historical accuracy for {pair}: {stats['correct']}/{stats['graded']} "
            f"({stats['hit_rate_pct']:.0f}%)."
        )
    else:
        print(
            f"  Logged this call as recommendation #{rec_id}. "
            f"No graded history for {pair} yet — run the seeder or wait for "
            f"`verify` to grade it."
        )
    print()
    return result


def run_verify(window_hours: float = config.DEFAULT_WINDOW_HOURS, *, db_path=config.DB_PATH) -> int:
    """Grade pending live recommendations older than the window against current prices."""
    print(display.header(f"Verifying outcomes older than {window_hours}h against live prices"))
    market = MarketData()
    conn = memory.connect(db_path)
    results = memory.verify_outcomes(conn, window_hours, market.current_price)

    if not results:
        print("  Nothing to verify yet (no recommendations are old enough).")
    else:
        for o in results:
            mark = "✅ CORRECT" if o["was_correct"] else "❌ WRONG"
            print(
                f"  rec #{o['recommendation_id']}: {o['actual_pct_change']:+.2f}% "
                f"-> {mark}"
            )
    conn.close()
    print()
    return len(results)
