"""Demo-friendly terminal output helpers.

These keep the loop/seeder code readable and give the screen-recorded demo a
clear visual story: market state -> retrieved memory -> calibration -> call.
"""

from __future__ import annotations

from datetime import datetime, timezone


def rule(char: str = "─", width: int = 70) -> str:
    return char * width


def header(title: str) -> str:
    return f"\n{rule('═')}\n  {title}\n{rule('═')}"


def section(title: str) -> str:
    return f"\n{title}\n{rule()}"


def fmt_time(ms_or_s: float, *, is_ms: bool = False) -> str:
    seconds = ms_or_s / 1000.0 if is_ms else ms_or_s
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def print_snapshot(snapshot: dict) -> None:
    print(section(f"📊 MARKET SNAPSHOT — {snapshot['pair']}"))
    print(f"  Time        : {fmt_time(snapshot['timestamp_ms'], is_ms=True)}")
    print(f"  Price       : {snapshot['price']}")
    print(f"  RSI(14)     : {snapshot['rsi']}  ({snapshot['rsi_signal']})")
    print(
        f"  MA crossover: fast {snapshot['sma_fast']} vs slow {snapshot['sma_slow']}  "
        f"-> gap {snapshot['ma_gap_pct']:+.2f}% ({snapshot['ma_signal']})"
    )


def print_retrieved(similar: list[dict]) -> None:
    print(section("🧠 RETRIEVED MEMORY (most similar past decisions)"))
    if not similar:
        print("  (memory is empty — this is a cold start with no history yet)")
        return
    for i, d in enumerate(similar, 1):
        snap = d["snapshot"]
        cond = f"RSI={snap.get('rsi')}, MAgap={snap.get('ma_gap_pct')}%"
        if d["outcome"] is None:
            verdict = "outcome not yet known"
        else:
            mark = "✅ CORRECT" if d["outcome"]["was_correct"] else "❌ WRONG"
            verdict = f"{d['outcome']['actual_pct_change']:+.2f}% -> {mark}"
        print(
            f"  {i}. [{d['action']:<4}] conf {d['confidence']:>3}  "
            f"dist {d['distance']:.3f}  ({cond})  →  {verdict}"
        )


def print_calibration(raw: int, calibrated: int, explanation: str) -> None:
    print(section("⚖️  CONFIDENCE CALIBRATION (the self-learning step)"))
    arrow = "→"
    print(f"  Raw confidence       : {raw}")
    print(f"  Calibrated confidence: {calibrated}   ({raw} {arrow} {calibrated})")
    print(f"  Why: {explanation}")


def print_recommendation(rec_action: str, calibrated: int, direction: str, reasoning: str) -> None:
    print(section("🤖 FINAL RECOMMENDATION"))
    print(f"  Action     : {rec_action}")
    print(f"  Confidence : {calibrated}/100  (calibrated)")
    print(f"  Direction  : {direction}")
    print(f"  Reasoning  : {reasoning}")
