"""CryptoMind — interactive Streamlit front-end for the demo.

This is a thin UI over the SAME core logic the CLI uses (cryptomind.loop.analyze,
cryptomind.seed.run_seed, cryptomind.memory). It exists purely to make the
self-learning memory loop interactive and screen-recording friendly; none of the
assessed core modules are changed.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from cryptomind import config, memory
from cryptomind.agent import get_agent
from cryptomind.data_layer import MarketData
from cryptomind.loop import analyze
from cryptomind.seed import run_seed

st.set_page_config(page_title="CryptoMind", page_icon="🧠", layout="wide")


# --- Shared resources (created once, reused across reruns) -------------------
@st.cache_resource(show_spinner=False)
def get_market() -> MarketData:
    """A single ccxt client, reused so every interaction doesn't re-init it."""
    return MarketData()


def llm_key_available() -> bool:
    """True if either supported LLM backend has a key configured."""
    return bool(config.get_gemini_key() or config.get_anthropic_key())


# --- Header -----------------------------------------------------------------
st.title("🧠 CryptoMind")
st.caption(
    "A single-agent crypto advisor with a **self-learning memory loop**: it logs "
    "every call, verifies the outcome against real prices, and retrieves similar "
    "past decisions to calibrate its confidence."
)

# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    pair = st.selectbox("Trading pair", config.DEFAULT_PAIRS, index=0)

    engine_label = st.radio(
        "Reasoning engine",
        ["Rule-based (free, offline)", "Gemini LLM (free API key)"],
        index=0,
        help="Rule-based needs no key. Gemini uses GEMINI_API_KEY (free at aistudio.google.com).",
    )
    engine = "llm" if engine_label.startswith("Gemini") else "rule"

    if engine == "llm" and not llm_key_available():
        st.warning(
            "No GEMINI_API_KEY set — LLM calls will fail. Get a free key at "
            "aistudio.google.com and add it in Secrets, or use the rule-based engine."
        )
    else:
        st.success(f"Engine ready: {'Gemini LLM' if engine == 'llm' else 'Rule-based'}")

    st.divider()
    stats = memory.connect()
    overall = memory.accuracy_stats(stats)
    stats.close()
    st.metric("Decisions in memory (graded)", overall["graded"])
    st.metric("Overall historical accuracy", f"{overall['hit_rate_pct']:.0f}%")


# --- Tabs -------------------------------------------------------------------
tab_live, tab_seed, tab_memory = st.tabs(
    ["▶️  Live decision", "🌱 Build memory (seed)", "📚 Memory browser"]
)


# ===========================================================================
# TAB 1 — Live decision (the main demo)
# ===========================================================================
with tab_live:
    st.subheader(f"Live decision for {pair}")
    st.write(
        "Pulls **live** market data, retrieves similar past decisions, asks the "
        "agent, then calibrates the confidence using historical accuracy."
    )
    if st.button("▶️  Run live analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching live data, retrieving memory, reasoning, calibrating…"):
                result = analyze(pair, engine, market=get_market(), agent=get_agent(engine))
        except Exception as exc:
            st.error(f"Could not complete the loop: {exc}")
            st.stop()

        snap = result["snapshot"]

        # --- Stage 1: market snapshot ---
        st.markdown("#### 📊 Market snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{snap['price']:,}")
        c2.metric("RSI(14)", f"{snap['rsi']:.1f}", snap["rsi_signal"])
        c3.metric("MA gap", f"{snap['ma_gap_pct']:+.2f}%", snap["ma_signal"])
        c4.metric("Fast / Slow MA", f"{snap['sma_fast']:,.0f}", f"vs {snap['sma_slow']:,.0f}")

        # --- Stage 2: retrieved memory ---
        st.markdown("#### 🧠 Retrieved memory (most similar past decisions)")
        if not result["similar"]:
            st.info("Memory is empty for this pair — a cold start. Seed history first (next tab).")
        else:
            rows = []
            for d in result["similar"]:
                s = d["snapshot"]
                if d["outcome"] is None:
                    verdict = "⏳ not yet known"
                else:
                    mark = "✅ correct" if d["outcome"]["was_correct"] else "❌ wrong"
                    verdict = f"{d['outcome']['actual_pct_change']:+.2f}%  →  {mark}"
                rows.append({
                    "action": d["action"],
                    "confidence": d["confidence"],
                    "similarity dist": round(d["distance"], 3),
                    "RSI then": s.get("rsi"),
                    "MA gap then": s.get("ma_gap_pct"),
                    "outcome": verdict,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # --- Stage 3: calibration ---
        st.markdown("#### ⚖️ Confidence calibration (the self-learning step)")
        raw = result["raw_confidence"]
        cal = result["calibrated_confidence"]
        cc1, cc2 = st.columns([1, 2])
        cc1.metric("Confidence", f"{cal}/100", f"{cal - raw:+d} vs raw {raw}")
        cc2.info(result["calibration_explanation"])

        # --- Stage 4: final recommendation ---
        st.markdown("#### 🤖 Final recommendation")
        action = result["action"]
        banner = {"BUY": st.success, "SELL": st.error, "HOLD": st.warning}.get(action, st.info)
        banner(f"### {action}  ·  confidence {cal}/100")
        st.write(f"**Predicted direction:** {result['predicted_direction']}")
        st.write(f"**Reasoning:** {result['reasoning']}")
        st.caption(f"Logged as recommendation #{result['recommendation_id']} (source: live).")


# ===========================================================================
# TAB 2 — Seeding
# ===========================================================================
with tab_seed:
    st.subheader("Build memory from REAL historical data")
    st.write(
        "Replays real past candles, makes a decision at each step, and grades it "
        "against the **known** subsequent price. Every row is tagged "
        "`REAL-HISTORICAL` — real prices, replayed history, nothing fabricated."
    )
    sc1, sc2, sc3 = st.columns(3)
    seed_pairs = sc1.multiselect("Pairs", config.DEFAULT_PAIRS, default=config.DEFAULT_PAIRS)
    days = sc2.slider("Days of history", 7, 60, 30)
    limit = sc3.slider("Decisions per pair", 10, 100, 60)
    seed_engine_label = st.radio(
        "Seeding engine",
        ["Rule-based (fast, free)", "Gemini LLM (slower, one call per decision)"],
        index=0, horizontal=True,
    )
    seed_engine = "llm" if seed_engine_label.startswith("Gemini") else "rule"
    if seed_engine == "llm":
        st.warning(
            f"This makes up to ~{limit * len(seed_pairs)} Gemini API calls and is slow. "
            f"Gemini's free tier is rate-limited, so rule-based is recommended for seeding."
        )

    if st.button("🌱 Seed memory now", type="primary", use_container_width=True):
        if not seed_pairs:
            st.error("Pick at least one pair.")
        else:
            try:
                with st.spinner("Fetching real historical candles and grading decisions…"):
                    summary = run_seed(
                        seed_pairs, engine=seed_engine, days=days, limit=limit,
                    )
                st.success(
                    f"Seeded {summary['total']} REAL-HISTORICAL decisions. "
                    f"Overall accuracy: {summary['overall']['correct']}/"
                    f"{summary['overall']['graded']} "
                    f"({summary['overall']['hit_rate_pct']:.0f}%)."
                )
                st.dataframe(
                    [
                        {
                            "pair": p["pair"],
                            "seeded": p["seeded"],
                            "accuracy": (
                                f"{p['accuracy']['correct']}/{p['accuracy']['graded']} "
                                f"({p['accuracy']['hit_rate_pct']:.0f}%)"
                                if p.get("accuracy") else p.get("error", "—")
                            ),
                        }
                        for p in summary["per_pair"]
                    ],
                    use_container_width=True, hide_index=True,
                )
                st.info("Now switch to the **Live decision** tab and watch this memory calibrate a fresh call.")
            except Exception as exc:
                st.error(f"Seeding failed: {exc}")


# ===========================================================================
# TAB 3 — Memory browser
# ===========================================================================
with tab_memory:
    st.subheader("What the agent remembers")
    conn = memory.connect()
    try:
        view_pair = st.selectbox("Filter by pair", ["(all)"] + config.DEFAULT_PAIRS, index=0)
        filt = None if view_pair == "(all)" else view_pair

        s = memory.accuracy_stats(conn, filt)
        m1, m2, m3 = st.columns(3)
        m1.metric("Graded decisions", s["graded"])
        m2.metric("Correct", s["correct"])
        m3.metric("Hit-rate", f"{s['hit_rate_pct']:.0f}%")

        recent = memory.recent_recommendations(conn, filt, limit=50)
        if not recent:
            st.info("Memory is empty. Seed history in the previous tab.")
        else:
            rows = []
            for r in recent:
                if r["was_correct"] is None:
                    outcome = "⏳ pending"
                else:
                    outcome = (
                        f"{r['actual_pct_change']:+.2f}% "
                        f"{'✅' if r['was_correct'] else '❌'}"
                    )
                rows.append({
                    "id": r["id"],
                    "pair": r["pair"],
                    "action": r["action"],
                    "conf": r["confidence"],
                    "raw": r["raw_confidence"],
                    "RSI": r["rsi"],
                    "MA gap%": r["ma_gap_pct"],
                    "entry": r["entry_price"],
                    "outcome": outcome,
                    "source": r["source"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
    finally:
        conn.close()

    st.divider()
    st.write("**Verify pending live calls** against current prices (grades calls older than the window):")
    win = st.number_input("Window (hours)", 0.0, 72.0, float(config.DEFAULT_WINDOW_HOURS), step=1.0)
    if st.button("✅ Verify outcomes now"):
        try:
            with st.spinner("Fetching current prices and grading…"):
                conn = memory.connect()
                graded = memory.verify_outcomes(conn, win, get_market().current_price)
                conn.close()
            if graded:
                st.success(f"Graded {len(graded)} recommendation(s).")
            else:
                st.info("Nothing old enough to grade yet.")
        except Exception as exc:
            st.error(f"Verification failed: {exc}")
