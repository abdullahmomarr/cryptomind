"""Tests for the agent layer: response parsing, validation, caching and retry.

The caching and retry behaviour matters more than it looks. An LLM evaluation
makes hundreds of sequential calls against a rate limited free tier, so a single
throttle response must not destroy a run, and a re-run must not spend the quota
again or silently produce different numbers.
"""

import json

import pytest

from cryptomind import agent, config
from cryptomind.agent import (
    Recommendation,
    RuleBasedAgent,
    _parse_json_response,
    cached_call,
    get_agent,
    with_retry,
)
from cryptomind.data_layer import MarketSnapshot


# --- response parsing --------------------------------------------------------
def test_parses_bare_json():
    data = _parse_json_response('{"action": "BUY", "confidence": 70}')
    assert data["action"] == "BUY"


def test_parses_json_wrapped_in_a_code_fence():
    """Models routinely wrap JSON in a fence despite being told not to."""
    text = '```json\n{"action": "SELL", "confidence": 40}\n```'
    assert _parse_json_response(text)["action"] == "SELL"


def test_parses_json_buried_in_prose():
    text = 'Sure! Here is my answer:\n{"action": "HOLD", "confidence": 55}\nHope that helps.'
    assert _parse_json_response(text)["action"] == "HOLD"


def test_recovers_a_reply_truncated_by_the_token_limit():
    """A real failure seen during the Groq evaluation.

    The model hit its token ceiling mid-sentence, so the object never closed.
    Brace matching then finds no closing brace at all and a strict parse fails
    on a reply whose decision is perfectly readable.
    """
    truncated = (
        '{\n  "action": "HOLD",\n  "confidence": 60,\n'
        '  "predicted_direction": "slight downturn",\n'
        '  "reasoning": "The RSI is neutral and the MA signal is bearish, so I am'
    )
    data = _parse_json_response(truncated)
    assert data["action"] == "HOLD"
    assert data["confidence"] == 60
    assert data["predicted_direction"] == "slight downturn"


def test_recovers_a_decision_from_a_reply_that_is_not_json_at_all():
    text = 'My answer: "action": "SELL", "confidence": 35, and that is my view.'
    data = _parse_json_response(text)
    assert data["action"] == "SELL"
    assert data["confidence"] == 35


def test_refuses_to_invent_a_missing_decision():
    """Guessing at a missing action would fabricate evaluation data."""
    with pytest.raises((ValueError, KeyError)):
        _parse_json_response('{"predicted_direction": "up", "reasoning": "vibes"}')


# --- validation --------------------------------------------------------------
def test_recommendation_normalises_case():
    assert Recommendation("buy", 50, "up", "x").action == "BUY"


def test_recommendation_clamps_confidence_out_of_range():
    """A model that ignores the 0-100 bound must not poison the metrics."""
    assert Recommendation("BUY", 150, "up", "x").confidence == 100
    assert Recommendation("BUY", -20, "up", "x").confidence == 0


def test_recommendation_accepts_a_confidence_the_model_quoted():
    """A real failure seen during the Groq evaluation.

    Asked for JSON, a model will occasionally quote a number, returning
    "confidence": "80" instead of 80. Rounding a string raises TypeError, and
    one such reply in 1367 aborted a whole pair's run after it had already
    spent most of the day's token quota. A quoted number is unambiguous, so
    coerce it rather than throwing the run away.
    """
    assert Recommendation("BUY", "80", "up", "x").confidence == 80
    assert Recommendation("BUY", "72.6", "up", "x").confidence == 73


def test_recommendation_still_rejects_a_confidence_that_is_not_a_number():
    """Coercing quoted digits must not become coercing anything at all."""
    with pytest.raises(ValueError):
        Recommendation("BUY", "high", "up", "x")


def test_recommendation_rejects_an_invented_action():
    """Rejecting beats coercing: a bad action would break grading much later."""
    with pytest.raises(ValueError):
        Recommendation("MOON", 90, "up", "x")


# --- caching -----------------------------------------------------------------
def test_cached_call_hits_the_api_once_then_serves_from_disk(tmp_path):
    calls = []

    def fake():
        calls.append(1)
        return '{"action": "BUY"}'

    first = cached_call("prompt-a", "model-x", fake, cache_dir=tmp_path)
    second = cached_call("prompt-a", "model-x", fake, cache_dir=tmp_path)

    assert first == second
    assert len(calls) == 1, "second identical call should have been served from cache"


def test_cache_distinguishes_prompts_and_models(tmp_path):
    calls = []

    def fake():
        calls.append(1)
        return "reply"

    cached_call("prompt-a", "model-x", fake, cache_dir=tmp_path)
    cached_call("prompt-b", "model-x", fake, cache_dir=tmp_path)
    cached_call("prompt-a", "model-y", fake, cache_dir=tmp_path)

    assert len(calls) == 3, "different prompt or model must not share a cache entry"


def test_cache_survives_an_entry_left_empty_by_an_interrupted_write(tmp_path):
    """A real failure seen during the Groq evaluation.

    The machine ran out of disk mid-run, leaving a zero-byte cache file. Every
    later run that reached that prompt died on a JSON decode error, so one
    unreadable file out of a thousand permanently poisoned the cache. An entry
    that cannot be read is worth no more than a missing one: refetch and repair.
    """
    from cryptomind.agent import _cache_path

    path = _cache_path("prompt-a", "model-x", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    assert cached_call("prompt-a", "model-x", lambda: "refetched", cache_dir=tmp_path) == "refetched"
    assert json.loads(path.read_text())["response"] == "refetched"
    # and the repaired entry is a normal cache hit from then on
    assert cached_call("prompt-a", "model-x", lambda: "boom", cache_dir=tmp_path) == "refetched"


def test_cache_write_leaves_no_partial_file_behind(tmp_path):
    """Entries are written to a temp file and renamed, so a kill cannot truncate one."""
    cached_call("prompt-a", "model-x", lambda: "reply", cache_dir=tmp_path)
    assert not list(tmp_path.glob("**/*.tmp")), "a temporary file was left in the cache"


def test_cache_can_be_disabled(tmp_path):
    calls = []

    def fake():
        calls.append(1)
        return "reply"

    cached_call("p", "m", fake, cache_dir=tmp_path, use_cache=False)
    cached_call("p", "m", fake, cache_dir=tmp_path, use_cache=False)
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.json"))


# --- retry -------------------------------------------------------------------
def test_retry_recovers_from_a_rate_limit(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKOFF_SECONDS", 0.0)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Error code: 429 - rate limit exceeded")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert len(attempts) == 3


def test_retry_does_not_mask_a_real_error(monkeypatch):
    """An auth failure must surface immediately, not after five slow retries."""
    monkeypatch.setattr(config, "LLM_BACKOFF_SECONDS", 0.0)
    attempts = []

    def broken():
        attempts.append(1)
        raise RuntimeError("401 invalid api key")

    with pytest.raises(RuntimeError, match="401"):
        with_retry(broken)
    assert len(attempts) == 1, "a non-retryable error should not be retried"


def test_retry_gives_up_after_the_configured_limit(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 3)
    attempts = []

    def always_throttled():
        attempts.append(1)
        raise RuntimeError("429 rate limit")

    with pytest.raises(RuntimeError):
        with_retry(always_throttled)
    assert len(attempts) == 3


# --- factory -----------------------------------------------------------------
def test_factory_returns_the_rule_agent_without_any_key():
    assert isinstance(get_agent("rule"), RuleBasedAgent)


def test_factory_rejects_an_unknown_engine():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_agent("nonsense")


def test_groq_engine_reports_a_helpful_error_when_the_key_is_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_agent("groq")


def test_rule_agent_does_not_read_memory_but_llm_agents_do():
    """The evaluation uses this flag to decide if a retrieval ablation is real."""
    assert RuleBasedAgent().uses_memory is False
    assert agent.OpenAICompatibleAgent.uses_memory is True
    assert agent.LLMAgent.uses_memory is True


# --- the rule agent's own logic ----------------------------------------------
def _snap(rsi, gap):
    return MarketSnapshot(
        pair="BTC/USDT", timestamp_ms=0, price=100.0, rsi=rsi,
        sma_fast=100.0, sma_slow=100.0, ma_gap_pct=gap,
        rsi_signal="neutral", ma_signal="bullish" if gap >= 0 else "bearish",
    )


def test_rule_agent_buys_an_oversold_bullish_setup():
    rec = RuleBasedAgent().recommend(_snap(rsi=20, gap=1.5), [])
    assert rec.action == "BUY"


def test_rule_agent_sells_an_overbought_bearish_setup():
    rec = RuleBasedAgent().recommend(_snap(rsi=85, gap=-1.5), [])
    assert rec.action == "SELL"


def test_rule_agent_is_deterministic():
    """It is the experimental control, so identical input must give identical output."""
    a = RuleBasedAgent().recommend(_snap(55, 0.4), [])
    b = RuleBasedAgent().recommend(_snap(55, 0.4), [])
    assert (a.action, a.confidence) == (b.action, b.confidence)
