"""Agent reasoning layer.

A single agent turns a market snapshot (plus retrieved memory) into a
recommendation: action, confidence, predicted direction, and plain-language
reasoning.

The agent is defined behind a small interface (`Agent`) so the LLM backend is
swappable. Two implementations are provided:

  * LLMAgent      -> Anthropic Claude (the live reasoning layer).
  * RuleBasedAgent -> deterministic logic from the indicators. This doubles as
                      (a) a free/offline engine for seeding history, and
                      (b) the seam for plugging in a local Ollama model later
                          (an OllamaAgent would implement the same interface).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from . import config
from .data_layer import MarketSnapshot

VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


@dataclass
class Recommendation:
    """What the agent produces for a single market snapshot."""

    action: str               # BUY | SELL | HOLD
    confidence: int           # 0-100 (this is the RAW confidence from the agent)
    predicted_direction: str  # short phrase, e.g. "up ~1-2% over next few hours"
    reasoning: str            # plain-language explanation

    def __post_init__(self) -> None:
        self.action = self.action.upper().strip()
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{self.action}', expected one of {VALID_ACTIONS}")
        # A language model asked for JSON will occasionally quote a number, so
        # confidence can arrive as "80" rather than 80. Smaller models do this
        # more often. Coerce before rounding: one quoted value out of a
        # thousand would otherwise abort a whole evaluation run.
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError):
            raise ValueError(
                f"Confidence {self.confidence!r} is not a number"
            ) from None
        self.confidence = int(max(0, min(100, round(confidence))))


class Agent(ABC):
    """Interface every reasoning backend implements."""

    name: str = "agent"

    # Whether `recommend` actually reads `similar_past`. The evaluation uses
    # this to decide if a "no retrieval" ablation is meaningful: for a backend
    # that ignores memory, disabling retrieval reproduces the same actions and
    # the arm would be a duplicate column rather than a result.
    uses_memory: bool = False

    @abstractmethod
    def recommend(
        self, snapshot: MarketSnapshot, similar_past: list[dict]
    ) -> Recommendation:
        """Produce a recommendation for `snapshot`, optionally informed by memory.

        `similar_past` is a list of retrieved past decisions (see
        memory.retrieve_similar); backends may use it or ignore it.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule-based agent (deterministic, no network)
# ---------------------------------------------------------------------------
class RuleBasedAgent(Agent):
    """A transparent indicator-driven agent.

    Logic (intentionally simple and explainable for the report):
      * Oversold RSI + bullish MA crossover  -> BUY
      * Overbought RSI + bearish MA crossover -> SELL
      * Otherwise lean on the MA crossover with lower confidence, or HOLD when
        signals are mixed / weak.
    Confidence scales with how strong/aligned the signals are.
    """

    name = "rule-based"
    uses_memory = False  # decides purely from the indicators; memory enters via calibration

    def recommend(self, snapshot: MarketSnapshot, similar_past: list[dict]) -> Recommendation:
        rsi = snapshot.rsi
        gap = snapshot.ma_gap_pct
        bullish = gap >= 0

        # Distance of RSI from the neutral 50 line, scaled to [0,1].
        rsi_strength = min(abs(rsi - 50) / 50.0, 1.0)
        ma_strength = min(abs(gap) / 2.0, 1.0)  # 2% gap counts as "strong"

        if rsi <= config.RSI_OVERSOLD and bullish:
            action = "BUY"
            direction = "up — oversold bounce with bullish MA crossover"
            confidence = 55 + 40 * (rsi_strength + ma_strength) / 2
        elif rsi >= config.RSI_OVERBOUGHT and not bullish:
            action = "SELL"
            direction = "down — overbought with bearish MA crossover"
            confidence = 55 + 40 * (rsi_strength + ma_strength) / 2
        elif bullish and rsi < config.RSI_OVERBOUGHT:
            action = "BUY"
            direction = "mildly up — bullish MA crossover, RSI not yet overbought"
            confidence = 40 + 30 * ma_strength
        elif not bullish and rsi > config.RSI_OVERSOLD:
            action = "SELL"
            direction = "mildly down — bearish MA crossover, RSI not yet oversold"
            confidence = 40 + 30 * ma_strength
        else:
            action = "HOLD"
            direction = "sideways — signals are mixed or weak"
            confidence = 45

        reasoning = (
            f"RSI={rsi:.1f} ({snapshot.rsi_signal}), "
            f"MA gap={gap:+.2f}% ({snapshot.ma_signal}). {direction}."
        )
        return Recommendation(
            action=action,
            confidence=confidence,
            predicted_direction=direction,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# LLM agent (Anthropic Claude)
# ---------------------------------------------------------------------------
_JSON_ONLY_SYSTEM = (
    "You are a JSON API. You respond with exactly one JSON object and nothing "
    "else: no explanation, no reasoning, no markdown, no code fences, no text "
    "before or after the object. Begin your reply with '{' and end it with '}'."
)


PROMPT_TEMPLATE = """You are CryptoMind, a disciplined single-agent crypto trading advisor.
You are given the current market state for {pair} and a few of your own SIMILAR
PAST DECISIONS together with how they actually turned out. Use the past outcomes
to stay honest about your confidence.

CURRENT MARKET STATE:
{snapshot}

YOUR SIMILAR PAST DECISIONS (most similar first; may be empty on a fresh start):
{history}

Decide a single action and respond with ONLY a JSON object, no prose, no code
fences, exactly these keys:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <integer 0-100>,
  "predicted_direction": "<short phrase on expected near-term move>",
  "reasoning": "<ONE short sentence citing the indicators and, if relevant, the past outcomes>"
}}"""


# Fields worth putting in front of the model. The full snapshot also carries the
# raw moving averages and a millisecond timestamp, which are either redundant
# with ma_gap_pct or irrelevant to the decision. Trimming them roughly halves
# the prompt, and prompt size is the binding constraint on how many historical
# decisions a free tier will let us evaluate in a day.
PROMPT_SNAPSHOT_FIELDS = ("price", "rsi", "rsi_signal", "ma_gap_pct", "ma_signal")


def _compact_snapshot(snapshot: MarketSnapshot) -> str:
    """Render only the decision-relevant snapshot fields, compactly."""
    full = snapshot.to_dict()
    return json.dumps({k: full[k] for k in PROMPT_SNAPSHOT_FIELDS if k in full})


class LLMAgent(Agent):
    """Claude-backed agent. Requires ANTHROPIC_API_KEY in the environment."""

    name = "claude-llm"
    uses_memory = True  # retrieved outcomes go into the prompt

    def __init__(self, model: str = config.CLAUDE_MODEL, api_key: str | None = None):
        from anthropic import Anthropic  # imported lazily so offline use needs no SDK

        key = api_key or config.get_anthropic_key()
        if not key:
            raise RuntimeError(
                f"{config.ANTHROPIC_API_KEY_ENV} is not set. Export it or use the "
                f"rule-based engine instead."
            )
        self._client = Anthropic(api_key=key)
        self._model = model

    def recommend(self, snapshot: MarketSnapshot, similar_past: list[dict]) -> Recommendation:
        prompt = PROMPT_TEMPLATE.format(
            pair=snapshot.pair,
            snapshot=_compact_snapshot(snapshot),
            history=_format_history_for_prompt(similar_past),
        )
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text
        except Exception as exc:
            raise RuntimeError(f"Claude request failed: {exc}") from exc

        data = _parse_json_response(raw_text)
        return Recommendation(
            action=data["action"],
            confidence=data["confidence"],
            predicted_direction=data.get("predicted_direction", ""),
            reasoning=data.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible agent (Groq, OpenRouter, local Ollama…)
# ---------------------------------------------------------------------------
class OpenAICompatibleAgent(Agent):
    """Agent backed by any OpenAI-compatible chat endpoint.

    Groq and OpenRouter both expose OpenAI-compatible APIs, so we can reach their
    free tiers through the standard `openai` SDK just by pointing `base_url` at
    them. The same class works unchanged with a local Ollama server — this is the
    project's "swap the LLM" seam, and the free, cloud-deployable alternative to
    the Claude backend.
    """

    name = "openai-compatible"
    uses_memory = True  # retrieved outcomes go into the prompt

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        label: str = "llm",
        key_hint: str = "OPENROUTER_API_KEY (free at https://openrouter.ai/settings/keys)",
        use_cache: bool = True,
    ):
        if not api_key:
            raise RuntimeError(
                f"No API key for the '{label}' engine. Set {key_hint}, "
                f"or use the rule-based engine."
            )

        from openai import OpenAI  # imported lazily so offline use needs no SDK

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._use_cache = use_cache
        self.name = label

    @classmethod
    def for_groq(cls, use_cache: bool = True) -> "OpenAICompatibleAgent":
        """Build an agent for Groq's free OpenAI-compatible endpoint."""
        return cls(
            base_url=config.GROQ_OPENAI_BASE_URL,
            api_key=config.get_groq_key(),
            model=config.GROQ_MODEL,
            label="groq",
            key_hint="GROQ_API_KEY (free at https://console.groq.com)",
            use_cache=use_cache,
        )

    @classmethod
    def for_openrouter(cls, use_cache: bool = True) -> "OpenAICompatibleAgent":
        """Build an agent for OpenRouter's free OpenAI-compatible endpoint."""
        return cls(
            base_url=config.OPENROUTER_OPENAI_BASE_URL,
            api_key=config.get_openrouter_key(),
            model=config.OPENROUTER_MODEL,
            label="openrouter",
            key_hint="OPENROUTER_API_KEY (free at https://openrouter.ai/settings/keys)",
            use_cache=use_cache,
        )

    def recommend(self, snapshot: MarketSnapshot, similar_past: list[dict]) -> Recommendation:
        prompt = PROMPT_TEMPLATE.format(
            pair=snapshot.pair,
            snapshot=_compact_snapshot(snapshot),
            history=_format_history_for_prompt(similar_past),
        )

        # A JSON-only system instruction plus response_format push the model to
        # emit only the JSON object; some free models otherwise "think out loud"
        # before (or instead of) answering and the reply carries no recoverable
        # decision. response_format is best-effort — some providers reject it, so
        # we retry without it. (An assistant-turn "{" prefill was tried and
        # removed: this provider echoes the prefill back into the content and
        # corrupts the object, which made parsing worse, not better.)
        messages = [
            {"role": "system", "content": _JSON_ONLY_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        def call() -> str:
            # A larger token budget than the object needs, so a little prepended
            # reasoning does not truncate the JSON mid-object (350 was cutting
            # replies off before any closing brace).
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=512,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Provider/model does not accept response_format — fall back to
                # the plain call, still with the JSON-only system instruction.
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=512,
                    messages=messages,
                )
            return _content_from_response(resp)

        try:
            raw_text = cached_call(
                prompt, self._model,
                lambda: with_retry(call, label=self.name),
                use_cache=self._use_cache,
            )
        except Exception as exc:
            raise RuntimeError(f"{self.name} request failed: {exc}") from exc

        data = _parse_json_response(raw_text)
        return Recommendation(
            action=data["action"],
            confidence=data["confidence"],
            predicted_direction=data.get("predicted_direction", ""),
            reasoning=data.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# Caching and retry for LLM calls
# ---------------------------------------------------------------------------
def _cache_path(prompt: str, model: str, cache_dir: Path) -> Path:
    """Deterministic cache location for one (prompt, model) pair."""
    digest = hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()[:32]
    return Path(cache_dir) / f"{digest}.json"


def cached_call(
    prompt: str,
    model: str,
    call: "callable",
    cache_dir: Path = config.LLM_CACHE_DIR,
    use_cache: bool = True,
) -> str:
    """Return the model's raw reply for `prompt`, from disk if seen before.

    An evaluation replays hundreds of historical decisions. Without a cache each
    re-run spends the whole quota again and, worse, produces different numbers,
    because a language model is not deterministic even at fixed settings. Caching
    the raw reply makes an LLM evaluation reproducible in the same way the cached
    candles make the market data reproducible.
    """
    path = _cache_path(prompt, model, cache_dir)
    if use_cache and path.exists():
        # A cache entry can be truncated or empty if a previous run was killed
        # mid-write, or if the disk filled up. Treat an unreadable entry as a
        # miss and re-fetch rather than letting one bad file abort the run.
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (json.JSONDecodeError, KeyError, OSError):
            path.unlink(missing_ok=True)

    text = call()

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename, so an interrupted write leaves
        # the old entry (or no entry) instead of a half-written one. Rename is
        # atomic within a directory on both POSIX and Windows.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"model": model, "prompt": prompt, "response": text}),
            encoding="utf-8",
        )
        tmp.replace(path)
    return text


def with_retry(call: "callable", label: str = "llm"):
    """Call `call`, backing off on rate limits and transient server errors.

    Free tiers throttle hard, and an evaluation makes hundreds of sequential
    calls, so a single 429 twenty minutes into a run must not destroy it.
    """
    delay = config.LLM_BACKOFF_SECONDS
    last: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            return call()
        except Exception as exc:  # SDKs raise their own exception hierarchies
            last = exc
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in ("rate limit", "429", "timeout", "timed out",
                              "temporarily", "503", "502", "overloaded")
            )
            if not retryable or attempt == config.LLM_MAX_RETRIES - 1:
                raise
            print(f"  ~ {label}: {type(exc).__name__}, retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
    raise last  # pragma: no cover - loop always returns or raises above


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _content_from_response(resp) -> str:
    """Pull the message text out of a chat-completions response, defensively.

    An OpenAI-compatible endpoint does not always return the happy-path shape.
    OpenRouter in particular can return a 200 whose body carries an `error`
    object and no `choices`, or a `choices` entry whose `message.content` is
    null (an empty or moderation-filtered completion). Indexing `choices[0]`
    blindly then fails with a cryptic "'NoneType' object is not subscriptable".

    Turn each of those into a clear, catchable error (or an empty string the
    JSON parser will reject with a real message) instead.
    """
    # Some SDKs expose a provider error on the response object itself.
    err = getattr(resp, "error", None)
    if err:
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"provider returned an error: {message}")

    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(
            "response contained no choices "
            f"(provider returned: {str(resp)[:200]})"
        )

    content = choices[0].message.content
    # A null/absent completion is not an exception, but it carries no decision;
    # return "" so _parse_json_response raises its own descriptive error.
    return content or ""


def _format_history_for_prompt(similar_past: list[dict]) -> str:
    """Render retrieved memory compactly for the LLM prompt."""
    if not similar_past:
        return "(none yet)"
    lines = []
    for item in similar_past:
        outcome = item.get("outcome")
        if outcome is None:
            verdict = "outcome not yet known"
        else:
            verdict = (
                f"price moved {outcome['actual_pct_change']:+.2f}% -> "
                f"{'CORRECT' if outcome['was_correct'] else 'WRONG'}"
            )
        lines.append(
            f"- action={item['action']} conf={item['confidence']} "
            f"(RSI={item['snapshot'].get('rsi')}, MA gap={item['snapshot'].get('ma_gap_pct')}%): {verdict}"
        )
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """Parse a model's JSON reply as tolerantly as is safe.

    Three escalating strategies, because over hundreds of evaluation calls the
    models do all of these:

      1. strict JSON, after stripping any ```json fence and surrounding prose
      2. the same, after repairing an object truncated by the token limit
         (a reply cut off mid-sentence has no closing brace at all, so simple
         brace matching finds nothing and fails on an otherwise usable answer)
      3. field-wise extraction, which recovers a decision from a reply that is
         not valid JSON in any form

    What it will NOT do is invent a missing action or confidence. Those two
    fields are the decision itself, and guessing at them would quietly
    fabricate data rather than record a failure.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()

    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        candidates = []
        if end > start:
            candidates.append(text[start : end + 1])
        # Truncated mid-object: close the dangling string and the brace.
        candidates += [text[start:] + suffix for suffix in ('"}', "}", '"]}')]

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if _has_decision(data):
                return data
            # Parsed cleanly but carries no decision, so keep looking rather
            # than returning an object the caller cannot use.

    return _extract_fields(text)


def _has_decision(data: object) -> bool:
    """True when a parsed reply actually contains an action and a confidence."""
    return isinstance(data, dict) and "action" in data and "confidence" in data


def _extract_fields(text: str) -> dict:
    """Last-resort field-wise recovery from a reply that is not valid JSON."""
    action = re.search(r'"action"\s*:\s*"([A-Za-z]+)"', text)
    confidence = re.search(r'"confidence"\s*:\s*(\d+)', text)
    if not action or not confidence:
        raise ValueError(
            f"Could not recover a decision from the model reply: {text[:200]!r}"
        )

    direction = re.search(r'"predicted_direction"\s*:\s*"([^"]*)"', text)
    reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
    return {
        "action": action.group(1),
        "confidence": int(confidence.group(1)),
        "predicted_direction": direction.group(1) if direction else "",
        "reasoning": reasoning.group(1) if reasoning else "",
    }


def get_agent(engine: str) -> Agent:
    """Factory mapping an engine name to a concrete Agent.

    'rule'           -> RuleBasedAgent (no key, works everywhere)
    'groq'           -> Llama via Groq's OpenAI-compatible endpoint (free key)
    'openrouter'     -> free model via OpenRouter's OpenAI-compatible endpoint (free key)
    'claude'         -> Claude (Anthropic, paid key)
    'llm'            -> auto: whichever free key is set, else Claude
    """
    engine = engine.lower()
    if engine in ("rule", "rule-based", "rules"):
        return RuleBasedAgent()
    if engine in ("groq", "llama"):
        return OpenAICompatibleAgent.for_groq()
    if engine in ("openrouter", "or"):
        return OpenAICompatibleAgent.for_openrouter()
    if engine in ("claude", "anthropic"):
        return LLMAgent()
    if engine == "llm":  # convenience alias: prefer whichever free key is present
        if config.get_groq_key():
            return OpenAICompatibleAgent.for_groq()
        if config.get_openrouter_key():
            return OpenAICompatibleAgent.for_openrouter()
        return LLMAgent()
    raise ValueError(
        f"Unknown engine '{engine}', expected 'rule', 'groq', "
        f"'openrouter' or 'claude'"
    )
