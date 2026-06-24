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

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        self.confidence = int(max(0, min(100, round(self.confidence))))


class Agent(ABC):
    """Interface every reasoning backend implements."""

    name: str = "agent"

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
  "reasoning": "<2-3 sentences, plain language, referencing the indicators and—if relevant—the past outcomes>"
}}"""


class LLMAgent(Agent):
    """Claude-backed agent. Requires ANTHROPIC_API_KEY in the environment."""

    name = "claude-llm"

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
            snapshot=json.dumps(snapshot.to_dict(), indent=2),
            history=_format_history_for_prompt(similar_past),
        )
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=400,
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
# OpenAI-compatible agent (Gemini free tier, Groq, OpenRouter, local Ollama…)
# ---------------------------------------------------------------------------
class OpenAICompatibleAgent(Agent):
    """Agent backed by any OpenAI-compatible chat endpoint.

    Google Gemini exposes an OpenAI-compatible API, so we can reach its free
    tier through the standard `openai` SDK just by pointing `base_url` at it.
    The same class works unchanged with Groq, OpenRouter, or a local Ollama
    server — this is the project's "swap the LLM" seam, and the free,
    cloud-deployable alternative to the Claude backend.
    """

    name = "openai-compatible"

    def __init__(self, *, base_url: str, api_key: str | None, model: str, label: str = "llm"):
        from openai import OpenAI  # imported lazily so offline use needs no SDK

        if not api_key:
            raise RuntimeError(
                f"No API key for the '{label}' engine. Set GEMINI_API_KEY "
                f"(free at https://aistudio.google.com) or use the rule-based engine."
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.name = label

    @classmethod
    def for_gemini(cls) -> "OpenAICompatibleAgent":
        """Build an agent for Google Gemini's free OpenAI-compatible endpoint."""
        return cls(
            base_url=config.GEMINI_OPENAI_BASE_URL,
            api_key=config.get_gemini_key(),
            model=config.GEMINI_MODEL,
            label="gemini",
        )

    def recommend(self, snapshot: MarketSnapshot, similar_past: list[dict]) -> Recommendation:
        prompt = PROMPT_TEMPLATE.format(
            pair=snapshot.pair,
            snapshot=json.dumps(snapshot.to_dict(), indent=2),
            history=_format_history_for_prompt(similar_past),
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = resp.choices[0].message.content
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
# Helpers
# ---------------------------------------------------------------------------
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
    """Parse Claude's JSON reply, tolerating accidental code fences/extra text."""
    text = text.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if the model added one.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # Fall back to extracting the outermost {...} if there's stray prose.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def get_agent(engine: str) -> Agent:
    """Factory mapping an engine name to a concrete Agent.

    'rule'           -> RuleBasedAgent (no key, works everywhere)
    'gemini'         -> Gemini via OpenAI-compatible endpoint (free key)
    'claude'         -> Claude (Anthropic, paid key)
    'llm'            -> auto: Gemini if its key is set, else Claude
    """
    engine = engine.lower()
    if engine in ("rule", "rule-based", "rules"):
        return RuleBasedAgent()
    if engine in ("gemini", "google"):
        return OpenAICompatibleAgent.for_gemini()
    if engine in ("claude", "anthropic"):
        return LLMAgent()
    if engine == "llm":  # convenience alias: prefer the free Gemini key if present
        return OpenAICompatibleAgent.for_gemini() if config.get_gemini_key() else LLMAgent()
    raise ValueError(f"Unknown engine '{engine}', expected 'rule', 'gemini', or 'claude'")
