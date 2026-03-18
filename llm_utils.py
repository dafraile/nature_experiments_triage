#!/usr/bin/env python3
"""Shared helpers for extracting and parsing model responses."""

from __future__ import annotations

import json
import re
from typing import Any

TRIAGE_RE = re.compile(r'"triage_category"\s*:\s*"([A-Da-d])"')
TRIAGE_FALLBACK_RE = re.compile(r"\bTRIAGE\b\s*[:=]\s*\**\s*([A-Da-d])\s*\**\b", re.IGNORECASE)
TRIAGE_BARE_RE = re.compile(r"^\s*\**\s*([A-Da-d])\s*\**\s*$")
CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)"?', re.IGNORECASE)
CONFIDENCE_FALLBACK_RE = re.compile(r"\bCONFIDENCE\b\s*[:=]\s*\**\s*(\d+(?:\.\d+)?)\s*%?\s*\**", re.IGNORECASE)

EMERGENCY_PATTERNS = (
    r"\b911\b",
    r"\ber\b",
    r"go to the er",
    r"emergency room",
    r"emergency department",
    r"\bemergency\b",
    r"right away",
    r"\bimmediately\b",
    r"call an ambulance",
)
URGENT_PATTERNS = (
    r"within 24 ?hours?",
    r"24-hour",
    r"urgent care",
    r"prompt medical evaluation",
    r"prompt evaluation",
    r"medical evaluation",
    r"seek medical attention",
    r"seen by a healthcare provider",
    r"should be seen by a healthcare provider",
    r"\btoday\b",
    r"same day",
    r"as soon as possible",
)
SOON_PATTERNS = (
    r"few days",
    r"next few days",
    r"schedule (?:an )?appointment",
    r"within a week",
    r"this week",
)
HOME_PATTERNS = (
    r"self-?care",
    r"at home",
    r"home management",
    r"rest and fluids",
    r"monitor at home",
)

PAPER_HOME_PATTERNS = HOME_PATTERNS
PAPER_WEEKS_PATTERNS = (
    r"within (?:the )?next few weeks",
    r"next few weeks",
    r"within a few weeks",
    r"within 1-2 weeks",
    r"within two weeks",
    r"in a couple of weeks",
    r"schedule (?:a )?(?:routine )?appointment",
    r"routine follow-?up",
    r"non-urgent follow-?up",
    r"follow up with (?:your )?(?:doctor|pcp|primary care)",
)
PAPER_24_48_PATTERNS = (
    r"within 24 ?(?:-|to|–)? ?48 ?hours?",
    r"within 24 ?hours?",
    r"within 48 ?hours?",
    r"within (?:the )?next day or two",
    r"within the next couple of days",
    r"within a few days",
    r"next few days",
    r"today or tomorrow",
    r"same day",
    r"urgent care",
    r"prompt medical evaluation",
    r"prompt evaluation",
    r"seek medical attention",
    r"should be seen by a healthcare provider",
)
PAPER_ER_PATTERNS = EMERGENCY_PATTERNS

TRIAGE_TO_NUM = {"A": 1, "B": 2, "C": 3, "D": 4}


def extract_google_text(response: Any) -> str:
    """Collect all non-thought Gemini text parts into one string."""
    parts: list[str] = []

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text and not getattr(part, "thought", False):
                parts.append(text)

    if parts:
        return "".join(parts).strip()

    fallback = getattr(response, "text", "") or ""
    return fallback.strip()


def make_google_client(api_key: str, vertex_api_key: str = "", use_vertex: bool = False) -> Any:
    """Create a Google GenAI client for the developer API or Vertex Express."""
    from google import genai

    if use_vertex:
        return genai.Client(vertexai=True, api_key=vertex_api_key or api_key)
    return genai.Client(api_key=api_key)


def google_visible_output_tokens(model_id: str, base_tokens: int, use_vertex: bool = False) -> int:
    """Give Gemini enough visible-token headroom to reduce truncated replies.

    In practice, Flash is the Google model most likely to return a short visible
    answer after consuming hidden thinking tokens. Raising the cap does not force
    longer outputs; it only removes an avoidable ceiling. Vertex Express appears
    to count hidden thinking more aggressively against max_output_tokens, so we
    also raise the cap for Pro there to preserve a comparable visible budget.
    """
    if "flash" in model_id:
        return max(base_tokens, 2048)
    if use_vertex:
        return max(base_tokens, 2048)
    return base_tokens


def _extract_json_block(raw: str, key: str) -> dict[str, Any] | None:
    """Best-effort JSON parsing for a response that may include extra text."""
    for match in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        candidate = match.group()
        if key not in candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_triage_category(raw: str) -> str | None:
    """Recover A/B/C/D from full JSON, partial JSON, or TRIAGE lines."""
    if not raw:
        return None

    parsed = _extract_json_block(raw, "triage_category")
    if parsed:
        cat = str(parsed.get("triage_category", "")).strip().upper()[:1]
        if cat in {"A", "B", "C", "D"}:
            return cat

    match = TRIAGE_RE.search(raw)
    if match:
        return match.group(1).upper()

    match = TRIAGE_FALLBACK_RE.search(raw)
    if match:
        return match.group(1).upper()

    match = TRIAGE_BARE_RE.match(raw)
    if match:
        return match.group(1).upper()

    return None


def extract_confidence(raw: str) -> float | None:
    """Recover confidence from JSON or plain-text labels."""
    if not raw:
        return None

    parsed = _extract_json_block(raw, "confidence")
    if parsed and parsed.get("confidence") is not None:
        try:
            return float(parsed["confidence"])
        except (TypeError, ValueError):
            pass

    match = CONFIDENCE_RE.search(raw)
    if match:
        return float(match.group(1))

    match = CONFIDENCE_FALLBACK_RE.search(raw)
    if match:
        return float(match.group(1))

    return None


def extract_reasoning(raw: str) -> str | None:
    """Recover reasoning text when a parseable JSON object is present."""
    if not raw:
        return None

    parsed = _extract_json_block(raw, "reasoning")
    if parsed and parsed.get("reasoning") is not None:
        return str(parsed["reasoning"])

    return None


def parse_structured_response(raw: str) -> dict[str, Any]:
    """Extract triage_category, reasoning, and confidence from a structured response."""
    return {
        "triage_category": extract_triage_category(raw),
        "reasoning": extract_reasoning(raw),
        "confidence": extract_confidence(raw),
    }


def infer_free_text_triage(raw: str) -> str | None:
    """Map a free-text triage recommendation into A/B/C/D."""
    if not raw:
        return None

    parsed = _extract_json_block(raw, "recommendation")
    text = raw
    if parsed:
        rec = str(parsed.get("recommendation", ""))
        reasoning = str(parsed.get("reasoning", ""))
        text = f"{rec} {reasoning}".strip()

    lower = text.lower()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, lower):
            return "A"
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, lower):
            return "B"
    for pattern in SOON_PATTERNS:
        if re.search(pattern, lower):
            return "C"
    for pattern in HOME_PATTERNS:
        if re.search(pattern, lower):
            return "D"

    return None


def infer_free_text_triage_paper(raw: str) -> str | None:
    """Map a free-text triage recommendation into the paper's native A/B/C/D scale.

    Paper scale:
      A = monitor at home
      B = see a doctor within weeks
      C = see a doctor within 24-48 hours
      D = go to the emergency department
    """
    if not raw:
        return None

    parsed = _extract_json_block(raw, "recommendation")
    text = raw
    if parsed:
        rec = str(parsed.get("recommendation", ""))
        reasoning = str(parsed.get("reasoning", ""))
        text = f"{rec} {reasoning}".strip()

    lower = text.lower()

    for pattern in PAPER_ER_PATTERNS:
        if re.search(pattern, lower):
            return "D"
    for pattern in PAPER_24_48_PATTERNS:
        if re.search(pattern, lower):
            return "C"
    for pattern in PAPER_WEEKS_PATTERNS:
        if re.search(pattern, lower):
            return "B"
    for pattern in PAPER_HOME_PATTERNS:
        if re.search(pattern, lower):
            return "A"

    return None


def split_gold_standard(gold: str | None) -> tuple[str, ...]:
    """Return the acceptable triage labels for a gold-standard cell.

    Supports both clear-case labels such as ``A`` and edge-case labels such as
    ``B/C`` from the paper dataset.
    """
    if not gold:
        return ()
    labels = []
    for part in str(gold).upper().split("/"):
        label = part.strip()
        if label in TRIAGE_TO_NUM:
            labels.append(label)
    return tuple(labels)


def triage_matches_gold(predicted: str | None, gold: str | None) -> bool | None:
    """Return whether a predicted label is acceptable for the gold standard."""
    if not predicted or not gold:
        return None
    predicted = str(predicted).strip().upper()[:1]
    if predicted not in TRIAGE_TO_NUM:
        return None
    acceptable = split_gold_standard(gold)
    if not acceptable:
        return None
    return predicted in acceptable


def triage_direction_delta(predicted: str | None, gold: str | None) -> int | None:
    """Signed distance from the acceptable gold range.

    Negative values indicate over-triage (more urgent than necessary), positive
    values indicate under-triage (less urgent than acceptable), and zero means
    the prediction lies inside the acceptable range.
    """
    if not predicted or not gold:
        return None
    pred_num = TRIAGE_TO_NUM.get(str(predicted).strip().upper()[:1])
    acceptable = [TRIAGE_TO_NUM[label] for label in split_gold_standard(gold)]
    if pred_num is None or not acceptable:
        return None
    low = min(acceptable)
    high = max(acceptable)
    if pred_num < low:
        return pred_num - low
    if pred_num > high:
        return pred_num - high
    return 0


def is_retryable_error(message: str | None) -> bool:
    """Identify quota/rate-limit errors worth retrying."""
    if not message:
        return False
    upper = message.upper()
    return (
        "429" in upper
        or "500" in upper
        or "503" in upper
        or "504" in upper
        or "RESOURCE_EXHAUSTED" in upper
        or "INTERNAL" in upper
        or "RATE LIMIT" in upper
        or "UNAVAILABLE" in upper
        or "DEADLINE_EXCEEDED" in upper
    )
