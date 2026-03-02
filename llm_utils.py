#!/usr/bin/env python3
"""Shared helpers for extracting and parsing model responses."""

from __future__ import annotations

import json
import re
from typing import Any

TRIAGE_RE = re.compile(r'"triage_category"\s*:\s*"([A-Da-d])"')
TRIAGE_FALLBACK_RE = re.compile(r"\bTRIAGE\b\s*[:=]\s*([A-Da-d])\b", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)"?', re.IGNORECASE)
CONFIDENCE_FALLBACK_RE = re.compile(r"\bCONFIDENCE\b\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

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


def google_visible_output_tokens(model_id: str, base_tokens: int) -> int:
    """Give Gemini Flash extra visible-token headroom to reduce truncated replies.

    In practice, Flash is the Google model most likely to return a short visible
    answer after consuming hidden thinking tokens. Raising the cap does not force
    longer outputs; it only removes an avoidable ceiling.
    """
    if "flash" in model_id:
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


def is_retryable_error(message: str | None) -> bool:
    """Identify quota/rate-limit errors worth retrying."""
    if not message:
        return False
    upper = message.upper()
    return (
        "429" in upper
        or "503" in upper
        or "504" in upper
        or "RESOURCE_EXHAUSTED" in upper
        or "RATE LIMIT" in upper
        or "UNAVAILABLE" in upper
        or "DEADLINE_EXCEEDED" in upper
    )
