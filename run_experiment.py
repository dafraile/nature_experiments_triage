#!/usr/bin/env python3
"""
Triage Replication Experiment Runner
=====================================

Tests LLM triage accuracy across different prompt formats to demonstrate
that the structured-vignette approach used in published evaluations does
not generalize to real-world patient interactions.

Usage:
    python run_experiment.py                    # Run all models, all formats
    python run_experiment.py --models gpt-5.2-thinking-high claude-sonnet-4.6   # Specific models
    python run_experiment.py --formats patient_realistic patient_minimal  # Specific formats
    python run_experiment.py --dry-run          # Print prompts without calling APIs
"""

import json
import os
import sys
import time
import re
import argparse
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ── Add project root to path ──
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    MODELS, NUM_RUNS, TEMPERATURE, MAX_TOKENS, TIMEOUT_SECONDS,
    PROMPT_FORMATS, TRIAGE_CATEGORIES, DATA_DIR, RESULTS_DIR,
)
from llm_utils import (
    extract_google_text,
    google_visible_output_tokens,
    infer_free_text_triage,
    parse_structured_response,
)

# ═══════════════════════════════════════════════
# SYSTEM PROMPT — Identical for all models
# ═══════════════════════════════════════════════

# VERSION A: Structured system prompt (matches the original paper's approach)
SYSTEM_PROMPT_STRUCTURED = """You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}
"""

# VERSION B: Natural system prompt (as a real health chatbot would use)
SYSTEM_PROMPT_NATURAL = """You are a helpful health assistant. A user is describing their symptoms to you. Based on what they tell you, help them understand the urgency of their situation. Categorize their situation as one of:

A: They should call 911 or go to the emergency room right away
B: They should see a doctor or go to urgent care within the next 24 hours
C: They should schedule a doctor's appointment within the next few days
D: They can likely manage this at home with self-care

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief explanation in plain language>",
    "confidence": <number between 0 and 100>
}
"""


# ═══════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════

@dataclass
class TrialResult:
    """One trial = one (model, vignette, format, run) combination."""
    case_id: str
    case_title: str
    gold_standard: str
    model: str
    provider: str
    prompt_format: str
    system_prompt_version: str
    run_number: int
    predicted_triage: Optional[str]
    reasoning: Optional[str]
    confidence: Optional[float]
    raw_response: str
    is_correct: Optional[bool]
    error: Optional[str]
    latency_seconds: float
    timestamp: str


# ═══════════════════════════════════════════════
# API call wrappers
# ═══════════════════════════════════════════════

def call_openai(model_id: str, system_prompt: str, user_message: str,
                model_config: dict = None) -> str:
    """
    Call OpenAI API. Handles reasoning_effort parameter.
    NOTE: When reasoning_effort != "none", temperature and top_p are NOT supported.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    reasoning_effort = (model_config or {}).get("reasoning_effort", None)

    kwargs = dict(
        model=model_id,
        max_completion_tokens=MAX_TOKENS,   # GPT-5.2 requires max_completion_tokens, not max_tokens
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    if reasoning_effort and reasoning_effort != "none":
        # Thinking mode: temperature is NOT supported
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        # No reasoning: temperature is supported
        if (model_config or {}).get("supports_temperature_override", True):
            kwargs["temperature"] = TEMPERATURE

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_anthropic(model_id: str, system_prompt: str, user_message: str,
                   model_config: dict = None) -> str:
    """
    Call Anthropic API. Thinking is OFF for Claude models per experimental design
    (performs well without it, sometimes detrimental).
    """
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )
    return response.content[0].text


def call_google(model_id: str, system_prompt: str, user_message: str,
                model_config: dict = None) -> str:
    """
    Call Google Gemini API with thinking_level configuration.
    Gemini 3 series uses thinking_level (not thinkingBudget).
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)

    thinking_level = (model_config or {}).get("thinking_level", None)

    config_kwargs = dict(
        system_instruction=system_prompt,
        temperature=TEMPERATURE,
        max_output_tokens=google_visible_output_tokens(model_id, MAX_TOKENS),
        http_options=types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000),
    )

    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level
        )

    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return extract_google_text(response)


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


# ═══════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════

def parse_triage_response(raw: str) -> dict:
    """
    Extract triage_category, reasoning, and confidence from LLM response.
    Handles JSON responses and also tries to extract from free text.
    """
    parsed = parse_structured_response(raw or "")
    if not parsed["triage_category"]:
        parsed["triage_category"] = infer_free_text_triage(raw or "")
    return parsed


# ═══════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════

def run_single_trial(
    case: dict,
    model_name: str,
    model_config: dict,
    prompt_format: str,
    run_number: int,
    dry_run: bool = False,
) -> TrialResult:
    """Execute a single trial and return the result."""

    # Select system prompt based on format
    if prompt_format == "original_structured":
        system_prompt = SYSTEM_PROMPT_STRUCTURED
        sys_version = "structured"
    else:
        system_prompt = SYSTEM_PROMPT_NATURAL
        sys_version = "natural"

    user_message = case.get(prompt_format, "")
    if not user_message:
        # Fallback for cases that might not have all formats
        return TrialResult(
            case_id=case["id"], case_title=case["title"],
            gold_standard=case["gold_standard_triage"], model=model_name,
            provider=model_config["provider"], prompt_format=prompt_format,
            system_prompt_version=sys_version, run_number=run_number,
            predicted_triage=None, reasoning=None, confidence=None,
            raw_response="", is_correct=None,
            error=f"Format '{prompt_format}' not found for case {case['id']}",
            latency_seconds=0.0, timestamp=datetime.now().isoformat(),
        )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name} | Case: {case['id']} | Format: {prompt_format} | Run: {run_number}")
        print(f"{'='*60}")
        print(f"  SYSTEM: {system_prompt[:100]}...")
        print(f"  USER: {user_message[:200]}...")
        print(f"  Gold standard: {case['gold_standard_triage']}")
        return TrialResult(
            case_id=case["id"], case_title=case["title"],
            gold_standard=case["gold_standard_triage"], model=model_name,
            provider=model_config["provider"], prompt_format=prompt_format,
            system_prompt_version=sys_version, run_number=run_number,
            predicted_triage=None, reasoning=None, confidence=None,
            raw_response="[DRY RUN]", is_correct=None, error=None,
            latency_seconds=0.0, timestamp=datetime.now().isoformat(),
        )

    # Call the API (pass full model_config for reasoning/thinking settings)
    call_fn = PROVIDER_DISPATCH[model_config["provider"]]
    error = None
    raw_response = ""
    latency = 0.0

    try:
        t0 = time.time()
        raw_response = call_fn(
            model_config["model_id"], system_prompt, user_message,
            model_config=model_config,
        )
        latency = time.time() - t0
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
        latency = time.time() - t0

    # Parse response
    parsed = parse_triage_response(raw_response) if raw_response else {}
    predicted = parsed.get("triage_category")
    is_correct = (predicted == case["gold_standard_triage"]) if predicted else None

    return TrialResult(
        case_id=case["id"],
        case_title=case["title"],
        gold_standard=case["gold_standard_triage"],
        model=model_name,
        provider=model_config["provider"],
        prompt_format=prompt_format,
        system_prompt_version=sys_version,
        run_number=run_number,
        predicted_triage=predicted,
        reasoning=parsed.get("reasoning"),
        confidence=parsed.get("confidence"),
        raw_response=raw_response,
        is_correct=is_correct,
        error=error,
        latency_seconds=round(latency, 2),
        timestamp=datetime.now().isoformat(),
    )


def run_experiment(
    models: list[str],
    formats: list[str],
    num_runs: int,
    case_ids: Optional[list[str]] = None,
    dry_run: bool = False,
) -> list[TrialResult]:
    """Run the full experiment matrix."""

    # Load vignettes
    vignettes_path = Path(__file__).parent / DATA_DIR / "vignettes.json"
    with open(vignettes_path) as f:
        cases = json.load(f)
    if case_ids:
        allowed = set(case_ids)
        cases = [case for case in cases if case["id"] in allowed]

    results = []
    total = len(models) * len(cases) * len(formats) * num_runs
    current = 0

    for model_name in models:
        model_config = MODELS[model_name]
        for case in cases:
            for fmt in formats:
                for run in range(1, num_runs + 1):
                    current += 1
                    print(f"[{current}/{total}] {model_name} | {case['id']} | {fmt} | run {run}")

                    result = run_single_trial(case, model_name, model_config, fmt, run, dry_run)
                    results.append(result)

                    # Brief log
                    if not dry_run and result.error:
                        print(f"  ✗ ERROR: {result.error}")
                    elif not dry_run:
                        correct_str = "✓" if result.is_correct else "✗"
                        print(f"  {correct_str} Predicted: {result.predicted_triage} "
                              f"(gold: {result.gold_standard}) "
                              f"confidence: {result.confidence} "
                              f"({result.latency_seconds}s)")

                    # Rate limiting: brief pause between calls
                    if not dry_run:
                        time.sleep(1.0)

    return results


def save_results(results: list[TrialResult], tag: str = ""):
    """Save results to CSV and JSON."""
    results_dir = Path(__file__).parent / RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results_{tag}_{timestamp}" if tag else f"results_{timestamp}"

    # CSV
    csv_path = results_dir / f"{base_name}.csv"
    fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # JSON (full including raw responses)
    json_path = results_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")
    return csv_path, json_path


# ═══════════════════════════════════════════════
# Quick summary
# ═══════════════════════════════════════════════

def print_summary(results: list[TrialResult]):
    """Print a quick accuracy summary table."""
    from collections import defaultdict

    # Group by (model, format)
    groups = defaultdict(list)
    for r in results:
        if r.is_correct is not None:
            groups[(r.model, r.prompt_format)].append(r.is_correct)

    print(f"\n{'='*70}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Format':<25} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*25} {'-'*10}")

    for (model, fmt), correct_list in sorted(groups.items()):
        acc = sum(correct_list) / len(correct_list) * 100
        print(f"  {model:<25} {fmt:<25} {acc:>9.1f}%")

    # Also show per-format aggregated
    print(f"\n  {'AGGREGATED BY FORMAT':}")
    print(f"  {'-'*25} {'-'*10}")
    format_groups = defaultdict(list)
    for r in results:
        if r.is_correct is not None:
            format_groups[r.prompt_format].append(r.is_correct)
    for fmt, correct_list in sorted(format_groups.items()):
        acc = sum(correct_list) / len(correct_list) * 100
        print(f"  {fmt:<25} {acc:>9.1f}%")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run triage replication experiment")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to test")
    parser.add_argument("--formats", nargs="+", default=PROMPT_FORMATS,
                        choices=PROMPT_FORMATS,
                        help="Prompt formats to test")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"Runs per combination (default: {NUM_RUNS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling APIs")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for output filenames")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Optional list of case ids to run (for targeted verification)")
    args = parser.parse_args()

    vignettes_path = Path(__file__).parent / DATA_DIR / "vignettes.json"
    with open(vignettes_path) as f:
        all_cases = json.load(f)
    if args.cases:
        allowed = set(args.cases)
        selected_cases = [case for case in all_cases if case["id"] in allowed]
        missing = sorted(allowed - {case["id"] for case in selected_cases})
        if missing:
            raise SystemExit(f"Unknown case ids: {', '.join(missing)}")
    else:
        selected_cases = all_cases

    print(f"\n{'='*70}")
    print(f"  TRIAGE REPLICATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Models:  {', '.join(args.models)}")
    print(f"  Formats: {', '.join(args.formats)}")
    if args.cases:
        print(f"  Cases:   {', '.join(args.cases)}")
    print(f"  Runs:    {args.runs}")
    print(f"  Total trials: {len(args.models) * len(selected_cases) * len(args.formats) * args.runs}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*70}\n")

    results = run_experiment(args.models, args.formats, args.runs, args.cases, args.dry_run)

    if not args.dry_run:
        save_results(results, args.tag)
        print_summary(results)
    else:
        print(f"\n[DRY RUN] {len(results)} trials would be executed.")


if __name__ == "__main__":
    main()
