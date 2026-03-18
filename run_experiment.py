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
    triage_matches_gold,
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

    max_completion_tokens = MAX_TOKENS
    if reasoning_effort == "xhigh":
        # GPT-5.4 can spend the full visible budget on hidden reasoning unless
        # we leave more headroom for the actual answer.
        max_completion_tokens = max(MAX_TOKENS, 8192)

    kwargs = dict(
        model=model_id,
        max_completion_tokens=max_completion_tokens,
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
    Call Anthropic API. Supports optional adaptive thinking via the model config.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    max_tokens = MAX_TOKENS
    kwargs = dict(
        model=model_id,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )

    thinking_mode = (model_config or {}).get("thinking")
    if thinking_mode == "adaptive":
        kwargs["max_tokens"] = max(MAX_TOKENS, 2048)
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {
            "effort": (model_config or {}).get("thinking_effort", "high"),
        }
        kwargs["temperature"] = 1

    response = client.messages.create(**kwargs)
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


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
    is_correct = triage_matches_gold(predicted, case["gold_standard_triage"])

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


def load_cases(vignettes_path: Path, case_ids: Optional[list[str]] = None) -> list[dict]:
    with open(vignettes_path) as f:
        cases = json.load(f)

    if not case_ids:
        return cases

    allowed = set(case_ids)
    selected = [case for case in cases if case["id"] in allowed]
    missing = sorted(allowed - {case["id"] for case in selected})
    if missing:
        raise SystemExit(f"Unknown case ids: {', '.join(missing)}")
    return selected


def trial_key(result: TrialResult | dict) -> tuple[str, str, str, int]:
    return (
        str(result["model"] if isinstance(result, dict) else result.model),
        str(result["case_id"] if isinstance(result, dict) else result.case_id),
        str(result["prompt_format"] if isinstance(result, dict) else result.prompt_format),
        int(result["run_number"] if isinstance(result, dict) else result.run_number),
    )


def prepare_output_paths(tag: str = "", output_stem: Optional[str] = None,
                         output_dir: Optional[Path] = None) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = Path(__file__).parent / RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_stem:
        base_name = output_stem
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"results_{tag}_{timestamp}" if tag else f"results_{timestamp}"
    return output_dir / f"{base_name}.csv", output_dir / f"{base_name}.json"


def save_results(results: list[TrialResult], csv_path: Path, json_path: Path) -> None:
    rows = [asdict(r) for r in results]
    fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def load_existing_results(json_path: Path) -> list[TrialResult]:
    if not json_path.exists():
        return []
    with open(json_path) as f:
        rows = json.load(f)
    return [TrialResult(**row) for row in rows]


def run_experiment(
    models: list[str],
    formats: list[str],
    num_runs: int,
    vignettes_path: Path,
    case_ids: Optional[list[str]] = None,
    dry_run: bool = False,
    tag: str = "",
    output_stem: Optional[str] = None,
    output_dir: Optional[Path] = None,
    call_wait: float = 1.0,
) -> tuple[list[TrialResult], Optional[Path], Optional[Path]]:
    """Run the full experiment matrix with per-trial checkpointing."""

    cases = load_cases(vignettes_path, case_ids)
    total = len(models) * len(cases) * len(formats) * num_runs
    current = 0

    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    results_by_key: dict[tuple[str, str, str, int], TrialResult] = {}

    if not dry_run:
        csv_path, json_path = prepare_output_paths(tag=tag, output_stem=output_stem, output_dir=output_dir)
        existing_results = load_existing_results(json_path)
        results_by_key = {trial_key(result): result for result in existing_results}
        print(f"Checkpoint CSV:  {csv_path}")
        print(f"Checkpoint JSON: {json_path}")
        if existing_results:
            print(f"Resuming from {len(existing_results)} saved trial(s)")

    for model_name in models:
        model_config = MODELS[model_name]
        for case in cases:
            for fmt in formats:
                for run in range(1, num_runs + 1):
                    current += 1
                    key = (model_name, case["id"], fmt, run)
                    if key in results_by_key and not results_by_key[key].error:
                        print(
                            f"[{current}/{total}] {model_name} | {case['id']} | {fmt} | run {run} "
                            "(skip: already saved)",
                        )
                        continue

                    print(f"[{current}/{total}] {model_name} | {case['id']} | {fmt} | run {run}")
                    result = run_single_trial(case, model_name, model_config, fmt, run, dry_run)
                    results_by_key[key] = result

                    if not dry_run and result.error:
                        print(f"  ✗ ERROR: {result.error}")
                    elif not dry_run:
                        correct_str = "✓" if result.is_correct else "✗"
                        print(
                            f"  {correct_str} Predicted: {result.predicted_triage} "
                            f"(gold: {result.gold_standard}) confidence: {result.confidence} "
                            f"({result.latency_seconds}s)"
                        )

                    if not dry_run and csv_path and json_path:
                        ordered_results = [results_by_key[item] for item in sorted(results_by_key)]
                        save_results(ordered_results, csv_path, json_path)

                    if not dry_run and call_wait > 0:
                        time.sleep(call_wait)

    ordered_results = [results_by_key[item] for item in sorted(results_by_key)]
    return ordered_results, csv_path, json_path


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
    parser.add_argument("--vignettes-path", type=str, default=None,
                        help="Optional path to a vignettes JSON file")
    parser.add_argument("--output-stem", type=str, default=None,
                        help="Optional fixed output stem (without extension) for resumable runs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional directory for CSV/JSON outputs")
    parser.add_argument("--call-wait", type=float, default=1.0,
                        help="Seconds to wait between API calls (default: 1)")
    args = parser.parse_args()

    vignettes_path = Path(args.vignettes_path) if args.vignettes_path else Path(__file__).parent / DATA_DIR / "vignettes.json"
    selected_cases = load_cases(vignettes_path, args.cases)

    print(f"\n{'='*70}")
    print(f"  TRIAGE REPLICATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Models:  {', '.join(args.models)}")
    print(f"  Formats: {', '.join(args.formats)}")
    print(f"  Vignettes: {vignettes_path}")
    if args.cases:
        print(f"  Cases:   {', '.join(args.cases)}")
    print(f"  Runs:    {args.runs}")
    print(f"  Total trials: {len(args.models) * len(selected_cases) * len(args.formats) * args.runs}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*70}\n")

    results, csv_path, json_path = run_experiment(
        args.models,
        args.formats,
        args.runs,
        vignettes_path=vignettes_path,
        case_ids=args.cases,
        dry_run=args.dry_run,
        tag=args.tag,
        output_stem=args.output_stem,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        call_wait=args.call_wait,
    )

    if not args.dry_run:
        if csv_path and json_path:
            print(f"\nResults saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")
        print_summary(results)
    else:
        print(f"\n[DRY RUN] {len(results)} trials would be executed.")


if __name__ == "__main__":
    main()
