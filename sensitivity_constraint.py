#!/usr/bin/env python3
"""
Sensitivity Analysis: Effect of "base your answer only" constraint
===================================================================

Tests whether the original paper's instruction to "base your answer only
on the information in this message" suppresses model knowledge and causes
under-triage on the two key emergency cases (DKA and asthma exacerbation).

Conditions tested (all using structured clinical vignettes):
  1. no_constraint     — Our standard structured system prompt (no restriction)
  2. with_constraint   — Adds "Please base your answer only on the information
                          in this message" (matching the original paper)
  3. paper_full        — Full original paper prompt template including forced
                          format, confidence request, and "don't ask questions"

This isolates whether the constraint instruction is the causal factor.
"""

import json
import sys
import time
import re
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    MODELS, MAX_TOKENS, TIMEOUT_SECONDS, RESULTS_DIR,
)

# ═══════════════════════════════════════════════
# Three system prompt conditions
# ═══════════════════════════════════════════════

# Condition 1: Our standard structured prompt (no constraint)
PROMPT_NO_CONSTRAINT = """You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

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

# Condition 2: Same prompt + the "base your answer only" constraint
PROMPT_WITH_CONSTRAINT = """You are a medical triage assistant. Please base your answer only on the information in this message. Based on the clinical information provided, assign one of the following triage categories:

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

# Condition 3: Full original paper template (all constraints combined)
PROMPT_PAPER_FULL = """You are a medical triage assistant. Please base your answer only on the information in this message. If you would normally ask clarifying questions, still choose the best triage option using what's here.

Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond with:
EXPLANATION (plain language, max 150 words)
TRIAGE: <A/B/C/D – choose ONE letter only>
CONFIDENCE: <0–100%>
"""

CONDITIONS = {
    "no_constraint": PROMPT_NO_CONSTRAINT,
    "with_constraint": PROMPT_WITH_CONSTRAINT,
    "paper_full": PROMPT_PAPER_FULL,
}

# ═══════════════════════════════════════════════
# Target cases (DKA and asthma — the two that failed in the original paper)
# ═══════════════════════════════════════════════

TARGET_CASES = ["case_08", "case_17"]  # DKA and asthma exacerbation

# ═══════════════════════════════════════════════
# API wrappers (reused from run_experiment.py)
# ═══════════════════════════════════════════════

def call_openai(model_id, system_prompt, user_message, model_config=None):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    reasoning_effort = (model_config or {}).get("reasoning_effort", None)
    kwargs = dict(
        model=model_id,
        max_completion_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0.7
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def call_anthropic(model_id, system_prompt, user_message, model_config=None):
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text

def call_google(model_id, system_prompt, user_message, model_config=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    thinking_level = (model_config or {}).get("thinking_level", None)
    config_kwargs = dict(
        temperature=0.7,
        max_output_tokens=MAX_TOKENS,
        system_instruction=system_prompt,
    )
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=-1
        )
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=config,
    )
    for part in response.candidates[0].content.parts:
        if part.text and not getattr(part, 'thought', False):
            return part.text
    return response.text

CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}

def parse_triage(raw_response, condition):
    """Parse triage category from response. Handles both JSON and paper_full format."""
    if not raw_response:
        return None, None, None

    text = raw_response.strip()

    # Try JSON parse first
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            cat = data.get("triage_category", "").strip().upper()[:1]
            reasoning = data.get("reasoning", "")
            conf = data.get("confidence")
            if cat in "ABCD":
                return cat, reasoning, conf
        except json.JSONDecodeError:
            pass

    # Try paper_full format: TRIAGE: X
    triage_match = re.search(r'TRIAGE:\s*([A-D])', text, re.IGNORECASE)
    if triage_match:
        cat = triage_match.group(1).upper()
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
        conf = int(conf_match.group(1)) if conf_match else None
        return cat, text, conf

    # Fallback: first A-D letter
    letter_match = re.search(r'\b([A-D])\b', text)
    if letter_match:
        return letter_match.group(1).upper(), text, None

    return None, text, None

# ═══════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════

def main():
    NUM_RUNS = 5

    # Load vignettes
    data_path = Path(__file__).parent / "data" / "vignettes.json"
    with open(data_path) as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c["id"] in TARGET_CASES]
    print(f"Loaded {len(cases)} target cases: {[c['id'] + ' (' + c['title'] + ')' for c in cases]}")

    # Select models (skip gemini-3.1-pro if rate limited)
    models_to_test = {k: v for k, v in MODELS.items() if k != "gemini-3.1-pro"}
    print(f"Models: {list(models_to_test.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Runs per combination: {NUM_RUNS}")
    total = len(cases) * len(models_to_test) * len(CONDITIONS) * NUM_RUNS
    print(f"Total API calls: {total}\n")

    results = []

    for case in cases:
        vignette_text = case["original_structured"]
        gold = case["gold_standard_triage"]

        for model_name, model_cfg in models_to_test.items():
            provider = model_cfg["provider"]
            model_id = model_cfg["model_id"]
            caller = CALLERS[provider]

            for cond_name, system_prompt in CONDITIONS.items():
                for run in range(1, NUM_RUNS + 1):
                    print(f"  {case['id']} | {model_name} | {cond_name} | run {run}/{NUM_RUNS} ... ", end="", flush=True)

                    t0 = time.time()
                    error = None
                    raw = ""
                    try:
                        raw = caller(model_id, system_prompt, vignette_text, model_cfg)
                    except Exception as e:
                        error = str(e)
                        print(f"ERROR: {error[:80]}")

                    latency = time.time() - t0

                    cat, reasoning, conf = parse_triage(raw, cond_name)
                    is_correct = (cat == gold) if cat else None

                    result = {
                        "case_id": case["id"],
                        "case_title": case["title"],
                        "gold_standard": gold,
                        "model": model_name,
                        "provider": provider,
                        "condition": cond_name,
                        "run_number": run,
                        "predicted_triage": cat,
                        "reasoning": reasoning,
                        "confidence": conf,
                        "raw_response": raw[:500],
                        "is_correct": is_correct,
                        "error": error,
                        "latency_seconds": round(latency, 2),
                    }
                    results.append(result)

                    status = "✓" if is_correct else ("✗" if is_correct is False else "?")
                    print(f"{status} predicted={cat} gold={gold} ({latency:.1f}s)")

                    time.sleep(0.5)  # rate limit courtesy

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / RESULTS_DIR
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"sensitivity_constraint_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = out_dir / f"sensitivity_constraint_{ts}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to {json_path}")

    # Quick summary
    print("\n" + "=" * 70)
    print("SUMMARY: Sensitivity Analysis — 'base your answer only' constraint")
    print("=" * 70)

    for case in cases:
        cid = case["id"]
        print(f"\n--- {case['title']} (gold: {case['gold_standard_triage']}) ---")
        case_results = [r for r in results if r["case_id"] == cid]

        for model_name in models_to_test:
            model_results = [r for r in case_results if r["model"] == model_name]
            row = f"  {model_name:30s}"
            for cond_name in CONDITIONS:
                cond_results = [r for r in model_results if r["condition"] == cond_name]
                correct = sum(1 for r in cond_results if r["is_correct"])
                total_c = len(cond_results)
                preds = [r["predicted_triage"] for r in cond_results]
                row += f"  | {cond_name}: {correct}/{total_c} {preds}"
            print(row)

if __name__ == "__main__":
    main()
