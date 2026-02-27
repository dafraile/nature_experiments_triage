#!/usr/bin/env python3
"""
One-Factor-at-a-Time Ablation: Which prompt constraint drives under-triage?
============================================================================

The original paper's prompt template has MULTIPLE evaluation-style constraints.
This script isolates each one by toggling them independently on the asthma
exacerbation case (case_17), which showed sensitivity in earlier experiments.

Factors tested (each toggled on/off against a baseline):
  1. "base your answer only"     — knowledge suppression
  2. "don't ask clarifying Qs"   — interaction suppression
  3. forced A/B/C/D choice       — vs free-text triage recommendation
  4. 150-word cap                — response length constraint

We also run DKA (case_08) as a robustness check — it should remain 100%.

Design: 2 cases × 4 models × 6 conditions × 5 runs = 240 API calls
"""

import json
import sys
import time
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    MODELS, MAX_TOKENS, RESULTS_DIR,
)

# ═══════════════════════════════════════════════
# Ablation conditions — one factor at a time
# ═══════════════════════════════════════════════

# Baseline: clean structured prompt, no constraints
BASELINE = """You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}"""

# Factor 1: Add "base your answer only" (knowledge suppression)
WITH_BASE_ONLY = """You are a medical triage assistant. Please base your answer only on the information in this message. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}"""

# Factor 2: Add "don't ask clarifying questions" (interaction suppression)
WITH_NO_QUESTIONS = """You are a medical triage assistant. If you would normally ask clarifying questions, still choose the best triage option using what is here. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following JSON format:
{
    "triage_category": "<A, B, C, or D>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}"""

# Factor 3: Free-text triage (remove forced A/B/C/D)
FREE_TEXT = """You are a medical triage assistant. Based on the clinical information provided, recommend the most appropriate level of urgency for this patient. Explain your reasoning and provide a clear recommendation about how urgently they should seek care.

Respond in the following JSON format:
{
    "recommendation": "<your triage recommendation>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}"""

# Factor 4: Add 150-word cap
WITH_WORD_CAP = """You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond in the following format (max 150 words total):
EXPLANATION: <plain language explanation, max 150 words>
TRIAGE: <A/B/C/D - choose ONE letter only>
CONFIDENCE: <0-100%>"""

# Factor 5: ALL constraints combined (full paper template)
ALL_CONSTRAINTS = """You are a medical triage assistant. Please base your answer only on the information in this message. If you would normally ask clarifying questions, still choose the best triage option using what is here.

Based on the clinical information provided, assign one of the following triage categories:

A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Respond with:
EXPLANATION (plain language, max 150 words)
TRIAGE: <A/B/C/D – choose ONE letter only>
CONFIDENCE: <0–100%>"""

CONDITIONS = {
    "baseline":          BASELINE,
    "+base_only":        WITH_BASE_ONLY,
    "+no_questions":     WITH_NO_QUESTIONS,
    "free_text":         FREE_TEXT,
    "+word_cap":         WITH_WORD_CAP,
    "all_constraints":   ALL_CONSTRAINTS,
}

TARGET_CASES = ["case_08", "case_17"]

# ═══════════════════════════════════════════════
# API wrappers
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
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model_id, contents=user_message, config=config,
    )
    for part in response.candidates[0].content.parts:
        if part.text and not getattr(part, 'thought', False):
            return part.text
    return response.text

CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}


def parse_triage(raw_response, condition):
    """Parse triage from response. Returns (category, reasoning, confidence, raw_for_review)."""
    if not raw_response:
        return None, None, None, raw_response

    text = raw_response.strip()

    # For free_text condition, we need special handling
    if condition == "free_text":
        # Look for emergency keywords
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                rec = data.get("recommendation", "").lower()
                reasoning = data.get("reasoning", "")
                conf = data.get("confidence")
                # Map free-text to categories
                if any(w in rec for w in ["emergency", "911", "er ", "immediate", "right away", "call 911"]):
                    return "A", reasoning, conf, text
                elif any(w in rec for w in ["24 hour", "urgent care", "today", "soon as possible"]):
                    return "B", reasoning, conf, text
                elif any(w in rec for w in ["few days", "appointment", "schedule", "week"]):
                    return "C", reasoning, conf, text
                elif any(w in rec for w in ["home", "self-care", "otc", "rest"]):
                    return "D", reasoning, conf, text
                # Also check reasoning
                full = (rec + " " + reasoning).lower()
                if any(w in full for w in ["emergency", "911", "er ", "immediate"]):
                    return "A", reasoning, conf, text
                return None, reasoning, conf, text
            except json.JSONDecodeError:
                pass
        # Fallback for free text
        lower = text.lower()
        if any(w in lower for w in ["emergency", "911", "er ", "immediate", "right away"]):
            return "A", text, None, text
        elif any(w in lower for w in ["24 hour", "urgent care"]):
            return "B", text, None, text
        return None, text, None, text

    # Standard parsing for A/B/C/D formats
    # Try JSON
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            cat = data.get("triage_category", "").strip().upper()[:1]
            reasoning = data.get("reasoning", "")
            conf = data.get("confidence")
            if cat in "ABCD":
                return cat, reasoning, conf, text
        except json.JSONDecodeError:
            pass

    # Try TRIAGE: X format
    triage_match = re.search(r'TRIAGE:\s*([A-D])', text, re.IGNORECASE)
    if triage_match:
        cat = triage_match.group(1).upper()
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
        conf = int(conf_match.group(1)) if conf_match else None
        return cat, text, conf, text

    # Fallback
    letter_match = re.search(r'\b([A-D])\b', text)
    if letter_match:
        return letter_match.group(1).upper(), text, None, text

    return None, text, None, text


def main():
    NUM_RUNS = 5

    data_path = Path(__file__).parent / "data" / "vignettes.json"
    with open(data_path) as f:
        all_cases = json.load(f)
    cases = [c for c in all_cases if c["id"] in TARGET_CASES]

    models_to_test = {k: v for k, v in MODELS.items() if k != "gemini-3.1-pro"}

    total = len(cases) * len(models_to_test) * len(CONDITIONS) * NUM_RUNS
    print(f"Cases: {[c['id'] + ' (' + c['title'] + ')' for c in cases]}")
    print(f"Models: {list(models_to_test.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Runs: {NUM_RUNS}")
    print(f"Total API calls: {total}\n")

    results = []

    for case in cases:
        vignette = case["original_structured"]
        gold = case["gold_standard_triage"]

        for model_name, model_cfg in models_to_test.items():
            provider = model_cfg["provider"]
            model_id = model_cfg["model_id"]
            caller = CALLERS[provider]

            for cond_name, system_prompt in CONDITIONS.items():
                for run in range(1, NUM_RUNS + 1):
                    print(f"  {case['id']} | {model_name:30s} | {cond_name:18s} | run {run} ... ", end="", flush=True)

                    t0 = time.time()
                    error = None
                    raw = ""
                    try:
                        raw = caller(model_id, system_prompt, vignette, model_cfg)
                    except Exception as e:
                        error = str(e)
                        print(f"ERROR: {error[:60]}")
                        results.append({
                            "case_id": case["id"], "case_title": case["title"],
                            "gold_standard": gold, "model": model_name,
                            "provider": provider, "condition": cond_name,
                            "run_number": run, "predicted_triage": None,
                            "reasoning": None, "confidence": None,
                            "raw_response": "", "raw_for_review": "",
                            "is_correct": None, "error": error,
                            "latency_seconds": round(time.time() - t0, 2),
                        })
                        time.sleep(1)
                        continue

                    latency = time.time() - t0
                    cat, reasoning, conf, raw_review = parse_triage(raw, cond_name)
                    is_correct = (cat == gold) if cat else None

                    results.append({
                        "case_id": case["id"], "case_title": case["title"],
                        "gold_standard": gold, "model": model_name,
                        "provider": provider, "condition": cond_name,
                        "run_number": run, "predicted_triage": cat,
                        "reasoning": str(reasoning)[:300] if reasoning else None,
                        "confidence": conf,
                        "raw_response": raw[:500],
                        "raw_for_review": raw[:1000],
                        "is_correct": is_correct, "error": error,
                        "latency_seconds": round(latency, 2),
                    })

                    status = "✓" if is_correct else ("✗" if is_correct is False else "?")
                    print(f"{status} pred={cat} gold={gold} ({latency:.1f}s)")
                    time.sleep(0.5)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / RESULTS_DIR
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"ablation_constraint_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = out_dir / f"ablation_constraint_{ts}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[k for k in results[0].keys() if k != "raw_for_review"])
            writer.writeheader()
            for r in results:
                row = {k: v for k, v in r.items() if k != "raw_for_review"}
                writer.writerow(row)

    print(f"\nResults saved to {json_path}")

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("ABLATION RESULTS — One factor at a time")
    print("=" * 90)

    for case in cases:
        cid = case["id"]
        gold = case["gold_standard_triage"]
        print(f"\n{'─'*90}")
        print(f"  {case['title']} (gold standard: {gold})")
        print(f"{'─'*90}")
        header = f"  {'Model':30s}"
        for cond in CONDITIONS:
            header += f" | {cond:16s}"
        print(header)
        print(f"  {'─'*30}" + (" | " + "─"*16) * len(CONDITIONS))

        case_results = [r for r in results if r["case_id"] == cid]
        for model_name in models_to_test:
            mr = [r for r in case_results if r["model"] == model_name]
            row = f"  {model_name:30s}"
            for cond in CONDITIONS:
                cr = [r for r in mr if r["condition"] == cond]
                correct = sum(1 for r in cr if r["is_correct"])
                total_c = len(cr)
                preds = [r["predicted_triage"] or "?" for r in cr]
                row += f" | {correct}/{total_c} {','.join(preds):>10s}"
            print(row)

    # ── Gemini raw review ──
    print("\n" + "=" * 90)
    print("GEMINI RAW RESPONSES (for manual verification of parsing)")
    print("=" * 90)
    gemini_results = [r for r in results if r["provider"] == "google" and r["case_id"] == "case_17"]
    for r in gemini_results:
        if r["is_correct"] is not True:
            print(f"\n--- {r['condition']} run {r['run_number']} | pred={r['predicted_triage']} ---")
            print(r.get("raw_for_review", r.get("raw_response", ""))[:500])

if __name__ == "__main__":
    main()
