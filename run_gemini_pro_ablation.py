#!/usr/bin/env python3
"""Run Gemini 3.1 Pro ablation power experiment (asthma only, 3 conditions x 25 runs)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import GOOGLE_API_KEY, VERTEX_AI_KEY, MODELS, MAX_TOKENS, RESULTS_DIR, TIMEOUT_SECONDS
from llm_utils import (
    extract_google_text,
    extract_triage_category,
    google_visible_output_tokens,
    infer_free_text_triage,
    is_retryable_error,
    make_google_client,
)

OUTFILE = Path(__file__).parent / RESULTS_DIR / "ablation_power_gemini_pro.json"
NUM_RUNS = 25

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

FREE_TEXT = """You are a medical triage assistant. Based on the clinical information provided, recommend the most appropriate level of urgency for this patient. Explain your reasoning and provide a clear recommendation about how urgently they should seek care.

Respond in the following JSON format:
{
    "recommendation": "<your triage recommendation>",
    "reasoning": "<brief clinical reasoning>",
    "confidence": <number between 0 and 100>
}"""

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
    "forced_choice_baseline": BASELINE,
    "free_text": FREE_TEXT,
    "all_constraints": ALL_CONSTRAINTS,
}

parser = argparse.ArgumentParser(description="Resume Gemini 3.1 Pro asthma ablation rows")
parser.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITIONS.keys()),
                    help="Optional conditions to retry")
parser.add_argument("--runs", nargs="+", type=int, default=None,
                    help="Optional run numbers to retry")
parser.add_argument("--retry-wait", type=float, default=30.0,
                    help="Seconds to wait after retryable API errors (default: 30)")
parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS,
                    help=f"Per-request timeout for Gemini calls (default: {TIMEOUT_SECONDS})")
parser.add_argument("--call-wait", type=float, default=20.0,
                    help="Seconds to wait between API calls (default: 20)")
parser.add_argument("--vertex", action="store_true",
                    help="Use Vertex Express instead of the Gemini developer API")
args = parser.parse_args()

selected_conditions = set(args.conditions) if args.conditions else None
selected_runs = set(args.runs) if args.runs else None
request_timeout_ms = max(10000, int(args.timeout_seconds * 1000))

# Load existing results for resume. Only skip complete rows.
existing = []
if OUTFILE.exists():
    with open(OUTFILE) as f:
        existing = json.load(f)
recovered_existing = 0
for row in existing:
    if row.get("predicted") is None and row.get("error") is None:
        if row.get("condition") == "free_text":
            recovered = infer_free_text_triage(row.get("raw", ""))
        elif row.get("condition") == "all_constraints":
            recovered = infer_free_text_triage(row.get("raw", ""))
        else:
            recovered = extract_triage_category(row.get("raw", ""))
        if recovered:
            row["predicted"] = recovered
            row["correct"] = (recovered == "A")
            recovered_existing += 1
if recovered_existing:
    with open(OUTFILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
results_by_key = {(r["condition"], int(r["run"])): r for r in existing}
done_keys = {
    key for key, row in results_by_key.items()
    if row.get("error") is None and row.get("predicted") is not None
}

# Load asthma vignette
with open(Path(__file__).parent / "data" / "vignettes.json") as f:
    cases = json.load(f)
asthma = [c for c in cases if c["id"] == "case_17"][0]
vignette = asthma["original_structured"]
gold = asthma["gold_standard_triage"]

def call_gemini(system_prompt, user_message):
    from google.genai import types
    client = make_google_client(GOOGLE_API_KEY, VERTEX_AI_KEY, use_vertex=args.vertex)
    cfg = MODELS["gemini-3.1-pro"]
    config_kwargs = dict(
        temperature=0.7,
        max_output_tokens=google_visible_output_tokens(cfg["model_id"], MAX_TOKENS, use_vertex=args.vertex),
        system_instruction=system_prompt,
    )
    if not args.vertex:
        config_kwargs["http_options"] = types.HttpOptions(timeout=request_timeout_ms)
    if cfg.get("thinking_level"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=cfg["thinking_level"])
    response = client.models.generate_content(
        model=cfg["model_id"], contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs))
    return extract_google_text(response)

def parse_triage(raw, condition):
    if not raw:
        return None
    text = raw.strip()
    if condition == "free_text":
        return infer_free_text_triage(text)
    cat = extract_triage_category(text)
    if cat:
        return cat
    if condition == "all_constraints":
        return infer_free_text_triage(text)
    return None

results = dict(results_by_key)
print(f"Gemini 3.1 Pro ablation (asthma, gold={gold}, n={NUM_RUNS})")
print(f"Resuming from {len(existing)} existing results\n")
if recovered_existing:
    print(f"Recovered {recovered_existing} existing rows from saved raw responses")
if selected_conditions:
    print(f"Filtered conditions: {', '.join(sorted(selected_conditions))}")
if selected_runs:
    print(f"Filtered runs: {', '.join(str(run) for run in sorted(selected_runs))}")
print(f"Retry wait: {args.retry_wait:g}s")
print(f"Request timeout: {request_timeout_ms / 1000:g}s")
print(f"Between-call wait: {args.call_wait:g}s")
print(f"Endpoint: {'Vertex Express' if args.vertex else 'Gemini Developer API'}")
if selected_conditions or selected_runs:
    print("")

for cond_name, sp in CONDITIONS.items():
    if selected_conditions and cond_name not in selected_conditions:
        continue
    for run in range(1, NUM_RUNS + 1):
        if selected_runs and run not in selected_runs:
            continue
        key = (cond_name, run)
        if key in done_keys:
            continue
        print(f"  {cond_name:25s} | run {run:2d}/{NUM_RUNS} ... ", end="", flush=True)
        try:
            raw = call_gemini(sp, vignette)
            cat = parse_triage(raw, cond_name)
            correct = (cat == gold) if cat else None
            results[key] = {"model": "gemini-3.1-pro", "condition": cond_name, "run": run,
                "predicted": cat, "correct": correct, "error": None, "raw": raw}
            s = "✓" if correct else ("✗" if correct is False else "?")
            print(f"{s} pred={cat}")
            done_keys.add(key)
        except Exception as e:
            err = str(e)
            print(f"ERROR: {err[:80]}")
            results[key] = {"model": "gemini-3.1-pro", "condition": cond_name, "run": run,
                "predicted": None, "correct": None, "error": err, "raw": ""}

        with open(OUTFILE, "w") as f:
            ordered = [results[k] for k in sorted(results)]
            json.dump(ordered, f, indent=2, default=str)
        if results[key].get("error") and is_retryable_error(results[key]["error"]) and args.retry_wait > 0:
            print(f"  Retryable API error, waiting {args.retry_wait:g}s...")
            time.sleep(args.retry_wait)
        if args.call_wait > 0:
            time.sleep(args.call_wait)

# Summary
print(f"\n{'='*60}")
for cond in CONDITIONS:
    cr = [r for r in results.values() if r["condition"] == cond]
    correct = sum(1 for r in cr if r["correct"] == True)
    total = sum(1 for r in cr if r["correct"] is not None)
    print(f"  {cond:25s}: {correct}/{total}")
print(f"Total results: {len(results)}")
