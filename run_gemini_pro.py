#!/usr/bin/env python3
"""Run Gemini 3.1 Pro trials with incremental saving. Resumes and retries incomplete rows."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import GOOGLE_API_KEY, MODELS, MAX_TOKENS, RESULTS_DIR, PROMPT_FORMATS, TIMEOUT_SECONDS
from llm_utils import (
    extract_google_text,
    extract_triage_category,
    google_visible_output_tokens,
    infer_free_text_triage,
    is_retryable_error,
)

OUTFILE = Path(__file__).parent / RESULTS_DIR / "results_gemini_pro_resumed.json"
NUM_RUNS = 5

parser = argparse.ArgumentParser(description="Resume Gemini 3.1 Pro main experiment rows")
parser.add_argument("--cases", nargs="+", default=None, help="Optional case ids to retry")
parser.add_argument("--formats", nargs="+", default=None, choices=PROMPT_FORMATS,
                    help="Optional prompt formats to retry")
parser.add_argument("--runs", nargs="+", type=int, default=None,
                    help="Optional run numbers to retry")
parser.add_argument("--retry-wait", type=float, default=30.0,
                    help="Seconds to wait after retryable API errors (default: 30)")
parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS,
                    help=f"Per-request timeout for Gemini calls (default: {TIMEOUT_SECONDS})")
args = parser.parse_args()

selected_case_ids = set(args.cases) if args.cases else None
selected_formats = set(args.formats) if args.formats else None
selected_runs = set(args.runs) if args.runs else None
request_timeout_ms = max(10000, int(args.timeout_seconds * 1000))

# Load existing results and only skip complete rows.
existing = []
if OUTFILE.exists():
    with open(OUTFILE) as f:
        existing = json.load(f)
recovered_existing = 0
for row in existing:
    if row.get("predicted") is None and row.get("error") is None:
        recovered = extract_triage_category(row.get("raw_response", ""))
        if not recovered:
            recovered = infer_free_text_triage(row.get("raw_response", ""))
        if recovered:
            row["predicted"] = recovered
            gold = row.get("gold_standard")
            row["correct"] = (recovered == gold) if gold else None
            recovered_existing += 1
if recovered_existing:
    with open(OUTFILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
results_by_key = {
    (r["case_id"], r["prompt_format"], int(r["run"])): r
    for r in existing
}
done_keys = {
    key for key, row in results_by_key.items()
    if row.get("error") is None and row.get("predicted") is not None
}

# Load vignettes
with open(Path(__file__).parent / "data" / "vignettes.json") as f:
    cases = json.load(f)
if selected_case_ids:
    available_case_ids = {case["id"] for case in cases}
    missing_case_ids = sorted(selected_case_ids - available_case_ids)
    if missing_case_ids:
        raise SystemExit(f"Unknown case ids: {', '.join(missing_case_ids)}")

# System prompt
SYSTEM_PROMPT = """You are a medical triage assistant. Based on the clinical information provided, assign one of the following triage categories:

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

def call_gemini(user_message):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    cfg = MODELS["gemini-3.1-pro"]
    config_kwargs = dict(
        temperature=0.7,
        max_output_tokens=google_visible_output_tokens(cfg["model_id"], MAX_TOKENS),
        system_instruction=SYSTEM_PROMPT,
        http_options=types.HttpOptions(timeout=request_timeout_ms),
    )
    if cfg.get("thinking_level"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=cfg["thinking_level"])
    response = client.models.generate_content(
        model=cfg["model_id"], contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs))
    return extract_google_text(response)

def parse_triage(raw):
    category = extract_triage_category(raw)
    if category:
        return category
    return infer_free_text_triage(raw)

results = dict(results_by_key)
total_new = 0
errors = 0

print("Gemini 3.1 Pro main experiment resume")
print(f"Resuming from {len(existing)} existing rows")
if recovered_existing:
    print(f"Recovered {recovered_existing} existing rows from saved raw responses")
if selected_case_ids:
    print(f"Filtered cases: {', '.join(sorted(selected_case_ids))}")
if selected_formats:
    print(f"Filtered formats: {', '.join(sorted(selected_formats))}")
if selected_runs:
    print(f"Filtered runs: {', '.join(str(run) for run in sorted(selected_runs))}")
print(f"Retry wait: {args.retry_wait:g}s")
print(f"Request timeout: {request_timeout_ms / 1000:g}s")
print("")

for case in cases:
    if selected_case_ids and case["id"] not in selected_case_ids:
        continue
    for fmt in PROMPT_FORMATS:
        if selected_formats and fmt not in selected_formats:
            continue
        vignette = case.get(fmt, "")
        if not vignette:
            continue
        for run in range(1, NUM_RUNS + 1):
            if selected_runs and run not in selected_runs:
                continue
            key = (case["id"], fmt, run)
            if key in done_keys:
                continue

            print(f"  {case['id']:10s} | {fmt:22s} | run {run} ... ", end="", flush=True)
            try:
                raw = call_gemini(vignette)
                cat = parse_triage(raw)
                correct = (cat == case["gold_standard_triage"]) if cat else None
                results[key] = {
                    "case_id": case["id"], "prompt_format": fmt, "run": run,
                    "model": "gemini-3.1-pro",
                    "predicted": cat, "correct": correct, "error": None,
                    "raw_response": raw,
                    "gold_standard": case["gold_standard_triage"],
                }
                s = "✓" if correct else ("✗" if correct is False else "?")
                print(f"{s} pred={cat}")
                done_keys.add(key)
                total_new += 1
            except Exception as e:
                err = str(e)
                print(f"ERROR: {err[:80]}")
                results[key] = {
                    "case_id": case["id"], "prompt_format": fmt, "run": run,
                    "model": "gemini-3.1-pro",
                    "predicted": None, "correct": None, "error": err,
                    "raw_response": "", "gold_standard": case["gold_standard_triage"],
                }
                errors += 1

            # Save after every trial
            with open(OUTFILE, "w") as f:
                ordered = [results[k] for k in sorted(results)]
                json.dump(ordered, f, indent=2, default=str)

            if results[key].get("error") and is_retryable_error(results[key]["error"]) and args.retry_wait > 0:
                print(f"  Retryable API error, waiting {args.retry_wait:g}s...")
                time.sleep(args.retry_wait)

            time.sleep(1.5)

print(f"\nDone. New trials: {total_new}, Errors: {errors}, Total: {len(results)}")
print(f"Results saved to {OUTFILE}")
