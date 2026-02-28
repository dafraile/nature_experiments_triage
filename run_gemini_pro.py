#!/usr/bin/env python3
"""Run Gemini 3.1 Pro trials with incremental saving. Resumes from existing results."""

import json, sys, time, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import GOOGLE_API_KEY, MODELS, MAX_TOKENS, RESULTS_DIR, PROMPT_FORMATS

OUTFILE = Path(__file__).parent / RESULTS_DIR / "results_gemini_pro_resumed.json"
NUM_RUNS = 5

# Load existing results to skip completed trials
existing = []
if OUTFILE.exists():
    with open(OUTFILE) as f:
        existing = json.load(f)
done_keys = set()
for r in existing:
    done_keys.add((r["case_id"], r["prompt_format"], r["run"]))

# Load vignettes
with open(Path(__file__).parent / "data" / "vignettes.json") as f:
    cases = json.load(f)

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
    config_kwargs = dict(temperature=0.7, max_output_tokens=MAX_TOKENS, system_instruction=SYSTEM_PROMPT)
    if cfg.get("thinking_level"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    response = client.models.generate_content(
        model=cfg["model_id"], contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs))
    for part in response.candidates[0].content.parts:
        if part.text and not getattr(part, 'thought', False):
            return part.text
    return response.text

def parse_triage(raw):
    if not raw:
        return None
    jm = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if jm:
        try:
            d = json.loads(jm.group())
            cat = d.get("triage_category", "").strip().upper()[:1]
            if cat in "ABCD":
                return cat
        except json.JSONDecodeError:
            pass
    tm = re.search(r'TRIAGE:\s*([A-D])', raw, re.IGNORECASE)
    if tm:
        return tm.group(1).upper()
    return None

results = list(existing)
total_new = 0
errors = 0

for case in cases:
    for fmt in PROMPT_FORMATS:
        vignette = case.get(fmt, "")
        if not vignette:
            continue
        for run in range(1, NUM_RUNS + 1):
            key = (case["id"], fmt, run)
            if key in done_keys:
                continue

            print(f"  {case['id']:10s} | {fmt:22s} | run {run} ... ", end="", flush=True)
            try:
                raw = call_gemini(vignette)
                cat = parse_triage(raw)
                correct = (cat == case["gold_standard_triage"]) if cat else None
                results.append({
                    "case_id": case["id"], "prompt_format": fmt, "run": run,
                    "model": "gemini-3.1-pro",
                    "predicted": cat, "correct": correct, "error": None,
                    "raw_response": raw[:500],
                    "gold_standard": case["gold_standard_triage"],
                })
                s = "✓" if correct else ("✗" if correct is False else "?")
                print(f"{s} pred={cat}")
                total_new += 1
            except Exception as e:
                err = str(e)
                print(f"ERROR: {err[:80]}")
                results.append({
                    "case_id": case["id"], "prompt_format": fmt, "run": run,
                    "model": "gemini-3.1-pro",
                    "predicted": None, "correct": None, "error": err,
                    "raw_response": "", "gold_standard": case["gold_standard_triage"],
                })
                errors += 1
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print("  Rate limited, waiting 30s...")
                    time.sleep(30)

            # Save after every trial
            with open(OUTFILE, "w") as f:
                json.dump(results, f, indent=2, default=str)

            time.sleep(1.5)

print(f"\nDone. New trials: {total_new}, Errors: {errors}, Total: {len(results)}")
print(f"Results saved to {OUTFILE}")
