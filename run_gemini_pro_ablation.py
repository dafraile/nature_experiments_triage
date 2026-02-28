#!/usr/bin/env python3
"""Run Gemini 3.1 Pro ablation power experiment (asthma only, 3 conditions x 25 runs)."""

import json, sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import GOOGLE_API_KEY, MODELS, MAX_TOKENS, RESULTS_DIR

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

# Load existing results for resume
existing = []
if OUTFILE.exists():
    with open(OUTFILE) as f:
        existing = json.load(f)
done_keys = set((r["condition"], r["run"]) for r in existing)

# Load asthma vignette
with open(Path(__file__).parent / "data" / "vignettes.json") as f:
    cases = json.load(f)
asthma = [c for c in cases if c["id"] == "case_17"][0]
vignette = asthma["original_structured"]
gold = asthma["gold_standard_triage"]

def call_gemini(system_prompt, user_message):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    cfg = MODELS["gemini-3.1-pro"]
    config_kwargs = dict(temperature=0.7, max_output_tokens=MAX_TOKENS, system_instruction=system_prompt)
    if cfg.get("thinking_level"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    response = client.models.generate_content(
        model=cfg["model_id"], contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs))
    for part in response.candidates[0].content.parts:
        if part.text and not getattr(part, 'thought', False):
            return part.text
    return response.text

def parse_triage(raw, condition):
    if not raw:
        return None
    text = raw.strip()
    if condition == "free_text":
        jm = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if jm:
            try:
                d = json.loads(jm.group())
                rec = d.get("recommendation", "").lower()
                reas = d.get("reasoning", "").lower()
                full = rec + " " + reas
                if any(w in full for w in ["emergency", "911", "er ", "er.", "immediate", "right away", "call 911", "go to the er", "emergency room", "emergency department"]):
                    return "A"
            except json.JSONDecodeError:
                pass
        lower = text.lower()
        if any(w in lower for w in ["emergency", "911", "er ", "er.", "immediate", "right away", "emergency room", "emergency department", "call 911"]):
            return "A"
        if any(w in lower for w in ["24 hour", "urgent care", "today"]):
            return "B"
        return None
    jm = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if jm:
        try:
            d = json.loads(jm.group())
            cat = d.get("triage_category", "").strip().upper()[:1]
            if cat in "ABCD":
                return cat
        except json.JSONDecodeError:
            pass
    tm = re.search(r'TRIAGE:\s*([A-D])', text, re.IGNORECASE)
    if tm:
        return tm.group(1).upper()
    lm = re.search(r'\b([A-D])\b', text)
    if lm:
        return lm.group(1).upper()
    return None

results = list(existing)
print(f"Gemini 3.1 Pro ablation (asthma, gold={gold}, n={NUM_RUNS})")
print(f"Resuming from {len(existing)} existing results\n")

for cond_name, sp in CONDITIONS.items():
    for run in range(1, NUM_RUNS + 1):
        if (cond_name, run) in done_keys:
            continue
        print(f"  {cond_name:25s} | run {run:2d}/{NUM_RUNS} ... ", end="", flush=True)
        try:
            raw = call_gemini(sp, vignette)
            cat = parse_triage(raw, cond_name)
            correct = (cat == gold) if cat else None
            results.append({"model": "gemini-3.1-pro", "condition": cond_name, "run": run,
                "predicted": cat, "correct": correct, "error": None, "raw": raw[:500]})
            s = "✓" if correct else ("✗" if correct is False else "?")
            print(f"{s} pred={cat}")
        except Exception as e:
            err = str(e)
            print(f"ERROR: {err[:80]}")
            results.append({"model": "gemini-3.1-pro", "condition": cond_name, "run": run,
                "predicted": None, "correct": None, "error": err, "raw": ""})
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("  Rate limited, waiting 30s...")
                time.sleep(30)

        with open(OUTFILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        time.sleep(1.5)

# Summary
print(f"\n{'='*60}")
for cond in CONDITIONS:
    cr = [r for r in results if r["condition"] == cond]
    correct = sum(1 for r in cr if r["correct"] == True)
    total = sum(1 for r in cr if r["correct"] is not None)
    print(f"  {cond:25s}: {correct}/{total}")
print(f"Total results: {len(results)}")
