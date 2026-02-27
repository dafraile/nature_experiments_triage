#!/usr/bin/env python3
"""
High-power ablation: 25 runs on asthma case for forced-choice vs free-text.
Focused on the three key conditions for statistical testing.
"""

import json, sys, time, re, csv
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    MODELS, MAX_TOKENS, RESULTS_DIR,
)

# Only the 3 conditions that matter for the causal claim
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

# API wrappers
def call_openai(model_id, system_prompt, user_message, model_config=None):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    reasoning_effort = (model_config or {}).get("reasoning_effort", None)
    kwargs = dict(model=model_id, max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0.7
    return client.chat.completions.create(**kwargs).choices[0].message.content

def call_anthropic(model_id, system_prompt, user_message, model_config=None):
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return client.messages.create(model=model_id, max_tokens=MAX_TOKENS, temperature=0.7,
        system=system_prompt, messages=[{"role": "user", "content": user_message}]).content[0].text

def call_google(model_id, system_prompt, user_message, model_config=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    thinking_level = (model_config or {}).get("thinking_level", None)
    config_kwargs = dict(temperature=0.7, max_output_tokens=MAX_TOKENS, system_instruction=system_prompt)
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    response = client.models.generate_content(model=model_id, contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs))
    for part in response.candidates[0].content.parts:
        if part.text and not getattr(part, 'thought', False):
            return part.text
    return response.text

CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}

def parse_triage(raw, condition):
    if not raw:
        return None, raw
    text = raw.strip()
    if condition == "free_text":
        # Check JSON first
        jm = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if jm:
            try:
                d = json.loads(jm.group())
                rec = d.get("recommendation", "").lower()
                reas = d.get("reasoning", "").lower()
                full = rec + " " + reas
                if any(w in full for w in ["emergency", "911", "er ", "er.", "immediate", "right away", "call 911", "go to the er", "emergency room", "emergency department"]):
                    return "A", text
            except json.JSONDecodeError:
                pass
        lower = text.lower()
        if any(w in lower for w in ["emergency", "911", "er ", "er.", "immediate", "right away", "emergency room", "emergency department", "call 911"]):
            return "A", text
        if any(w in lower for w in ["24 hour", "urgent care", "today"]):
            return "B", text
        return None, text
    # Standard A/B/C/D
    jm = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if jm:
        try:
            d = json.loads(jm.group())
            cat = d.get("triage_category", "").strip().upper()[:1]
            if cat in "ABCD":
                return cat, text
        except json.JSONDecodeError:
            pass
    tm = re.search(r'TRIAGE:\s*([A-D])', text, re.IGNORECASE)
    if tm:
        return tm.group(1).upper(), text
    lm = re.search(r'\b([A-D])\b', text)
    if lm:
        return lm.group(1).upper(), text
    return None, text

def main():
    NUM_RUNS = 25
    data_path = Path(__file__).parent / "data" / "vignettes.json"
    with open(data_path) as f:
        all_cases = json.load(f)
    asthma = [c for c in all_cases if c["id"] == "case_17"][0]
    vignette = asthma["original_structured"]
    gold = asthma["gold_standard_triage"]

    models_to_test = {k: v for k, v in MODELS.items() if k != "gemini-3.1-pro"}
    total = len(models_to_test) * len(CONDITIONS) * NUM_RUNS
    print(f"Asthma exacerbation (gold: {gold})")
    print(f"Models: {list(models_to_test.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Runs: {NUM_RUNS}, Total: {total}\n")

    results = []
    for model_name, model_cfg in models_to_test.items():
        provider = model_cfg["provider"]
        model_id = model_cfg["model_id"]
        caller = CALLERS[provider]
        for cond_name, sp in CONDITIONS.items():
            for run in range(1, NUM_RUNS + 1):
                print(f"  {model_name:30s} | {cond_name:25s} | run {run:2d}/{NUM_RUNS} ... ", end="", flush=True)
                t0 = time.time()
                error = None
                raw = ""
                try:
                    raw = caller(model_id, sp, vignette, model_cfg)
                except Exception as e:
                    error = str(e)
                    print(f"ERROR: {error[:60]}")
                    results.append({"model": model_name, "condition": cond_name, "run": run,
                        "predicted": None, "correct": None, "error": error, "raw": ""})
                    time.sleep(2)
                    continue
                cat, raw_text = parse_triage(raw, cond_name)
                correct = (cat == gold) if cat else None
                results.append({"model": model_name, "condition": cond_name, "run": run,
                    "predicted": cat, "correct": correct, "error": None, "raw": raw[:500]})
                s = "✓" if correct else ("✗" if correct is False else "?")
                print(f"{s} pred={cat} ({time.time()-t0:.1f}s)")
                time.sleep(0.3)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / RESULTS_DIR
    out.mkdir(exist_ok=True)
    with open(out / f"ablation_power_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary + Fisher's exact test
    from scipy import stats
    print("\n" + "=" * 80)
    print(f"HIGH-POWER ABLATION — Asthma exacerbation (n={NUM_RUNS} per cell)")
    print("=" * 80)
    for model_name in models_to_test:
        mr = [r for r in results if r["model"] == model_name]
        print(f"\n  {model_name}:")
        for cond in CONDITIONS:
            cr = [r for r in mr if r["condition"] == cond]
            correct = sum(1 for r in cr if r["correct"] == True)
            total = len(cr)
            preds = [r["predicted"] or "?" for r in cr]
            print(f"    {cond:25s}: {correct}/{total} ({correct/total*100:.0f}%) — {','.join(preds)}")

        # Fisher's exact: forced_choice_baseline vs free_text
        fc = [r for r in mr if r["condition"] == "forced_choice_baseline"]
        ft = [r for r in mr if r["condition"] == "free_text"]
        fc_correct = sum(1 for r in fc if r["correct"] == True)
        fc_wrong = len(fc) - fc_correct
        ft_correct = sum(1 for r in ft if r["correct"] == True)
        ft_wrong = len(ft) - ft_correct
        table = [[fc_correct, fc_wrong], [ft_correct, ft_wrong]]
        odds, p = stats.fisher_exact(table)
        print(f"    Fisher's exact (forced vs free): p = {p:.2e}, OR = {odds:.2f}")

if __name__ == "__main__":
    main()
