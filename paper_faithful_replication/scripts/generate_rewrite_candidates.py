#!/usr/bin/env python3
"""Generate model-authored naturalistic rewrite candidates for a canonical case."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    MODELS,
    OPENAI_API_KEY,
    TIMEOUT_SECONDS,
    VERTEX_AI_KEY,
)
from llm_utils import extract_google_text, make_google_client  # noqa: E402


DEFAULT_MODELS = [
    "gpt-5.2-thinking-high",
    "claude-opus-4.6",
    "gemini-3.1-pro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rewrite candidates for one case")
    parser.add_argument("--case-id", required=True, help="Canonical case id, e.g. E1 or F13")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to use",
    )
    parser.add_argument(
        "--google-vertex",
        action="store_true",
        help="Use Vertex transport for Google models",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path",
    )
    return parser.parse_args()


def load_case(case_id: str) -> dict:
    path = WORKSPACE_DIR / "data" / "canonical_source_bank.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["case_id"] == case_id:
            return row
    raise SystemExit(f"Unknown case_id: {case_id}")


def load_guide() -> str:
    return (WORKSPACE_DIR / "docs" / "NATURALISTIC_REWRITE_GUIDE.md").read_text()


def build_prompt(case: dict, guide: str) -> str:
    return f"""You are helping author stimuli for a paper-faithful replication study of clinical-triage prompts.

Task:
Write ONE patient-like, single-turn naturalistic rewrite of the source row below.

Why we are doing this:
We are testing whether the paper's exam-style prompt construction changes model behavior. We need a more realistic patient-authored message, but we must preserve the source row's information content exactly.

Follow this rewrite guide strictly:

{guide}

Additional style target:
- The output should read like a believable patient or caregiver message, not a chart summary.
- Avoid a telegraphic list of symptoms.
- Prefer a natural narrative flow with ordinary language, some uncertainty, and realistic phrasing.
- You may still include test results and measured findings when the source row includes them, but weave them in naturally as things the patient was told, saw in the portal, or remembers from the visit.
- Do not use bullets or headings in the final rewrite.

Case metadata:
- case_id: {case['case_id']}
- diagnosis: {case['diagnosis']}
- source_version: {case['source_version']}
- gold_triage: {case['gold_triage']}

Source prompt opening:
{case['source_opening']}

Source "About me":
{case['source_about_me']}

Source clinical block:
{case['source_clinical_block']}

Output rules:
- Produce only one rewrite candidate.
- Keep the message as a plausible single patient/caregiver message.
- Keep all clinically material facts from the source row.
- Do not add or remove objective data.
- End with a clear timing/disposition ask.

Respond in valid JSON with exactly these keys:
- natural_singleturn
- style_notes
- preservation_check
"""


def call_openai(model_name: str, prompt: str) -> str:
    from openai import OpenAI

    model_config = MODELS[model_name]
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 1200,
    }
    reasoning_effort = model_config.get("reasoning_effort")
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_anthropic(model_name: str, prompt: str) -> str:
    import anthropic

    model_config = MODELS[model_name]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_config["model_id"],
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def call_google(model_name: str, prompt: str, use_vertex: bool) -> str:
    from google.genai import types

    model_config = MODELS[model_name]
    client = make_google_client(GOOGLE_API_KEY, VERTEX_AI_KEY, use_vertex=use_vertex)
    config_kwargs = {}
    if not use_vertex:
        config_kwargs["http_options"] = types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000)
    thinking_level = model_config.get("thinking_level")
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    response = client.models.generate_content(
        model=model_config["model_id"],
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return extract_google_text(response)


def parse_jsonish(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


def main() -> None:
    args = parse_args()
    case = load_case(args.case_id)
    guide = load_guide()
    prompt = build_prompt(case, guide)

    out_path = Path(args.output) if args.output else (
        WORKSPACE_DIR / "results" / f"rewrite_candidates_{args.case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at": datetime.now().isoformat(),
        "case_id": args.case_id,
        "prompt": prompt,
        "candidates": [],
    }

    for model_name in args.models:
        provider = MODELS[model_name]["provider"]
        if provider == "openai":
            raw = call_openai(model_name, prompt)
        elif provider == "anthropic":
            raw = call_anthropic(model_name, prompt)
        elif provider == "google":
            raw = call_google(model_name, prompt, args.google_vertex)
        else:
            raise SystemExit(f"Unsupported provider: {provider}")

        parsed = None
        parse_error = None
        try:
            parsed = parse_jsonish(raw)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

        results["candidates"].append(
            {
                "model": model_name,
                "provider": provider,
                "raw": raw,
                "parsed": parsed,
                "parse_error": parse_error,
            }
        )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results['candidates'])} candidates to {out_path}")


if __name__ == "__main__":
    main()
