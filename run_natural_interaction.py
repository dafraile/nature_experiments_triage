#!/usr/bin/env python3
"""
Natural Interaction Experiment Runner
=====================================

Runs a separate experiment family intended to approximate a real user message
as closely as the APIs allow:

- no system prompt
- user message only
- no temperature / top_p overrides
- no output-length override for OpenAI unless explicitly requested
- raw responses stored verbatim

This is intentionally separate from the structured replication so we do not
contaminate the original protocol or its results.

By default this targets GPT-5.2 only, because Anthropic's API requires an
explicit max_tokens value; Claude can still be used, but only if the caller
supplies a transport-level token cap explicitly.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    VERTEX_AI_KEY,
    MODELS,
    NUM_RUNS,
    OPENAI_API_KEY,
    PROMPT_FORMATS,
    RESULTS_DIR,
    TIMEOUT_SECONDS,
)
from llm_utils import extract_google_text, extract_triage_category, infer_free_text_triage, make_google_client  # noqa: E402


DEFAULT_FORMATS = ["patient_realistic", "patient_minimal"]


@dataclass
class NaturalTrialResult:
    case_id: str
    case_title: str
    gold_standard: str
    model: str
    provider: str
    prompt_format: str
    run_number: int
    raw_response: str
    best_effort_triage: Optional[str]
    best_effort_is_correct: Optional[bool]
    error: Optional[str]
    latency_seconds: float
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run near-unconstrained natural-interaction trials",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.2-thinking-high"],
        help="Models to run (defaults to GPT-5.2 only)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FORMATS,
        choices=PROMPT_FORMATS,
        help="Prompt formats to use (defaults to patient-like prompts only)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional case ids to run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of repeated runs per cell (default: {NUM_RUNS})",
    )
    parser.add_argument(
        "--call-wait",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2)",
    )
    parser.add_argument(
        "--openai-max-completion-tokens",
        type=int,
        default=None,
        help="Optional transport-level fallback if OpenAI rejects omitted max_completion_tokens",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=None,
        help="Required only if you include Claude models; Anthropic's API needs max_tokens",
    )
    parser.add_argument(
        "--google-vertex",
        action="store_true",
        help="Use Vertex Express for Google models instead of the Gemini developer API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling APIs",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional fixed output stem (without extension) for resumable runs",
    )
    return parser.parse_args()


def load_cases(selected_case_ids: Optional[set[str]]) -> list[dict]:
    with open(Path(__file__).parent / "data" / "vignettes.json") as f:
        cases = json.load(f)

    if not selected_case_ids:
        return cases

    available = {case["id"] for case in cases}
    missing = sorted(selected_case_ids - available)
    if missing:
        raise SystemExit(f"Unknown case ids: {', '.join(missing)}")
    return [case for case in cases if case["id"] in selected_case_ids]


def call_openai_natural(model_id: str, user_message: str, model_config: dict,
                        max_completion_tokens: Optional[int]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }
    reasoning_effort = model_config.get("reasoning_effort")
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_anthropic_natural(model_id: str, user_message: str, max_tokens: Optional[int]) -> str:
    if max_tokens is None:
        raise SystemExit(
            "Claude natural-interaction runs require --anthropic-max-tokens because "
            "Anthropic's API will not accept a request without max_tokens.",
        )

    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def call_google_natural(model_id: str, user_message: str, model_config: dict,
                        use_vertex: bool = False) -> str:
    from google.genai import types

    client = make_google_client(GOOGLE_API_KEY, VERTEX_AI_KEY, use_vertex=use_vertex)
    config_kwargs = {}
    if not use_vertex:
        config_kwargs["http_options"] = types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000)
    thinking_level = model_config.get("thinking_level")
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
        )

    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return extract_google_text(response)


def best_effort_label(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None
    return extract_triage_category(raw_response) or infer_free_text_triage(raw_response)


def run_trial(case: dict, model_name: str, model_config: dict, prompt_format: str,
              run_number: int, args: argparse.Namespace) -> NaturalTrialResult:
    user_message = case.get(prompt_format, "")
    if not user_message:
        return NaturalTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=model_config["provider"],
            prompt_format=prompt_format,
            run_number=run_number,
            raw_response="",
            best_effort_triage=None,
            best_effort_is_correct=None,
            error=f"Format '{prompt_format}' not found for case {case['id']}",
            latency_seconds=0.0,
            timestamp=datetime.now().isoformat(),
        )

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name} | Case: {case['id']} | Format: {prompt_format} | Run: {run_number}")
        print("System prompt: <none>")
        print(f"User prompt: {user_message}")
        return NaturalTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=model_config["provider"],
            prompt_format=prompt_format,
            run_number=run_number,
            raw_response="",
            best_effort_triage=None,
            best_effort_is_correct=None,
            error=None,
            latency_seconds=0.0,
            timestamp=datetime.now().isoformat(),
        )

    start = time.time()
    try:
        provider = model_config["provider"]
        if provider == "openai":
            raw = call_openai_natural(
                model_config["model_id"],
                user_message,
                model_config,
                args.openai_max_completion_tokens,
            )
        elif provider == "anthropic":
            raw = call_anthropic_natural(
                model_config["model_id"],
                user_message,
                args.anthropic_max_tokens,
            )
        elif provider == "google":
            raw = call_google_natural(
                model_config["model_id"],
                user_message,
                model_config,
                args.google_vertex,
            )
        else:
            raise SystemExit(
                f"Provider '{provider}' is not supported by this runner. "
                "This runner currently supports OpenAI, Anthropic, and Google.",
            )

        latency = time.time() - start
        triage = best_effort_label(raw)
        return NaturalTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=provider,
            prompt_format=prompt_format,
            run_number=run_number,
            raw_response=raw,
            best_effort_triage=triage,
            best_effort_is_correct=(triage == case["gold_standard_triage"]) if triage else None,
            error=None,
            latency_seconds=latency,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        latency = time.time() - start
        return NaturalTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=model_config["provider"],
            prompt_format=prompt_format,
            run_number=run_number,
            raw_response="",
            best_effort_triage=None,
            best_effort_is_correct=None,
            error=str(exc),
            latency_seconds=latency,
            timestamp=datetime.now().isoformat(),
        )


def validate_models(model_names: list[str]) -> None:
    unknown = [name for name in model_names if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")


def trial_key(result: NaturalTrialResult | dict) -> tuple[str, str, str, int]:
    return (
        str(result["model"] if isinstance(result, dict) else result.model),
        str(result["case_id"] if isinstance(result, dict) else result.case_id),
        str(result["prompt_format"] if isinstance(result, dict) else result.prompt_format),
        int(result["run_number"] if isinstance(result, dict) else result.run_number),
    )


def prepare_output_paths(output_stem: Optional[str]) -> tuple[Path, Path]:
    out_dir = Path(__file__).parent / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_stem:
        json_path = out_dir / f"{output_stem}.json"
        csv_path = out_dir / f"{output_stem}.csv"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"natural_interaction_{ts}.json"
        csv_path = out_dir / f"natural_interaction_{ts}.csv"
    return json_path, csv_path


def save_results(results: list[NaturalTrialResult], json_path: Path, csv_path: Path) -> None:
    rows = [asdict(result) for result in results]
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_existing_results(json_path: Path) -> list[NaturalTrialResult]:
    if not json_path.exists():
        return []
    with open(json_path) as f:
        rows = json.load(f)
    return [NaturalTrialResult(**row) for row in rows]

def main() -> None:
    args = parse_args()
    validate_models(args.models)
    cases = load_cases(set(args.cases) if args.cases else None)

    total = len(cases) * len(args.models) * len(args.formats) * args.runs
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    existing_results: list[NaturalTrialResult] = []
    results_by_key: dict[tuple[str, str, str, int], NaturalTrialResult] = {}

    print("Natural interaction experiment")
    print("Protocol: user-only prompt, no system prompt, no temperature overrides")
    if args.openai_max_completion_tokens is None:
        print("OpenAI max_completion_tokens: omitted")
    else:
        print(f"OpenAI max_completion_tokens: {args.openai_max_completion_tokens}")
    if any(MODELS[name]["provider"] == "anthropic" for name in args.models):
        print(f"Anthropic max_tokens: {args.anthropic_max_tokens}")
    if any(MODELS[name]["provider"] == "google" for name in args.models):
        print("Google config: no system prompt, no temperature override, no output cap override")
    print(f"Total planned trials: {total}")
    if not args.dry_run:
        json_path, csv_path = prepare_output_paths(args.output_stem)
        existing_results = load_existing_results(json_path)
        results_by_key = {trial_key(result): result for result in existing_results}
        print(f"Checkpoint JSON: {json_path}")
        print(f"Checkpoint CSV:  {csv_path}")
        if existing_results:
            print(f"Resuming from {len(existing_results)} saved trial(s)")

    counter = 0
    for model_name in args.models:
        model_config = MODELS[model_name]
        for case in cases:
            for prompt_format in args.formats:
                for run_number in range(1, args.runs + 1):
                    counter += 1
                    planned_key = (model_name, case["id"], prompt_format, run_number)
                    if planned_key in results_by_key and not results_by_key[planned_key].error:
                        print(
                            f"[{counter:03d}/{total:03d}] "
                            f"{model_name:22s} | {case['id']:8s} | {prompt_format:18s} | run {run_number} "
                            f"(skip: already saved)",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{counter:03d}/{total:03d}] "
                        f"{model_name:22s} | {case['id']:8s} | {prompt_format:18s} | run {run_number}",
                        flush=True,
                    )
                    result = run_trial(case, model_name, model_config, prompt_format, run_number, args)
                    if result.error:
                        print(f"  ERROR: {result.error[:160]}")
                    elif result.best_effort_triage:
                        marker = "✓" if result.best_effort_is_correct else "✗"
                        print(f"  {marker} best_effort={result.best_effort_triage}")
                    else:
                        print("  ? unclassified raw response")
                    results_by_key[planned_key] = result
                    if not args.dry_run and json_path and csv_path:
                        ordered_results = [results_by_key[key] for key in sorted(results_by_key)]
                        save_results(ordered_results, json_path, csv_path)
                    if not args.dry_run and args.call_wait > 0:
                        time.sleep(args.call_wait)

    if args.dry_run:
        print("\nDry run only; no files written.")
        return

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
