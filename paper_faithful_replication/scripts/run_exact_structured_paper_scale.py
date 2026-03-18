#!/usr/bin/env python3
"""Run the paper's exact structured prompts with paper-native A/B/C/D semantics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS  # noqa: E402
from llm_utils import (  # noqa: E402
    extract_triage_category,
    infer_free_text_triage_paper,
    parse_structured_response,
    triage_matches_gold,
)
from run_natural_interaction import (  # noqa: E402
    call_anthropic_natural,
    call_google_natural,
    call_openai_natural,
)


DEFAULT_DATASET = WORKSPACE_DIR / "data" / "canonical_singleturn_vignettes.json"
DEFAULT_RESULTS = WORKSPACE_DIR / "results"


@dataclass
class StructuredPaperTrialResult:
    case_id: str
    case_title: str
    gold_standard: str
    model: str
    provider: str
    prompt_format: str
    system_prompt_version: str
    run_number: int
    source_user_message: str
    predicted_triage: Optional[str]
    reasoning: Optional[str]
    confidence: Optional[float]
    raw_response: str
    is_correct: Optional[bool]
    error: Optional[str]
    latency_seconds: float
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact paper prompts without an added system prompt")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models to run")
    parser.add_argument("--cases", nargs="+", default=None, help="Optional case ids to run")
    parser.add_argument("--runs", type=int, default=2, help="Repeated runs per case/model")
    parser.add_argument("--call-wait", type=float, default=1.0, help="Seconds to wait between API calls")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Run-ready canonical dataset JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS, help="Output directory")
    parser.add_argument("--output-stem", default=None, help="Fixed output stem (without extension)")
    parser.add_argument("--openai-max-completion-tokens", type=int, default=8192, help="Transport-level cap for OpenAI")
    parser.add_argument("--anthropic-max-tokens", type=int, default=2048, help="Transport-level cap for Anthropic")
    parser.add_argument("--google-vertex", action="store_true", help="Use Vertex Express for Google models")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling APIs")
    return parser.parse_args()


def validate_models(model_names: list[str]) -> None:
    unknown = [name for name in model_names if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")


def load_cases(dataset_path: Path, selected_case_ids: Optional[list[str]]) -> list[dict]:
    cases = json.loads(dataset_path.read_text())
    if not selected_case_ids:
        return cases
    selected = set(selected_case_ids)
    filtered = [case for case in cases if case["id"] in selected]
    missing = sorted(selected - {case["id"] for case in filtered})
    if missing:
        raise SystemExit(f"Unknown case ids: {', '.join(missing)}")
    return filtered


def structured_label_paper(raw_response: str) -> Optional[str]:
    return extract_triage_category(raw_response) or infer_free_text_triage_paper(raw_response)


def trial_key(result: StructuredPaperTrialResult | dict) -> tuple[str, str, int]:
    return (
        str(result["model"] if isinstance(result, dict) else result.model),
        str(result["case_id"] if isinstance(result, dict) else result.case_id),
        int(result["run_number"] if isinstance(result, dict) else result.run_number),
    )


def output_paths(output_dir: Path, output_stem: Optional[str]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"paper_exact_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def save_results(results: list[StructuredPaperTrialResult], json_path: Path, csv_path: Path) -> None:
    rows = [asdict(result) for result in results]
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_existing(json_path: Path) -> list[StructuredPaperTrialResult]:
    if not json_path.exists():
        return []
    rows = json.loads(json_path.read_text())
    return [StructuredPaperTrialResult(**row) for row in rows]


def call_model(model_name: str, prompt: str, args: argparse.Namespace) -> str:
    model_config = MODELS[model_name]
    provider = model_config["provider"]
    if provider == "openai":
        return call_openai_natural(
            model_config["model_id"],
            prompt,
            model_config,
            args.openai_max_completion_tokens,
        )
    if provider == "anthropic":
        return call_anthropic_natural(
            model_config["model_id"],
            prompt,
            model_config,
            args.anthropic_max_tokens,
        )
    if provider == "google":
        return call_google_natural(
            model_config["model_id"],
            prompt,
            model_config,
            use_vertex=args.google_vertex,
        )
    raise SystemExit(f"Unsupported provider: {provider}")


def run_trial(case: dict, model_name: str, run_number: int,
              args: argparse.Namespace) -> StructuredPaperTrialResult:
    prompt = case["original_structured"]
    model_config = MODELS[model_name]

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name} | Case: {case['id']} | Run: {run_number}")
        print("System prompt: <none>")
        print(f"User prompt: {prompt}")
        return StructuredPaperTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=model_config["provider"],
            prompt_format="original_structured",
            system_prompt_version="none_user_only",
            run_number=run_number,
            source_user_message=prompt,
            predicted_triage=None,
            reasoning=None,
            confidence=None,
            raw_response="",
            is_correct=None,
            error=None,
            latency_seconds=0.0,
            timestamp=datetime.now().isoformat(),
        )

    start = time.time()
    raw_response = ""
    error = None
    try:
        raw_response = call_model(model_name, prompt, args)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    latency = time.time() - start

    parsed = parse_structured_response(raw_response) if raw_response else {}
    predicted = parsed.get("triage_category") or structured_label_paper(raw_response)
    is_correct = triage_matches_gold(predicted, case["gold_standard_triage"])

    return StructuredPaperTrialResult(
        case_id=case["id"],
        case_title=case["title"],
        gold_standard=case["gold_standard_triage"],
        model=model_name,
        provider=model_config["provider"],
        prompt_format="original_structured",
        system_prompt_version="none_user_only",
        run_number=run_number,
        source_user_message=prompt,
        predicted_triage=predicted,
        reasoning=parsed.get("reasoning"),
        confidence=parsed.get("confidence"),
        raw_response=raw_response,
        is_correct=is_correct,
        error=error,
        latency_seconds=round(latency, 2),
        timestamp=datetime.now().isoformat(),
    )


def main() -> None:
    args = parse_args()
    validate_models(args.models)
    cases = load_cases(args.dataset, args.cases)
    json_path, csv_path = output_paths(args.output_dir, args.output_stem)

    print("Exact paper structured run")
    print("Protocol: user-only prompt, no added system prompt, paper-native A/B/C/D labels")
    print(f"Dataset: {args.dataset}")
    print(f"Cases: {len(cases)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Runs: {args.runs}")
    print(f"Total planned trials: {len(cases) * len(args.models) * args.runs}")
    if not args.dry_run:
        print(f"Checkpoint JSON: {json_path}")
        print(f"Checkpoint CSV:  {csv_path}")

    existing = [] if args.dry_run else load_existing(json_path)
    by_key = {trial_key(row): row for row in existing}
    counter = 0
    total = len(cases) * len(args.models) * args.runs

    for model_name in args.models:
        for case in cases:
            for run_number in range(1, args.runs + 1):
                counter += 1
                key = (model_name, case["id"], run_number)
                if key in by_key and not by_key[key].error:
                    print(f"[{counter:03d}/{total:03d}] {model_name:22s} | {case['id']:7s} | run {run_number} (skip: already saved)")
                    continue

                print(f"[{counter:03d}/{total:03d}] {model_name:22s} | {case['id']:7s} | run {run_number}", flush=True)
                result = run_trial(case, model_name, run_number, args)
                by_key[key] = result
                if not args.dry_run:
                    if result.error:
                        print(f"  ERROR: {result.error[:180]}")
                    elif result.predicted_triage:
                        marker = "✓" if result.is_correct else "✗"
                        print(f"  {marker} predicted={result.predicted_triage}")
                    else:
                        print("  ? unclassified raw response")
                    ordered = [by_key[item] for item in sorted(by_key)]
                    save_results(ordered, json_path, csv_path)
                    if args.call_wait > 0:
                        time.sleep(args.call_wait)

    if args.dry_run:
        print("\nDry run only; no files written.")
        return

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
