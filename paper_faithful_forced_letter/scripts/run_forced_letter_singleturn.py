#!/usr/bin/env python3
"""Run the paper-faithful forced-letter single-turn study."""

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
from llm_utils import extract_triage_category, infer_free_text_triage_paper, triage_matches_gold  # noqa: E402
from run_natural_interaction import call_anthropic_natural, call_google_natural, call_openai_natural  # noqa: E402


DEFAULT_DATASET = WORKSPACE_DIR / "data" / "canonical_forced_letter_vignettes.json"
DEFAULT_RESULTS = WORKSPACE_DIR / "results"
DEFAULT_FORMATS = ["structured_forced_letter", "natural_forced_letter"]


@dataclass
class ForcedLetterTrialResult:
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
    raw_response: str
    is_correct: Optional[bool]
    error: Optional[str]
    latency_seconds: float
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-faithful forced-letter trials")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models to run")
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS, choices=DEFAULT_FORMATS, help="Input styles to run")
    parser.add_argument("--cases", nargs="+", default=None, help="Optional case ids to run")
    parser.add_argument("--runs", type=int, default=2, help="Repeated runs per model/case/format")
    parser.add_argument("--call-wait", type=float, default=1.0, help="Seconds to wait between API calls")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Forced-letter dataset JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS, help="Output directory")
    parser.add_argument("--output-stem", default=None, help="Fixed output stem (without extension)")
    parser.add_argument("--openai-max-completion-tokens", type=int, default=4096, help="Transport-level cap for OpenAI")
    parser.add_argument("--anthropic-max-tokens", type=int, default=1024, help="Transport-level cap for Anthropic")
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


def label_from_response(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None
    return extract_triage_category(raw_response) or infer_free_text_triage_paper(raw_response)


def trial_key(result: ForcedLetterTrialResult | dict) -> tuple[str, str, str, int]:
    return (
        str(result["model"] if isinstance(result, dict) else result.model),
        str(result["prompt_format"] if isinstance(result, dict) else result.prompt_format),
        str(result["case_id"] if isinstance(result, dict) else result.case_id),
        int(result["run_number"] if isinstance(result, dict) else result.run_number),
    )


def output_paths(output_dir: Path, output_stem: Optional[str]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"paper_forced_letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def save_results(results: list[ForcedLetterTrialResult], json_path: Path, csv_path: Path) -> None:
    rows = [asdict(result) for result in results]
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_existing(json_path: Path) -> list[ForcedLetterTrialResult]:
    if not json_path.exists():
        return []
    rows = json.loads(json_path.read_text())
    return [ForcedLetterTrialResult(**row) for row in rows]


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


def run_trial(case: dict, model_name: str, prompt_format: str, run_number: int,
              args: argparse.Namespace) -> ForcedLetterTrialResult:
    prompt = case[prompt_format]
    model_config = MODELS[model_name]

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name} | Case: {case['id']} | Format: {prompt_format} | Run: {run_number}")
        print("System prompt: <none>")
        print(f"User prompt: {prompt}")
        return ForcedLetterTrialResult(
            case_id=case["id"],
            case_title=case["title"],
            gold_standard=case["gold_standard_triage"],
            model=model_name,
            provider=model_config["provider"],
            prompt_format=prompt_format,
            system_prompt_version="none_user_only",
            run_number=run_number,
            source_user_message=prompt,
            predicted_triage=None,
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

    predicted = label_from_response(raw_response)
    is_correct = triage_matches_gold(predicted, case["gold_standard_triage"])

    return ForcedLetterTrialResult(
        case_id=case["id"],
        case_title=case["title"],
        gold_standard=case["gold_standard_triage"],
        model=model_name,
        provider=model_config["provider"],
        prompt_format=prompt_format,
        system_prompt_version="none_user_only",
        run_number=run_number,
        source_user_message=prompt,
        predicted_triage=predicted,
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

    total = len(cases) * len(args.models) * len(args.formats) * args.runs
    print("Paper-faithful forced-letter run")
    print("Protocol: user-only prompt, no added system prompt, paper-native A/B/C/D labels, one-letter output")
    print(f"Dataset: {args.dataset}")
    print(f"Cases: {len(cases)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Runs: {args.runs}")
    print(f"Total planned trials: {total}")
    if not args.dry_run:
        print(f"Checkpoint JSON: {json_path}")
        print(f"Checkpoint CSV:  {csv_path}")

    existing = [] if args.dry_run else load_existing(json_path)
    by_key = {trial_key(row): row for row in existing}
    counter = 0

    for model_name in args.models:
        for prompt_format in args.formats:
            for case in cases:
                for run_number in range(1, args.runs + 1):
                    counter += 1
                    key = (model_name, prompt_format, case["id"], run_number)
                    if key in by_key and not by_key[key].error:
                        print(
                            f"[{counter:03d}/{total:03d}] {model_name:22s} | {prompt_format:23s} | "
                            f"{case['id']:7s} | run {run_number} (skip: already saved)"
                        )
                        continue

                    print(
                        f"[{counter:03d}/{total:03d}] {model_name:22s} | {prompt_format:23s} | "
                        f"{case['id']:7s} | run {run_number}",
                        flush=True,
                    )
                    result = run_trial(case, model_name, prompt_format, run_number, args)
                    by_key[key] = result
                    if not args.dry_run:
                        if result.error:
                            print(f"  ERROR: {result.error[:180]}")
                        elif result.predicted_triage:
                            marker = "✓" if result.is_correct else "✗"
                            print(f"  {marker} predicted={result.predicted_triage}")
                        else:
                            print("  WARNING: no triage label parsed")
                        ordered = [by_key[k] for k in sorted(by_key)]
                        save_results(ordered, json_path, csv_path)

                    if args.call_wait > 0 and not args.dry_run:
                        time.sleep(args.call_wait)

    if not args.dry_run:
        print("\nFinished.")
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
