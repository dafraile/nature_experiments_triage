#!/usr/bin/env python3
"""
Targeted replication of the paper's released user-facing prompts for the two
headline emergency failure cases (asthma and DKA).

This runner:
- loads the exact released `prompt_text` rows from DataOriginal_FINAL.csv
- focuses on the two emergency scenarios only
- generates controlled prompt ablations that remove specific confounds
- calls the same model backends used elsewhere in this repo
- saves outputs in an adjudication-friendly format compatible with the
  natural-interaction second-pass scorer
"""

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

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, NUM_RUNS, RESULTS_DIR  # noqa: E402
from llm_utils import extract_triage_category, infer_free_text_triage  # noqa: E402
from run_natural_interaction import (  # noqa: E402
    call_anthropic_natural,
    call_google_natural,
    call_openai_natural,
)


DEFAULT_DIAGNOSES = ["Asthma exacerbation", "DKA"]
DEFAULT_VARIANTS = ["WW", "WW-AX"]
DEFAULT_CONDITIONS = [
    "paper_exact",
    "no_base_only",
    "no_no_questions",
    "no_context",
    "no_schema",
    "fully_deconfounded",
    "natural_ask",
    "user_only",
]

PAPER_SCALE_CONDITIONS = {
    "paper_exact",
    "no_base_only",
    "no_no_questions",
    "no_context",
}

PAPER_TO_INTERNAL = {
    "A": "D",
    "B": "C",
    "C": "B",
    "D": "A",
}


@dataclass
class PaperFailureResult:
    case_id: str
    case_title: str
    gold_standard: str
    source_model: str
    source_provider: str
    prompt_format: str
    run_number: int
    source_user_message: str
    raw_response: str
    best_effort_triage: Optional[str]
    best_effort_is_correct: Optional[bool]
    error: Optional[str]
    paper_case_id: str
    paper_case_pair: str
    paper_variant_code: str
    paper_prompt_type: int
    source_diagnosis: str
    source_anchor_type: str
    source_barrier_type: str
    paper_scale_triage: Optional[str]
    source_condition: str
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt-faithful paper subset experiments on asthma/DKA",
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Path to the paper's DataOriginal_FINAL.csv file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.2-thinking-high"],
        help="Models to run",
    )
    parser.add_argument(
        "--diagnoses",
        nargs="+",
        default=DEFAULT_DIAGNOSES,
        help="Diagnoses to include (defaults: asthma + DKA)",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional specific paper case ids (for example E9 F9 E13 F13)",
    )
    parser.add_argument(
        "--variant-codes",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Factorial variants to include (defaults: WW and WW-AX)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=DEFAULT_CONDITIONS,
        help="Prompt conditions to run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeated runs per model/source-condition cell (default: 1)",
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
        help="Optional transport-level fallback for OpenAI",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=None,
        help="Required only if Claude models are included",
    )
    parser.add_argument(
        "--google-vertex",
        action="store_true",
        help="Use Vertex Express for Google models",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional fixed output stem (without extension) for resumable runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling APIs",
    )
    return parser.parse_args()


def sanitize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_blank_lines(text: str) -> str:
    lines = [line.rstrip() for line in sanitize(text).split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            cleaned.append(line.strip())
            blank = False
            continue
        if not blank and cleaned:
            cleaned.append("")
            blank = True
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned)


def remove_exact_line(text: str, target: str) -> str:
    target = sanitize(target).strip()
    kept = [line for line in sanitize(text).split("\n") if line.strip() != target]
    return "\n".join(kept)


def remove_schema_block(text: str) -> str:
    marker = "Please answer in exactly this format:"
    if marker not in text:
        return text.strip()
    prefix = text.split(marker, 1)[0].rstrip()
    suffix = (
        "Please explain in plain language what you think is going on and "
        "what I should do next."
    )
    return f"{prefix}\n\n{suffix}"


def extract_user_only_prompt(text: str, row: dict, keep_opening: bool = False) -> str:
    prompt = sanitize(text)
    if not keep_opening:
        prompt = remove_exact_line(
            prompt,
            "I'm looking for help understanding what these results might mean and how urgently I should act.",
        )
    prompt = remove_exact_line(
        prompt,
        "Please base your answer only on the information in this message.",
    )
    if str(row.get("anchor_text", "")).strip() and str(row.get("anchor_text", "")).lower() != "nan":
        prompt = remove_exact_line(prompt, str(row["anchor_text"]))
    if str(row.get("barrier_text", "")).strip() and str(row.get("barrier_text", "")).lower() != "nan":
        prompt = remove_exact_line(prompt, str(row["barrier_text"]))

    marker = "Please answer in exactly this format:"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0].rstrip()
    prompt = remove_exact_line(
        prompt,
        "Please explain in plain language what you think is going on and what I should do next.",
    )

    cleaned_lines: list[str] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "My situation:":
            continue
        if line.startswith("About me:"):
            line = line.replace("About me:", "", 1).strip()
        if line.startswith("- "):
            line = line[2:].strip()
        cleaned_lines.append(line)

    return collapse_blank_lines("\n".join(cleaned_lines))


def build_condition_prompt(row: dict, condition: str) -> str:
    prompt = sanitize(str(row["prompt_text"]))
    if condition == "paper_exact":
        return collapse_blank_lines(prompt)

    if condition == "no_base_only":
        prompt = remove_exact_line(
            prompt,
            "Please base your answer only on the information in this message.",
        )
        return collapse_blank_lines(prompt)

    if condition == "no_no_questions":
        prompt = remove_exact_line(
            prompt,
            "If you would normally ask clarifying questions, still choose the best TRIAGE option using what's here.",
        )
        return collapse_blank_lines(prompt)

    if condition == "no_context":
        if str(row.get("anchor_text", "")).strip() and str(row.get("anchor_text", "")).lower() != "nan":
            prompt = remove_exact_line(prompt, str(row["anchor_text"]))
        if str(row.get("barrier_text", "")).strip() and str(row.get("barrier_text", "")).lower() != "nan":
            prompt = remove_exact_line(prompt, str(row["barrier_text"]))
        return collapse_blank_lines(prompt)

    if condition == "no_schema":
        return collapse_blank_lines(remove_schema_block(prompt))

    if condition == "fully_deconfounded":
        prompt = remove_exact_line(
            prompt,
            "Please base your answer only on the information in this message.",
        )
        if str(row.get("anchor_text", "")).strip() and str(row.get("anchor_text", "")).lower() != "nan":
            prompt = remove_exact_line(prompt, str(row["anchor_text"]))
        if str(row.get("barrier_text", "")).strip() and str(row.get("barrier_text", "")).lower() != "nan":
            prompt = remove_exact_line(prompt, str(row["barrier_text"]))
        prompt = remove_schema_block(prompt)
        return collapse_blank_lines(prompt)

    if condition == "user_only":
        return extract_user_only_prompt(prompt, row)

    if condition == "natural_ask":
        return extract_user_only_prompt(prompt, row, keep_opening=True)

    raise ValueError(f"Unsupported condition: {condition}")


def read_source_rows(path: Path, args: argparse.Namespace) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    diagnoses = set(args.diagnoses)
    case_ids = set(args.case_ids) if args.case_ids else None
    variant_codes = set(args.variant_codes)

    filtered: list[dict] = []
    for row in rows:
        if row["diagnosis"] not in diagnoses:
            continue
        if case_ids and row["case_id"] not in case_ids:
            continue
        if row["variant_code"] not in variant_codes:
            continue
        filtered.append(row)

    filtered.sort(
        key=lambda r: (
            r["diagnosis"],
            r["case_id"],
            r["variant_code"],
        )
    )
    if not filtered:
        raise SystemExit("No rows matched the requested diagnoses/case ids/variant codes.")
    return filtered


def paper_scale_to_internal(letter: Optional[str]) -> Optional[str]:
    if not letter:
        return None
    return PAPER_TO_INTERNAL.get(letter.upper())


def map_gold_to_internal(gold: str) -> str:
    letter = gold.strip().upper()
    if letter not in PAPER_TO_INTERNAL:
        raise SystemExit(f"Only clear-case single-letter gold labels are supported here; got {gold}")
    mapped = paper_scale_to_internal(letter)
    if not mapped:
        raise SystemExit(f"Could not map gold label: {gold}")
    return mapped


def label_response(raw_response: str, condition: str) -> tuple[Optional[str], Optional[str]]:
    if not raw_response:
        return None, None

    if condition in PAPER_SCALE_CONDITIONS:
        paper_label = extract_triage_category(raw_response)
        return paper_scale_to_internal(paper_label), paper_label

    return infer_free_text_triage(raw_response), None


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


def result_key(row: dict) -> tuple[str, str, str, str, int]:
    return (
        row["source_model"],
        row["paper_case_id"],
        row["paper_variant_code"],
        row["source_condition"],
        int(row["run_number"]),
    )


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = args.output_stem
    if not stem:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"paper_failure_cases_{stamp}"
    out_dir = Path(__file__).parent / RESULTS_DIR
    return out_dir / f"{stem}.json", out_dir / f"{stem}.csv"


def save_results(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, default=str)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_existing(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    with json_path.open() as f:
        return json.load(f)


def is_completed_row(row: dict) -> bool:
    return not row.get("error")


def planned_cells(source_rows: list[dict], args: argparse.Namespace) -> list[tuple[str, dict, str, int]]:
    cells: list[tuple[str, dict, str, int]] = []
    for model_name in args.models:
        if model_name not in MODELS:
            raise SystemExit(f"Unknown model: {model_name}")
        for row in source_rows:
            for condition in args.conditions:
                for run_number in range(1, args.runs + 1):
                    cells.append((model_name, row, condition, run_number))
    return cells


def dry_run_preview(cells: list[tuple[str, dict, str, int]]) -> None:
    print(f"Planned cells: {len(cells)}")
    for idx, (model_name, row, condition, run_number) in enumerate(cells[:3], start=1):
        prompt = build_condition_prompt(row, condition)
        print("\n" + "=" * 70)
        print(
            f"[{idx}] {model_name} | {row['diagnosis']} | {row['case_id']} | "
            f"{row['variant_code']} | {condition} | run {run_number}"
        )
        print(prompt)
    if len(cells) > 3:
        print(f"\n... {len(cells) - 3} additional planned cells not shown")


def main() -> None:
    args = parse_args()
    source_rows = read_source_rows(Path(args.source_csv), args)
    cells = planned_cells(source_rows, args)

    if args.dry_run:
        dry_run_preview(cells)
        return

    json_path, csv_path = output_paths(args)
    results = load_existing(json_path)
    by_key = {result_key(row): row for row in results}

    print(
        "Running paper failure-case subset:",
        f"{len(source_rows)} source rows, {len(args.conditions)} conditions,",
        f"{len(args.models)} model(s), {args.runs} run(s) each",
    )

    completed = 0
    for idx, (model_name, row, condition, run_number) in enumerate(cells, start=1):
        key = (
            model_name,
            row["case_id"],
            row["variant_code"],
            condition,
            run_number,
        )
        existing = by_key.get(key)
        if existing and is_completed_row(existing):
            continue

        prompt = build_condition_prompt(row, condition)
        gold = map_gold_to_internal(row["gold_triage"])
        model_config = MODELS[model_name]
        print(
            f"[{idx}/{len(cells)}] {model_name} | {row['case_id']} | "
            f"{row['variant_code']} | {condition} | run {run_number}"
        )

        raw_response = ""
        error = None
        try:
            raw_response = call_model(model_name, prompt, args)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        best_effort_triage, paper_scale_triage = label_response(raw_response, condition)
        is_correct = None
        if best_effort_triage:
            is_correct = best_effort_triage == gold

        result = PaperFailureResult(
            case_id=row["case_id"],
            case_title=row["diagnosis"],
            gold_standard=gold,
            source_model=model_name,
            source_provider=model_config["provider"],
            prompt_format=condition,
            run_number=run_number,
            source_user_message=prompt,
            raw_response=raw_response,
            best_effort_triage=best_effort_triage,
            best_effort_is_correct=is_correct,
            error=error,
            paper_case_id=row["case_id"],
            paper_case_pair=row["case_pair"],
            paper_variant_code=row["variant_code"],
            paper_prompt_type=int(row["prompt_type"]),
            source_diagnosis=row["diagnosis"],
            source_anchor_type=row["anchor_type"],
            source_barrier_type=row["barrier_type"],
            paper_scale_triage=paper_scale_triage,
            source_condition=condition,
            timestamp=datetime.now().isoformat(),
        )
        by_key[key] = asdict(result)
        results = list(by_key.values())
        save_results(results, json_path, csv_path)

        if error:
            print(f"  error: {error}")
        else:
            print(
                f"  label={best_effort_triage or 'unclassified'}",
                f"(paper={paper_scale_triage})" if paper_scale_triage else "",
            )
        completed += 1
        time.sleep(args.call_wait)

    print(f"\nCompleted new rows: {completed}")
    print(f"Results saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
