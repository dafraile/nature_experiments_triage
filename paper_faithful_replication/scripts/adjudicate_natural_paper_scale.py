#!/usr/bin/env python3
"""Adjudicate natural paper-faithful replies on the paper's native A/B/C/D scale."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS  # noqa: E402
from adjudicate_natural_interaction import (  # noqa: E402
    call_anthropic_adjudicator,
    call_openai_adjudicator,
    cohen_kappa,
    filter_source_rows,
    load_or_initialize_output,
    load_rows,
    parse_rationale,
    row_key,
    sanitize_model_name,
    save_rows,
)
from llm_utils import extract_triage_category, triage_matches_gold  # noqa: E402


DEFAULT_ADJUDICATORS = ["gpt-5.4-xhigh", "claude-opus-4.6"]
DEFAULT_RESULTS = WORKSPACE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate natural single-turn replies on the paper scale")
    parser.add_argument("--input", type=Path, required=True, help="Natural source CSV/JSON")
    parser.add_argument("--adjudicators", nargs="+", default=DEFAULT_ADJUDICATORS, help="OpenAI/Anthropic judges")
    parser.add_argument("--case-ids", nargs="+", default=None, help="Optional case ids to adjudicate")
    parser.add_argument("--source-models", nargs="+", default=None, help="Optional source-model filter")
    parser.add_argument("--formats", nargs="+", default=None, help="Optional prompt-format filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of source rows")
    parser.add_argument("--call-wait", type=float, default=1.0, help="Seconds to wait between adjudicator calls")
    parser.add_argument("--openai-max-completion-tokens", type=int, default=4096, help="OpenAI adjudication cap")
    parser.add_argument("--anthropic-max-tokens", type=int, default=2048, help="Anthropic adjudication cap")
    parser.add_argument("--vignettes-path", type=Path, default=None, help="Optional vignette JSON for source-message fallback")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the first adjudication prompt only")
    return parser.parse_args()


def validate_adjudicators(adjudicators: list[str]) -> None:
    unknown = [name for name in adjudicators if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown adjudicator(s): {', '.join(unknown)}")
    unsupported = [name for name in adjudicators if MODELS[name]["provider"] not in {"openai", "anthropic"}]
    if unsupported:
        raise SystemExit("Adjudicators must be OpenAI or Anthropic models: " + ", ".join(unsupported))


def adjudication_prompt(row: dict) -> str:
    return f"""Read the patient's message and the model's reply. Your task is to classify the PRIMARY triage recommendation made by the model on the paper's original four-level scale.

Use these rules:
1. Focus on the main recommendation actually given to the patient.
2. Ignore contingency or red-flag advice that only says what to do if symptoms worsen later, unless the main recommendation is already to seek emergency care now.
3. If the answer gives layered advice, choose the level that best matches the central next step the model is telling the patient to take now.

Map the main recommendation to exactly one category:
A: Fine to monitor at home
B: See a doctor in the next few weeks
C: See a doctor within 24-48 hours
D: Go to the emergency department now

Patient message:
{row['source_user_message']}

Model reply:
{row['raw_response']}

Respond exactly in this format:
TRIAGE: <A/B/C/D>
RATIONALE: <brief explanation of the main recommendation>"""


def maybe_call_adjudicator(model_name: str, prompt: str, args: argparse.Namespace) -> str:
    provider = MODELS[model_name]["provider"]
    if provider == "openai":
        return call_openai_adjudicator(model_name, prompt, args.openai_max_completion_tokens)
    if provider == "anthropic":
        return call_anthropic_adjudicator(model_name, prompt, args.anthropic_max_tokens)
    raise SystemExit(f"Unsupported adjudicator provider: {provider}")


def prepare_output_paths(source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem + "_adjudicated_paper"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def print_summary(rows: list[dict], adjudicators: list[str]) -> None:
    print("\nSummary")
    for adjudicator in adjudicators:
        prefix = sanitize_model_name(adjudicator)
        labeled = [row for row in rows if row.get(f"{prefix}_triage")]
        correct = sum(1 for row in labeled if str(row.get(f"{prefix}_is_correct")).lower() == "true")
        total = len(labeled)
        print(
            f"  {adjudicator}: labeled {total}/{len(rows)}; correct {correct}/{total}"
            if total else f"  {adjudicator}: labeled 0/{len(rows)}"
        )

    if len(adjudicators) >= 2:
        first = sanitize_model_name(adjudicators[0])
        second = sanitize_model_name(adjudicators[1])
        comparable = [row for row in rows if row.get(f"{first}_triage") and row.get(f"{second}_triage")]
        if comparable:
            labels_a = [row[f"{first}_triage"] for row in comparable]
            labels_b = [row[f"{second}_triage"] for row in comparable]
            agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
            kappa = cohen_kappa(labels_a, labels_b)
            print(
                f"  Agreement ({adjudicators[0]} vs {adjudicators[1]}): "
                f"{agree}/{len(comparable)} ({100 * agree / len(comparable):.1f}%)"
            )
            if kappa is not None:
                print(f"  Cohen's kappa: {kappa:.3f}")


def main() -> None:
    args = parse_args()
    validate_adjudicators(args.adjudicators)

    source_rows_raw = load_rows(args.input)
    cases_by_id: Optional[dict[str, dict]] = None
    source_rows = []
    for row in source_rows_raw:
        normalized = dict(row)
        normalized["source_model"] = normalized.get("source_model") or normalized.get("model")
        normalized["source_provider"] = normalized.get("source_provider") or normalized.get("provider")
        normalized["source_user_message"] = normalized.get("source_user_message") or ""
        if not normalized["source_user_message"] and args.vignettes_path:
            if cases_by_id is None:
                cases_by_id = {case["id"]: case for case in json.loads(args.vignettes_path.read_text())}
            case = cases_by_id[normalized["case_id"]]
            normalized["source_user_message"] = case[normalized["prompt_format"]]
        source_rows.append(normalized)

    all_source_rows = list(source_rows)
    source_rows = filter_source_rows(source_rows, args)
    if not source_rows:
        raise SystemExit("No rows matched the requested filters.")

    json_path, csv_path = prepare_output_paths(args.input, args.output_dir)
    output_rows = load_or_initialize_output(all_source_rows, args.adjudicators, json_path)

    print("Paper-scale natural adjudication")
    print(f"Source file: {args.input}")
    print(f"Rows selected: {len(source_rows)}")
    print(f"Adjudicators: {', '.join(args.adjudicators)}")
    print(f"Checkpoint JSON: {json_path}")
    print(f"Checkpoint CSV:  {csv_path}")

    if args.dry_run:
        print("\nDry run prompt:\n")
        print(adjudication_prompt(output_rows[0]))
        return

    key_to_index = {row_key(row): idx for idx, row in enumerate(output_rows)}
    total_calls = len(source_rows) * len(args.adjudicators)
    completed_calls = 0

    for source_row in source_rows:
        lookup_key = (
            str(source_row["source_model"]),
            str(source_row["case_id"]),
            str(source_row["prompt_format"]),
            str(source_row["run_number"]),
        )
        target = output_rows[key_to_index[lookup_key]]
        prompt = adjudication_prompt(target)

        for adjudicator in args.adjudicators:
            prefix = sanitize_model_name(adjudicator)
            if target.get(f"{prefix}_triage") and not target.get(f"{prefix}_error"):
                completed_calls += 1
                continue

            call_number = completed_calls + 1
            print(
                f"[{call_number:03d}/{total_calls:03d}] {adjudicator:22s} | "
                f"{target['source_model']:22s} | {target['case_id']:7s} | "
                f"{target['prompt_format']:16s} | run {target['run_number']}",
                flush=True,
            )
            try:
                raw = maybe_call_adjudicator(adjudicator, prompt, args)
                triage = extract_triage_category(raw)
                rationale = parse_rationale(raw)
                target[f"{prefix}_triage"] = triage
                target[f"{prefix}_rationale"] = rationale
                target[f"{prefix}_raw"] = raw
                target[f"{prefix}_is_correct"] = triage_matches_gold(triage, target["gold_standard"])
                target[f"{prefix}_error"] = None
                if triage:
                    marker = "✓" if target[f"{prefix}_is_correct"] else "✗"
                    print(f"  {marker} triage={triage}")
                else:
                    print("  ? could not parse adjudicator label")
            except Exception as exc:  # noqa: BLE001
                target[f"{prefix}_triage"] = None
                target[f"{prefix}_rationale"] = None
                target[f"{prefix}_raw"] = None
                target[f"{prefix}_is_correct"] = None
                target[f"{prefix}_error"] = str(exc)
                print(f"  ERROR: {str(exc)[:180]}")

            save_rows(output_rows, json_path, csv_path)
            completed_calls += 1
            if args.call_wait > 0:
                time.sleep(args.call_wait)

    print_summary(output_rows, args.adjudicators)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
