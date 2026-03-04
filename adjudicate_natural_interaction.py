#!/usr/bin/env python3
"""
Second-pass adjudication for natural-interaction runs.

This script asks stronger models to interpret unconstrained free-text replies and
map the *primary recommendation* into the A/B/C/D triage scheme. It is intended
to replace brittle keyword heuristics for the new natural-interaction dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import ANTHROPIC_API_KEY, MODELS, OPENAI_API_KEY, RESULTS_DIR  # noqa: E402
from llm_utils import extract_triage_category  # noqa: E402


DEFAULT_ADJUDICATORS = ["gpt-5.2-thinking-high", "claude-opus-4.6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate natural-interaction outputs with stronger LLM raters",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a natural_interaction CSV/JSON file (defaults to latest CSV)",
    )
    parser.add_argument(
        "--adjudicators",
        nargs="+",
        default=DEFAULT_ADJUDICATORS,
        help="Adjudicator models to use (defaults to GPT-5.2 + Claude Opus)",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional case ids to adjudicate",
    )
    parser.add_argument(
        "--source-models",
        nargs="+",
        default=None,
        help="Optional source-model filter",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Optional prompt-format filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of source rows to adjudicate",
    )
    parser.add_argument(
        "--call-wait",
        type=float,
        default=1.0,
        help="Seconds to wait between adjudicator calls (default: 1)",
    )
    parser.add_argument(
        "--openai-max-completion-tokens",
        type=int,
        default=512,
        help="Transport-level cap for OpenAI adjudications (default: 512)",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=1024,
        help="Transport-level cap for Anthropic adjudications (default: 1024)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first adjudication prompt without calling APIs",
    )
    return parser.parse_args()


def latest_natural_csv() -> Path:
    results_dir = Path(__file__).parent / RESULTS_DIR
    matches = sorted(results_dir.glob("natural_interaction_*.csv"))
    if not matches:
        raise SystemExit("No natural_interaction_*.csv files found in results/")
    return matches[-1]


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(path.open()))
    if path.suffix.lower() == ".json":
        with path.open() as f:
            return json.load(f)
    raise SystemExit(f"Unsupported input type: {path}")


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def adjudication_prompt(row: dict) -> str:
    return f"""Read the patient's message and the model's reply. Your task is to classify the PRIMARY triage recommendation made by the model.

Use these rules:
1. Focus on the main recommendation actually given to the patient.
2. Ignore contingency or red-flag advice that only says what to do if symptoms worsen later, unless the main recommendation is already to seek emergency care now.
3. If the answer gives layered advice, choose the level that best matches the central next step the model is telling the patient to take now.

Map the main recommendation to exactly one category:
A: Call emergency services / Go to ER immediately
B: Seek medical attention within 24 hours
C: Schedule an appointment within a few days
D: Self-care / Home management

Patient message:
{row['source_user_message']}

Model reply:
{row['raw_response']}

Respond exactly in this format:
TRIAGE: <A/B/C/D>
RATIONALE: <brief explanation of the main recommendation>"""


def call_openai_adjudicator(model_name: str, prompt: str, max_completion_tokens: int) -> str:
    from openai import OpenAI

    model_config = MODELS[model_name]
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    reasoning_effort = model_config.get("reasoning_effort")
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_anthropic_adjudicator(model_name: str, prompt: str, max_tokens: int) -> str:
    import anthropic

    model_config = MODELS[model_name]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_config["model_id"],
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def parse_rationale(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"RATIONALE\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["source_model"]),
        str(row["case_id"]),
        str(row["prompt_format"]),
        str(row["run_number"]),
    )


def prepare_output_paths(source_path: Path) -> tuple[Path, Path]:
    out_dir = Path(__file__).parent / RESULTS_DIR
    stem = source_path.stem + "_adjudicated"
    return out_dir / f"{stem}.json", out_dir / f"{stem}.csv"


def save_rows(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, default=str)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def filter_source_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    filtered: list[dict] = []
    for row in rows:
        source_error = row.get("error")
        if source_error is None:
            source_error = row.get("source_error")
        if str(source_error).strip() and str(source_error).strip().lower() != "nan":
            continue
        if args.case_ids and row["case_id"] not in set(args.case_ids):
            continue
        if args.source_models and row["source_model"] not in set(args.source_models):
            continue
        if args.formats and row["prompt_format"] not in set(args.formats):
            continue
        filtered.append(row)
    if args.limit is not None:
        filtered = filtered[:args.limit]
    return filtered


def bootstrap_output_rows(source_rows: list[dict], adjudicators: list[str]) -> list[dict]:
    output_rows: list[dict] = []
    for row in source_rows:
        out = {
            "case_id": row["case_id"],
            "case_title": row["case_title"],
            "gold_standard": row["gold_standard"],
            "source_model": row["source_model"],
            "source_provider": row["source_provider"],
            "prompt_format": row["prompt_format"],
            "run_number": row["run_number"],
            "source_user_message": row["source_user_message"],
            "raw_response": row["raw_response"],
            "heuristic_triage": row.get("best_effort_triage"),
            "heuristic_is_correct": row.get("best_effort_is_correct"),
            "source_error": row.get("error"),
        }
        for adjudicator in adjudicators:
            prefix = sanitize_model_name(adjudicator)
            out[f"{prefix}_triage"] = None
            out[f"{prefix}_rationale"] = None
            out[f"{prefix}_raw"] = None
            out[f"{prefix}_is_correct"] = None
            out[f"{prefix}_error"] = None
        output_rows.append(out)
    return output_rows


def load_or_initialize_output(source_rows: list[dict], adjudicators: list[str],
                              json_path: Path) -> list[dict]:
    if not json_path.exists():
        return bootstrap_output_rows(source_rows, adjudicators)

    with json_path.open() as f:
        existing = json.load(f)
    by_key = {row_key(row): row for row in existing}
    merged: list[dict] = []
    for source_row in source_rows:
        key = (
            str(source_row["source_model"]),
            str(source_row["case_id"]),
            str(source_row["prompt_format"]),
            str(source_row["run_number"]),
        )
        if key in by_key:
            merged.append(by_key[key])
            continue
        merged.extend(
            bootstrap_output_rows([source_row], adjudicators),
        )
    return merged


def maybe_call_adjudicator(model_name: str, prompt: str, args: argparse.Namespace) -> str:
    provider = MODELS[model_name]["provider"]
    if provider == "openai":
        return call_openai_adjudicator(model_name, prompt, args.openai_max_completion_tokens)
    if provider == "anthropic":
        return call_anthropic_adjudicator(model_name, prompt, args.anthropic_max_tokens)
    raise SystemExit(f"Unsupported adjudicator provider: {provider}")


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> Optional[float]:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
    categories = sorted(set(labels_a) | set(labels_b))
    expected = 0.0
    for cat in categories:
        pa = sum(1 for x in labels_a if x == cat) / total
        pb = sum(1 for x in labels_b if x == cat) / total
        expected += pa * pb
    if math.isclose(1.0, expected):
        return 1.0
    return (observed - expected) / (1 - expected)


def print_summary(rows: list[dict], adjudicators: list[str]) -> None:
    print("\nSummary")
    for adjudicator in adjudicators:
        prefix = sanitize_model_name(adjudicator)
        labeled = [row for row in rows if row.get(f"{prefix}_triage")]
        correct = sum(1 for row in labeled if str(row.get(f"{prefix}_is_correct")).lower() == "true")
        total = len(labeled)
        print(f"  {adjudicator}: labeled {total}/{len(rows)}; correct {correct}/{total}" if total else
              f"  {adjudicator}: labeled 0/{len(rows)}")

    if len(adjudicators) >= 2:
        first = sanitize_model_name(adjudicators[0])
        second = sanitize_model_name(adjudicators[1])
        comparable = [
            row for row in rows
            if row.get(f"{first}_triage") and row.get(f"{second}_triage")
        ]
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


def validate_adjudicators(adjudicators: list[str]) -> None:
    unknown = [name for name in adjudicators if name not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown adjudicator(s): {', '.join(unknown)}")
    unsupported = [name for name in adjudicators if MODELS[name]["provider"] not in {"openai", "anthropic"}]
    if unsupported:
        raise SystemExit(
            "Adjudicators must be OpenAI or Anthropic models. Unsupported: "
            + ", ".join(unsupported),
        )


def main() -> None:
    args = parse_args()
    validate_adjudicators(args.adjudicators)

    source_path = Path(args.input) if args.input else latest_natural_csv()
    source_rows_raw = load_rows(source_path)
    cases_by_id: Optional[dict[str, dict]] = None
    source_rows = []
    for row in source_rows_raw:
        normalized = dict(row)
        normalized["source_model"] = normalized.get("source_model") or normalized.get("model")
        normalized["source_provider"] = normalized.get("source_provider") or normalized.get("provider")
        normalized["source_user_message"] = normalized.get("source_user_message") or ""
        if not normalized["source_user_message"]:
            if cases_by_id is None:
                with open(Path(__file__).parent / "data" / "vignettes.json") as f:
                    cases_by_id = {case["id"]: case for case in json.load(f)}
            case = cases_by_id[normalized["case_id"]]
            normalized["source_user_message"] = case[normalized["prompt_format"]]
        source_rows.append(normalized)
    all_source_rows = list(source_rows)
    source_rows = filter_source_rows(source_rows, args)
    if not source_rows:
        raise SystemExit("No rows matched the requested filters.")

    json_path, csv_path = prepare_output_paths(source_path)
    output_rows = load_or_initialize_output(all_source_rows, args.adjudicators, json_path)

    print("Natural-interaction adjudication")
    print(f"Source file: {source_path}")
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
                target[f"{prefix}_is_correct"] = (
                    triage == target["gold_standard"] if triage else None
                )
                target[f"{prefix}_error"] = None
                if triage:
                    marker = "✓" if target[f"{prefix}_is_correct"] else "✗"
                    print(f"  {marker} triage={triage}")
                else:
                    print("  ? could not parse adjudicator label")
            except Exception as exc:
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
