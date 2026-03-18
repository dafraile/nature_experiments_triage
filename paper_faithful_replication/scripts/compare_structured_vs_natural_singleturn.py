#!/usr/bin/env python3
"""Compare exact structured paper prompts against natural single-turn rewrites."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from scipy.stats import binomtest, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))
from llm_utils import triage_matches_gold  # noqa: E402


DEFAULT_JUDGES = ["gpt-5.4-xhigh", "claude-opus-4.6"]


@dataclass
class RowComparison:
    source_model: str
    case_id: str
    run_number: int
    gold_standard: str
    structured_correct: int
    structured_predicted: str
    natural_first_label: str
    natural_second_label: str
    natural_first_correct: int
    natural_second_correct: int
    natural_mean_correct: float
    natural_consensus_label: str
    natural_consensus_correct: Optional[int]
    judge_disagreement: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare structured vs natural single-turn paper-faithful runs")
    parser.add_argument("--structured", type=Path, required=True, help="Structured CSV from run_experiment.py")
    parser.add_argument("--natural", type=Path, required=True, help="Adjudicated natural CSV from adjudicate_natural_interaction.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison artifacts")
    parser.add_argument("--run-label", default="paper_faithful_singleturn", help="Filename label for output artifacts")
    parser.add_argument("--structured-format", default="original_structured", help="Structured prompt format label")
    parser.add_argument("--natural-format", default="patient_realistic", help="Natural prompt format label")
    parser.add_argument("--judge-models", nargs=2, default=DEFAULT_JUDGES, help="Two adjudicator model names in CSV column order")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def to_bool_int(value: str | bool | None) -> Optional[int]:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "true":
        return 1
    if lowered == "false":
        return 0
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_row_comparisons(args: argparse.Namespace) -> tuple[list[RowComparison], dict[str, int]]:
    first_prefix = sanitize_model_name(args.judge_models[0])
    second_prefix = sanitize_model_name(args.judge_models[1])

    structured_rows = [
        row for row in read_csv(args.structured)
        if row["prompt_format"] == args.structured_format
        and row.get("is_correct", "") != ""
    ]
    natural_rows = [
        row for row in read_csv(args.natural)
        if row["prompt_format"] == args.natural_format
        and row.get(f"{first_prefix}_triage")
        and row.get(f"{second_prefix}_triage")
    ]

    structured_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
        for row in structured_rows
    }
    natural_by_key = {
        (row["source_model"], row["case_id"], int(row["run_number"])): row
        for row in natural_rows
    }

    shared_keys = sorted(set(structured_by_key) & set(natural_by_key))
    diagnostics = {
        "structured_rows": len(structured_rows),
        "natural_rows": len(natural_rows),
        "shared_row_pairs": len(shared_keys),
        "missing_in_natural": len(set(structured_by_key) - set(natural_by_key)),
        "missing_in_structured": len(set(natural_by_key) - set(structured_by_key)),
    }

    comparisons: list[RowComparison] = []
    for key in shared_keys:
        s = structured_by_key[key]
        n = natural_by_key[key]
        first_label = n[f"{first_prefix}_triage"]
        second_label = n[f"{second_prefix}_triage"]
        first_correct = to_bool_int(n[f"{first_prefix}_is_correct"])
        second_correct = to_bool_int(n[f"{second_prefix}_is_correct"])
        if first_correct is None or second_correct is None:
            continue
        consensus_label = first_label if first_label == second_label else ""
        consensus_correct = None
        if consensus_label:
            consensus_match = triage_matches_gold(consensus_label, n["gold_standard"])
            consensus_correct = 1 if consensus_match else 0

        comparisons.append(
            RowComparison(
                source_model=s["model"],
                case_id=s["case_id"],
                run_number=int(s["run_number"]),
                gold_standard=s["gold_standard"],
                structured_correct=to_bool_int(s["is_correct"]) or 0,
                structured_predicted=s.get("predicted_triage", ""),
                natural_first_label=first_label,
                natural_second_label=second_label,
                natural_first_correct=first_correct,
                natural_second_correct=second_correct,
                natural_mean_correct=(first_correct + second_correct) / 2.0,
                natural_consensus_label=consensus_label,
                natural_consensus_correct=consensus_correct,
                judge_disagreement=0 if first_label == second_label else 1,
            )
        )

    diagnostics["paired_after_parse_filter"] = len(comparisons)
    return comparisons, diagnostics


def group_cell_stats(comparisons: list[RowComparison]) -> list[dict]:
    buckets: dict[tuple[str, str], list[RowComparison]] = {}
    for row in comparisons:
        buckets.setdefault((row.source_model, row.case_id), []).append(row)

    out: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key]
        structured_accuracy = sum(row.structured_correct for row in items) / len(items)
        natural_first_accuracy = sum(row.natural_first_correct for row in items) / len(items)
        natural_second_accuracy = sum(row.natural_second_correct for row in items) / len(items)
        natural_mean_accuracy = sum(row.natural_mean_correct for row in items) / len(items)
        consensus_values = [row.natural_consensus_correct for row in items if row.natural_consensus_correct is not None]
        out.append(
            {
                "source_model": key[0],
                "case_id": key[1],
                "n_runs": len(items),
                "structured_accuracy": structured_accuracy,
                "natural_first_accuracy": natural_first_accuracy,
                "natural_second_accuracy": natural_second_accuracy,
                "natural_mean_accuracy": natural_mean_accuracy,
                "natural_consensus_accuracy": (sum(consensus_values) / len(consensus_values)) if consensus_values else None,
                "delta_mean_minus_structured": natural_mean_accuracy - structured_accuracy,
                "judge_disagreements": sum(row.judge_disagreement for row in items),
            }
        )
    return out


def summarize_by_model(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in sorted({row.source_model for row in comparisons}):
        rows = [row for row in comparisons if row.source_model == model]
        out[model] = {
            "n": len(rows),
            "structured_accuracy": sum(row.structured_correct for row in rows) / len(rows),
            "natural_first_accuracy": sum(row.natural_first_correct for row in rows) / len(rows),
            "natural_second_accuracy": sum(row.natural_second_correct for row in rows) / len(rows),
            "natural_mean_accuracy": sum(row.natural_mean_correct for row in rows) / len(rows),
            "mean_delta": (
                sum(row.natural_mean_correct for row in rows) / len(rows)
                - sum(row.structured_correct for row in rows) / len(rows)
            ),
            "judge_disagreements": sum(row.judge_disagreement for row in rows),
        }
    return out


def compute_wilcoxon(cell_rows: list[dict], field: str) -> dict:
    structured = [row["structured_accuracy"] for row in cell_rows]
    natural = [row[field] for row in cell_rows]
    deltas = [n - s for n, s in zip(natural, structured)]
    if not any(delta != 0 for delta in deltas):
        return {
            "field": field,
            "n_cells": len(cell_rows),
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": len(cell_rows),
            "mean_delta": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }
    result = wilcoxon(natural, structured, zero_method="wilcox", alternative="two-sided")
    return {
        "field": field,
        "n_cells": len(cell_rows),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_negative": sum(1 for d in deltas if d < 0),
        "n_zero": sum(1 for d in deltas if d == 0),
        "mean_delta": sum(deltas) / len(deltas),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def compute_exact_mcnemar(comparisons: list[RowComparison], field: str) -> dict:
    b = 0
    c = 0
    for row in comparisons:
        natural_correct = getattr(row, field)
        if row.structured_correct == 1 and natural_correct == 0:
            b += 1
        elif row.structured_correct == 0 and natural_correct == 1:
            c += 1
    p_value = 1.0
    if b + c:
        p_value = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
    return {
        "field": field,
        "b_structured_only": b,
        "c_natural_only": c,
        "discordant_total": b + c,
        "exact_p_value": float(p_value),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparisons, diagnostics = build_row_comparisons(args)
    if not comparisons:
        raise SystemExit("No paired structured/natural rows were available for comparison.")

    row_rows = [asdict(row) for row in comparisons]
    cell_rows = group_cell_stats(comparisons)

    row_path = args.output_dir / f"{args.run_label}_rowwise.csv"
    cell_path = args.output_dir / f"{args.run_label}_cell_summary.csv"
    report_path = args.output_dir / f"{args.run_label}_comparison.json"

    write_csv(row_path, row_rows)
    write_csv(cell_path, cell_rows)

    report = {
        "comparison_scope": {
            "structured_file": str(args.structured),
            "natural_file": str(args.natural),
            "structured_format": args.structured_format,
            "natural_format": args.natural_format,
            "judges": args.judge_models,
            "n_row_pairs": len(comparisons),
            "n_cell_pairs": len(cell_rows),
        },
        "pairing_diagnostics": diagnostics,
        "overall": {
            "structured_accuracy": sum(row.structured_correct for row in comparisons) / len(comparisons),
            "natural_first_accuracy": sum(row.natural_first_correct for row in comparisons) / len(comparisons),
            "natural_second_accuracy": sum(row.natural_second_correct for row in comparisons) / len(comparisons),
            "natural_mean_accuracy": sum(row.natural_mean_correct for row in comparisons) / len(comparisons),
            "judge_disagreements": sum(row.judge_disagreement for row in comparisons),
        },
        "by_model": summarize_by_model(comparisons),
        "wilcoxon": {
            "natural_mean_vs_structured": compute_wilcoxon(cell_rows, "natural_mean_accuracy"),
            "natural_first_vs_structured": compute_wilcoxon(cell_rows, "natural_first_accuracy"),
            "natural_second_vs_structured": compute_wilcoxon(cell_rows, "natural_second_accuracy"),
        },
        "mcnemar_exact_row_level": {
            "natural_first_vs_structured": compute_exact_mcnemar(comparisons, "natural_first_correct"),
            "natural_second_vs_structured": compute_exact_mcnemar(comparisons, "natural_second_correct"),
        },
    }

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("Overall")
    print(f"  structured: {100 * report['overall']['structured_accuracy']:.1f}%")
    print(f"  natural ({args.judge_models[0]} judge): {100 * report['overall']['natural_first_accuracy']:.1f}%")
    print(f"  natural ({args.judge_models[1]} judge): {100 * report['overall']['natural_second_accuracy']:.1f}%")
    print(f"  natural (two-judge mean): {100 * report['overall']['natural_mean_accuracy']:.1f}%")
    print(f"  judge disagreements: {report['overall']['judge_disagreements']}/{len(comparisons)}")
    print(f"Saved rowwise CSV: {row_path}")
    print(f"Saved cell CSV: {cell_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
