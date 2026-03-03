#!/usr/bin/env python3
"""Compare constrained structured runs against natural-interaction runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from scipy.stats import binomtest, wilcoxon


RESULTS_DIR = Path(__file__).parent / "results"
STRUCTURED_PATH = RESULTS_DIR / "main_experiment_reconciled.csv"
NATURAL_PATH = RESULTS_DIR / "natural_interaction_5run_all_models_adjudicated.csv"

TARGET_MODELS = [
    "gpt-5.2-thinking-high",
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gemini-3-flash",
    "gemini-3.1-pro",
]
TARGET_FORMATS = ["patient_realistic", "patient_minimal"]


@dataclass
class RowComparison:
    source_model: str
    case_id: str
    prompt_format: str
    run_number: int
    gold_standard: str
    structured_correct: int
    structured_predicted: str
    natural_gpt_label: str
    natural_opus_label: str
    natural_gpt_correct: int
    natural_opus_correct: int
    natural_mean_correct: float
    natural_consensus_label: str
    natural_consensus_correct: Optional[int]
    judge_disagreement: int


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_bool_int(value: str) -> int:
    return 1 if str(value).lower() == "true" else 0


def build_row_comparisons() -> list[RowComparison]:
    structured_rows = [
        row for row in read_csv(STRUCTURED_PATH)
        if row["model"] in TARGET_MODELS
        and row["prompt_format"] in TARGET_FORMATS
        and row["is_correct"] != ""
    ]
    natural_rows = read_csv(NATURAL_PATH)

    structured_by_key = {
        (row["model"], row["case_id"], row["prompt_format"], int(row["run_number"])): row
        for row in structured_rows
    }
    natural_by_key = {
        (row["source_model"], row["case_id"], row["prompt_format"], int(row["run_number"])): row
        for row in natural_rows
    }

    if set(structured_by_key) != set(natural_by_key):
        missing_in_natural = sorted(set(structured_by_key) - set(natural_by_key))
        missing_in_structured = sorted(set(natural_by_key) - set(structured_by_key))
        raise SystemExit(
            "Key mismatch between structured and natural datasets.\n"
            f"Missing in natural: {missing_in_natural[:5]}\n"
            f"Missing in structured: {missing_in_structured[:5]}"
        )

    comparisons: list[RowComparison] = []
    for key in sorted(structured_by_key):
        s = structured_by_key[key]
        n = natural_by_key[key]
        gpt_correct = to_bool_int(n["gpt_5_2_thinking_high_is_correct"])
        opus_correct = to_bool_int(n["claude_opus_4_6_is_correct"])
        gpt_label = n["gpt_5_2_thinking_high_triage"]
        opus_label = n["claude_opus_4_6_triage"]
        consensus_label = gpt_label if gpt_label == opus_label else ""
        consensus_correct = None
        if consensus_label:
            consensus_correct = 1 if consensus_label == n["gold_standard"] else 0

        comparisons.append(
            RowComparison(
                source_model=s["model"],
                case_id=s["case_id"],
                prompt_format=s["prompt_format"],
                run_number=int(s["run_number"]),
                gold_standard=s["gold_standard"],
                structured_correct=to_bool_int(s["is_correct"]),
                structured_predicted=s["predicted_triage"],
                natural_gpt_label=gpt_label,
                natural_opus_label=opus_label,
                natural_gpt_correct=gpt_correct,
                natural_opus_correct=opus_correct,
                natural_mean_correct=(gpt_correct + opus_correct) / 2.0,
                natural_consensus_label=consensus_label,
                natural_consensus_correct=consensus_correct,
                judge_disagreement=0 if gpt_label == opus_label else 1,
            )
        )
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_cell_stats(comparisons: list[RowComparison]) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[RowComparison]] = {}
    for row in comparisons:
        key = (row.source_model, row.case_id, row.prompt_format)
        buckets.setdefault(key, []).append(row)

    cell_rows: list[dict] = []
    for key in sorted(buckets):
        items = buckets[key]
        structured_accuracy = sum(row.structured_correct for row in items) / len(items)
        natural_gpt_accuracy = sum(row.natural_gpt_correct for row in items) / len(items)
        natural_opus_accuracy = sum(row.natural_opus_correct for row in items) / len(items)
        natural_mean_accuracy = sum(row.natural_mean_correct for row in items) / len(items)
        if all(row.natural_consensus_correct is not None for row in items):
            consensus_accuracy = sum(int(row.natural_consensus_correct) for row in items) / len(items)
        else:
            consensus_accuracy = None
        cell_rows.append(
            {
                "source_model": key[0],
                "case_id": key[1],
                "prompt_format": key[2],
                "n_runs": len(items),
                "structured_accuracy": structured_accuracy,
                "natural_gpt_accuracy": natural_gpt_accuracy,
                "natural_opus_accuracy": natural_opus_accuracy,
                "natural_mean_accuracy": natural_mean_accuracy,
                "natural_consensus_accuracy": consensus_accuracy,
                "delta_mean_minus_structured": natural_mean_accuracy - structured_accuracy,
                "judge_disagreements": sum(row.judge_disagreement for row in items),
            }
        )
    return cell_rows


def summarize_by_model(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in TARGET_MODELS:
        rows = [row for row in comparisons if row.source_model == model]
        out[model] = {
            "n": len(rows),
            "structured_accuracy": sum(row.structured_correct for row in rows) / len(rows),
            "natural_gpt_accuracy": sum(row.natural_gpt_correct for row in rows) / len(rows),
            "natural_opus_accuracy": sum(row.natural_opus_correct for row in rows) / len(rows),
            "natural_mean_accuracy": sum(row.natural_mean_correct for row in rows) / len(rows),
            "judge_disagreements": sum(row.judge_disagreement for row in rows),
        }
    return out


def summarize_by_model_format(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in TARGET_MODELS:
        for prompt_format in TARGET_FORMATS:
            rows = [
                row for row in comparisons
                if row.source_model == model and row.prompt_format == prompt_format
            ]
            key = f"{model}::{prompt_format}"
            out[key] = {
                "n": len(rows),
                "structured_accuracy": sum(row.structured_correct for row in rows) / len(rows),
                "natural_gpt_accuracy": sum(row.natural_gpt_correct for row in rows) / len(rows),
                "natural_opus_accuracy": sum(row.natural_opus_correct for row in rows) / len(rows),
                "natural_mean_accuracy": sum(row.natural_mean_correct for row in rows) / len(rows),
                "judge_disagreements": sum(row.judge_disagreement for row in rows),
            }
    return out


def compute_wilcoxon(cell_rows: list[dict], field: str) -> dict:
    structured = [row["structured_accuracy"] for row in cell_rows]
    natural = [row[field] for row in cell_rows]
    result = wilcoxon(natural, structured, zero_method="wilcox", alternative="two-sided")
    deltas = [n - s for n, s in zip(natural, structured)]
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
    b = 0  # structured correct, natural incorrect
    c = 0  # structured incorrect, natural correct
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
    comparisons = build_row_comparisons()
    row_rows = [asdict(row) for row in comparisons]
    cell_rows = group_cell_stats(comparisons)

    write_csv(RESULTS_DIR / "natural_vs_structured_rowwise.csv", row_rows)
    write_csv(RESULTS_DIR / "natural_vs_structured_cell_summary.csv", cell_rows)

    report = {
        "comparison_scope": {
            "models": TARGET_MODELS,
            "prompt_formats": TARGET_FORMATS,
            "n_row_pairs": len(comparisons),
            "n_cell_pairs": len(cell_rows),
            "natural_scoring_primary": (
                "Mean of two adjudicators per response (0, 0.5, or 1), "
                "aggregated to matched cell accuracies over 5 runs."
            ),
        },
        "overall": {
            "structured_accuracy": sum(row.structured_correct for row in comparisons) / len(comparisons),
            "natural_gpt_accuracy": sum(row.natural_gpt_correct for row in comparisons) / len(comparisons),
            "natural_opus_accuracy": sum(row.natural_opus_correct for row in comparisons) / len(comparisons),
            "natural_mean_accuracy": sum(row.natural_mean_correct for row in comparisons) / len(comparisons),
            "judge_disagreements": sum(row.judge_disagreement for row in comparisons),
        },
        "by_model": summarize_by_model(comparisons),
        "by_model_and_format": summarize_by_model_format(comparisons),
        "wilcoxon": {
            "natural_mean_vs_structured": compute_wilcoxon(cell_rows, "natural_mean_accuracy"),
            "natural_gpt_vs_structured": compute_wilcoxon(cell_rows, "natural_gpt_accuracy"),
            "natural_opus_vs_structured": compute_wilcoxon(cell_rows, "natural_opus_accuracy"),
        },
        "mcnemar_exact_row_level": {
            "natural_gpt_vs_structured": compute_exact_mcnemar(comparisons, "natural_gpt_correct"),
            "natural_opus_vs_structured": compute_exact_mcnemar(comparisons, "natural_opus_correct"),
        },
    }

    report_path = RESULTS_DIR / "natural_vs_structured_comparison.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("Overall")
    print(f"  structured: {100 * report['overall']['structured_accuracy']:.1f}%")
    print(f"  natural (GPT judge): {100 * report['overall']['natural_gpt_accuracy']:.1f}%")
    print(f"  natural (Opus judge): {100 * report['overall']['natural_opus_accuracy']:.1f}%")
    print(f"  natural (two-judge mean): {100 * report['overall']['natural_mean_accuracy']:.1f}%")
    print(f"  judge disagreements: {report['overall']['judge_disagreements']}/{len(comparisons)}")

    print(f"\nWilcoxon ({len(cell_rows)} matched model x case x format cells)")
    for label, stats in report["wilcoxon"].items():
        print(
            f"  {label}: mean delta {stats['mean_delta']:+.3f}, "
            f"W={stats['statistic']:.3f}, p={stats['p_value']:.6g}, "
            f"+/-/0={stats['n_positive']}/{stats['n_negative']}/{stats['n_zero']}"
        )

    print("\nExact McNemar-style row-level discordance")
    for label, stats in report["mcnemar_exact_row_level"].items():
        print(
            f"  {label}: structured_only={stats['b_structured_only']}, "
            f"natural_only={stats['c_natural_only']}, "
            f"p={stats['exact_p_value']:.6g}"
        )

    print(f"\nSaved report: {report_path}")
    print(f"Saved rowwise CSV: {RESULTS_DIR / 'natural_vs_structured_rowwise.csv'}")
    print(f"Saved cell CSV: {RESULTS_DIR / 'natural_vs_structured_cell_summary.csv'}")


if __name__ == "__main__":
    main()
