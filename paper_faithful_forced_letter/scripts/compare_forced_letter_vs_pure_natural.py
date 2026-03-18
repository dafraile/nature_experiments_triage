#!/usr/bin/env python3
"""Compare natural forced-letter outputs against the faithful free-text natural run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from scipy.stats import binomtest, wilcoxon


DEFAULT_JUDGES = ["gpt-5.4-xhigh", "claude-opus-4.6"]


@dataclass
class RowComparison:
    source_model: str
    case_id: str
    run_number: int
    gold_standard: str
    forced_correct: int
    forced_predicted: str
    pure_first_label: str
    pure_second_label: str
    pure_first_correct: int
    pure_second_correct: int
    pure_mean_correct: float
    judge_disagreement: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare forced-letter natural outputs vs pure natural adjudication")
    parser.add_argument("--forced", type=Path, required=True, help="Forced-letter response CSV")
    parser.add_argument("--pure-natural", type=Path, required=True, help="Faithful free-text natural adjudicated CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison artifacts")
    parser.add_argument("--run-label", default="paper_faithful_forced_vs_pure_natural", help="Filename label")
    parser.add_argument("--forced-format", default="natural_forced_letter", help="Forced-letter natural format label")
    parser.add_argument("--pure-format", default="patient_realistic", help="Faithful free-text natural format label")
    parser.add_argument("--judge-models", nargs=2, default=DEFAULT_JUDGES, help="Two adjudicator model names")
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

    forced_rows = [
        row for row in read_csv(args.forced)
        if row["prompt_format"] == args.forced_format
        and row.get("is_correct", "") != ""
    ]
    pure_rows = [
        row for row in read_csv(args.pure_natural)
        if row["prompt_format"] == args.pure_format
        and row.get(f"{first_prefix}_triage")
        and row.get(f"{second_prefix}_triage")
    ]

    forced_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
        for row in forced_rows
    }
    pure_by_key = {
        (row["source_model"], row["case_id"], int(row["run_number"])): row
        for row in pure_rows
    }

    shared_keys = sorted(set(forced_by_key) & set(pure_by_key))
    diagnostics = {
        "forced_rows": len(forced_rows),
        "pure_rows": len(pure_rows),
        "shared_row_pairs": len(shared_keys),
        "missing_in_pure": len(set(forced_by_key) - set(pure_by_key)),
        "missing_in_forced": len(set(pure_by_key) - set(forced_by_key)),
    }

    comparisons: list[RowComparison] = []
    for key in shared_keys:
        forced = forced_by_key[key]
        pure = pure_by_key[key]
        forced_correct = to_bool_int(forced["is_correct"])
        pure_first_correct = to_bool_int(pure[f"{first_prefix}_is_correct"])
        pure_second_correct = to_bool_int(pure[f"{second_prefix}_is_correct"])
        if forced_correct is None or pure_first_correct is None or pure_second_correct is None:
            continue
        first_label = pure[f"{first_prefix}_triage"]
        second_label = pure[f"{second_prefix}_triage"]
        comparisons.append(
            RowComparison(
                source_model=forced["model"],
                case_id=forced["case_id"],
                run_number=int(forced["run_number"]),
                gold_standard=forced["gold_standard"],
                forced_correct=forced_correct,
                forced_predicted=forced.get("predicted_triage", ""),
                pure_first_label=first_label,
                pure_second_label=second_label,
                pure_first_correct=pure_first_correct,
                pure_second_correct=pure_second_correct,
                pure_mean_correct=(pure_first_correct + pure_second_correct) / 2.0,
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
        forced_accuracy = sum(row.forced_correct for row in items) / len(items)
        pure_first_accuracy = sum(row.pure_first_correct for row in items) / len(items)
        pure_second_accuracy = sum(row.pure_second_correct for row in items) / len(items)
        pure_mean_accuracy = sum(row.pure_mean_correct for row in items) / len(items)
        out.append(
            {
                "source_model": key[0],
                "case_id": key[1],
                "n_runs": len(items),
                "forced_accuracy": forced_accuracy,
                "pure_first_accuracy": pure_first_accuracy,
                "pure_second_accuracy": pure_second_accuracy,
                "pure_mean_accuracy": pure_mean_accuracy,
                "delta_forced_minus_pure_mean": forced_accuracy - pure_mean_accuracy,
                "judge_disagreements": sum(row.judge_disagreement for row in items),
            }
        )
    return out


def summarize_by_model(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in sorted({row.source_model for row in comparisons}):
        rows = [row for row in comparisons if row.source_model == model]
        forced_accuracy = sum(row.forced_correct for row in rows) / len(rows)
        pure_first_accuracy = sum(row.pure_first_correct for row in rows) / len(rows)
        pure_second_accuracy = sum(row.pure_second_correct for row in rows) / len(rows)
        pure_mean_accuracy = sum(row.pure_mean_correct for row in rows) / len(rows)
        out[model] = {
            "n": len(rows),
            "forced_accuracy": forced_accuracy,
            "pure_first_accuracy": pure_first_accuracy,
            "pure_second_accuracy": pure_second_accuracy,
            "pure_mean_accuracy": pure_mean_accuracy,
            "mean_delta": forced_accuracy - pure_mean_accuracy,
            "judge_disagreements": sum(row.judge_disagreement for row in rows),
        }
    return out


def compute_wilcoxon(cell_rows: list[dict]) -> dict:
    forced = [row["forced_accuracy"] for row in cell_rows]
    pure = [row["pure_mean_accuracy"] for row in cell_rows]
    deltas = [f - p for f, p in zip(forced, pure)]
    if not any(delta != 0 for delta in deltas):
        return {
            "n_cells": len(cell_rows),
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": len(cell_rows),
            "mean_delta": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }
    result = wilcoxon(forced, pure, zero_method="wilcox", alternative="two-sided")
    return {
        "n_cells": len(cell_rows),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_negative": sum(1 for d in deltas if d < 0),
        "n_zero": sum(1 for d in deltas if d == 0),
        "mean_delta": sum(deltas) / len(deltas),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def compute_exact_mcnemar(comparisons: list[RowComparison], field: str) -> dict:
    forced_only = 0
    pure_only = 0
    for row in comparisons:
        pure_correct = getattr(row, field)
        if row.forced_correct == 1 and pure_correct == 0:
            forced_only += 1
        elif row.forced_correct == 0 and pure_correct == 1:
            pure_only += 1
    p_value = 1.0
    if forced_only + pure_only:
        p_value = binomtest(
            min(forced_only, pure_only),
            n=forced_only + pure_only,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    return {
        "field": field,
        "forced_only": forced_only,
        "pure_only": pure_only,
        "discordant_total": forced_only + pure_only,
        "exact_p_value": float(p_value),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparisons, diagnostics = build_row_comparisons(args)
    row_rows = [asdict(row) for row in comparisons]
    cell_rows = group_cell_stats(comparisons)
    by_model = summarize_by_model(comparisons)

    overall = {
        "forced_accuracy": (
            sum(row.forced_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
        "pure_first_accuracy": (
            sum(row.pure_first_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
        "pure_second_accuracy": (
            sum(row.pure_second_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
        "pure_mean_accuracy": (
            sum(row.pure_mean_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
        "judge_disagreements": sum(row.judge_disagreement for row in comparisons),
    }

    report = {
        "overall": overall,
        "pairing_diagnostics": diagnostics,
        "wilcoxon": compute_wilcoxon(cell_rows) if cell_rows else None,
        "mcnemar_forced_vs_pure_first": (
            compute_exact_mcnemar(comparisons, "pure_first_correct") if comparisons else None
        ),
        "mcnemar_forced_vs_pure_second": (
            compute_exact_mcnemar(comparisons, "pure_second_correct") if comparisons else None
        ),
        "by_model": by_model,
    }

    row_path = args.output_dir / f"{args.run_label}_rowwise.csv"
    cell_path = args.output_dir / f"{args.run_label}_cell_summary.csv"
    json_path = args.output_dir / f"{args.run_label}_comparison.json"

    write_csv(row_path, row_rows)
    write_csv(cell_path, cell_rows)
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote rowwise comparison to {row_path}")
    print(f"Wrote cell summary to {cell_path}")
    print(f"Wrote comparison report to {json_path}")
    if overall["forced_accuracy"] is not None:
        print(
            "Overall matched-row accuracy: "
            f"forced={overall['forced_accuracy']:.4f}, "
            f"pure_mean={overall['pure_mean_accuracy']:.4f}"
        )
        print(f"Matched cells: {report['wilcoxon']['n_cells']}")
        print(f"Wilcoxon p-value: {report['wilcoxon']['p_value']:.6g}")


if __name__ == "__main__":
    main()
