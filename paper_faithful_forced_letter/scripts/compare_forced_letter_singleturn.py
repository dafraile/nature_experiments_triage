#!/usr/bin/env python3
"""Compare structured-vs-natural forced-letter results on the canonical bank."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from scipy.stats import binomtest, wilcoxon


@dataclass
class RowComparison:
    source_model: str
    case_id: str
    run_number: int
    gold_standard: str
    structured_correct: int
    structured_predicted: str
    natural_correct: int
    natural_predicted: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare structured vs natural forced-letter results")
    parser.add_argument("--responses", type=Path, required=True, help="Forced-letter response CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison artifacts")
    parser.add_argument("--run-label", default="paper_faithful_forced_letter", help="Filename label for output artifacts")
    parser.add_argument("--structured-format", default="structured_forced_letter", help="Structured prompt format label")
    parser.add_argument("--natural-format", default="natural_forced_letter", help="Natural prompt format label")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
    rows = read_csv(args.responses)
    structured_rows = [
        row for row in rows
        if row["prompt_format"] == args.structured_format
        and row.get("is_correct", "") != ""
    ]
    natural_rows = [
        row for row in rows
        if row["prompt_format"] == args.natural_format
        and row.get("is_correct", "") != ""
    ]

    structured_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
        for row in structured_rows
    }
    natural_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
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
        structured_correct = to_bool_int(s["is_correct"])
        natural_correct = to_bool_int(n["is_correct"])
        if structured_correct is None or natural_correct is None:
            continue
        comparisons.append(
            RowComparison(
                source_model=s["model"],
                case_id=s["case_id"],
                run_number=int(s["run_number"]),
                gold_standard=s["gold_standard"],
                structured_correct=structured_correct,
                structured_predicted=s.get("predicted_triage", ""),
                natural_correct=natural_correct,
                natural_predicted=n.get("predicted_triage", ""),
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
        natural_accuracy = sum(row.natural_correct for row in items) / len(items)
        out.append(
            {
                "source_model": key[0],
                "case_id": key[1],
                "n_runs": len(items),
                "structured_accuracy": structured_accuracy,
                "natural_accuracy": natural_accuracy,
                "delta_natural_minus_structured": natural_accuracy - structured_accuracy,
            }
        )
    return out


def summarize_by_model(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in sorted({row.source_model for row in comparisons}):
        rows = [row for row in comparisons if row.source_model == model]
        structured_accuracy = sum(row.structured_correct for row in rows) / len(rows)
        natural_accuracy = sum(row.natural_correct for row in rows) / len(rows)
        out[model] = {
            "n": len(rows),
            "structured_accuracy": structured_accuracy,
            "natural_accuracy": natural_accuracy,
            "mean_delta": natural_accuracy - structured_accuracy,
        }
    return out


def compute_wilcoxon(cell_rows: list[dict]) -> dict:
    structured = [row["structured_accuracy"] for row in cell_rows]
    natural = [row["natural_accuracy"] for row in cell_rows]
    deltas = [n - s for n, s in zip(natural, structured)]
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
    result = wilcoxon(natural, structured, zero_method="wilcox", alternative="two-sided")
    return {
        "n_cells": len(cell_rows),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_negative": sum(1 for d in deltas if d < 0),
        "n_zero": sum(1 for d in deltas if d == 0),
        "mean_delta": sum(deltas) / len(deltas),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def compute_exact_mcnemar(comparisons: list[RowComparison]) -> dict:
    structured_only = 0
    natural_only = 0
    for row in comparisons:
        if row.structured_correct == 1 and row.natural_correct == 0:
            structured_only += 1
        elif row.structured_correct == 0 and row.natural_correct == 1:
            natural_only += 1
    p_value = 1.0
    if structured_only + natural_only:
        p_value = binomtest(
            min(structured_only, natural_only),
            n=structured_only + natural_only,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    return {
        "structured_only": structured_only,
        "natural_only": natural_only,
        "discordant_total": structured_only + natural_only,
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
        "structured_accuracy": (
            sum(row.structured_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
        "natural_accuracy": (
            sum(row.natural_correct for row in comparisons) / len(comparisons) if comparisons else None
        ),
    }

    report = {
        "overall": overall,
        "pairing_diagnostics": diagnostics,
        "wilcoxon": compute_wilcoxon(cell_rows) if cell_rows else None,
        "mcnemar": compute_exact_mcnemar(comparisons) if comparisons else None,
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
    if overall["structured_accuracy"] is not None:
        print(
            "Overall matched-row accuracy: "
            f"structured={overall['structured_accuracy']:.4f}, "
            f"natural={overall['natural_accuracy']:.4f}"
        )
        print(f"Matched cells: {report['wilcoxon']['n_cells']}")
        print(f"Wilcoxon p-value: {report['wilcoxon']['p_value']:.6g}")


if __name__ == "__main__":
    main()
