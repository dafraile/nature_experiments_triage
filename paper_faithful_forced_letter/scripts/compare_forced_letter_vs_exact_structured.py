#!/usr/bin/env python3
"""Compare natural forced-letter outputs against the faithful exact structured run."""

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
    forced_correct: int
    forced_predicted: str
    structured_correct: int
    structured_predicted: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare forced-letter natural outputs vs exact structured outputs")
    parser.add_argument("--forced", type=Path, required=True, help="Forced-letter response CSV")
    parser.add_argument("--structured", type=Path, required=True, help="Exact structured response CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison artifacts")
    parser.add_argument("--run-label", default="paper_faithful_forced_vs_exact_structured", help="Filename label")
    parser.add_argument("--forced-format", default="natural_forced_letter", help="Forced-letter natural format label")
    parser.add_argument("--structured-format", default="original_structured", help="Exact structured prompt format label")
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
    forced_rows = [
        row for row in read_csv(args.forced)
        if row["prompt_format"] == args.forced_format
        and row.get("is_correct", "") != ""
    ]
    structured_rows = [
        row for row in read_csv(args.structured)
        if row["prompt_format"] == args.structured_format
        and row.get("is_correct", "") != ""
    ]

    forced_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
        for row in forced_rows
    }
    structured_by_key = {
        (row["model"], row["case_id"], int(row["run_number"])): row
        for row in structured_rows
    }

    shared_keys = sorted(set(forced_by_key) & set(structured_by_key))
    diagnostics = {
        "forced_rows": len(forced_rows),
        "structured_rows": len(structured_rows),
        "shared_row_pairs": len(shared_keys),
        "missing_in_structured": len(set(forced_by_key) - set(structured_by_key)),
        "missing_in_forced": len(set(structured_by_key) - set(forced_by_key)),
    }

    comparisons: list[RowComparison] = []
    for key in shared_keys:
        forced = forced_by_key[key]
        structured = structured_by_key[key]
        forced_correct = to_bool_int(forced["is_correct"])
        structured_correct = to_bool_int(structured["is_correct"])
        if forced_correct is None or structured_correct is None:
            continue
        comparisons.append(
            RowComparison(
                source_model=forced["model"],
                case_id=forced["case_id"],
                run_number=int(forced["run_number"]),
                gold_standard=forced["gold_standard"],
                forced_correct=forced_correct,
                forced_predicted=forced.get("predicted_triage", ""),
                structured_correct=structured_correct,
                structured_predicted=structured.get("predicted_triage", ""),
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
        structured_accuracy = sum(row.structured_correct for row in items) / len(items)
        out.append(
            {
                "source_model": key[0],
                "case_id": key[1],
                "n_runs": len(items),
                "forced_accuracy": forced_accuracy,
                "structured_accuracy": structured_accuracy,
                "delta_forced_minus_structured": forced_accuracy - structured_accuracy,
            }
        )
    return out


def summarize_by_model(comparisons: list[RowComparison]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in sorted({row.source_model for row in comparisons}):
        rows = [row for row in comparisons if row.source_model == model]
        forced_accuracy = sum(row.forced_correct for row in rows) / len(rows)
        structured_accuracy = sum(row.structured_correct for row in rows) / len(rows)
        out[model] = {
            "n": len(rows),
            "forced_accuracy": forced_accuracy,
            "structured_accuracy": structured_accuracy,
            "mean_delta": forced_accuracy - structured_accuracy,
        }
    return out


def compute_wilcoxon(cell_rows: list[dict]) -> dict:
    forced = [row["forced_accuracy"] for row in cell_rows]
    structured = [row["structured_accuracy"] for row in cell_rows]
    deltas = [f - s for f, s in zip(forced, structured)]
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
    result = wilcoxon(forced, structured, zero_method="wilcox", alternative="two-sided")
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
    forced_only = 0
    structured_only = 0
    for row in comparisons:
        if row.forced_correct == 1 and row.structured_correct == 0:
            forced_only += 1
        elif row.forced_correct == 0 and row.structured_correct == 1:
            structured_only += 1
    p_value = 1.0
    if forced_only + structured_only:
        p_value = binomtest(
            min(forced_only, structured_only),
            n=forced_only + structured_only,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    return {
        "forced_only": forced_only,
        "structured_only": structured_only,
        "discordant_total": forced_only + structured_only,
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
        "structured_accuracy": (
            sum(row.structured_correct for row in comparisons) / len(comparisons) if comparisons else None
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
    if overall["forced_accuracy"] is not None:
        print(
            "Overall matched-row accuracy: "
            f"forced={overall['forced_accuracy']:.4f}, "
            f"structured={overall['structured_accuracy']:.4f}"
        )
        print(f"Matched cells: {report['wilcoxon']['n_cells']}")
        print(f"Wilcoxon p-value: {report['wilcoxon']['p_value']:.6g}")


if __name__ == "__main__":
    main()
