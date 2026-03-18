#!/usr/bin/env python3
"""Build a run-ready vignette JSON from the canonical rewrite workbook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = WORKSPACE_DIR / "data" / "canonical_rewrite_workbook.csv"
DEFAULT_OUTPUT = WORKSPACE_DIR / "data" / "canonical_singleturn_vignettes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-faithful single-turn dataset JSON")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK, help="Canonical rewrite workbook CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sort_key(row: dict) -> tuple[int, str]:
    return int(row["case_num"]), row["case_id"]


def title_for(row: dict) -> str:
    return f"{row['case_id']} {row['diagnosis']}"


def validate_row(row: dict) -> None:
    missing = []
    for field in ("case_id", "diagnosis", "gold_triage", "source_prompt_text", "natural_singleturn"):
        if not str(row.get(field, "")).strip():
            missing.append(field)
    if missing:
        raise SystemExit(f"Workbook row {row.get('case_id', '<unknown>')} is missing: {', '.join(missing)}")


def build_rows(rows: list[dict]) -> list[dict]:
    built: list[dict] = []
    for row in sorted(rows, key=sort_key):
        validate_row(row)
        built.append(
            {
                "id": row["case_id"],
                "title": title_for(row),
                "gold_standard_triage": row["gold_triage"],
                "original_structured": row["source_prompt_text"].strip(),
                "patient_realistic": row["natural_singleturn"].strip(),
                "paper_metadata": {
                    "case_num": row["case_num"],
                    "case_pair": row["case_pair"],
                    "scenario_num": row["scenario_num"],
                    "source_version": row["source_version"],
                    "prompt_type": row["prompt_type"],
                    "domain": row["domain"],
                    "diagnosis": row["diagnosis"],
                    "triage_boundary": row["triage_boundary"],
                    "acuity": row["acuity"],
                    "is_edge_case": row["is_edge_case"],
                    "variant_code": row["variant_code"],
                    "race": row["race"],
                    "gender": row["gender"],
                    "has_anchor": row["has_anchor"],
                    "has_barrier": row["has_barrier"],
                    "rewrite_status": row.get("rewrite_status", ""),
                },
            }
        )
    return built


def main() -> None:
    args = parse_args()
    rows = load_rows(args.workbook)
    built = build_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(built, indent=2))
    print(f"Wrote {len(built)} rows to {args.output}")


if __name__ == "__main__":
    main()
