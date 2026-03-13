#!/usr/bin/env python3
"""Apply rewrite draft JSON entries onto the canonical rewrite workbook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = WORKSPACE_DIR / "data" / "canonical_rewrite_workbook.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply rewrite drafts to workbook")
    parser.add_argument("--drafts", required=True, help="Path to draft JSON file")
    parser.add_argument(
        "--workbook",
        default=str(DEFAULT_WORKBOOK),
        help="Path to canonical rewrite workbook CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draft_path = Path(args.drafts)
    workbook_path = Path(args.workbook)

    with workbook_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    with draft_path.open() as f:
        drafts = json.load(f)

    by_case = {row["case_id"]: row for row in rows}
    updated = 0
    for draft in drafts:
        case_id = draft["case_id"]
        if case_id not in by_case:
            raise SystemExit(f"Unknown case_id in drafts: {case_id}")
        target = by_case[case_id]
        for key, value in draft.items():
            if key == "case_id":
                continue
            if key not in target:
                raise SystemExit(f"Unknown workbook field in draft: {key}")
            target[key] = value
        updated += 1

    ordered_rows = [by_case[row["case_id"]] for row in rows]
    with workbook_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)

    print(f"Updated {updated} rows in {workbook_path}")


if __name__ == "__main__":
    main()
