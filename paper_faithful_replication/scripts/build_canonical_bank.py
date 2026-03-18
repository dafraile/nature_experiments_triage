#!/usr/bin/env python3
"""Build the canonical 60-row paper-faithful source bank and rewrite workbook."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parents[1]
PAPER_DATA_PATH = PROJECT_ROOT / "gpt_health_eval.X1pVTe" / "data" / "DataOriginal_FINAL.csv"
OUTPUT_DIR = WORKSPACE_DIR / "data"


SOURCE_FIELDNAMES = [
    "case_num",
    "case_id",
    "case_pair",
    "scenario_num",
    "source_version",
    "prompt_type",
    "domain",
    "diagnosis",
    "gold_triage",
    "triage_boundary",
    "acuity",
    "is_edge_case",
    "variant_code",
    "race",
    "gender",
    "has_anchor",
    "has_barrier",
    "source_opening",
    "source_about_me",
    "source_clinical_block",
    "source_prompt_text",
    "prompt_text_E",
    "prompt_text_F",
]

WORKBOOK_EXTRA_FIELDS = [
    "rewrite_status",
    "natural_singleturn",
    "natural_singleturn_review_notes",
    "multiturn_seed_question",
    "multiturn_seed_notes",
]


def extract_prompt_parts(prompt_text: str) -> tuple[str, str, str]:
    lines = [line.rstrip() for line in prompt_text.splitlines()]

    opening = lines[0].strip() if lines else ""

    about_me = ""
    clinical_lines: list[str] = []
    in_clinical_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("About me:"):
            about_me = stripped.removeprefix("About me:").strip()
            continue
        if stripped == "My situation:":
            in_clinical_block = True
            continue
        if stripped.startswith("Please answer in exactly this format:"):
            break
        if in_clinical_block:
            clinical_lines.append(stripped)

    clinical_block = "\n".join(line for line in clinical_lines if line)
    return opening, about_me, clinical_block


def read_reference_rows() -> list[dict]:
    with PAPER_DATA_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    reference_rows = [
        row for row in rows
        if row["variant_code"] == "WM"
        and row["has_anchor"] == "no"
        and row["has_barrier"] == "no"
    ]

    if len(reference_rows) != 60:
        raise SystemExit(f"Expected 60 reference rows, found {len(reference_rows)}")

    out: list[dict] = []
    for row in sorted(reference_rows, key=lambda item: int(item["case_num"])):
        opening, about_me, clinical_block = extract_prompt_parts(row["prompt_text"])
        out.append(
            {
                "case_num": row["case_num"],
                "case_id": row["case_id"],
                "case_pair": row["case_pair"],
                "scenario_num": row["scenario_num"],
                "source_version": row["case_id"][0],
                "prompt_type": row["prompt_type"],
                "domain": row["domain"],
                "diagnosis": row["diagnosis"],
                "gold_triage": row["gold_triage"],
                "triage_boundary": row["triage_boundary"],
                "acuity": row["acuity"],
                "is_edge_case": row["is_edge_case"],
                "variant_code": row["variant_code"],
                "race": row["race"],
                "gender": row["gender"],
                "has_anchor": row["has_anchor"],
                "has_barrier": row["has_barrier"],
                "source_opening": opening,
                "source_about_me": about_me,
                "source_clinical_block": clinical_block,
                "source_prompt_text": row["prompt_text"],
                "prompt_text_E": row["prompt_text_E"],
                "prompt_text_F": row["prompt_text_F"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = read_reference_rows()

    source_csv = OUTPUT_DIR / "canonical_source_bank.csv"
    source_json = OUTPUT_DIR / "canonical_source_bank.json"
    workbook_csv = OUTPUT_DIR / "canonical_rewrite_workbook.csv"

    write_csv(source_csv, source_rows, SOURCE_FIELDNAMES)
    source_json.write_text(json.dumps(source_rows, indent=2))

    workbook_rows = []
    for row in source_rows:
        workbook_rows.append(
            row | {field: "" for field in WORKBOOK_EXTRA_FIELDS},
        )
    write_csv(workbook_csv, workbook_rows, SOURCE_FIELDNAMES + WORKBOOK_EXTRA_FIELDS)

    print(f"Wrote {len(source_rows)} rows to {source_csv}")
    print(f"Wrote {len(source_rows)} rows to {source_json}")
    print(f"Wrote {len(workbook_rows)} rows to {workbook_csv}")


if __name__ == "__main__":
    main()
