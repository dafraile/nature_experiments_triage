#!/usr/bin/env python3
"""Generate Opus-led naturalistic rewrites for the canonical workbook."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ANTHROPIC_API_KEY, MODELS  # noqa: E402

from generate_rewrite_candidates import build_prompt, load_guide  # noqa: E402


MODEL_NAME = "claude-opus-4.6"
WORKBOOK_PATH = WORKSPACE_DIR / "data" / "canonical_rewrite_workbook.csv"
RESULTS_DIR = WORKSPACE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate Opus rewrites for workbook")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional subset of case ids",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite rows that already have natural_singleturn text",
    )
    parser.add_argument(
        "--call-wait",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls",
    )
    return parser.parse_args()


def read_workbook() -> tuple[list[str], list[dict]]:
    with WORKBOOK_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    return fieldnames, rows


def write_workbook(fieldnames: list[str], rows: list[dict]) -> None:
    with WORKBOOK_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def call_anthropic(prompt: str) -> str:
    import anthropic

    model_config = MODELS[MODEL_NAME]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_config["model_id"],
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )
    chunks: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def parse_jsonish(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def select_rows(rows: list[dict], case_ids: set[str] | None, overwrite: bool) -> list[dict]:
    selected = []
    for row in rows:
        if case_ids and row["case_id"] not in case_ids:
            continue
        if row.get("natural_singleturn") and not overwrite:
            continue
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_workbook()
    guide = load_guide()
    selected = select_rows(rows, set(args.case_ids) if args.case_ids else None, args.overwrite)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = RESULTS_DIR / f"rewrite_candidates_{MODEL_NAME.replace('.', '_')}_batch_{stamp}.json"
    checkpoint = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "rows_total": len(selected),
        "results": [],
    }

    by_case = {row["case_id"]: row for row in rows}

    print(f"Generating {len(selected)} Opus rewrites")
    print(f"Checkpoint: {checkpoint_path}")

    for idx, row in enumerate(selected, start=1):
        print(f"[{idx:03d}/{len(selected):03d}] {row['case_id']} {row['diagnosis']}", flush=True)
        prompt = build_prompt(row, guide)
        raw = call_anthropic(prompt)
        parse_error = None
        parsed = None
        try:
            parsed = parse_jsonish(raw)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

        checkpoint["results"].append(
            {
                "case_id": row["case_id"],
                "diagnosis": row["diagnosis"],
                "raw": raw,
                "parsed": parsed,
                "parse_error": parse_error,
            }
        )
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

        if parsed:
            target = by_case[row["case_id"]]
            target["rewrite_status"] = "opus_generated_v1"
            target["natural_singleturn"] = parsed.get("natural_singleturn", "")
            target["natural_singleturn_review_notes"] = (
                "Generated with Claude Opus 4.6 using the canonical rewrite guide. "
                + str(parsed.get("style_notes", "")).strip()
            ).strip()
            write_workbook(fieldnames, rows)

        if args.call_wait > 0:
            time.sleep(args.call_wait)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Updated workbook: {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()
