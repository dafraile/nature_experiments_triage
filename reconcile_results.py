#!/usr/bin/env python3
"""Rebuild canonical datasets from raw result files and recover parseable rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_utils import infer_free_text_triage, parse_structured_response

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data" / "vignettes.json"

MAIN_COLUMNS = [
    "case_id",
    "case_title",
    "gold_standard",
    "model",
    "provider",
    "prompt_format",
    "system_prompt_version",
    "run_number",
    "predicted_triage",
    "reasoning",
    "confidence",
    "raw_response",
    "is_correct",
    "error",
    "latency_seconds",
    "timestamp",
]


def latest_result(pattern: str) -> Path | None:
    matches = sorted(RESULTS_DIR.glob(pattern))
    return matches[-1] if matches else None


def case_lookup() -> dict[str, dict]:
    rows = json.loads(DATA_PATH.read_text())
    return {row["id"]: row for row in rows}


def coerce_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def recover_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    recovered = 0

    for idx, row in df.iterrows():
        raw_response = row.get("raw_response")
        if pd.isna(raw_response):
            raw_response = ""
        raw_response = str(raw_response)

        error = row.get("error")
        has_error = isinstance(error, str) and error.strip()
        predicted = row.get("predicted_triage")

        if has_error or (isinstance(predicted, str) and predicted.strip()):
            continue

        parsed = parse_structured_response(raw_response)
        cat = parsed["triage_category"] or infer_free_text_triage(raw_response)
        if not cat:
            continue

        gold = row.get("gold_standard")
        df.at[idx, "predicted_triage"] = cat
        if pd.isna(row.get("reasoning")) and parsed["reasoning"]:
            df.at[idx, "reasoning"] = parsed["reasoning"]
        if pd.isna(row.get("confidence")) and parsed["confidence"] is not None:
            df.at[idx, "confidence"] = parsed["confidence"]
        if isinstance(gold, str) and gold:
            df.at[idx, "is_correct"] = (cat == gold)
        recovered += 1

    return df, recovered


def load_csv_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in MAIN_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[MAIN_COLUMNS].copy()


def load_gemini_resumed(path: Path, lookup: dict[str, dict]) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    normalized = []

    for row in rows:
        meta = lookup.get(row["case_id"], {})
        normalized.append(
            {
                "case_id": row["case_id"],
                "case_title": meta.get("title"),
                "gold_standard": row.get("gold_standard") or meta.get("gold_standard_triage"),
                "model": row.get("model", "gemini-3.1-pro"),
                "provider": "google",
                "prompt_format": row.get("prompt_format"),
                "system_prompt_version": "structured" if row.get("prompt_format") == "original_structured" else "natural",
                "run_number": int(row.get("run")),
                "predicted_triage": row.get("predicted"),
                "reasoning": None,
                "confidence": None,
                "raw_response": row.get("raw_response", ""),
                "is_correct": row.get("correct"),
                "error": row.get("error"),
                "latency_seconds": None,
                "timestamp": None,
            }
        )

    return pd.DataFrame(normalized, columns=MAIN_COLUMNS)


def load_main_dataset() -> tuple[pd.DataFrame, dict[str, Path]]:
    lookup = case_lookup()
    sources: dict[str, Path] = {}
    frames: list[pd.DataFrame] = []

    patterns = {
        "claude_opus": "results_claude_opus_20*.csv",
        "claude_sonnet": "results_claude_sonnet_20*.csv",
        "gemini_flash": "results_gemini_flash_20*.csv",
        "gpt52": "results_gpt52_20*.csv",
        "case17": "results_case17_triage_20*.csv",
    }

    for label, pattern in patterns.items():
        path = latest_result(pattern)
        if not path:
            continue
        sources[label] = path
        frames.append(load_csv_result(path))

    gemini_resumed = RESULTS_DIR / "results_gemini_pro_resumed.json"
    if gemini_resumed.exists():
        sources["gemini_pro_resumed"] = gemini_resumed
        frames.append(load_gemini_resumed(gemini_resumed, lookup))

    if not frames:
        raise FileNotFoundError("No result files found in results/")

    combined = pd.concat(frames, ignore_index=True)
    combined["is_correct"] = combined["is_correct"].apply(coerce_bool)
    combined, recovered = recover_rows(combined)
    combined.attrs["recovered"] = recovered

    return combined, sources


def print_summary(df: pd.DataFrame, sources: dict[str, Path], case_ids: list[str] | None):
    print("Sources:")
    for label, path in sources.items():
        print(f"  {label}: {path.name}")

    total = len(df)
    errors = sum(1 for value in df["error"] if isinstance(value, str) and value.strip())
    scored = sum(value is not None for value in df["is_correct"])
    predicted = sum(1 for value in df["predicted_triage"] if isinstance(value, str) and value)
    recovered = df.attrs.get("recovered", 0)

    print("\nMain experiment:")
    print(f"  total rows: {total}")
    print(f"  predicted rows: {predicted}")
    print(f"  scored rows: {scored}")
    print(f"  error rows: {errors}")
    print(f"  recovered rows: {recovered}")

    print("\nBy model and format:")
    grouped = df.groupby(["model", "prompt_format"]).agg(
        total=("case_id", "count"),
        scored=("is_correct", lambda s: sum(value is not None for value in s)),
        correct=("is_correct", lambda s: sum(value is True for value in s)),
    )
    print(grouped.to_string())

    if case_ids:
        subset = df[df["case_id"].isin(case_ids)].copy()
        if subset.empty:
            return
        print("\nSelected cases:")
        grouped = subset.groupby(["case_id", "model", "prompt_format"]).agg(
            total=("case_id", "count"),
            scored=("is_correct", lambda s: sum(value is not None for value in s)),
            correct=("is_correct", lambda s: sum(value is True for value in s)),
        )
        print(grouped.to_string())


def write_outputs(df: pd.DataFrame):
    csv_path = RESULTS_DIR / "main_experiment_reconciled.csv"
    json_path = RESULTS_DIR / "main_experiment_reconciled.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\nWrote {csv_path.name} and {json_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Reconcile raw result files into canonical datasets")
    parser.add_argument("--write", action="store_true", help="Write reconciled CSV/JSON into results/")
    parser.add_argument("--cases", nargs="+", default=["case_08", "case_17"],
                        help="Case ids to include in the printed focused summary")
    args = parser.parse_args()

    df, sources = load_main_dataset()
    print_summary(df, sources, args.cases)
    if args.write:
        write_outputs(df)


if __name__ == "__main__":
    main()
