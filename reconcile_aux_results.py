#!/usr/bin/env python3
"""Re-score the sensitivity and ablation result files with current parsing rules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_utils import extract_triage_category, infer_free_text_triage

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def latest_result(pattern: str) -> Path | None:
    matches = sorted(
        path for path in RESULTS_DIR.glob(pattern)
        if "_reconciled" not in path.name
    )
    return matches[-1] if matches else None


def latest_ablation_power_base() -> Path | None:
    matches = sorted(
        path for path in RESULTS_DIR.glob("ablation_power_*.json")
        if path.name not in {"ablation_power_gemini_pro.json", "ablation_power_reconciled.json"}
    )
    return matches[-1] if matches else None


def reconcile_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    recovered = 0
    for idx, row in df.iterrows():
        if not pd.isna(row["predicted_triage"]) or (isinstance(row["error"], str) and row["error"].strip()):
            continue
        raw = "" if pd.isna(row["raw_response"]) else str(row["raw_response"])
        cat = extract_triage_category(raw)
        if not cat and row["condition"] == "paper_full":
            cat = infer_free_text_triage(raw)
        if not cat:
            continue
        df.at[idx, "predicted_triage"] = cat
        df.at[idx, "is_correct"] = (cat == row["gold_standard"])
        recovered += 1
    return df, recovered


def reconcile_ablation_constraint(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    recovered = 0
    for idx, row in df.iterrows():
        if not pd.isna(row["predicted_triage"]) or (isinstance(row["error"], str) and row["error"].strip()):
            continue
        raw = "" if pd.isna(row["raw_response"]) else str(row["raw_response"])
        if row["condition"] in {"free_text", "all_constraints"}:
            cat = infer_free_text_triage(raw)
        else:
            cat = extract_triage_category(raw)
        if not cat:
            continue
        df.at[idx, "predicted_triage"] = cat
        df.at[idx, "is_correct"] = (cat == row["gold_standard"])
        recovered += 1
    return df, recovered


def reconcile_ablation_power(rows: list[dict]) -> tuple[pd.DataFrame, int]:
    recovered = 0
    for row in rows:
        if row.get("predicted") is not None or row.get("error"):
            continue
        raw = row.get("raw", "")
        if row["condition"] in {"free_text", "all_constraints"}:
            cat = infer_free_text_triage(raw)
        else:
            cat = extract_triage_category(raw)
        if not cat:
            continue
        row["predicted"] = cat
        row["correct"] = (cat == "A")
        recovered += 1
    return pd.DataFrame(rows), recovered


def write_frame(df: pd.DataFrame, stem: str):
    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"  wrote {csv_path.name} and {json_path.name}")


def main():
    print("Reconciling auxiliary result files...")

    sensitivity_path = latest_result("sensitivity_constraint_*.csv")
    if sensitivity_path:
        sensitivity = pd.read_csv(sensitivity_path)
        sensitivity, recovered = reconcile_sensitivity(sensitivity)
        print(f"\nSensitivity: {sensitivity_path.name}")
        print(f"  recovered rows: {recovered}")
        print(f"  unresolved rows: {int(sensitivity['is_correct'].isna().sum())}")
        print(sensitivity.groupby(['model', 'condition'])['is_correct'].agg(['count', 'sum']).to_string())
        write_frame(sensitivity, "sensitivity_constraint_reconciled")

    ablation_constraint_path = latest_result("ablation_constraint_*.csv")
    if ablation_constraint_path:
        ablation_constraint = pd.read_csv(ablation_constraint_path)
        ablation_constraint, recovered = reconcile_ablation_constraint(ablation_constraint)
        print(f"\nAblation (one-factor): {ablation_constraint_path.name}")
        print(f"  recovered rows: {recovered}")
        print(f"  unresolved rows: {int(ablation_constraint['is_correct'].isna().sum())}")
        print(ablation_constraint.groupby(['model', 'condition'])['is_correct'].agg(['count', 'sum']).to_string())
        write_frame(ablation_constraint, "ablation_constraint_reconciled")

    ablation_power_path = latest_ablation_power_base()
    gemini_power_path = RESULTS_DIR / "ablation_power_gemini_pro.json"
    if ablation_power_path and gemini_power_path.exists():
        rows = json.loads(ablation_power_path.read_text()) + json.loads(gemini_power_path.read_text())
        ablation_power, recovered = reconcile_ablation_power(rows)
        print(f"\nAblation (high-power): {ablation_power_path.name} + {gemini_power_path.name}")
        print(f"  recovered rows: {recovered}")
        print(f"  unresolved rows: {int(ablation_power['correct'].isna().sum())}")
        print(ablation_power.groupby(['model', 'condition'])['correct'].agg(['count', 'sum']).to_string())
        write_frame(ablation_power, "ablation_power_reconciled")


if __name__ == "__main__":
    main()
