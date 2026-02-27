#!/usr/bin/env python3
"""
Analysis & Visualization for the Triage Replication Experiment
================================================================

Reads results CSV(s) and produces:
1. Accuracy comparison tables (model × format)
2. Per-case breakdown showing where format matters most
3. Triage category confusion analysis (over-triage vs under-triage by format)
4. "Confidence" analysis — showing these numbers are meaningless
5. Statistical tests (McNemar's, chi-squared) for format effect
6. Publication-ready figures

Usage:
    python analyze_results.py results/results_20260227_143000.csv
    python analyze_results.py results/*.csv   # combine multiple runs
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

# ── Visualization imports ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

TRIAGE_ORDER = ["A", "B", "C", "D"]
TRIAGE_LABELS = {
    "A": "Emergency (A)",
    "B": "Within 24h (B)",
    "C": "Few days (C)",
    "D": "Self-care (D)",
}
FORMAT_LABELS = {
    "original_structured": "Structured\n(Paper format)",
    "patient_realistic": "Patient\n(Realistic)",
    "patient_minimal": "Patient\n(Minimal)",
}
FORMAT_COLORS = {
    "original_structured": "#2196F3",
    "patient_realistic": "#FF9800",
    "patient_minimal": "#F44336",
}


def load_results(paths: list[str]) -> pd.DataFrame:
    """Load and concatenate one or more results CSVs."""
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} trials from {len(paths)} file(s)")
    return df


# ═══════════════════════════════════════════════
# 1. ACCURACY TABLE
# ═══════════════════════════════════════════════

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy by (model, prompt_format)."""
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(bool)

    table = valid.groupby(["model", "prompt_format"]).agg(
        n_trials=("is_correct", "count"),
        n_correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
    ).reset_index()
    table["accuracy_pct"] = (table["accuracy"] * 100).round(1)

    # Pivot for display
    pivot = table.pivot_table(
        index="model", columns="prompt_format", values="accuracy_pct"
    )
    pivot["delta_structured_vs_realistic"] = pivot.get("original_structured", 0) - pivot.get("patient_realistic", 0)

    print("\n" + "=" * 70)
    print("  ACCURACY BY MODEL × PROMPT FORMAT (%)")
    print("=" * 70)
    print(pivot.to_string())
    return table, pivot


# ═══════════════════════════════════════════════
# 2. PER-CASE BREAKDOWN
# ═══════════════════════════════════════════════

def per_case_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Show accuracy per case across formats — identifies where format matters most."""
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(bool)

    case_table = valid.groupby(["case_id", "case_title", "gold_standard", "prompt_format"]).agg(
        accuracy=("is_correct", "mean"),
        n=("is_correct", "count"),
    ).reset_index()

    pivot = case_table.pivot_table(
        index=["case_id", "case_title", "gold_standard"],
        columns="prompt_format",
        values="accuracy",
    )
    if "original_structured" in pivot.columns and "patient_realistic" in pivot.columns:
        pivot["delta"] = pivot["original_structured"] - pivot["patient_realistic"]
        pivot = pivot.sort_values("delta", ascending=False)

    print("\n" + "=" * 70)
    print("  PER-CASE ACCURACY ACROSS FORMATS")
    print("=" * 70)
    print(pivot.round(2).to_string())
    return pivot


# ═══════════════════════════════════════════════
# 3. TRIAGE DIRECTION ANALYSIS (over vs under triage)
# ═══════════════════════════════════════════════

def triage_direction_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze whether format changes cause over-triage or under-triage."""
    valid = df[df["predicted_triage"].notna()].copy()

    triage_to_num = {"A": 1, "B": 2, "C": 3, "D": 4}
    valid["gold_num"] = valid["gold_standard"].map(triage_to_num)
    valid["pred_num"] = valid["predicted_triage"].map(triage_to_num)
    valid["direction"] = valid["pred_num"] - valid["gold_num"]
    valid["direction_label"] = valid["direction"].apply(
        lambda x: "Under-triage" if x > 0 else ("Over-triage" if x < 0 else "Correct")
    )

    summary = valid.groupby(["prompt_format", "direction_label"]).size().reset_index(name="count")
    summary_pivot = summary.pivot_table(index="prompt_format", columns="direction_label", values="count", fill_value=0)

    print("\n" + "=" * 70)
    print("  TRIAGE DIRECTION BY FORMAT (over-triage vs under-triage)")
    print("=" * 70)
    print(summary_pivot.to_string())
    return summary_pivot


# ═══════════════════════════════════════════════
# 4. CONFIDENCE ANALYSIS (showing it's meaningless)
# ═══════════════════════════════════════════════

def confidence_analysis(df: pd.DataFrame):
    """
    Analyze the 'confidence' scores models produce.
    Key claim: these numbers are not calibrated and change with prompt format.
    """
    valid = df[df["confidence"].notna()].copy()
    if valid.empty:
        print("\n  [No confidence data found]")
        return

    # Confidence by correctness
    correct_conf = valid[valid["is_correct"] == True]["confidence"]
    incorrect_conf = valid[valid["is_correct"] == False]["confidence"]

    print("\n" + "=" * 70)
    print("  CONFIDENCE SCORE ANALYSIS")
    print("  (Demonstrating these are generated text, not calibrated probabilities)")
    print("=" * 70)
    print(f"  Correct predictions:   mean={correct_conf.mean():.1f}, std={correct_conf.std():.1f}, n={len(correct_conf)}")
    print(f"  Incorrect predictions: mean={incorrect_conf.mean():.1f}, std={incorrect_conf.std():.1f}, n={len(incorrect_conf)}")

    # Confidence by format
    print(f"\n  Confidence by prompt format:")
    for fmt in valid["prompt_format"].unique():
        subset = valid[valid["prompt_format"] == fmt]["confidence"]
        print(f"    {fmt:<25}: mean={subset.mean():.1f}, std={subset.std():.1f}")

    # Key test: does the SAME model give different confidence for the SAME case
    # just because of format change?
    print(f"\n  Same model, same case — confidence varies by format:")
    for model in valid["model"].unique():
        model_data = valid[valid["model"] == model]
        for case_id in model_data["case_id"].unique():
            case_data = model_data[model_data["case_id"] == case_id]
            confs = case_data.groupby("prompt_format")["confidence"].mean()
            if len(confs) > 1:
                spread = confs.max() - confs.min()
                if spread > 10:  # Only show large swings
                    print(f"    {model} / {case_id}: "
                          f"spread={spread:.0f} points | {dict(confs.round(0))}")


# ═══════════════════════════════════════════════
# 5. STATISTICAL TESTS
# ═══════════════════════════════════════════════

def statistical_tests(df: pd.DataFrame):
    """
    Test whether prompt format significantly affects accuracy.
    - Chi-squared test: overall format effect
    - McNemar's test: pairwise structured vs realistic (within same cases)
    - Cohen's kappa: agreement between formats
    """
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(int)

    print("\n" + "=" * 70)
    print("  STATISTICAL TESTS")
    print("=" * 70)

    # Chi-squared: does format affect accuracy?
    contingency = pd.crosstab(valid["prompt_format"], valid["is_correct"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n  Chi-squared test (format → accuracy):")
    print(f"    χ² = {chi2:.3f}, df = {dof}, p = {p_chi:.4f}")
    print(f"    {'Significant' if p_chi < 0.05 else 'Not significant'} at α = 0.05")

    # McNemar's test: pairwise comparison on matched cases
    formats = valid["prompt_format"].unique()
    if len(formats) >= 2 and "original_structured" in formats and "patient_realistic" in formats:
        # Match by (model, case_id, run_number)
        struct = valid[valid["prompt_format"] == "original_structured"].set_index(
            ["model", "case_id", "run_number"])["is_correct"]
        realist = valid[valid["prompt_format"] == "patient_realistic"].set_index(
            ["model", "case_id", "run_number"])["is_correct"]
        matched = pd.DataFrame({"structured": struct, "realistic": realist}).dropna()

        if len(matched) > 0:
            # Build 2×2 table for McNemar
            b = ((matched["structured"] == 1) & (matched["realistic"] == 0)).sum()  # struct correct, real wrong
            c = ((matched["structured"] == 0) & (matched["realistic"] == 1)).sum()  # struct wrong, real correct

            if (b + c) > 0:
                mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
                p_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                print(f"\n  McNemar's test (structured vs realistic, matched pairs):")
                print(f"    Structured-only correct: {b}, Realistic-only correct: {c}")
                print(f"    McNemar χ² = {mcnemar_stat:.3f}, p = {p_mcnemar:.4f}")
                print(f"    {'Significant' if p_mcnemar < 0.05 else 'Not significant'} at α = 0.05")

                # Cohen's kappa
                a = ((matched["structured"] == 1) & (matched["realistic"] == 1)).sum()
                d = ((matched["structured"] == 0) & (matched["realistic"] == 0)).sum()
                total = a + b + c + d
                po = (a + d) / total
                pe = ((a + b) * (a + c) + (c + d) * (b + d)) / total ** 2
                kappa = (po - pe) / (1 - pe) if pe < 1 else 0
                print(f"\n  Cohen's kappa (structured vs realistic agreement): {kappa:.3f}")
                print(f"    Interpretation: {'Poor' if kappa < 0.2 else 'Fair' if kappa < 0.4 else 'Moderate' if kappa < 0.6 else 'Good' if kappa < 0.8 else 'Excellent'}")


# ═══════════════════════════════════════════════
# 6. FIGURES
# ═══════════════════════════════════════════════

def plot_accuracy_by_format(df: pd.DataFrame, output_dir: Path):
    """Bar chart: accuracy by model grouped by format."""
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(bool)

    acc = valid.groupby(["model", "prompt_format"])["is_correct"].mean().reset_index()
    acc["accuracy_pct"] = acc["is_correct"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    models = sorted(acc["model"].unique())
    formats = [f for f in ["original_structured", "patient_realistic", "patient_minimal"]
               if f in acc["prompt_format"].unique()]

    x = np.arange(len(models))
    width = 0.25

    for i, fmt in enumerate(formats):
        subset = acc[acc["prompt_format"] == fmt]
        vals = [subset[subset["model"] == m]["accuracy_pct"].values[0]
                if m in subset["model"].values else 0 for m in models]
        bars = ax.bar(x + i * width, vals, width,
                      label=FORMAT_LABELS.get(fmt, fmt),
                      color=FORMAT_COLORS.get(fmt, f"C{i}"),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Triage Accuracy (%)")
    ax.set_title("LLM Triage Accuracy by Prompt Format\n"
                 "(Same clinical information, different presentation)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="Chance")

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_by_format.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "accuracy_by_format.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: accuracy_by_format.png/pdf")


def plot_per_case_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap: accuracy per case × format, showing where format matters."""
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(bool)

    pivot = valid.groupby(["case_title", "prompt_format"])["is_correct"].mean().reset_index()
    pivot["accuracy_pct"] = pivot["is_correct"] * 100
    heatmap_data = pivot.pivot_table(
        index="case_title", columns="prompt_format", values="accuracy_pct"
    )

    # Reorder columns
    col_order = [c for c in ["original_structured", "patient_realistic", "patient_minimal"]
                 if c in heatmap_data.columns]
    heatmap_data = heatmap_data[col_order]
    heatmap_data.columns = [FORMAT_LABELS.get(c, c).replace("\n", " ") for c in col_order]

    fig, ax = plt.subplots(figsize=(10, max(8, len(heatmap_data) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, center=50, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"}, ax=ax)
    ax.set_title("Triage Accuracy by Clinical Case × Prompt Format\n"
                 "(Aggregated across all models and runs)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "per_case_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "per_case_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: per_case_heatmap.png/pdf")


def plot_confidence_scatter(df: pd.DataFrame, output_dir: Path):
    """
    Scatter plot: confidence vs correctness by format.
    The point is to show confidence is NOT calibrated.
    """
    valid = df[df["confidence"].notna() & df["is_correct"].notna()].copy()
    if valid.empty:
        return

    fig, axes = plt.subplots(1, len(valid["prompt_format"].unique()),
                             figsize=(5 * len(valid["prompt_format"].unique()), 5),
                             sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, fmt in zip(axes, sorted(valid["prompt_format"].unique())):
        subset = valid[valid["prompt_format"] == fmt]
        correct = subset[subset["is_correct"] == True]["confidence"]
        incorrect = subset[subset["is_correct"] == False]["confidence"]

        ax.boxplot([correct, incorrect], labels=["Correct", "Incorrect"],
                   patch_artist=True,
                   boxprops=dict(facecolor=FORMAT_COLORS.get(fmt, "C0"), alpha=0.6))
        ax.set_title(FORMAT_LABELS.get(fmt, fmt).replace("\n", " "))
        ax.set_ylabel("Reported Confidence")
        ax.set_ylim(0, 105)

    fig.suptitle("LLM-Reported 'Confidence' by Correctness and Format\n"
                 "(These numbers are generated text, not calibrated probabilities)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "confidence_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "confidence_analysis.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: confidence_analysis.png/pdf")


def plot_triage_direction(df: pd.DataFrame, output_dir: Path):
    """Stacked bar: over-triage vs under-triage vs correct by format and model."""
    valid = df[df["predicted_triage"].notna()].copy()
    triage_to_num = {"A": 1, "B": 2, "C": 3, "D": 4}
    valid["direction"] = valid["predicted_triage"].map(triage_to_num) - valid["gold_standard"].map(triage_to_num)
    valid["direction_label"] = valid["direction"].apply(
        lambda x: "Under-triage\n(less urgent)" if x > 0
        else ("Over-triage\n(more urgent)" if x < 0 else "Correct")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ct = pd.crosstab(
        [valid["model"], valid["prompt_format"]],
        valid["direction_label"],
        normalize="index"
    ) * 100

    ct = ct.reindex(columns=["Over-triage\n(more urgent)", "Correct", "Under-triage\n(less urgent)"], fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#2196F3", "#4CAF50", "#F44336"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Percentage of Responses")
    ax.set_title("Triage Direction by Model × Format\n"
                 "(Under-triage = potentially dangerous misses)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Direction", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "triage_direction.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "triage_direction.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: triage_direction.png/pdf")


def plot_format_delta(df: pd.DataFrame, output_dir: Path):
    """
    The money plot: for each model, show the accuracy DELTA between
    structured and patient-realistic format.
    """
    valid = df[df["is_correct"].notna()].copy()
    valid["is_correct"] = valid["is_correct"].astype(bool)

    acc = valid.groupby(["model", "prompt_format"])["is_correct"].mean().reset_index()
    acc["accuracy_pct"] = acc["is_correct"] * 100

    struct = acc[acc["prompt_format"] == "original_structured"].set_index("model")["accuracy_pct"]
    realist = acc[acc["prompt_format"] == "patient_realistic"].set_index("model")["accuracy_pct"]
    delta = (struct - realist).dropna().sort_values(ascending=True)

    if delta.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in delta.values]
    bars = ax.barh(range(len(delta)), delta.values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(delta.index)
    ax.set_xlabel("Accuracy Delta (percentage points)\n"
                  "Positive = structured format scored HIGHER (format inflates performance)")
    ax.set_title("Performance Gap: Structured Clinical Format vs. Realistic Patient Language\n"
                 "(Positive values mean the paper's format flatters the model)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)

    for bar, val in zip(bars, delta.values):
        x_pos = bar.get_width() + (0.5 if val >= 0 else -0.5)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}pp", ha="left" if val >= 0 else "right",
                va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "format_delta.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "format_delta.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: format_delta.png/pdf")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze triage experiment results")
    parser.add_argument("csv_files", nargs="+", help="Path(s) to results CSV file(s)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    df = load_results(args.csv_files)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Run all analyses
    print("\n" + "=" * 70)
    print("  RUNNING ANALYSES")
    print("=" * 70)

    accuracy_table(df)
    per_case_analysis(df)
    triage_direction_analysis(df)
    confidence_analysis(df)
    statistical_tests(df)

    # Generate figures
    print(f"\n  Generating figures in {output_dir}/...")
    plot_accuracy_by_format(df, output_dir)
    plot_per_case_heatmap(df, output_dir)
    plot_confidence_scatter(df, output_dir)
    plot_triage_direction(df, output_dir)
    plot_format_delta(df, output_dir)

    print(f"\n  Done! All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
