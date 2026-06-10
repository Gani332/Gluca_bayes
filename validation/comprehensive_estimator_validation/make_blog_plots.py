#!/usr/bin/env python3
"""
Generate blog-ready plots from the frozen Gluca validation outputs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTDIR = SCRIPT_DIR / "comprehensive_latest"
PLOTS_DIR = OUTDIR / "plots"
ESTIMATOR_DIR = REPO_ROOT / "validation" / "estimator_baseline_validation" / "estimator_baseline_latest"

COLORS = {
    "gluca": "#0F766E",
    "baseline": "#334155",
    "positive": "#0F766E",
    "neutral": "#64748B",
    "negative": "#B91C1C",
    "accent": "#2563EB",
    "gold": "#B45309",
    "grid": "#CBD5E1",
    "text": "#111827",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def short_label(label: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def plot_pump_proxy_improvements() -> None:
    rows = pd.read_csv(OUTDIR / "pump_proxy_claim_rows.csv")
    rows = rows.copy()
    rows["plot_label"] = rows["domain"].map(lambda x: short_label(x, 24))
    rows["color"] = np.where(
        rows["value_add_supported"].astype(str).str.lower() == "true",
        COLORS["positive"],
        np.where(rows["comparison_supported"].astype(str).str.lower() == "true", COLORS["neutral"], COLORS["negative"]),
    )

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    y = np.arange(len(rows))
    ax.barh(y, rows["gluca_improvement_vs_baseline_pct"], color=rows["color"], height=0.68)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Improvement vs pump-style baseline (%)")
    ax.set_title("Gluca Value-Add vs Current Pump-Style Proxies")
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.7, linewidth=0.8)
    for idx, value in enumerate(rows["gluca_improvement_vs_baseline_pct"]):
        ha = "left" if value >= 0 else "right"
        offset = 1.2 if value >= 0 else -1.2
        ax.text(value + offset, idx, f"{value:.1f}%", va="center", ha=ha, fontweight="bold")
    ax.set_xlim(min(-15, rows["gluca_improvement_vs_baseline_pct"].min() - 6), max(100, rows["gluca_improvement_vs_baseline_pct"].max() + 8))
    save(fig, "01_pump_proxy_value_add.png")


def plot_isf_clean_mae() -> None:
    rows = pd.read_csv(ESTIMATOR_DIR / "clean_isf_estimator_summary.csv")
    wanted = ["clinical_prior", "empirical_mean", "gluca_modular_trust", "robust_huber_center"]
    labels = {
        "clinical_prior": "Clinical/profile proxy",
        "empirical_mean": "Empirical mean",
        "gluca_modular_trust": "Gluca Bayesian trust",
        "robust_huber_center": "Robust Huber",
    }
    cohorts = ["adolescents", "adults"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, cohort in zip(axes, cohorts):
        subset = rows[(rows["cohort"] == cohort) & (rows["estimator"].isin(wanted))].copy()
        subset["estimator"] = pd.Categorical(subset["estimator"], categories=wanted, ordered=True)
        subset = subset.sort_values("estimator")
        colors = [COLORS["gluca"] if item == "gluca_modular_trust" else COLORS["baseline"] for item in subset["estimator"]]
        x = np.arange(len(subset))
        ax.bar(x, subset["net_mae"], color=colors, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(labels[item], 13) for item in subset["estimator"]], rotation=0)
        ax.set_title(cohort.title())
        ax.set_ylabel("Holdout net-drop MAE (mg/dL)")
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.7)
        for idx, value in enumerate(subset["net_mae"]):
            ax.text(idx, value + max(subset["net_mae"]) * 0.025, f"{value:.1f}", ha="center", fontweight="bold")
    fig.suptitle("ISF Personalization: Future Correction-Response Error")
    save(fig, "02_isf_clean_correction_mae.png")


def plot_parameter_recovery() -> None:
    rows = pd.read_csv(OUTDIR / "pump_proxy_claim_rows.csv")
    rows = rows[rows["domain"].isin(["CR recovery", "Basal recovery", "Overall parameter recovery"])].copy()
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width / 2, rows["baseline_value"], width=width, color=COLORS["baseline"], label="Configured profile proxy")
    ax.bar(x + width / 2, rows["gluca_value"], width=width, color=COLORS["gluca"], label="Gluca Bayesian trust")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["domain"])
    ax.set_ylabel("Median absolute percent error")
    ax.set_title("Known Therapy-Parameter Recovery in RL4BG Adolescents")
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.7)
    ax.legend(frameon=False, loc="upper right")
    for idx, row in rows.reset_index(drop=True).iterrows():
        ax.text(idx - width / 2, row["baseline_value"] + 0.8, f"{row['baseline_value']:.1f}%", ha="center", fontweight="bold")
        ax.text(idx + width / 2, row["gluca_value"] + 0.8, f"{row['gluca_value']:.1f}%", ha="center", fontweight="bold")
    save(fig, "03_parameter_recovery_vs_profile_proxy.png")


def plot_significance_forest() -> None:
    rows = pd.read_csv(OUTDIR / "paired_significance_checks.csv")
    keep = [
        ("clean_isf_adolescents", "clinical_prior", "ISF adolescents\nvs profile"),
        ("clean_isf_adults", "clinical_prior", "ISF adults\nvs profile"),
        ("rl4bg_cr_recovery", "population_prior", "CR recovery\nvs profile"),
        ("rl4bg_basal_recovery", "population_prior", "Basal recovery\nvs profile"),
        ("rl4bg_overall_parameter_recovery", "population_prior", "Overall params\nvs profile"),
        ("carb_absorption_timing", "fixed_3h", "Absorption\nvs fixed 3h"),
        ("dawn_morning_rise", "no_dawn", "Dawn\nvs no-dawn"),
    ]
    selected = []
    for domain, baseline, label in keep:
        match = rows[(rows["domain"] == domain) & (rows["baseline"] == baseline)].iloc[0].to_dict()
        match["label"] = label
        selected.append(match)
    plot_rows = pd.DataFrame(selected)
    y = np.arange(len(plot_rows))
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    colors = np.where(plot_rows["mean_error_reduction"] > 0, COLORS["positive"], COLORS["negative"])
    xerr = np.vstack(
        [
            plot_rows["mean_error_reduction"] - plot_rows["ci95_low"],
            plot_rows["ci95_high"] - plot_rows["mean_error_reduction"],
        ]
    )
    ax.errorbar(
        plot_rows["mean_error_reduction"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#64748B",
        elinewidth=1.6,
        capsize=4,
        zorder=1,
    )
    ax.scatter(plot_rows["mean_error_reduction"], y, color=colors, s=58, zorder=2)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Paired mean error reduction, in each task's native units")
    ax.set_title("Paired Bootstrap Checks: Positive Values Favor Gluca")
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.7)
    for idx, row in plot_rows.iterrows():
        ax.text(
            row["ci95_high"] + 0.55,
            idx,
            f"P(+)= {row['bootstrap_prob_reduction_gt_zero']:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color="#334155",
        )
    ax.set_xlim(
        min(plot_rows["ci95_low"].min() - 2.0, -2.5),
        plot_rows["ci95_high"].max() + 4.8,
    )
    save(fig, "04_paired_error_reduction_forest.png")


def plot_absorption_and_meal() -> None:
    absorption = pd.read_csv(OUTDIR / "carb_absorption_summary.csv")
    absorption = absorption[absorption["estimator"].isin(["fixed_3h", "modular_curve_history"])].copy()
    absorption["label"] = absorption["estimator"].map({"fixed_3h": "Fixed 3h", "modular_curve_history": "Gluca curve history"})

    meal = pd.read_csv(OUTDIR / "meal_inference_detector_summary.csv")
    meal = meal[meal["detector"].isin(["bolus_only", "gluca_contextual_meal", "gluca_latent_meal", "schedule_prior"])].copy()
    meal["label"] = meal["detector"].map(
        {
            "bolus_only": "Bolus context",
            "gluca_contextual_meal": "Gluca contextual",
            "gluca_latent_meal": "Gluca latent only",
            "schedule_prior": "Schedule prior",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    colors = [COLORS["gluca"] if item == "modular_curve_history" else COLORS["baseline"] for item in absorption["estimator"]]
    ax.bar(absorption["label"], absorption["mae_hours"], color=colors, width=0.65)
    ax.set_title("Carb Absorption Timing")
    ax.set_ylabel("MAE (hours)")
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.7)
    for idx, value in enumerate(absorption["mae_hours"]):
        ax.text(idx, value + 0.04, f"{value:.2f}h", ha="center", fontweight="bold")

    ax = axes[1]
    colors = [COLORS["gluca"] if "Gluca" in label else COLORS["baseline"] for label in meal["label"]]
    ax.bar(meal["label"], meal["f1"], color=colors, width=0.65)
    ax.set_title("Meal Inference Event F1")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.7)
    for idx, value in enumerate(meal["f1"]):
        ax.text(idx, value + 0.03, f"{value:.2f}", ha="center", fontweight="bold")
    save(fig, "05_absorption_and_meal_inference.png")


def plot_strong_sanity_baselines() -> None:
    rows = pd.read_csv(OUTDIR / "strong_baseline_claim_rows.csv")
    rows = rows.copy()
    rows["plot_label"] = rows["domain"].map(lambda x: short_label(x, 24))
    rows["color"] = np.where(
        rows["value_add_supported"].astype(str).str.lower() == "true",
        COLORS["positive"],
        np.where(rows["comparison_supported"].astype(str).str.lower() == "true", COLORS["neutral"], COLORS["negative"]),
    )

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    y = np.arange(len(rows))
    ax.barh(y, rows["gluca_improvement_vs_baseline_pct"], color=rows["color"], height=0.68)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Improvement vs strongest non-Gluca sanity baseline (%)")
    ax.set_title("Sanity Check: Gluca vs Strong Non-Gluca Baselines")
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.7)
    for idx, value in enumerate(rows["gluca_improvement_vs_baseline_pct"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.8 if value >= 0 else -0.8
        ax.text(value + offset, idx, f"{value:.1f}%", va="center", ha=ha, fontweight="bold")
    ax.set_xlim(min(-15, rows["gluca_improvement_vs_baseline_pct"].min() - 6), max(45, rows["gluca_improvement_vs_baseline_pct"].max() + 8))
    save(fig, "06_strong_sanity_baselines.png")


def main() -> int:
    setup_style()
    plot_pump_proxy_improvements()
    plot_isf_clean_mae()
    plot_parameter_recovery()
    plot_significance_forest()
    plot_absorption_and_meal()
    plot_strong_sanity_baselines()
    print(f"Wrote plots to {PLOTS_DIR}")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
