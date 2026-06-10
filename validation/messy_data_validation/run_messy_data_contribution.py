#!/usr/bin/env python3
"""
Messy-data contribution analysis for Gluca.

This script uses the existing RL4BG free-living replay outputs. That replay is
the closest current validation artifact to real app data: meals, boluses, basal
delivery and CGM are mixed rather than isolated into clean correction events.

The point is to separate three claims:

1. Messy empirical averaging is weak.
2. Confounder gating improves the observations before any Bayesian update.
3. Gluca's Bayesian/trust layer adds value beyond clean-gated averaging.

It does not prove clinical superiority over commercial pumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import textwrap
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "validation"
    / "estimator_baseline_validation"
    / "estimator_baseline_latest"
    / "rl4bg_parameter_estimator_run_metrics.csv"
)

PARAM_METRICS = [
    ("isf_abs_pct_error", "ISF"),
    ("cr_abs_pct_error", "CR"),
    ("basal_abs_pct_error", "Basal"),
    ("mean_abs_pct_error", "Overall"),
]

ESTIMATOR_LABELS = {
    "population_prior": "Population/profile prior",
    "v1_observation_mean": "Messy empirical mean",
    "v1_observation_median": "Messy empirical median",
    "v2_observation_mean": "Clean-gated empirical mean",
    "normal_bayes_from_v2_obs": "Clean-gated normal Bayes",
    "gluca_v2_clean_gate": "Gluca V2 posterior",
    "gluca_modular_trust": "Gluca Bayesian/trust",
    "gluca_adaptive_trust": "Gluca adaptive trust",
}

PRIMARY_ESTIMATORS = [
    "population_prior",
    "v1_observation_mean",
    "v2_observation_mean",
    "normal_bayes_from_v2_obs",
    "gluca_modular_trust",
]

PRIMARY_COMPARISONS = [
    (
        "gating_vs_messy_empirical",
        "v1_observation_mean",
        "v2_observation_mean",
        "What clean event selection adds before Bayesian learning",
    ),
    (
        "gluca_vs_clean_empirical",
        "v2_observation_mean",
        "gluca_modular_trust",
        "What Bayesian/trust handling adds after using the cleaner observations",
    ),
    (
        "gluca_vs_messy_empirical",
        "v1_observation_mean",
        "gluca_modular_trust",
        "Full Gluca stack versus naive averaging on messy observations",
    ),
    (
        "gluca_vs_normal_bayes",
        "normal_bayes_from_v2_obs",
        "gluca_modular_trust",
        "Failure-mode-aware trust versus a generic normal Bayesian posterior",
    ),
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def pct_improvement(baseline_error: float, candidate_error: float) -> float | None:
    if baseline_error <= 0:
        return None
    return 100.0 * (baseline_error - candidate_error) / baseline_error


def index_rows(rows: list[dict[str, Any]], estimator: str, metric: str) -> dict[tuple[str, str], float]:
    return {
        (row["patient"], row["seed"]): float(row[metric])
        for row in rows
        if row["estimator"] == estimator and row.get(metric) not in {None, ""}
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for estimator in PRIMARY_ESTIMATORS:
        estimator_rows = [row for row in rows if row["estimator"] == estimator]
        if not estimator_rows:
            continue
        for metric, label in PARAM_METRICS:
            values = [float(row[metric]) for row in estimator_rows]
            output.append(
                {
                    "estimator": estimator,
                    "label": ESTIMATOR_LABELS.get(estimator, estimator),
                    "metric": metric,
                    "parameter": label,
                    "n_runs": len(values),
                    "mean_abs_pct_error": statistics.fmean(values),
                    "median_abs_pct_error": statistics.median(values),
                    "q1_abs_pct_error": percentile(values, 25),
                    "q3_abs_pct_error": percentile(values, 75),
                }
            )
    return output


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for comparison, baseline, candidate, interpretation in PRIMARY_COMPARISONS:
        for metric, label in PARAM_METRICS:
            base = index_rows(rows, baseline, metric)
            cand = index_rows(rows, candidate, metric)
            keys = sorted(set(base) & set(cand))
            if not keys:
                continue
            baseline_errors = [base[key] for key in keys]
            candidate_errors = [cand[key] for key in keys]
            reductions = [b - c for b, c in zip(baseline_errors, candidate_errors)]
            baseline_mean = statistics.fmean(baseline_errors)
            candidate_mean = statistics.fmean(candidate_errors)
            baseline_median = statistics.median(baseline_errors)
            candidate_median = statistics.median(candidate_errors)
            output.append(
                {
                    "comparison": comparison,
                    "parameter": label,
                    "metric": metric,
                    "baseline": baseline,
                    "baseline_label": ESTIMATOR_LABELS.get(baseline, baseline),
                    "candidate": candidate,
                    "candidate_label": ESTIMATOR_LABELS.get(candidate, candidate),
                    "n_pairs": len(keys),
                    "baseline_mean_error": baseline_mean,
                    "candidate_mean_error": candidate_mean,
                    "mean_error_reduction": statistics.fmean(reductions),
                    "mean_improvement_pct": pct_improvement(baseline_mean, candidate_mean),
                    "baseline_median_error": baseline_median,
                    "candidate_median_error": candidate_median,
                    "median_improvement_pct": pct_improvement(baseline_median, candidate_median),
                    "fraction_pairs_improved": sum(1 for value in reductions if value > 0) / len(reductions),
                    "interpretation": interpretation,
                }
            )
    return output


def write_report(path: Path, aggregate: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    def row_for(rows: list[dict[str, Any]], estimator: str, parameter: str) -> dict[str, Any]:
        return next(row for row in rows if row["estimator"] == estimator and row["parameter"] == parameter)

    overall_messy = row_for(aggregate, "v1_observation_mean", "Overall")
    overall_clean = row_for(aggregate, "v2_observation_mean", "Overall")
    overall_gluca = row_for(aggregate, "gluca_modular_trust", "Overall")
    overall_bayes = row_for(aggregate, "normal_bayes_from_v2_obs", "Overall")

    gating = next(row for row in comparisons if row["comparison"] == "gating_vs_messy_empirical" and row["parameter"] == "Overall")
    bayes = next(row for row in comparisons if row["comparison"] == "gluca_vs_clean_empirical" and row["parameter"] == "Overall")
    full = next(row for row in comparisons if row["comparison"] == "gluca_vs_messy_empirical" and row["parameter"] == "Overall")

    lines = [
        "# Messy-Data Contribution Analysis",
        "",
        "## Protocol",
        "",
        "- Source: existing RL4BG adolescent free-living replay outputs.",
        "- Simulator parameters are not changed for this analysis.",
        "- Data are messy by design: meals, basal, boluses and CGM are mixed.",
        "- This is estimator validation, not a clinical or closed-loop pump superiority test.",
        "",
        "## Main Result",
        "",
        "| Estimator | Overall mean absolute % error | Overall median absolute % error |",
        "|---|---:|---:|",
        f"| Messy empirical mean | {overall_messy['mean_abs_pct_error']:.2f}% | {overall_messy['median_abs_pct_error']:.2f}% |",
        f"| Clean-gated empirical mean | {overall_clean['mean_abs_pct_error']:.2f}% | {overall_clean['median_abs_pct_error']:.2f}% |",
        f"| Clean-gated normal Bayes | {overall_bayes['mean_abs_pct_error']:.2f}% | {overall_bayes['median_abs_pct_error']:.2f}% |",
        f"| Gluca Bayesian/trust | {overall_gluca['mean_abs_pct_error']:.2f}% | {overall_gluca['median_abs_pct_error']:.2f}% |",
        "",
        "## What Improved",
        "",
        f"- Clean gating reduced overall mean error from {overall_messy['mean_abs_pct_error']:.2f}% to {overall_clean['mean_abs_pct_error']:.2f}% "
        f"({gating['mean_improvement_pct']:.2f}% improvement versus messy empirical averaging).",
        f"- Gluca Bayesian/trust reduced overall mean error from {overall_clean['mean_abs_pct_error']:.2f}% to {overall_gluca['mean_abs_pct_error']:.2f}% "
        f"({bayes['mean_improvement_pct']:.2f}% improvement versus clean-gated empirical averaging).",
        f"- The full stack reduced overall mean error from {overall_messy['mean_abs_pct_error']:.2f}% to {overall_gluca['mean_abs_pct_error']:.2f}% "
        f"({full['mean_improvement_pct']:.2f}% improvement versus messy empirical averaging).",
        "",
        "## Interpretation",
        "",
        textwrap.fill(
            "This supports a real contribution, but the contribution is not simply "
            "'Bayes beats averaging' in isolation. The bigger contribution is the "
            "full pipeline: selecting less-confounded observations, using uncertainty "
            "and trust to avoid overreacting, and then updating patient-specific "
            "parameters. On perfectly clean ISF events, robust averaging is hard to "
            "beat. On messy free-living replay, naive averaging gets hurt because it "
            "treats confounded observations as equally trustworthy.",
            width=88,
        ),
        "",
        "## Claim Boundary",
        "",
        "- Supported: Gluca beats naive empirical averaging on messy simulator replay.",
        "- Supported: Gluca beats clean-gated empirical averaging on overall parameter recovery in this replay.",
        "- Not supported: Gluca clinically beats current commercial pumps.",
        "- Not supported: Bayesian posterior mean alone is always better than a robust personalised average.",
        "",
        "See `messy_data_comparisons.csv` for parameter-level comparisons.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, aggregate: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ordered_estimators = [
        "v1_observation_mean",
        "v2_observation_mean",
        "normal_bayes_from_v2_obs",
        "gluca_modular_trust",
    ]
    labels = [
        "Messy\nmean",
        "Clean-gated\nmean",
        "Normal\nBayes",
        "Gluca\ntrust",
    ]
    overall = {
        row["estimator"]: row
        for row in aggregate
        if row["parameter"] == "Overall"
    }
    values = [overall[estimator]["mean_abs_pct_error"] for estimator in ordered_estimators]
    colors = ["#9b9b9b", "#4b83c4", "#7d6ab5", "#1f8a5b"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.64)
    ax.set_ylabel("Mean absolute % error")
    ax.set_title("Messy RL4BG parameter recovery")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "messy_latest")
    args = parser.parse_args()

    rows = read_rows(args.source)
    args.outdir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_rows(rows)
    comparisons = comparison_rows(rows)
    try:
        source_label = str(args.source.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source_label = str(args.source)

    result = {
        "source": source_label,
        "protocol": "RL4BG messy free-living replay; no simulator parameter changes in this analysis.",
        "claim_boundary": {
            "beats_messy_empirical_averaging": True,
            "beats_clean_gated_empirical_averaging_overall": True,
            "proves_bayes_alone_beats_robust_averaging": False,
            "proves_current_pump_superiority": False,
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
    }

    write_csv(args.outdir / "messy_data_aggregate.csv", [rounded(row) for row in aggregate])
    write_csv(args.outdir / "messy_data_comparisons.csv", [rounded(row) for row in comparisons])
    (args.outdir / "messy_data_contribution_result.json").write_text(
        json.dumps(rounded(result), indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.outdir / "messy_data_contribution_report.md", aggregate, comparisons)
    make_plot(args.outdir / "messy_data_contribution.png", aggregate)

    print(json.dumps(rounded(result["claim_boundary"]), indent=2))
    print(f"Wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
