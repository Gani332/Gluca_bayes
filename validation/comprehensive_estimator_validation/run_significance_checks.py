#!/usr/bin/env python3
"""
Paired significance checks for the comprehensive Gluca validation outputs.

These checks do not rerun simulators. They consume the frozen event/run rows
from the estimator and comprehensive validation outputs and report paired
error reductions with bootstrap confidence intervals.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ESTIMATOR_DIR = REPO_ROOT / "validation" / "estimator_baseline_validation" / "estimator_baseline_latest"
COMPREHENSIVE_DIR = SCRIPT_DIR / "comprehensive_latest"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def round_float(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    return value


def rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return round_float(value)


def paired_stats(
    *,
    domain: str,
    metric: str,
    baseline: str,
    gluca: str,
    baseline_errors: list[float],
    gluca_errors: list[float],
    n_bootstrap: int = 10000,
    seed: int = 20260608,
) -> dict[str, Any]:
    if len(baseline_errors) != len(gluca_errors):
        raise ValueError("paired_stats requires equal-length paired errors")
    diffs = np.array([b - g for b, g in zip(baseline_errors, gluca_errors)], dtype=float)
    rng = np.random.default_rng(seed)
    if len(diffs) == 0:
        return {
            "domain": domain,
            "metric": metric,
            "baseline": baseline,
            "gluca": gluca,
            "n_pairs": 0,
        }
    samples = rng.choice(diffs, size=(n_bootstrap, len(diffs)), replace=True).mean(axis=1)
    baseline_mean = statistics.fmean(baseline_errors)
    gluca_mean = statistics.fmean(gluca_errors)
    reduction = statistics.fmean(diffs)
    return {
        "domain": domain,
        "metric": metric,
        "baseline": baseline,
        "gluca": gluca,
        "n_pairs": len(diffs),
        "baseline_mean_error": baseline_mean,
        "gluca_mean_error": gluca_mean,
        "mean_error_reduction": reduction,
        "improvement_pct": 100.0 * reduction / baseline_mean if baseline_mean > 0 else None,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "bootstrap_prob_reduction_gt_zero": float(np.mean(samples > 0.0)),
    }


def indexed_errors(rows: list[dict[str, str]], key_fields: tuple[str, ...], estimator: str, error_field: str) -> dict[tuple[str, ...], float]:
    output: dict[tuple[str, ...], float] = {}
    for row in rows:
        if row["estimator"] != estimator:
            continue
        key = tuple(row[field] for field in key_fields)
        output[key] = float(row[error_field])
    return output


def add_paired_comparison(
    results: list[dict[str, Any]],
    *,
    domain: str,
    metric: str,
    baseline: str,
    gluca: str,
    baseline_map: dict[tuple[str, ...], float],
    gluca_map: dict[tuple[str, ...], float],
) -> None:
    keys = sorted(set(baseline_map) & set(gluca_map))
    results.append(
        paired_stats(
            domain=domain,
            metric=metric,
            baseline=baseline,
            gluca=gluca,
            baseline_errors=[baseline_map[key] for key in keys],
            gluca_errors=[gluca_map[key] for key in keys],
        )
    )


def main() -> int:
    results: list[dict[str, Any]] = []

    clean_rows = read_csv(ESTIMATOR_DIR / "clean_isf_holdout_predictions.csv")
    clean_key = ("cohort", "patient", "event_index")
    gluca_clean = indexed_errors(clean_rows, clean_key, "gluca_modular_trust", "net_abs_error")
    for cohort in ["adolescents", "adults"]:
        cohort_gluca = {key: value for key, value in gluca_clean.items() if key[0] == cohort}
        for baseline in [
            "fixed_population_50",
            "clinical_prior",
            "empirical_mean",
            "empirical_median",
            "empirical_trimmed_mean",
            "robust_winsorized_mean",
            "robust_huber_center",
        ]:
            baseline_errors = indexed_errors(clean_rows, clean_key, baseline, "net_abs_error")
            cohort_baseline = {key: value for key, value in baseline_errors.items() if key[0] == cohort}
            add_paired_comparison(
                results,
                domain=f"clean_isf_{cohort}",
                metric="net_abs_error_mgdl",
                baseline=baseline,
                gluca="gluca_modular_trust",
                baseline_map=cohort_baseline,
                gluca_map=cohort_gluca,
            )

    rl4bg_rows = read_csv(ESTIMATOR_DIR / "rl4bg_parameter_estimator_run_metrics.csv")
    rl4bg_key = ("patient", "seed")
    for metric_field, domain in [
        ("cr_abs_pct_error", "rl4bg_cr_recovery"),
        ("basal_abs_pct_error", "rl4bg_basal_recovery"),
        ("mean_abs_pct_error", "rl4bg_overall_parameter_recovery"),
    ]:
        gluca_errors = indexed_errors(rl4bg_rows, rl4bg_key, "gluca_modular_trust", metric_field)
        for baseline in ["population_prior", "v2_observation_mean", "normal_bayes_from_v2_obs"]:
            add_paired_comparison(
                results,
                domain=domain,
                metric=metric_field,
                baseline=baseline,
                gluca="gluca_modular_trust",
                baseline_map=indexed_errors(rl4bg_rows, rl4bg_key, baseline, metric_field),
                gluca_map=gluca_errors,
            )

    absorption_rows = read_csv(COMPREHENSIVE_DIR / "carb_absorption_event_rows.csv")
    absorption_key = ("trace", "meal_minute")
    add_paired_comparison(
        results,
        domain="carb_absorption_timing",
        metric="abs_error_hours",
        baseline="fixed_3h",
        gluca="modular_curve_history",
        baseline_map=indexed_errors(absorption_rows, absorption_key, "fixed_3h", "abs_error_hours"),
        gluca_map=indexed_errors(absorption_rows, absorption_key, "modular_curve_history", "abs_error_hours"),
    )

    dawn_rows = read_csv(COMPREHENSIVE_DIR / "dawn_event_rows.csv")
    dawn_key = ("trace", "day")
    add_paired_comparison(
        results,
        domain="dawn_morning_rise",
        metric="abs_error_mgdl",
        baseline="no_dawn",
        gluca="gluca_adaptive_trust",
        baseline_map=indexed_errors(dawn_rows, dawn_key, "no_dawn", "abs_error_mgdl"),
        gluca_map=indexed_errors(dawn_rows, dawn_key, "gluca_adaptive_trust", "abs_error_mgdl"),
    )

    out_csv = COMPREHENSIVE_DIR / "paired_significance_checks.csv"
    out_json = COMPREHENSIVE_DIR / "paired_significance_checks.json"
    write_csv(out_csv, [rounded(row) for row in results])
    out_json.write_text(json.dumps(rounded(results), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded(results), indent=2), flush=True)
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
