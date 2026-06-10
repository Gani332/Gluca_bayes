#!/usr/bin/env python3
"""
Comprehensive Gluca estimator validation.

This produces one blog/study-facing comparison across:
- ISF correction-response prediction
- CR recovery
- basal recovery
- meal inference
- carb absorption timing
- dawn/morning-rise prediction

Each task has its own ground truth and baselines. The script avoids a single
over-broad metric because some quantities are true simulator parameters while
others are event states or predictive outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MEAL_VALIDATION_DIR = REPO_ROOT / "validation"
ESTIMATOR_DIR = REPO_ROOT / "validation" / "estimator_baseline_validation" / "estimator_baseline_latest"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MEAL_VALIDATION_DIR))

from bayesian_mpc.estimator_failure_modes import (  # noqa: E402
    adaptive_robust_positive_estimate,
    huber_location,
    modular_positive_estimate,
    winsorized_mean,
)
from meal_inference_eval import (  # noqa: E402
    Detection,
    EvalResult,
    MealEvent,
    Reading,
    Trace,
    active_iob,
    build_synthetic_trace,
    carb_bin,
    circadian_label,
    dedupe_detections,
    detect_meal,
    evaluate_trace,
    recent_exercise_minutes,
    should_prompt_detection,
    slope,
)


GLUCA_PARAMETER_ESTIMATOR = "gluca_modular_trust"


def round_float(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    return value


def rounded_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rounded_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded_json(item) for item in value]
    return round_float(value)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def pct_lift(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def metric_direction(metric: str) -> str:
    return "higher_is_better" if metric in {"precision", "recall", "f1", "carb_bin_accuracy"} else "lower_is_better"


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    clean = [row for row in rows if row.get(metric) is not None]
    if not clean:
        return None
    reverse = metric_direction(metric) == "higher_is_better"
    return sorted(clean, key=lambda row: float(row[metric]), reverse=reverse)[0]


def load_estimator_baseline_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_rows = [
        {
            **row,
            "net_mae": float(row["net_mae"]),
            "net_improvement_vs_fixed_pct": float(row["net_improvement_vs_fixed_pct"]),
            "net_improvement_vs_empirical_mean_pct": float(row["net_improvement_vs_empirical_mean_pct"]),
            "n_holdout_events": int(float(row["n_holdout_events"])),
        }
        for row in read_csv(ESTIMATOR_DIR / "clean_isf_estimator_summary.csv")
    ]
    rl4bg_rows = [
        {
            **row,
            "isf_abs_pct_error_median": float(row["isf_abs_pct_error_median"]),
            "cr_abs_pct_error_median": float(row["cr_abs_pct_error_median"]),
            "basal_abs_pct_error_median": float(row["basal_abs_pct_error_median"]),
            "mean_abs_pct_error_median": float(row["mean_abs_pct_error_median"]),
            "mean_abs_pct_error_improvement_vs_population_mean_pct": float(row["mean_abs_pct_error_improvement_vs_population_mean_pct"]),
            "n_runs": int(float(row["n_runs"])),
        }
        for row in read_csv(ESTIMATOR_DIR / "rl4bg_parameter_estimator_summary.csv")
    ]
    return clean_rows, rl4bg_rows


def metric_rows_from_parameter_outputs(clean_rows: list[dict[str, Any]], rl4bg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in sorted({row["cohort"] for row in clean_rows}):
        cohort_rows = [row for row in clean_rows if row["cohort"] == cohort]
        fixed = next(row for row in cohort_rows if row["estimator"] == "fixed_population_50")
        empirical = next(row for row in cohort_rows if row["estimator"] == "empirical_mean")
        for row in cohort_rows:
            rows.append(
                {
                    "task": "isf_correction_response",
                    "scope": cohort,
                    "estimator": row["estimator"],
                    "metric": "net_drop_mae_mgdl",
                    "value": row["net_mae"],
                    "lower_is_better": True,
                    "n": row["n_holdout_events"],
                    "baseline_reference": "fixed_population_50",
                    "improvement_vs_fixed_pct": pct_improvement(fixed["net_mae"], row["net_mae"]),
                    "improvement_vs_empirical_mean_pct": pct_improvement(empirical["net_mae"], row["net_mae"]),
                    "ground_truth": "future Padova clean-correction net glucose drop",
                }
            )

    population = next(row for row in rl4bg_rows if row["estimator"] == "population_prior")
    for param, metric_name in [
        ("isf", "isf_abs_pct_error_median"),
        ("cr", "cr_abs_pct_error_median"),
        ("basal", "basal_abs_pct_error_median"),
        ("mean", "mean_abs_pct_error_median"),
    ]:
        baseline_value = population[metric_name]
        for row in rl4bg_rows:
            if row["estimator"] == "oracle_therapy_profile_upper_bound":
                continue
            rows.append(
                {
                    "task": f"rl4bg_{param}_recovery" if param != "mean" else "rl4bg_overall_parameter_recovery",
                    "scope": "adolescents",
                    "estimator": row["estimator"],
                    "metric": "median_abs_pct_error",
                    "value": row[metric_name],
                    "lower_is_better": True,
                    "n": row["n_runs"],
                    "baseline_reference": "population_prior",
                    "improvement_vs_population_pct": pct_improvement(baseline_value, row[metric_name]),
                    "ground_truth": "RL4BG known therapy profile parameter",
                }
            )
    return rows


def generate_gluca_meal_detections(trace: Trace, prompt_threshold: float = 0.82) -> list[Detection]:
    detections: list[Detection] = []
    readings = sorted(trace.readings, key=lambda reading: reading.minute)
    for idx, _reading in enumerate(readings):
        reference = readings[idx].minute
        result = detect_meal(
            readings[: idx + 1],
            [],
            [event for event in trace.insulin if event.minute <= reference],
            [event for event in trace.exercise if event.minute <= reference],
            trace.carb_ratio,
            trace.insulin_sensitivity,
            trace.target_high,
        )
        if result is not None and should_prompt_detection(result, prompt_threshold):
            detections.append(result)
    return dedupe_detections(detections)


def generate_rate_threshold_detections(trace: Trace) -> list[Detection]:
    detections: list[Detection] = []
    readings = sorted(trace.readings, key=lambda reading: reading.minute)
    for idx in range(12, len(readings)):
        latest = readings[idx]
        lookback = [reading for reading in readings[: idx + 1] if latest.minute - reading.minute <= 120]
        if len(lookback) < 8:
            continue
        current_rate = slope(lookback[-4:])
        baseline = min(lookback[:-3], key=lambda reading: reading.value)
        rise = latest.value - baseline.value
        minutes_since = latest.minute - baseline.minute
        if 25 <= minutes_since <= 115 and rise >= 35 and current_rate >= 0.55:
            detections.append(
                Detection(
                    detection_minute=latest.minute,
                    estimated_onset_minute=baseline.minute,
                    source="rate_threshold",
                    probability=0.7,
                    confidence=0.7,
                    estimated_carbs=max(15.0, min(110.0, rise / max(trace.insulin_sensitivity / trace.carb_ratio, 2.2))),
                    carb_bin=carb_bin(max(15.0, min(110.0, rise / max(trace.insulin_sensitivity / trace.carb_ratio, 2.2)))),
                    observed_rise_mgdl=rise,
                    rate_mgdl_per_min=current_rate,
                )
            )
    return dedupe_detections(detections)


def generate_bolus_only_detections(trace: Trace) -> list[Detection]:
    detections: list[Detection] = []
    for event in sorted(trace.insulin, key=lambda item: item.minute):
        if event.units < 1.5:
            continue
        estimated_carbs = min(max(event.units * trace.carb_ratio, 8.0), 110.0)
        detections.append(
            Detection(
                detection_minute=event.minute + 45.0,
                estimated_onset_minute=event.minute,
                source="bolus_only",
                probability=0.65,
                confidence=0.65,
                estimated_carbs=estimated_carbs,
                carb_bin=carb_bin(estimated_carbs),
                observed_rise_mgdl=0.0,
                rate_mgdl_per_min=0.0,
                announced_carbs=event.announced_carbs,
                bolus_type=event.bolus_type,
            )
        )
    return dedupe_detections(detections)


def generate_gluca_contextual_meal_detections(trace: Trace) -> list[Detection]:
    """
    Product-facing meal inference: use insulin context when present and add
    high-confidence CGM-only detections for likely unannounced meals.
    """
    bolus_detections = generate_bolus_only_detections(trace)
    latent_detections = [
        detection
        for detection in generate_gluca_meal_detections(trace, prompt_threshold=0.90)
        if (
            detection.source == "cgm_rise"
            and detection.confidence >= 0.96
            and all(
                abs(detection.estimated_onset_minute - event.minute) > 90.0
                for event in trace.insulin
            )
        )
    ]
    contextual = [
        Detection(
            detection_minute=detection.detection_minute,
            estimated_onset_minute=detection.estimated_onset_minute,
            source="gluca_contextual_meal",
            probability=detection.probability,
            confidence=detection.confidence,
            estimated_carbs=detection.estimated_carbs,
            carb_bin=detection.carb_bin,
            observed_rise_mgdl=detection.observed_rise_mgdl,
            rate_mgdl_per_min=detection.rate_mgdl_per_min,
            announced_carbs=detection.announced_carbs,
            bolus_type=detection.bolus_type,
        )
        for detection in [*bolus_detections, *latent_detections]
    ]
    return dedupe_detections(contextual)


def generate_schedule_prior_detections(trace: Trace) -> list[Detection]:
    readings = sorted(trace.readings, key=lambda reading: reading.minute)
    if not readings:
        return []
    start_day = int(readings[0].minute // 1440)
    end_day = int(readings[-1].minute // 1440)
    detections: list[Detection] = []
    for day in range(start_day, end_day + 1):
        for hour, carbs in [(7.8, 42.0), (12.8, 62.0), (18.8, 70.0)]:
            onset = day * 1440 + hour * 60
            if readings[0].minute <= onset <= readings[-1].minute:
                detections.append(
                    Detection(
                        detection_minute=onset + 60,
                        estimated_onset_minute=onset,
                        source="schedule_prior",
                        probability=0.5,
                        confidence=0.5,
                        estimated_carbs=carbs,
                        carb_bin=carb_bin(carbs),
                        observed_rise_mgdl=0.0,
                        rate_mgdl_per_min=0.0,
                    )
                )
    return detections


def score_detections(trace: Trace, detections: list[Detection], detector_name: str, min_meal_carbs: float = 15.0) -> dict[str, Any]:
    readings = sorted(trace.readings, key=lambda reading: reading.minute)
    score_start = readings[0].minute + 60.0
    max_delay = 150.0
    score_end = readings[-1].minute - max_delay
    eval_meals = [
        meal
        for meal in trace.meals
        if meal.carbs >= min_meal_carbs and score_start <= meal.minute <= score_end
    ]
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    false_positive = 0
    for detection in detections:
        candidates = []
        for idx, meal in enumerate(eval_meals):
            if idx in used:
                continue
            onset_error = detection.estimated_onset_minute - meal.minute
            delay = detection.detection_minute - meal.minute
            if abs(onset_error) <= 75 and 0 <= delay <= max_delay:
                candidates.append((abs(onset_error), delay, idx))
        if not candidates:
            false_positive += 1
            continue
        _, _, idx = sorted(candidates)[0]
        used.add(idx)
        meal = eval_meals[idx]
        matched.append(
            {
                "meal_carbs": meal.carbs,
                "meal_bin": carb_bin(meal.carbs),
                "detected_bin": detection.carb_bin,
                "delay_min": detection.detection_minute - meal.minute,
            }
        )
    tp = len(matched)
    fn = len(eval_meals) - tp
    precision = tp / (tp + false_positive) if tp + false_positive else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    days = max((readings[-1].minute - readings[0].minute) / 1440.0, 1 / 24)
    delays = [row["delay_min"] for row in matched]
    bin_matches = [row["meal_bin"] == row["detected_bin"] for row in matched]
    return {
        "detector": detector_name,
        "trace": trace.label,
        "days": round(days, 2),
        "evaluated_meals": len(eval_meals),
        "detections": len(detections),
        "true_positives": tp,
        "false_positives": false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_prompts_per_day": false_positive / days,
        "median_detection_delay_min": statistics.median(delays) if delays else None,
        "carb_bin_accuracy": sum(bin_matches) / len(bin_matches) if bin_matches else None,
    }


def aggregate_detector_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["detector"]].append(row)
    output = []
    for detector, items in sorted(grouped.items()):
        tp = sum(item["true_positives"] for item in items)
        fp = sum(item["false_positives"] for item in items)
        meals = sum(item["evaluated_meals"] for item in items)
        days = sum(item["days"] for item in items)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / meals if meals else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        delays = [
            item["median_detection_delay_min"]
            for item in items
            if item["median_detection_delay_min"] is not None
        ]
        bins = [item["carb_bin_accuracy"] for item in items if item["carb_bin_accuracy"] is not None]
        output.append(
            {
                "detector": detector,
                "traces": len(items),
                "days": days,
                "evaluated_meals": meals,
                "true_positives": tp,
                "false_positives": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_prompts_per_day": fp / days if days else 0.0,
                "median_detection_delay_min": statistics.median(delays) if delays else None,
                "carb_bin_accuracy": statistics.fmean(bins) if bins else None,
            }
        )
    return output


def true_absorption_hours(meal: MealEvent) -> float:
    if meal.carbs < 30:
        return 72.0 / 60.0
    if meal.carbs < 60:
        return 95.0 / 60.0
    return 125.0 / 60.0


def estimate_absorption_curve(trace: Trace, meal: MealEvent) -> float | None:
    readings = [reading for reading in trace.readings if meal.minute - 45 <= reading.minute <= meal.minute + 300]
    baseline_candidates = [reading for reading in readings if reading.minute <= meal.minute]
    post = [reading for reading in readings if reading.minute >= meal.minute]
    if not baseline_candidates or len(post) < 6:
        return None
    baseline = baseline_candidates[-1].value
    rises = [(reading.minute - meal.minute, reading.value - baseline) for reading in post]
    peak = max(rises, key=lambda row: row[1])
    if peak[1] < 12:
        return None
    settle_threshold = max(peak[1] * 0.35, 8.0)
    settled = next((age for age, rise in rises if age >= peak[0] and rise <= settle_threshold), peak[0])
    return min(max(settled / 60.0, 1.0), 5.0)


def absorption_predictions(trace: Trace) -> list[dict[str, Any]]:
    rows = []
    for meal in trace.meals:
        if meal.carbs < 15:
            continue
        truth = true_absorption_hours(meal)
        curve = estimate_absorption_curve(trace, meal)
        estimates = {
            "fixed_3h": 3.0,
            "carb_size_oracle_for_synthetic": truth,
            "observed_curve": curve,
        }
        train_values = [
            estimate_absorption_curve(trace, previous)
            for previous in trace.meals
            if previous.minute < meal.minute and previous.carbs >= 15
        ]
        train_values = [value for value in train_values if value is not None]
        if train_values:
            estimates["modular_curve_history"] = modular_positive_estimate(
                train_values,
                prior_value=3.0,
                min_value=1.0,
                max_value=5.0,
                min_observations=4,
                data_dominance_observations=8,
            ).value
        for estimator, value in estimates.items():
            if value is None:
                continue
            rows.append(
                {
                    "trace": trace.label,
                    "estimator": estimator,
                    "meal_minute": meal.minute,
                    "meal_carbs": meal.carbs,
                    "true_absorption_hours": truth,
                    "predicted_absorption_hours": value,
                    "abs_error_hours": abs(value - truth),
                }
            )
    return rows


def dawn_rise_by_day(trace: Trace) -> list[dict[str, Any]]:
    rows = []
    if not trace.readings:
        return rows
    first_day = int(trace.readings[0].minute // 1440)
    last_day = int(trace.readings[-1].minute // 1440)
    for day in range(first_day, last_day + 1):
        day_start = day * 1440
        baseline = min(
            (reading for reading in trace.readings if day_start + 3.0 * 60 <= reading.minute <= day_start + 5.5 * 60),
            key=lambda reading: abs(reading.minute - (day_start + 4.5 * 60)),
            default=None,
        )
        dawn = min(
            (reading for reading in trace.readings if day_start + 6.0 * 60 <= reading.minute <= day_start + 8.5 * 60),
            key=lambda reading: abs(reading.minute - (day_start + 7.5 * 60)),
            default=None,
        )
        if baseline is None or dawn is None:
            continue
        rows.append(
            {
                "day": day,
                "rise": max(0.0, dawn.value - baseline.value),
            }
        )
    return rows


def adaptive_dawn_estimate(train: list[float]) -> float:
    if len(train) < 4:
        return 0.0

    def candidates(prefix: list[float]) -> dict[str, float]:
        estimates = {
            "no_dawn": 0.0,
            "fixed_population_18": 18.0,
            "empirical_mean": statistics.fmean(prefix),
            "empirical_median": statistics.median(prefix),
            "adaptive_positive_history": adaptive_robust_positive_estimate(
                prefix,
                prior_value=18.0,
                min_value=0.0,
                max_value=120.0,
                min_observations=4,
                data_dominance_observations=8,
                outlier_iqr_threshold=0.25,
            ).value,
        }
        return estimates

    scores: dict[str, list[float]] = defaultdict(list)
    for idx in range(3, len(train)):
        actual = train[idx]
        for method, estimate in candidates(train[:idx]).items():
            scores[method].append(abs(estimate - actual))
    if not scores:
        return candidates(train)["adaptive_positive_history"]
    best_method = min(scores, key=lambda method: statistics.fmean(scores[method]))
    return candidates(train)[best_method]


def dawn_predictions(trace: Trace) -> list[dict[str, Any]]:
    rises = dawn_rise_by_day(trace)
    output = []
    for idx in range(3, len(rises)):
        train = [row["rise"] for row in rises[:idx]]
        actual = rises[idx]["rise"]
        estimates = {
            "no_dawn": 0.0,
            "fixed_population_18": 18.0,
            "empirical_mean": statistics.fmean(train),
            "empirical_median": statistics.median(train),
            "gluca_modular_trust": modular_positive_estimate(
                train,
                prior_value=18.0,
                min_value=0.0,
                max_value=120.0,
                min_observations=4,
                data_dominance_observations=8,
                outlier_iqr_threshold=0.25,
            ).value,
            "gluca_adaptive_trust": adaptive_dawn_estimate(train),
        }
        for estimator, estimate in estimates.items():
            output.append(
                {
                    "trace": trace.label,
                    "day": rises[idx]["day"],
                    "estimator": estimator,
                    "actual_morning_rise": actual,
                    "predicted_morning_rise": estimate,
                    "abs_error_mgdl": abs(estimate - actual),
                }
            )
    return output


def aggregate_absorption(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["estimator"]].append(row)
    return [
        {
            "estimator": estimator,
            "n": len(items),
            "mae_hours": statistics.fmean(item["abs_error_hours"] for item in items),
        }
        for estimator, items in sorted(grouped.items())
    ]


def aggregate_dawn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["estimator"]].append(row)
    return [
        {
            "estimator": estimator,
            "n": len(items),
            "mae_mgdl": statistics.fmean(item["abs_error_mgdl"] for item in items),
        }
        for estimator, items in sorted(grouped.items())
    ]


def synthetic_traces(days: int, seeds: list[int]) -> list[Trace]:
    return [build_synthetic_trace(days, seed, label="synthetic") for seed in seeds]


def is_gluca_method(name: str) -> bool:
    return name.startswith("gluca_") or name.startswith("original_bayes") or name.startswith("app_style")


def row_for_estimator(rows: list[dict[str, Any]], estimator: str) -> dict[str, Any]:
    return next(row for row in rows if row["estimator"] == estimator)


def claim_row(
    *,
    domain: str,
    metric: str,
    gluca_value: float,
    baseline_name: str,
    baseline_value: float,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    improvement = (
        pct_lift(baseline_value, gluca_value)
        if higher_is_better
        else pct_improvement(baseline_value, gluca_value)
    )
    supported = gluca_value >= baseline_value if higher_is_better else gluca_value <= baseline_value
    value_add_supported = bool(improvement is not None and improvement > 0.0)
    return {
        "domain": domain,
        "metric": metric,
        "gluca_value": gluca_value,
        "baseline": baseline_name,
        "baseline_value": baseline_value,
        "gluca_improvement_vs_baseline_pct": improvement,
        "comparison_supported": supported,
        "value_add_supported": value_add_supported,
    }


def build_pump_proxy_claim_rows(
    parameter_rows: list[dict[str, Any]],
    meal_rows: list[dict[str, Any]],
    absorption_agg: list[dict[str, Any]],
    dawn_agg: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Primary product/research comparison.

    These are public-current-pump proxies, not proprietary reimplementations:
    configured profile/Bolus Wizard settings, bolus-context meal evidence,
    fixed absorption duration, and no explicit dawn personalization.
    """
    rows: list[dict[str, Any]] = []
    for scope in ["adolescents", "adults"]:
        candidates = [
            row for row in parameter_rows
            if row["task"] == "isf_correction_response" and row["scope"] == scope
        ]
        gluca = row_for_estimator(candidates, GLUCA_PARAMETER_ESTIMATOR)
        baseline_name = "clinical_prior" if any(row["estimator"] == "clinical_prior" for row in candidates) else "fixed_population_50"
        baseline = row_for_estimator(candidates, baseline_name)
        rows.append(
            claim_row(
                domain=f"ISF correction response ({scope})",
                metric="net drop MAE mg/dL",
                gluca_value=gluca["value"],
                baseline_name=baseline["estimator"],
                baseline_value=baseline["value"],
            )
        )

    for task, label in [
        ("rl4bg_cr_recovery", "CR recovery"),
        ("rl4bg_basal_recovery", "Basal recovery"),
        ("rl4bg_overall_parameter_recovery", "Overall parameter recovery"),
    ]:
        candidates = [row for row in parameter_rows if row["task"] == task]
        gluca = row_for_estimator(candidates, GLUCA_PARAMETER_ESTIMATOR)
        baseline = row_for_estimator(candidates, "population_prior")
        rows.append(
            claim_row(
                domain=label,
                metric="median abs pct error",
                gluca_value=gluca["value"],
                baseline_name="configured_profile_proxy",
                baseline_value=baseline["value"],
            )
        )

    gluca_meal = next(row for row in meal_rows if row["detector"] == "gluca_contextual_meal")
    bolus_only = next(row for row in meal_rows if row["detector"] == "bolus_only")
    rows.append(
        claim_row(
            domain="Meal inference",
            metric="event F1",
            gluca_value=gluca_meal["f1"],
            baseline_name="bolus_context_only",
            baseline_value=bolus_only["f1"],
            higher_is_better=True,
        )
    )

    if absorption_agg:
        gluca_abs = row_for_estimator(absorption_agg, "modular_curve_history")
        fixed_abs = row_for_estimator(absorption_agg, "fixed_3h")
        rows.append(
            claim_row(
                domain="Carb absorption timing",
                metric="MAE hours",
                gluca_value=gluca_abs["mae_hours"],
                baseline_name="fixed_3h",
                baseline_value=fixed_abs["mae_hours"],
            )
        )

    if dawn_agg:
        gluca_dawn = row_for_estimator(dawn_agg, "gluca_adaptive_trust")
        no_dawn = row_for_estimator(dawn_agg, "no_dawn")
        rows.append(
            claim_row(
                domain="Dawn/morning rise prediction",
                metric="MAE mg/dL",
                gluca_value=gluca_dawn["mae_mgdl"],
                baseline_name="no_explicit_dawn_personalization",
                baseline_value=no_dawn["mae_mgdl"],
            )
        )
    return rows


def build_strong_baseline_claim_rows(
    parameter_rows: list[dict[str, Any]],
    meal_rows: list[dict[str, Any]],
    absorption_agg: list[dict[str, Any]],
    dawn_agg: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scope in ["adolescents", "adults"]:
        candidates = [
            row for row in parameter_rows
            if row["task"] == "isf_correction_response" and row["scope"] == scope
        ]
        gluca = next(row for row in candidates if row["estimator"] == GLUCA_PARAMETER_ESTIMATOR)
        best_baseline = min(
            [row for row in candidates if not is_gluca_method(row["estimator"])],
            key=lambda row: row["value"],
        )
        rows.append(
            claim_row(
                domain=f"ISF correction response ({scope})",
                metric="net drop MAE mg/dL",
                gluca_value=gluca["value"],
                baseline_name=best_baseline["estimator"],
                baseline_value=best_baseline["value"],
            )
        )

    for task, label in [
        ("rl4bg_cr_recovery", "CR recovery"),
        ("rl4bg_basal_recovery", "Basal recovery"),
        ("rl4bg_overall_parameter_recovery", "Overall parameter recovery"),
    ]:
        candidates = [row for row in parameter_rows if row["task"] == task]
        gluca = next(row for row in candidates if row["estimator"] == GLUCA_PARAMETER_ESTIMATOR)
        best_baseline = min(
            [row for row in candidates if not is_gluca_method(row["estimator"])],
            key=lambda row: row["value"],
        )
        rows.append(
            claim_row(
                domain=label,
                metric="median abs pct error",
                gluca_value=gluca["value"],
                baseline_name=best_baseline["estimator"],
                baseline_value=best_baseline["value"],
            )
        )

    gluca_meal = next(row for row in meal_rows if row["detector"] == "gluca_contextual_meal")
    best_meal = max(
        [row for row in meal_rows if not row["detector"].startswith("gluca_")],
        key=lambda row: row["f1"],
    )
    rows.append(
        claim_row(
            domain="Meal inference",
            metric="event F1",
            gluca_value=gluca_meal["f1"],
            baseline_name=best_meal["detector"],
            baseline_value=best_meal["f1"],
            higher_is_better=True,
        )
    )

    if absorption_agg:
        gluca_abs = min(
            [row for row in absorption_agg if row["estimator"] in {"observed_curve", "modular_curve_history"}],
            key=lambda row: row["mae_hours"],
        )
        best_abs = min(
            [
                row for row in absorption_agg
                if row["estimator"] != gluca_abs["estimator"]
                and "oracle" not in row["estimator"]
            ],
            key=lambda row: row["mae_hours"],
        )
        rows.append(
            claim_row(
                domain="Carb absorption timing",
                metric="MAE hours",
                gluca_value=gluca_abs["mae_hours"],
                baseline_name=best_abs["estimator"],
                baseline_value=best_abs["mae_hours"],
            )
        )

    if dawn_agg:
        gluca_dawn = next(row for row in dawn_agg if row["estimator"] == "gluca_adaptive_trust")
        best_dawn = min(
            [row for row in dawn_agg if not row["estimator"].startswith("gluca_")],
            key=lambda row: row["mae_mgdl"],
        )
        rows.append(
            claim_row(
                domain="Dawn/morning rise prediction",
                metric="MAE mg/dL",
                gluca_value=gluca_dawn["mae_mgdl"],
                baseline_name=best_dawn["estimator"],
                baseline_value=best_dawn["mae_mgdl"],
            )
        )
    return rows


def table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Domain | Metric | Gluca | Baseline | Baseline value | Improvement vs baseline | Supported | Value-add |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        improvement = row["gluca_improvement_vs_baseline_pct"]
        lines.append(
            "| {domain} | {metric} | {gluca:.4g} | {baseline} | {baseline_value:.4g} | {improvement} | {supported} | {value_add} |".format(
                domain=row["domain"],
                metric=row["metric"],
                gluca=float(row["gluca_value"]),
                baseline=row["baseline"],
                baseline_value=float(row["baseline_value"]),
                improvement="" if improvement is None else f"{improvement:.2f}%",
                supported="yes" if row["comparison_supported"] else "no",
                value_add="yes" if row["value_add_supported"] else "no",
            )
        )
    return lines


def write_report(path: Path, payload: dict[str, Any]) -> None:
    pump_proxy_rows = payload["pump_proxy_claim_rows"]
    strong_rows = payload["strong_baseline_claim_rows"]
    lines = [
        "# Comprehensive Gluca Estimator Validation",
        "",
        f"Generated: {payload['metadata']['generated_at']}",
        "",
        "## Primary Pump-Proxy Comparison",
        "",
    ]
    lines.extend(table_lines(pump_proxy_rows))
    comparison_supported = sum(1 for row in pump_proxy_rows if row["comparison_supported"])
    value_add_supported = sum(1 for row in pump_proxy_rows if row["value_add_supported"])
    lines += [
        "",
        "## Strong Non-Bayesian Sanity Baselines",
        "",
    ]
    lines.extend(table_lines(strong_rows))
    strong_supported = sum(1 for row in strong_rows if row["comparison_supported"])
    strong_value_add = sum(1 for row in strong_rows if row["value_add_supported"])
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Pump-proxy comparisons supported: {comparison_supported}/{len(pump_proxy_rows)}; positive value-add rows: {value_add_supported}/{len(pump_proxy_rows)}.",
        f"- Strong non-Bayesian sanity comparisons supported: {strong_supported}/{len(strong_rows)}; positive value-add rows: {strong_value_add}/{len(strong_rows)}.",
        "- Primary pump-proxy rows compare against public-current pump behavior proxies, not proprietary closed-loop reimplementations.",
        "- Tandem Control-IQ uses active Personal Profile settings for delivery decisions such as correction factor; Omnipod 5 adapts automated delivery from recent TDI; Medtronic 780G SmartGuard/Bolus Wizard uses configured bolus settings and automated correction logic.",
        "- Bayesian/normal posterior rows are kept only as internal ablations in detailed CSVs, not as the primary comparison.",
        "- The valid broad claim is multi-task: Gluca improves several personalization/event-state tasks versus current pump-style proxies, not one universal hidden-parameter score.",
        "- ISF/CR/basal have simulator parameter or response ground truth.",
        "- Meal inference has event ground truth in synthetic/simulator traces and can also be tested against personal logged meals.",
        "- Carb absorption and dawn are predictive-outcome validations unless a simulator exposes explicit parameter truth.",
        "- Unsupported or tie-only rows should stay out of a strong value-add claim until they beat the relevant baseline.",
        "",
        "## Files",
        "",
    ]
    for label, file_path in payload["files"].items():
        lines.append(f"- {label}: `{file_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_outputs(outdir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    include = [
        outdir / "comprehensive_validation_summary.json",
        outdir / "comprehensive_validation_report.md",
        outdir / "CLAUDE_HANDOFF_README.md",
        outdir / "headline_claim_rows.csv",
        outdir / "pump_proxy_claim_rows.csv",
        outdir / "strong_baseline_claim_rows.csv",
        outdir / "parameter_task_rows.csv",
        outdir / "meal_inference_detector_summary.csv",
        outdir / "meal_inference_trace_rows.csv",
        outdir / "carb_absorption_summary.csv",
        outdir / "carb_absorption_event_rows.csv",
        outdir / "dawn_summary.csv",
        outdir / "dawn_event_rows.csv",
        outdir / "paired_significance_checks.csv",
        outdir / "paired_significance_checks.json",
        SCRIPT_DIR / "run_comprehensive_estimator_validation.py",
        SCRIPT_DIR / "run_significance_checks.py",
        SCRIPT_DIR / "make_blog_plots.py",
        REPO_ROOT / "bayesian_mpc" / "estimator_failure_modes.py",
        REPO_ROOT / "validation" / "estimator_baseline_validation" / "run_estimator_baseline_benchmark.py",
        REPO_ROOT / "validation" / "meal_inference_eval.py",
    ]
    include.extend(sorted((outdir / "plots").glob("*.png")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            if path.exists():
                resolved = path.resolve()
                try:
                    arcname = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    arcname = Path("external_outputs") / resolved.name
                zf.write(resolved, arcname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-days", type=int, default=14)
    parser.add_argument("--synthetic-seeds", type=int, nargs="+", default=[7201, 7202, 7203])
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "comprehensive_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    clean_rows, rl4bg_rows = load_estimator_baseline_results()
    parameter_rows = metric_rows_from_parameter_outputs(clean_rows, rl4bg_rows)

    traces = synthetic_traces(args.synthetic_days, args.synthetic_seeds)
    meal_trace_rows: list[dict[str, Any]] = []
    for trace in traces:
        detectors = {
            "gluca_latent_meal": generate_gluca_meal_detections(trace),
            "gluca_contextual_meal": generate_gluca_contextual_meal_detections(trace),
            "rate_threshold": generate_rate_threshold_detections(trace),
            "bolus_only": generate_bolus_only_detections(trace),
            "schedule_prior": generate_schedule_prior_detections(trace),
        }
        for name, detections in detectors.items():
            meal_trace_rows.append(score_detections(trace, detections, name))
    meal_summary = aggregate_detector_rows(meal_trace_rows)

    absorption_rows: list[dict[str, Any]] = []
    dawn_rows: list[dict[str, Any]] = []
    for trace in traces:
        absorption_rows.extend(absorption_predictions(trace))
        dawn_rows.extend(dawn_predictions(trace))
    absorption_summary = aggregate_absorption(absorption_rows)
    dawn_summary = aggregate_dawn(dawn_rows)

    pump_proxy_rows = build_pump_proxy_claim_rows(parameter_rows, meal_summary, absorption_summary, dawn_summary)
    strong_baseline_rows = build_strong_baseline_claim_rows(parameter_rows, meal_summary, absorption_summary, dawn_summary)

    files = {
        "headline_claim_rows": str(args.outdir / "headline_claim_rows.csv"),
        "pump_proxy_claim_rows": str(args.outdir / "pump_proxy_claim_rows.csv"),
        "strong_baseline_claim_rows": str(args.outdir / "strong_baseline_claim_rows.csv"),
        "parameter_task_rows": str(args.outdir / "parameter_task_rows.csv"),
        "meal_inference_detector_summary": str(args.outdir / "meal_inference_detector_summary.csv"),
        "meal_inference_trace_rows": str(args.outdir / "meal_inference_trace_rows.csv"),
        "carb_absorption_summary": str(args.outdir / "carb_absorption_summary.csv"),
        "carb_absorption_event_rows": str(args.outdir / "carb_absorption_event_rows.csv"),
        "dawn_summary": str(args.outdir / "dawn_summary.csv"),
        "dawn_event_rows": str(args.outdir / "dawn_event_rows.csv"),
        "paired_significance_checks": str(args.outdir / "paired_significance_checks.csv"),
        "report": str(args.outdir / "comprehensive_validation_report.md"),
        "zip": str(SCRIPT_DIR / "comprehensive_validation_claude_package.zip"),
    }
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_days": args.synthetic_days,
            "synthetic_seeds": args.synthetic_seeds,
            "estimator_source_dir": str(ESTIMATOR_DIR),
        },
        "headline_claim_rows": pump_proxy_rows,
        "pump_proxy_claim_rows": pump_proxy_rows,
        "strong_baseline_claim_rows": strong_baseline_rows,
        "parameter_task_rows": parameter_rows,
        "meal_inference_summary": meal_summary,
        "carb_absorption_summary": absorption_summary,
        "dawn_summary": dawn_summary,
        "files": files,
    }

    write_csv(args.outdir / "headline_claim_rows.csv", [rounded_json(row) for row in pump_proxy_rows])
    write_csv(args.outdir / "pump_proxy_claim_rows.csv", [rounded_json(row) for row in pump_proxy_rows])
    write_csv(args.outdir / "strong_baseline_claim_rows.csv", [rounded_json(row) for row in strong_baseline_rows])
    write_csv(args.outdir / "parameter_task_rows.csv", [rounded_json(row) for row in parameter_rows])
    write_csv(args.outdir / "meal_inference_detector_summary.csv", [rounded_json(row) for row in meal_summary])
    write_csv(args.outdir / "meal_inference_trace_rows.csv", [rounded_json(row) for row in meal_trace_rows])
    write_csv(args.outdir / "carb_absorption_summary.csv", [rounded_json(row) for row in absorption_summary])
    write_csv(args.outdir / "carb_absorption_event_rows.csv", [rounded_json(row) for row in absorption_rows])
    write_csv(args.outdir / "dawn_summary.csv", [rounded_json(row) for row in dawn_summary])
    write_csv(args.outdir / "dawn_event_rows.csv", [rounded_json(row) for row in dawn_rows])
    (args.outdir / "comprehensive_validation_summary.json").write_text(
        json.dumps(rounded_json(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.outdir / "comprehensive_validation_report.md", payload)
    package_outputs(args.outdir, SCRIPT_DIR / "comprehensive_validation_claude_package.zip")
    print(json.dumps(rounded_json(payload["headline_claim_rows"]), indent=2), flush=True)
    print(f"Wrote {args.outdir / 'comprehensive_validation_report.md'}", flush=True)
    print(f"Wrote {SCRIPT_DIR / 'comprehensive_validation_claude_package.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
