#!/usr/bin/env python3
"""
Offline validation harness for Gluca latent meal inference.

This intentionally scores event-level behavior, not just aggregate glucose
metrics. The detector below mirrors Gluca_ios/Gluca/Engine/
LatentMealInferenceService.swift closely enough for threshold sweeps and
simulation triage; app-facing changes should still be locked with XCTest.

Modes:
  synthetic  - controlled CGM traces with known hidden meals/confounders
  padova     - simglucose/UVA-Padova virtual patient traces
  plist      - replay a copied Gluca UserDefaults plist and mask meal logs
  tandem_csv - replay Tandem CSV exports with pump carb/bolus labels
"""

from __future__ import annotations

import argparse
import json
import math
import os
import plistlib
import statistics
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
MMOL_L_TO_MG_DL = 18.01559


@dataclass(frozen=True)
class Reading:
    minute: float
    value: float


@dataclass(frozen=True)
class MealEvent:
    minute: float
    carbs: float
    logged: bool = False
    label: str = "meal"


@dataclass(frozen=True)
class InsulinEvent:
    minute: float
    units: float
    announced_carbs: float = 0.0
    bolus_type: str = ""


@dataclass(frozen=True)
class ExerciseEvent:
    minute: float
    duration_minutes: float


@dataclass(frozen=True)
class Detection:
    detection_minute: float
    estimated_onset_minute: float
    source: str
    probability: float
    confidence: float
    estimated_carbs: float
    carb_bin: str
    observed_rise_mgdl: float
    rate_mgdl_per_min: float
    announced_carbs: float = 0.0
    bolus_type: str = ""


@dataclass
class Trace:
    label: str
    readings: list[Reading]
    meals: list[MealEvent]
    insulin: list[InsulinEvent]
    exercise: list[ExerciseEvent]
    carb_ratio: float = 10.0
    insulin_sensitivity: float = 45.0
    target_high: float = 180.0


@dataclass
class EvalResult:
    label: str
    days: float
    meal_count: int
    logged_meal_count: int
    evaluated_unlogged_meals: int
    evaluated_meals_30g_plus: int
    detections: int
    true_positives: int
    true_positives_30g_plus: int
    false_positives: int
    small_carb_context_matches: int
    false_prompts_per_day: float
    false_positive_rate_per_100_windows: float
    recall: float
    recall_30g_plus: float | None
    precision: float
    f1: float
    carb_bin_accuracy: float | None
    median_detection_delay_min: float | None
    median_onset_error_min: float | None
    p90_detection_delay_min: float | None
    matched: list[dict[str, Any]]
    false_positive_examples: list[dict[str, Any]]
    missed_examples: list[dict[str, Any]]


def carb_bin(carbs: float) -> str:
    if carbs < 15:
        return "0-15g"
    if carbs < 30:
        return "15-30g"
    if carbs < 60:
        return "30-60g"
    return "60g+"


def slope(readings: list[Reading]) -> float:
    if len(readings) < 2:
        return 0.0
    first, last = readings[0], readings[-1]
    minutes = last.minute - first.minute
    if minutes <= 0:
        return 0.0
    return (last.value - first.value) / minutes


def circadian_label(minute: float) -> str:
    hour = int((minute % (24 * 60)) // 60)
    if 4 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 11:
        return "late_morning"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "dinner"
    return "overnight"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def logit(probability: float) -> float:
    p = clamp(probability, 0.001, 0.999)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def meal_prior(circadian: str) -> float:
    return {
        "early_morning": 0.08,
        "late_morning": 0.16,
        "lunch": 0.30,
        "afternoon": 0.14,
        "dinner": 0.32,
        "overnight": 0.04,
    }.get(circadian, 0.10)


def remaining_insulin_fraction(age_hours: float, active_hours: float = 4.0) -> float:
    if age_hours >= active_hours:
        return 0.0
    progress = max(0.0, min(1.0, age_hours / active_hours))
    return (1.0 - progress) ** 2


def active_iob(insulin: list[InsulinEvent], reference_minute: float, active_hours: float = 4.0) -> float:
    total = 0.0
    for event in insulin:
        age_hours = (reference_minute - event.minute) / 60.0
        if age_hours < 0:
            continue
        total += event.units * remaining_insulin_fraction(age_hours, active_hours)
    return total


def recent_exercise_minutes(exercise: list[ExerciseEvent], reference_minute: float) -> float:
    return sum(
        event.duration_minutes
        for event in exercise
        if 0 <= reference_minute - event.minute <= 8 * 60
    )


def rise_consistency(readings: list[Reading]) -> float:
    if len(readings) < 3:
        return 0.0
    deltas = [readings[idx].value - readings[idx - 1].value for idx in range(1, len(readings))]
    return sum(1 for delta in deltas if delta >= -2.0) / len(deltas)


def closest_reading(readings: list[Reading], minute: float, tolerance_min: float = 20.0) -> Reading | None:
    if not readings:
        return None
    candidate = min(readings, key=lambda reading: abs(reading.minute - minute))
    return candidate if abs(candidate.minute - minute) <= tolerance_min else None


def insulin_action_fraction(age_min: float, active_hours: float = 4.0) -> float:
    age_hours = max(age_min / 60.0, 0.0)
    return clamp(1.0 - remaining_insulin_fraction(age_hours, active_hours), 0.0, 1.0)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def origin_for_datetime(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def datetime_to_minute(value: datetime, origin: datetime) -> float:
    return (value - origin).total_seconds() / 60.0


def detect_meal(
    readings: list[Reading],
    visible_meals: list[MealEvent],
    insulin: list[InsulinEvent],
    exercise: list[ExerciseEvent],
    carb_ratio: float,
    insulin_sensitivity: float,
    target_high: float,
) -> Detection | None:
    sorted_readings = sorted(readings, key=lambda r: r.minute)
    if len(sorted_readings) < 12:
        return None
    latest = sorted_readings[-1]

    if any(meal.logged and 0 <= latest.minute - meal.minute <= 105 for meal in visible_meals):
        return None

    lookback = [r for r in sorted_readings if latest.minute - r.minute <= 150]
    if len(lookback) < 8:
        return None

    current_rate = slope(sorted_readings[-4:])
    iob_units = active_iob(insulin, latest.minute)
    exercise_score = min(recent_exercise_minutes(exercise, latest.minute) / 45.0, 1.4)
    isf = max(insulin_sensitivity or 45.0, 20.0)
    cr = max(carb_ratio or 10.0, 5.0)
    carb_effect_per_gram = max(isf / cr, 2.2)

    candidates: list[dict[str, Any]] = []
    best_rise = None
    best_score = -math.inf
    for idx in range(2, len(lookback) - 2):
        baseline = lookback[idx]
        recent_slice = lookback[idx : min(idx + 4, len(lookback))]
        local_rate = slope(recent_slice)
        rise = latest.value - baseline.value
        minutes_ago = latest.minute - baseline.minute
        if not (20 <= minutes_ago <= 115):
            continue
        if local_rate < 0.3 or rise < 18:
            continue
        score = rise + (local_rate * 18) - abs(minutes_ago - 55) * 0.18
        if score > best_score:
            best_score = score
            best_rise = baseline

    if best_rise is not None:
        candidates.append(
            {
                "source": "cgm_rise",
                "onset_minute": best_rise.minute,
                "baseline_value": best_rise.value,
                "baseline_index": next(
                    (idx for idx, reading in enumerate(sorted_readings) if reading.minute == best_rise.minute),
                    0,
                ),
                "bolus_units": 0.0,
                "announced_carbs": 0.0,
                "bolus_type": "",
                "insulin_residual": None,
                "observed_delta": latest.value - best_rise.value,
            }
        )

    for event in insulin:
        age_min = latest.minute - event.minute
        if event.units < 1.5 or not (35 <= age_min <= 150):
            continue
        known_type = event.bolus_type.lower()
        known_zero_carb_correction = (
            event.announced_carbs <= 0
            and any(token in known_type for token in ("auto", "correction"))
            and "food" not in known_type
            and "override" not in known_type
        )
        if known_zero_carb_correction:
            continue
        baseline = closest_reading(sorted_readings, event.minute)
        if baseline is None:
            continue
        observed_delta = latest.value - baseline.value
        expected_drop = event.units * isf * clamp(insulin_action_fraction(age_min) * 0.62, 0.08, 0.72)
        insulin_residual = observed_delta + expected_drop
        if insulin_residual < 30 and current_rate < 0.15:
            continue
        if baseline.value > target_high and observed_delta < -30 and insulin_residual < 50:
            continue
        candidates.append(
            {
                "source": "bolus_residual",
                "onset_minute": event.minute,
                "baseline_value": baseline.value,
                "baseline_index": next(
                    (idx for idx, reading in enumerate(sorted_readings) if reading.minute == baseline.minute),
                    0,
                ),
                "bolus_units": event.units,
                "announced_carbs": event.announced_carbs,
                "bolus_type": event.bolus_type,
                "insulin_residual": insulin_residual,
                "observed_delta": observed_delta,
            }
        )

    best_detection: Detection | None = None
    best_probability = -math.inf

    for candidate in candidates:
        source = candidate["source"]
        onset_minute = float(candidate["onset_minute"])
        baseline_value = float(candidate["baseline_value"])
        observed_rise = latest.value - baseline_value
        baseline_index = int(candidate["baseline_index"])
        prior_rate = (
            slope(sorted_readings[max(0, baseline_index - 3) : baseline_index + 1])
            if baseline_index >= 3
            else None
        )
        acceleration = current_rate - prior_rate if prior_rate is not None else None
        circadian = circadian_label(onset_minute)
        prediction_lift = max(current_rate * 45.0, 0.0)
        minutes_since_onset = max(latest.minute - onset_minute, 10.0)
        observed_fraction = clamp(minutes_since_onset / 150.0, 0.22, 0.85)
        consistency = rise_consistency(
            [reading for reading in sorted_readings if onset_minute <= reading.minute <= latest.minute]
        )
        insulin_residual = candidate["insulin_residual"]

        if source == "bolus_residual" and insulin_residual is not None:
            correction_units = max((baseline_value - 120.0) / isf, 0.0)
            meal_units = max(float(candidate["bolus_units"]) - correction_units, 0.0)
            bolus_carbs = meal_units * cr
            residual_carbs = max(float(insulin_residual) / carb_effect_per_gram, 0.0)
            estimated_carbs = min(max(max(bolus_carbs, residual_carbs), 8.0), 110.0)
        else:
            adjusted_rise = observed_rise + (iob_units * isf * 0.22) + (exercise_score * 8.0)
            estimated_carbs = min(max((adjusted_rise / carb_effect_per_gram) / observed_fraction, 8.0), 110.0)

        log_odds = logit(meal_prior(circadian))
        log_odds += clamp((observed_rise - 25.0) / 35.0, -0.8, 1.4)
        log_odds += clamp((current_rate - 0.4) / 0.55, -0.7, 1.3)
        if acceleration is not None:
            log_odds += clamp(acceleration / 0.35, -0.45, 0.55)
        log_odds += clamp((consistency - 0.55) * 1.4, -0.5, 0.55)
        log_odds += min(prediction_lift / 55.0, 0.5)
        log_odds -= min(exercise_score * 0.95, 1.4)

        if source == "bolus_residual" and insulin_residual is not None:
            log_odds += clamp((float(insulin_residual) - 35.0) / 35.0, -0.5, 1.8)
            log_odds += clamp((float(candidate["bolus_units"]) - 1.5) / 2.5, 0.0, 0.8)
            log_odds += 0.45 if float(candidate["observed_delta"]) >= -10 else -0.35
            log_odds += 0.3 if current_rate >= -0.1 else -0.45
            if baseline_value > target_high and float(candidate["observed_delta"]) < -20:
                log_odds -= 1.0
        else:
            log_odds -= min(iob_units * 0.45, 1.4)

        if circadian == "early_morning":
            log_odds -= 0.55
        if estimated_carbs < 15:
            log_odds -= 0.5
        if observed_rise < 30 and source == "cgm_rise":
            log_odds -= 0.45
        if latest.value < target_high and observed_rise < 45 and source == "cgm_rise":
            log_odds -= 0.25

        meal_probability = sigmoid(log_odds)

        cgm_gate_passed = (
            source == "cgm_rise"
            and meal_probability >= 0.68
            and observed_rise >= 24
            and current_rate >= 0.4
            and consistency >= 0.55
        )
        bolus_gate_passed = (
            source == "bolus_residual"
            and meal_probability >= 0.68
            and insulin_residual is not None
            and float(insulin_residual) >= 35
            and current_rate >= -0.35
            and latest.value >= 70
        )
        if not cgm_gate_passed and not bolus_gate_passed:
            continue

        if meal_probability <= best_probability:
            continue

        confidence = min(max(meal_probability, 0.34), 0.94)
        best_probability = meal_probability
        best_detection = Detection(
            detection_minute=latest.minute,
            estimated_onset_minute=onset_minute,
            source=source,
            probability=meal_probability,
            confidence=confidence,
            estimated_carbs=estimated_carbs,
            carb_bin=carb_bin(estimated_carbs),
            observed_rise_mgdl=observed_rise,
            rate_mgdl_per_min=current_rate,
            announced_carbs=float(candidate["announced_carbs"]),
            bolus_type=str(candidate["bolus_type"]),
        )

    if best_detection is None:
        return None

    return best_detection


def dedupe_detections(detections: list[Detection], refractory_minutes: float = 75) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.detection_minute):
        if any(abs(detection.estimated_onset_minute - prev.estimated_onset_minute) <= refractory_minutes for prev in kept):
            continue
        kept.append(detection)
    return kept


def should_prompt_detection(
    detection: Detection,
    bolus_prompt_threshold: float,
    cgm_prompt_threshold: float = 0.96,
) -> bool:
    if detection.source == "bolus_residual":
        if detection.estimated_carbs < 15:
            return False
        if 0 < detection.announced_carbs < 15:
            return False
        return detection.confidence >= bolus_prompt_threshold
    return detection.confidence >= cgm_prompt_threshold


def evaluate_trace(
    trace: Trace,
    min_meal_carbs: float = 15.0,
    match_tolerance_min: float = 75.0,
    max_detection_delay_min: float = 150.0,
    mask_logged_meals: bool = True,
    prompt_threshold: float = 0.82,
) -> EvalResult:
    detections: list[Detection] = []
    windows = 0

    sorted_readings = sorted(trace.readings, key=lambda r: r.minute)
    for idx in range(len(sorted_readings)):
        reference = sorted_readings[idx].minute
        visible_meals = [
            meal
            for meal in trace.meals
            if meal.minute <= reference and (meal.logged and not mask_logged_meals)
        ]
        result = detect_meal(
            sorted_readings[: idx + 1],
            visible_meals,
            [event for event in trace.insulin if event.minute <= reference],
            [event for event in trace.exercise if event.minute <= reference],
            trace.carb_ratio,
            trace.insulin_sensitivity,
            trace.target_high,
        )
        windows += 1
        if result is not None and should_prompt_detection(result, prompt_threshold):
            detections.append(result)

    detections = dedupe_detections(detections)
    score_start = sorted_readings[0].minute + 60.0 if sorted_readings else 0.0
    score_end = sorted_readings[-1].minute - max_detection_delay_min if sorted_readings else 0.0
    eval_meals = [
        meal
        for meal in trace.meals
        if (
            meal.carbs >= min_meal_carbs
            and (mask_logged_meals or not meal.logged)
            and score_start <= meal.minute <= score_end
        )
    ]

    matched: list[dict[str, Any]] = []
    used_meal_indexes: set[int] = set()
    false_positives: list[Detection] = []
    small_carb_context_matches: list[dict[str, Any]] = []

    for detection in detections:
        candidate_indexes = []
        for meal_index, meal in enumerate(eval_meals):
            if meal_index in used_meal_indexes:
                continue
            onset_error = detection.estimated_onset_minute - meal.minute
            detection_delay = detection.detection_minute - meal.minute
            if (
                abs(onset_error) <= match_tolerance_min
                and 0 <= detection_delay <= max_detection_delay_min
            ):
                candidate_indexes.append((abs(onset_error), detection_delay, meal_index))
        if not candidate_indexes:
            small_carb_candidates = []
            for meal in trace.meals:
                if not (0 < meal.carbs < min_meal_carbs):
                    continue
                onset_error = detection.estimated_onset_minute - meal.minute
                detection_delay = detection.detection_minute - meal.minute
                if (
                    abs(onset_error) <= match_tolerance_min
                    and 0 <= detection_delay <= max_detection_delay_min
                ):
                    small_carb_candidates.append((abs(onset_error), detection_delay, meal))
            if small_carb_candidates:
                _, _, meal = sorted(small_carb_candidates)[0]
                small_carb_context_matches.append(
                    {
                        "meal_minute": round(meal.minute, 1),
                        "meal_carbs": meal.carbs,
                        "detected_minute": round(detection.detection_minute, 1),
                        "estimated_onset_minute": round(detection.estimated_onset_minute, 1),
                        "source": detection.source,
                        "estimated_carbs": round(detection.estimated_carbs, 1),
                        "confidence": round(detection.confidence, 3),
                    }
                )
                continue
            false_positives.append(detection)
            continue
        _, _, meal_index = sorted(candidate_indexes)[0]
        meal = eval_meals[meal_index]
        used_meal_indexes.add(meal_index)
        matched.append(
            {
                "meal_minute": round(meal.minute, 1),
                "meal_carbs": meal.carbs,
                "meal_bin": carb_bin(meal.carbs),
                "detected_minute": round(detection.detection_minute, 1),
                "estimated_onset_minute": round(detection.estimated_onset_minute, 1),
                "source": detection.source,
                "delay_min": round(detection.detection_minute - meal.minute, 1),
                "onset_error_min": round(detection.estimated_onset_minute - meal.minute, 1),
                "estimated_carbs": round(detection.estimated_carbs, 1),
                "detected_bin": detection.carb_bin,
                "confidence": round(detection.confidence, 3),
            }
        )

    missed = [
        meal
        for idx, meal in enumerate(eval_meals)
        if idx not in used_meal_indexes
    ]

    tp = len(matched)
    fp = len(false_positives)
    fn = len(missed)
    eval_meals_30g_plus = [meal for meal in eval_meals if meal.carbs >= 30]
    matched_30g_plus = [row for row in matched if row["meal_carbs"] >= 30]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    recall_30g_plus = len(matched_30g_plus) / len(eval_meals_30g_plus) if eval_meals_30g_plus else None
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    days = max((sorted_readings[-1].minute - sorted_readings[0].minute) / (24 * 60), 1 / 24) if sorted_readings else 0
    delays = [row["delay_min"] for row in matched]
    onset_errors = [abs(row["onset_error_min"]) for row in matched]
    carb_matches = [row["meal_bin"] == row["detected_bin"] for row in matched]

    return EvalResult(
        label=trace.label,
        days=round(days, 2),
        meal_count=len(trace.meals),
        logged_meal_count=sum(1 for meal in trace.meals if meal.logged),
        evaluated_unlogged_meals=len(eval_meals),
        evaluated_meals_30g_plus=len(eval_meals_30g_plus),
        detections=len(detections),
        true_positives=tp,
        true_positives_30g_plus=len(matched_30g_plus),
        false_positives=fp,
        small_carb_context_matches=len(small_carb_context_matches),
        false_prompts_per_day=round(fp / days, 3) if days else 0,
        false_positive_rate_per_100_windows=round((fp / max(windows, 1)) * 100, 3),
        recall=round(recall, 3),
        recall_30g_plus=round(recall_30g_plus, 3) if recall_30g_plus is not None else None,
        precision=round(precision, 3),
        f1=round(f1, 3),
        carb_bin_accuracy=round(sum(carb_matches) / len(carb_matches), 3) if carb_matches else None,
        median_detection_delay_min=round(statistics.median(delays), 1) if delays else None,
        median_onset_error_min=round(statistics.median(onset_errors), 1) if onset_errors else None,
        p90_detection_delay_min=round(float(np.percentile(delays, 90)), 1) if delays else None,
        matched=matched[:20],
        false_positive_examples=[
            {
                "detected_minute": round(detection.detection_minute, 1),
                "estimated_onset_minute": round(detection.estimated_onset_minute, 1),
                "source": detection.source,
                "estimated_carbs": round(detection.estimated_carbs, 1),
                "confidence": round(detection.confidence, 3),
                "announced_carbs": round(detection.announced_carbs, 1),
                "bolus_type": detection.bolus_type,
            }
            for detection in false_positives[:10]
        ],
        missed_examples=[
            {
                "meal_minute": round(meal.minute, 1),
                "meal_carbs": meal.carbs,
                "meal_bin": carb_bin(meal.carbs),
                "label": meal.label,
            }
            for meal in missed[:20]
        ],
    )


def meal_response_shape(age_min: float, absorption_min: float) -> float:
    if age_min < 0:
        return 0.0
    x = age_min / max(absorption_min, 1.0)
    return (x**2) * math.exp(2 - x) if x <= 5 else 0.0


def insulin_action_shape(age_min: float, peak_min: float = 75.0, duration_min: float = 300.0) -> float:
    if age_min < 0 or age_min > duration_min:
        return 0.0
    x = age_min / peak_min
    return (x**2) * math.exp(2 - x) if x <= duration_min / peak_min else 0.0


def build_synthetic_trace(days: int, seed: int, label: str = "synthetic") -> Trace:
    rng = np.random.default_rng(seed)
    cadence = 5
    total_minutes = days * 24 * 60
    carb_ratio = 10.5
    isf = 46.0
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    exercise: list[ExerciseEvent] = []

    for day in range(days):
        day_offset = day * 24 * 60
        templates = [
            ("breakfast", 7.7 * 60, 42, 0.62),
            ("lunch", 12.8 * 60, 62, 0.56),
            ("dinner", 18.8 * 60, 70, 0.52),
            ("snack", 21.3 * 60, 16, 0.25),
        ]
        for name, mean_time, mean_carbs, logged_prob in templates:
            if rng.random() < (0.88 if name != "snack" else 0.38):
                minute = day_offset + float(rng.normal(mean_time, 35 if name != "snack" else 22))
                carbs = max(6.0, float(rng.normal(mean_carbs, mean_carbs * 0.22)))
                logged = bool(rng.random() < logged_prob)
                meals.append(MealEvent(minute=minute, carbs=round(carbs, 1), logged=logged, label=name))
                if rng.random() < 0.82:
                    bolus_minute = minute + float(rng.normal(3, 9))
                    insulin.append(InsulinEvent(minute=bolus_minute, units=round(max(carbs / carb_ratio, 0.2), 2)))
        if rng.random() < 0.42:
            start = day_offset + float(rng.normal(17.2 * 60, 80))
            duration = float(rng.uniform(25, 65))
            exercise.append(ExerciseEvent(minute=start, duration_minutes=duration))

    readings: list[Reading] = []
    for minute in range(0, total_minutes + 1, cadence):
        minute_f = float(minute)
        day_minute = minute_f % (24 * 60)
        baseline = 112 + 5 * math.sin(2 * math.pi * (day_minute - 240) / (24 * 60))
        dawn = 18 * math.exp(-((day_minute - 6.7 * 60) / 85) ** 2)
        value = baseline + dawn

        for meal in meals:
            absorption = 72 if meal.carbs < 30 else 95 if meal.carbs < 60 else 125
            value += meal.carbs * (isf / carb_ratio) * 0.58 * meal_response_shape(minute_f - meal.minute, absorption)
        for event in insulin:
            value -= event.units * isf * 0.75 * insulin_action_shape(minute_f - event.minute)
        for event in exercise:
            age = minute_f - event.minute
            if 0 <= age <= 180:
                value -= min(35, event.duration_minutes * 0.55) * math.exp(-age / 95)

        noise = float(rng.normal(0, 5.5))
        readings.append(Reading(minute=minute_f, value=round(max(45.0, min(360.0, value + noise)), 1)))

    return Trace(
        label=f"{label}:days={days}:seed={seed}",
        readings=readings,
        meals=sorted(meals, key=lambda m: m.minute),
        insulin=sorted(insulin, key=lambda i: i.minute),
        exercise=sorted(exercise, key=lambda e: e.minute),
        carb_ratio=carb_ratio,
        insulin_sensitivity=isf,
        target_high=180.0,
    )


def build_padova_trace(days: int, seed: int, patient_name: str) -> Trace:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.base import Action
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario_gen import RandomScenario

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = RandomScenario(start_time=datetime(2024, 1, 1, 0, 0, 0), seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    carb_ratio = 10.0
    isf = 45.0
    basal_per_step = 0.8 * (3.0 / 60.0)
    last_meal_carbs = 0.0

    reset = env.reset()
    readings = [Reading(minute=0.0, value=float(reset.observation.CGM))]
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []

    total_steps = days * 480
    for step in range(total_steps):
        minute = step * 3.0
        bolus = 0.0
        if last_meal_carbs > 0:
            bolus = min(max(last_meal_carbs / carb_ratio, 0.0), 12.0)
            insulin.append(InsulinEvent(minute=minute, units=round(bolus, 3)))

        result = env.step(Action(basal=basal_per_step + bolus, bolus=0.0))
        cgm = float(result.observation.CGM)
        readings.append(Reading(minute=minute + 3.0, value=cgm))

        raw_meal_carbs = float(result.info.get("meal", 0.0) or 0.0)
        sample_time = float(result.info.get("sample_time", 3.0) or 3.0)
        meal_carbs = raw_meal_carbs * sample_time
        if meal_carbs > 0.5:
            meals.append(
                MealEvent(
                    minute=minute + 3.0,
                    carbs=round(meal_carbs, 1),
                    logged=False,
                    label=f"{patient_name}:random_scenario",
                )
            )
        last_meal_carbs = meal_carbs
        if bool(result.done):
            break

    return Trace(
        label=f"padova:{patient_name}:days={days}:seed={seed}",
        readings=readings,
        meals=meals,
        insulin=insulin,
        exercise=[],
        carb_ratio=carb_ratio,
        insulin_sensitivity=isf,
        target_high=180.0,
    )


def apple_time_to_minute(value: float, origin: float) -> float:
    return (value - origin) / 60.0


def build_plist_trace(path: Path, label: str = "plist") -> Trace:
    with path.open("rb") as handle:
        plist = plistlib.load(handle)
    raw = plist.get("com.railabs.gluca.timeline.events")
    if not isinstance(raw, bytes):
        raise ValueError("plist does not contain com.railabs.gluca.timeline.events")
    events = json.loads(raw.decode("utf-8"))
    timestamps = [event.get("timestamp") for event in events if isinstance(event.get("timestamp"), (int, float))]
    if not timestamps:
        raise ValueError("timeline has no timestamped events")
    first_datetime = APPLE_EPOCH + timedelta(seconds=float(min(timestamps)))
    origin = origin_for_datetime(first_datetime)

    readings: list[Reading] = []
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    exercise: list[ExerciseEvent] = []
    for event in events:
        kind = event.get("kind")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            continue
        minute = datetime_to_minute(APPLE_EPOCH + timedelta(seconds=float(timestamp)), origin)
        value = event.get("value") or {}
        amount = value.get("amount")
        if kind == "glucose" and isinstance(amount, (int, float)):
            readings.append(Reading(minute=minute, value=float(amount)))
        elif kind == "meal" and isinstance(amount, (int, float)) and amount > 0:
            meals.append(
                MealEvent(
                    minute=minute,
                    carbs=float(amount),
                    logged=False,
                    label="redacted_phone_meal",
                )
            )
        elif kind == "insulin" and isinstance(amount, (int, float)) and amount > 0:
            insulin.append(InsulinEvent(minute=minute, units=float(amount)))
        elif kind == "exercise" and isinstance(amount, (int, float)) and amount > 0:
            exercise.append(ExerciseEvent(minute=minute, duration_minutes=float(amount)))

    return Trace(
        label=label,
        readings=sorted(readings, key=lambda r: r.minute),
        meals=sorted(meals, key=lambda m: m.minute),
        insulin=sorted(insulin, key=lambda i: i.minute),
        exercise=sorted(exercise, key=lambda e: e.minute),
    )


def build_tandem_csv_trace(path: Path, label: str = "tandem-csv") -> Trace:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    try:
        table_start = next(index for index, line in enumerate(lines) if line.startswith("DeviceType,"))
    except StopIteration as exc:
        raise ValueError("Tandem CSV does not contain a DeviceType event table") from exc

    raw_rows = list(csv.reader(lines[table_start + 1:]))
    timestamps: list[datetime] = []
    for row in raw_rows:
        timestamp: datetime | None = None
        if len(row) == 5 and row[2] == "EGV":
            try:
                timestamp = datetime.fromisoformat(row[3])
            except ValueError:
                timestamp = None
        elif len(row) == 20 and row and row[0] == "Bolus":
            raw_timestamp = row[10] or row[5]
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                timestamp = None
        if timestamp is not None:
            timestamps.append(timestamp)

    if not timestamps:
        raise ValueError("Tandem CSV has no parseable EGV or bolus timestamps")

    origin = origin_for_datetime(min(timestamps))
    readings: list[Reading] = []
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    carb_ratios: list[float] = []
    correction_factors_mmol: list[float] = []

    for row in raw_rows:
        if len(row) == 5 and row[2] == "EGV":
            glucose_mmol = safe_float(row[4])
            if glucose_mmol is None:
                continue
            try:
                timestamp = datetime.fromisoformat(row[3])
            except ValueError:
                continue
            readings.append(
                Reading(
                    minute=datetime_to_minute(timestamp, origin),
                    value=round(glucose_mmol * MMOL_L_TO_MG_DL, 1),
                )
            )
            continue

        if len(row) != 20 or not row or row[0] != "Bolus":
            continue

        delivered_units = safe_float(row[6], 0.0) or 0.0
        extended_units = safe_float(row[12], 0.0) or 0.0
        carb_size = safe_float(row[16], 0.0) or 0.0
        carb_ratio = safe_float(row[19])
        correction_factor = safe_float(row[18])

        if carb_ratio is not None and carb_ratio > 0:
            carb_ratios.append(carb_ratio)
        if correction_factor is not None and correction_factor > 0:
            correction_factors_mmol.append(correction_factor)

        start_raw = row[10] or row[5]
        completion_raw = row[5] or row[10]
        try:
            start_timestamp = datetime.fromisoformat(start_raw)
            completion_timestamp = datetime.fromisoformat(completion_raw)
        except ValueError:
            continue

        if carb_size > 0:
            meals.append(
                MealEvent(
                    minute=datetime_to_minute(start_timestamp, origin),
                    carbs=round(carb_size, 1),
                    logged=False,
                    label="redacted_tandem_carb_entry",
                )
            )

        if delivered_units <= 0:
            continue

        upfront_units = max(delivered_units - extended_units, 0.0)
        if upfront_units > 0:
            insulin.append(
                InsulinEvent(
                    minute=datetime_to_minute(start_timestamp, origin),
                    units=round(upfront_units, 3),
                    announced_carbs=round(carb_size, 1),
                    bolus_type=row[1],
                )
            )
        if extended_units > 0 and completion_timestamp != start_timestamp:
            insulin.append(
                InsulinEvent(
                    minute=datetime_to_minute(completion_timestamp, origin),
                    units=round(extended_units, 3),
                    announced_carbs=round(carb_size, 1),
                    bolus_type=row[1],
                )
            )
        elif upfront_units == 0:
            insulin.append(
                InsulinEvent(
                    minute=datetime_to_minute(completion_timestamp, origin),
                    units=round(delivered_units, 3),
                    announced_carbs=round(carb_size, 1),
                    bolus_type=row[1],
                )
            )

    carb_ratio = statistics.median(carb_ratios) if carb_ratios else 10.0
    insulin_sensitivity = (
        statistics.median(correction_factors_mmol) * MMOL_L_TO_MG_DL
        if correction_factors_mmol
        else 45.0
    )

    return Trace(
        label=label,
        readings=sorted(readings, key=lambda r: r.minute),
        meals=sorted(meals, key=lambda m: m.minute),
        insulin=sorted(insulin, key=lambda i: i.minute),
        exercise=[],
        carb_ratio=carb_ratio,
        insulin_sensitivity=insulin_sensitivity,
        target_high=180.0,
    )


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {}
    total_tp = sum(result.true_positives for result in results)
    total_fp = sum(result.false_positives for result in results)
    total_small_context = sum(result.small_carb_context_matches for result in results)
    total_meals = sum(result.evaluated_unlogged_meals for result in results)
    total_meals_30g_plus = sum(result.evaluated_meals_30g_plus for result in results)
    total_tp_30g_plus = sum(result.true_positives_30g_plus for result in results)
    total_days = sum(result.days for result in results)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / total_meals if total_meals else 0.0
    recall_30g_plus = total_tp_30g_plus / total_meals_30g_plus if total_meals_30g_plus else None
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [
        row["delay_min"]
        for result in results
        for row in result.matched
    ]
    carb_checks = [
        row["meal_bin"] == row["detected_bin"]
        for result in results
        for row in result.matched
    ]
    return {
        "traces": len(results),
        "days": round(total_days, 2),
        "evaluated_unlogged_meals": total_meals,
        "evaluated_meals_30g_plus": total_meals_30g_plus,
        "true_positives": total_tp,
        "true_positives_30g_plus": total_tp_30g_plus,
        "false_positives": total_fp,
        "small_carb_context_matches": total_small_context,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "recall_30g_plus": round(recall_30g_plus, 3) if recall_30g_plus is not None else None,
        "f1": round(f1, 3),
        "false_prompts_per_day": round(total_fp / total_days, 3) if total_days else 0,
        "median_detection_delay_min": round(statistics.median(delays), 1) if delays else None,
        "carb_bin_accuracy": round(sum(carb_checks) / len(carb_checks), 3) if carb_checks else None,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Meal Inference Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(payload.get("summary", {}), indent=2),
        "```",
        "",
        "## Trace Results",
        "",
        "| Trace | Days | Meals | 30g+ meals | TP | FP | Small context | Recall | 30g+ recall | Precision | False prompts/day | Median delay | Carb-bin acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload.get("results", []):
        lines.append(
            "| {label} | {days} | {meals} | {meals_30} | {tp} | {fp} | {small} | {recall:.3f} | {recall_30} | {precision:.3f} | {fpd:.3f} | {delay} | {bin_acc} |".format(
                label=result["label"],
                days=result["days"],
                meals=result["evaluated_unlogged_meals"],
                meals_30=result["evaluated_meals_30g_plus"],
                tp=result["true_positives"],
                fp=result["false_positives"],
                small=result["small_carb_context_matches"],
                recall=result["recall"],
                recall_30=result["recall_30g_plus"],
                precision=result["precision"],
                fpd=result["false_prompts_per_day"],
                delay=result["median_detection_delay_min"],
                bin_acc=result["carb_bin_accuracy"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Gates",
            "",
            "- Ship to shadow mode only when false prompts are below 0.5/day on real replay and synthetic stress tests.",
            "- Treat Padova recall as physiology plausibility, not real-world truth; it does not model every logging gap, exercise artifact, sensor issue, or behavioral pattern.",
            "- Promote meal memory only from confirmed pump/manual meals or user-confirmed prompts, not from raw inferred meals.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "padova", "plist", "tandem_csv", "all"], default="synthetic")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patients", nargs="*", default=["adult#001", "adult#002", "adult#003"])
    parser.add_argument("--plist", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces: list[Trace] = []
    if args.mode in {"synthetic", "all"}:
        for offset in range(3):
            traces.append(build_synthetic_trace(args.days, args.seed + offset, label="synthetic"))
    if args.mode in {"padova", "all"}:
        for patient in args.patients:
            traces.append(build_padova_trace(args.days, args.seed, patient))
    if args.mode in {"plist", "all"}:
        if not args.plist:
            raise SystemExit("--plist is required for plist/all mode")
        traces.append(build_plist_trace(args.plist, label="gluca-replay"))
    if args.mode in {"tandem_csv", "all"}:
        if not args.csv:
            raise SystemExit("--csv is required for tandem_csv/all mode")
        traces.append(build_tandem_csv_trace(args.csv, label="tandem-csv-replay"))

    results = [evaluate_trace(trace) for trace in traces]
    payload = {
        "mode": args.mode,
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")
    if args.md_out:
        args.md_out.write_text(markdown_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
