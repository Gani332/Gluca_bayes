#!/usr/bin/env python3
"""
Benchmark Gluca's parameter estimators against estimator baselines.

This is an estimator validation, not a controller validation.

It runs two complementary checks:

1. Clean UVA/Padova correction-event holdout:
   - ISF is directly identifiable from isolated correction events.
   - Compares fixed, clinical, empirical, EWMA, original Bayes, V2 Bayes,
     and app-style log posterior estimates on the same future holdout events.

2. RL4BG adolescent free-living replay:
   - Meals, basal insulin, and boluses are mixed.
   - Compares final parameter recovery vs RL4BG known CF/CR/basal truth.
   - This is where confounder gates and "how much to trust" logic matter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RL4BG_VALIDATION_DIR = REPO_ROOT / "validation" / "rl4bg_adolescent_validation"
DEFAULT_RL4BG_ROOT = Path("/tmp/RL4BG")
POPULATION_ISF = 50.0
POPULATION_CR = 10.0
POPULATION_BASAL = 1.0

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RL4BG_VALIDATION_DIR))
os.environ.setdefault("GLUCA_ENABLE_RESEARCH_DOSING", "1")

from bayesian_mpc.bayesian_estimator import BayesianEstimator  # noqa: E402
from bayesian_mpc.bayesian_v2 import BayesianEstimatorV2  # noqa: E402
from bayesian_mpc.estimator_failure_modes import (  # noqa: E402
    adaptive_robust_positive_estimate,
    huber_location,
    modular_positive_estimate,
    winsorized_mean,
)
from run_rl4bg_adolescent_validation import (  # noqa: E402
    ADOLESCENTS,
    build_env,
    build_manual_bb,
    configure_paths,
    ensure_rl4bg,
)


@dataclass(frozen=True)
class CleanCorrectionEvent:
    cohort: str
    patient: str
    event_index: int
    minute: float
    dose_units: float
    true_bg_before: float
    true_bg_after: float
    true_bg_after_no_bolus: float
    net_drop_mgdl: float
    causal_drop_mgdl: float
    observed_net_isf: float
    observed_causal_isf: float
    clinical_prior_isf: float


def round_float(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
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


def pct_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def abs_pct_error(estimate: float, truth: float) -> float:
    if truth == 0:
        return float("nan")
    return abs(float(estimate) - float(truth)) / abs(float(truth)) * 100.0


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"mean": None, "median": None, "std": None, "q1": None, "q3": None}
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
    }


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def trimmed_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    trim = int(len(ordered) * trim_fraction)
    trimmed = ordered[trim:len(ordered) - trim] if trim and len(ordered) - trim > trim else ordered
    return statistics.fmean(trimmed)


def ewma(values: list[float], initial: float, alpha: float = 0.35) -> float:
    estimate = initial
    for value in values:
        estimate = alpha * value + (1.0 - alpha) * estimate
    return estimate


def log_posterior_estimate(
    observations: list[float],
    *,
    prior_value: float,
    prior_log_std: float,
    min_value: float,
    max_value: float,
    observed_floor_std: float = 0.18,
) -> float:
    clean = [float(v) for v in observations if math.isfinite(float(v)) and min_value <= float(v) <= max_value]
    if not clean:
        return prior_value
    logs = [math.log(value) for value in clean]
    prior_mean = math.log(max(prior_value, min_value))
    prior_var = prior_log_std * prior_log_std
    obs_std = max(sample_std(logs), observed_floor_std)
    obs_var = obs_std * obs_std
    precision = (1.0 / prior_var) + (len(logs) / obs_var)
    post_var = 1.0 / max(precision, 1e-9)
    post_mean = post_var * ((prior_mean / prior_var) + (sum(logs) / obs_var))
    return float(min(max(math.exp(post_mean), min_value), max_value))


def normal_posterior_estimate(
    observations: list[float],
    *,
    prior_value: float,
    prior_std: float,
    min_value: float,
    max_value: float,
    observed_floor_std: float,
) -> float:
    clean = [float(v) for v in observations if math.isfinite(float(v)) and min_value <= float(v) <= max_value]
    if not clean:
        return prior_value
    obs_std = max(sample_std(clean), observed_floor_std)
    prior_var = prior_std * prior_std
    obs_var = obs_std * obs_std
    precision = (1.0 / prior_var) + (len(clean) / obs_var)
    post_var = 1.0 / max(precision, 1e-9)
    post_mean = post_var * ((prior_value / prior_var) + (sum(clean) / obs_var))
    return float(min(max(post_mean, min_value), max_value))


def fit_v1_isf(events: list[CleanCorrectionEvent], prior: float = POPULATION_ISF) -> float:
    estimator = BayesianEstimator(isf_prior=prior, isf_sigma=25.0)
    for event in events:
        estimator.record_bg(event.minute, event.true_bg_before)
        estimator.record_insulin(event.minute, event.dose_units, "bolus")
        estimator.record_bg(event.minute + 90.0, (event.true_bg_before + event.true_bg_after) / 2.0)
        estimator.record_bg(event.minute + 180.0, event.true_bg_after)
        estimator.update(event.minute + 181.0)
    return float(estimator.get_params()["isf"])


def fit_v2_isf(events: list[CleanCorrectionEvent], prior: float = POPULATION_ISF) -> float:
    estimator = BayesianEstimatorV2(
        isf_prior=prior,
        cr_prior=POPULATION_CR,
        basal_prior=POPULATION_BASAL,
        use_covariates=False,
        cohort="adolescent",
    )
    for event in events:
        estimator.record_bg(event.minute, event.true_bg_before)
        estimator.record_insulin(event.minute, event.dose_units, "bolus")
        estimator.record_bg(event.minute + 90.0, (event.true_bg_before + event.true_bg_after) / 2.0)
        estimator.record_bg(event.minute + 180.0, event.true_bg_after)
        estimator.update(event.minute + 181.0)
    return float(estimator.get_params()["isf"])


def load_clean_events(path: Path, cohort: str) -> list[CleanCorrectionEvent]:
    if not path.exists():
        return []
    rows: list[CleanCorrectionEvent] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                CleanCorrectionEvent(
                    cohort=cohort,
                    patient=row["patient"],
                    event_index=int(float(row["event_index"])),
                    minute=float(row["minute"]),
                    dose_units=float(row["dose_units"]),
                    true_bg_before=float(row["true_bg_before"]),
                    true_bg_after=float(row["true_bg_after"]),
                    true_bg_after_no_bolus=float(row["true_bg_after_no_bolus"]),
                    net_drop_mgdl=float(row["net_drop_mgdl"]),
                    causal_drop_mgdl=float(row["causal_drop_mgdl"]),
                    observed_net_isf=float(row["observed_net_isf"]),
                    observed_causal_isf=float(row["observed_causal_isf"]),
                    clinical_prior_isf=float(row.get("clinical_prior_isf") or POPULATION_ISF),
                )
            )
    return rows


def clean_estimates_for_patient(train: list[CleanCorrectionEvent]) -> dict[str, float]:
    observed = [event.observed_net_isf for event in train]
    clinical_prior = train[0].clinical_prior_isf if train else POPULATION_ISF
    recent_count = min(3, len(observed))
    modular = modular_positive_estimate(
        observed,
        prior_value=POPULATION_ISF,
        min_value=5,
        max_value=200,
        min_observations=4,
        data_dominance_observations=5,
        outlier_iqr_threshold=0.10,
    )
    adaptive = adaptive_robust_positive_estimate(
        observed,
        prior_value=POPULATION_ISF,
        min_value=5,
        max_value=200,
        min_observations=4,
        data_dominance_observations=5,
        outlier_iqr_threshold=0.10,
    )
    return {
        "fixed_population_50": POPULATION_ISF,
        "clinical_prior": clinical_prior,
        "empirical_mean": statistics.fmean(observed),
        "empirical_median": statistics.median(observed),
        "empirical_trimmed_mean": trimmed_mean(observed),
        "robust_winsorized_mean": winsorized_mean(observed),
        "robust_huber_center": huber_location(observed),
        "gluca_modular_trust": modular.value,
        "gluca_adaptive_trust": adaptive.value,
        "recent_3_mean": statistics.fmean(observed[-recent_count:]),
        "ewma_alpha_0_35": ewma(observed, initial=POPULATION_ISF, alpha=0.35),
        "original_bayes_v1": fit_v1_isf(train, prior=POPULATION_ISF),
        "gluca_v2_clean_gate": fit_v2_isf(train, prior=POPULATION_ISF),
        "app_style_log_posterior": log_posterior_estimate(
            observed,
            prior_value=POPULATION_ISF,
            prior_log_std=0.7,
            min_value=10,
            max_value=200,
            observed_floor_std=0.18,
        ),
    }


def benchmark_clean_correction_events(events: list[CleanCorrectionEvent]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[CleanCorrectionEvent]] = defaultdict(list)
    for event in events:
        groups[(event.cohort, event.patient)].append(event)

    for (cohort, patient), patient_events in sorted(groups.items()):
        patient_events = sorted(patient_events, key=lambda event: event.event_index)
        if len(patient_events) < 5:
            continue
        split_idx = max(1, int(len(patient_events) * 0.7))
        if len(patient_events) - split_idx < 2:
            split_idx = len(patient_events) - 2
        train = patient_events[:split_idx]
        holdout = patient_events[split_idx:]
        estimates = clean_estimates_for_patient(train)

        for event in holdout:
            for estimator, isf in estimates.items():
                predicted_drop = event.dose_units * isf
                predictions.append(
                    {
                        "protocol": "clean_correction_challenge",
                        "cohort": cohort,
                        "patient": patient,
                        "event_index": event.event_index,
                        "estimator": estimator,
                        "n_train": len(train),
                        "n_holdout": len(holdout),
                        "isf_estimate": isf,
                        "dose_units": event.dose_units,
                        "actual_net_drop": event.net_drop_mgdl,
                        "actual_causal_drop": event.causal_drop_mgdl,
                        "predicted_drop": predicted_drop,
                        "net_abs_error": abs(predicted_drop - event.net_drop_mgdl),
                        "causal_abs_error": abs(predicted_drop - event.causal_drop_mgdl),
                    }
                )

    summary: list[dict[str, Any]] = []
    for (cohort, estimator), rows in sorted(
        defaultdict(list, {
            key: [row for row in predictions if (row["cohort"], row["estimator"]) == key]
            for key in {(row["cohort"], row["estimator"]) for row in predictions}
        }).items()
    ):
        baseline_rows = [
            row for row in predictions
            if row["cohort"] == cohort and row["estimator"] == "fixed_population_50"
        ]
        empirical_rows = [
            row for row in predictions
            if row["cohort"] == cohort and row["estimator"] == "empirical_mean"
        ]
        net_mae = statistics.fmean(row["net_abs_error"] for row in rows)
        causal_mae = statistics.fmean(row["causal_abs_error"] for row in rows)
        fixed_net_mae = statistics.fmean(row["net_abs_error"] for row in baseline_rows)
        fixed_causal_mae = statistics.fmean(row["causal_abs_error"] for row in baseline_rows)
        empirical_net_mae = statistics.fmean(row["net_abs_error"] for row in empirical_rows)
        summary.append(
            {
                "protocol": "clean_correction_challenge",
                "cohort": cohort,
                "estimator": estimator,
                "n_holdout_events": len(rows),
                "net_mae": net_mae,
                "causal_mae": causal_mae,
                "net_improvement_vs_fixed_pct": pct_improvement(fixed_net_mae, net_mae),
                "causal_improvement_vs_fixed_pct": pct_improvement(fixed_causal_mae, causal_mae),
                "net_improvement_vs_empirical_mean_pct": pct_improvement(empirical_net_mae, net_mae),
            }
        )
    return summary, predictions


def obs_values(param: Any) -> list[float]:
    values = []
    for observation in getattr(param, "observations", []):
        try:
            values.append(float(observation[1]))
        except Exception:
            continue
    return values


def fallback_mean(values: list[float], fallback: float) -> float:
    return statistics.fmean(values) if values else fallback


def fallback_median(values: list[float], fallback: float) -> float:
    return statistics.median(values) if values else fallback


def metric_row(
    *,
    patient: str,
    seed: int,
    estimator: str,
    estimates: dict[str, float],
    truths: dict[str, float],
    counts: dict[str, int],
    done_ever: bool,
    first_done_step: int | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol": "rl4bg_free_living_replay",
        "patient": patient,
        "seed": seed,
        "estimator": estimator,
        "done_ever": done_ever,
        "first_done_step": first_done_step,
    }
    errors = []
    for param in ("isf", "cr", "basal"):
        estimate = estimates[param]
        truth = truths[param]
        error = abs_pct_error(estimate, truth)
        errors.append(error)
        row[f"true_{param}"] = truth
        row[f"{param}_estimate"] = estimate
        row[f"{param}_abs_pct_error"] = error
        row[f"n_{param}_observations"] = counts.get(param, 0)
    row["mean_abs_pct_error"] = statistics.fmean(errors)
    return row


def run_rl4bg_single(
    *,
    patient: str,
    seed: int,
    days: int,
    target: float,
    rl4bg_root: Path,
    bg_source: str,
    update_interval_min: float,
) -> list[dict[str, Any]]:
    env = build_env(patient, seed, rl4bg_root, time_std=None)
    env.reset()
    sample_time = float(env.sample_time)
    total_steps = int(days * env.day)

    truths = {
        "isf": float(env.CF),
        "cr": float(env.CR),
        "basal": float(env.ideal_basal) * 60.0,
    }

    v1 = BayesianEstimator(
        isf_prior=POPULATION_ISF,
        cr_prior=POPULATION_CR,
        basal_prior=POPULATION_BASAL,
        bg_target=target,
    )
    v2 = BayesianEstimatorV2(
        isf_prior=POPULATION_ISF,
        cr_prior=POPULATION_CR,
        basal_prior=POPULATION_BASAL,
        bg_target=target,
        use_covariates=False,
        cohort="adolescent",
    )

    manual_bb = build_manual_bb(env, target)
    action_umin = float(manual_bb.manual_bb_policy(carbs=0.0, glucose=target).basal)
    update_steps = max(1, int(round(update_interval_min / sample_time)))
    done_ever = False
    first_done_step: int | None = None

    for step in range(total_steps):
        _, _, done, info = env.step(action=max(0.0, float(action_umin)))
        done_ever = done_ever or bool(done)
        if done and first_done_step is None:
            first_done_step = step

        time_min = (step + 1) * sample_time
        cgm_now = float(env.env.CGM_hist[-1])
        bg_now = float(env.env.BG_hist[-1])
        glucose = bg_now if bg_source == "true_bg" else cgm_now
        v1.record_bg(time_min, glucose)
        v2.record_bg(time_min, glucose)

        meal_grams = float(info["meal"]) * sample_time
        if meal_grams > 0:
            v1.record_meal(time_min, meal_grams)
            v2.record_meal(time_min, meal_grams)

        action = manual_bb.manual_bb_policy(carbs=meal_grams, glucose=cgm_now)
        basal_step_units = max(0.0, float(action.basal)) * sample_time
        bolus_step_units = max(0.0, float(action.bolus)) * sample_time
        if basal_step_units > 0:
            v1.record_insulin(time_min, basal_step_units, "basal")
            v2.record_insulin(time_min, basal_step_units, "basal")
        if bolus_step_units > 0:
            v1.record_insulin(time_min, bolus_step_units, "bolus")
            v2.record_insulin(time_min, bolus_step_units, "bolus")

        if (step + 1) % update_steps == 0 or step == total_steps - 1:
            v1.update(time_min)
            v2.update(time_min)

        action_umin = max(0.0, float(action.basal + action.bolus))

    v1_params = v1.get_params()
    v2_params = v2.get_params()
    v1_obs = {
        "isf": obs_values(v1.isf),
        "cr": obs_values(v1.cr),
        "basal": obs_values(v1.basal),
    }
    v2_obs = {
        "isf": obs_values(v2.isf),
        "cr": obs_values(v2.cr),
        "basal": obs_values(v2.basal),
    }
    modular_isf = modular_positive_estimate(
        v2_obs["isf"],
        prior_value=POPULATION_ISF,
        min_value=5,
        max_value=200,
        min_observations=4,
        data_dominance_observations=5,
        outlier_iqr_threshold=0.10,
    )
    modular_cr = modular_positive_estimate(
        v2_obs["cr"],
        prior_value=POPULATION_CR,
        min_value=2,
        max_value=50,
        min_observations=5,
        data_dominance_observations=10,
        outlier_iqr_threshold=0.12,
    )
    modular_basal = modular_positive_estimate(
        v2_obs["basal"],
        prior_value=POPULATION_BASAL,
        min_value=0.05,
        max_value=5,
        min_observations=4,
        data_dominance_observations=8,
        outlier_iqr_threshold=0.12,
    )
    adaptive_isf = adaptive_robust_positive_estimate(
        v2_obs["isf"],
        prior_value=POPULATION_ISF,
        min_value=5,
        max_value=200,
        min_observations=4,
        data_dominance_observations=5,
        outlier_iqr_threshold=0.10,
    )
    adaptive_cr = adaptive_robust_positive_estimate(
        v2_obs["cr"],
        prior_value=POPULATION_CR,
        min_value=2,
        max_value=50,
        min_observations=5,
        data_dominance_observations=10,
        outlier_iqr_threshold=0.12,
    )
    adaptive_basal = adaptive_robust_positive_estimate(
        v2_obs["basal"],
        prior_value=POPULATION_BASAL,
        min_value=0.05,
        max_value=5,
        min_observations=4,
        data_dominance_observations=8,
        outlier_iqr_threshold=0.12,
    )
    # Sparse clean-correction evidence is a first-class failure mode. When V2
    # has fewer than four clean ISF observations, preserve its clean-gated
    # posterior/prior rather than letting a generic robust center overreact to
    # one or two correction events.
    modular_isf_value = (
        float(v2_params["isf"])
        if len(v2_obs["isf"]) < 4
        else modular_isf.value
    )
    adaptive_isf_value = (
        float(v2_params["isf"])
        if len(v2_obs["isf"]) < 4
        else adaptive_isf.value
    )
    run_rows = [
        metric_row(
            patient=patient,
            seed=seed,
            estimator="population_prior",
            estimates={"isf": POPULATION_ISF, "cr": POPULATION_CR, "basal": POPULATION_BASAL},
            truths=truths,
            counts={"isf": 0, "cr": 0, "basal": 0},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="oracle_therapy_profile_upper_bound",
            estimates=truths,
            truths=truths,
            counts={"isf": 0, "cr": 0, "basal": 0},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="original_bayes_v1",
            estimates={
                "isf": float(v1_params["isf"]),
                "cr": float(v1_params["cr"]),
                "basal": float(v1_params["basal"]),
            },
            truths=truths,
            counts={key: len(value) for key, value in v1_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="gluca_v2_clean_gate",
            estimates={
                "isf": float(v2_params["isf"]),
                "cr": float(v2_params["cr"]),
                "basal": float(v2_params["basal"]),
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="v1_observation_mean",
            estimates={
                "isf": fallback_mean(v1_obs["isf"], POPULATION_ISF),
                "cr": fallback_mean(v1_obs["cr"], POPULATION_CR),
                "basal": fallback_mean(v1_obs["basal"], POPULATION_BASAL),
            },
            truths=truths,
            counts={key: len(value) for key, value in v1_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="v1_observation_median",
            estimates={
                "isf": fallback_median(v1_obs["isf"], POPULATION_ISF),
                "cr": fallback_median(v1_obs["cr"], POPULATION_CR),
                "basal": fallback_median(v1_obs["basal"], POPULATION_BASAL),
            },
            truths=truths,
            counts={key: len(value) for key, value in v1_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="v2_observation_mean",
            estimates={
                "isf": fallback_mean(v2_obs["isf"], POPULATION_ISF),
                "cr": fallback_mean(v2_obs["cr"], POPULATION_CR),
                "basal": fallback_mean(v2_obs["basal"], POPULATION_BASAL),
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="app_style_log_posterior_from_clean_obs",
            estimates={
                "isf": log_posterior_estimate(
                    v2_obs["isf"],
                    prior_value=POPULATION_ISF,
                    prior_log_std=0.7,
                    min_value=10,
                    max_value=200,
                ),
                "cr": log_posterior_estimate(
                    # Mirrors app's medium-confidence meal weighting.
                    [value for value in v2_obs["cr"] for _ in range(2)],
                    prior_value=POPULATION_CR,
                    prior_log_std=0.6,
                    min_value=2,
                    max_value=50,
                ),
                "basal": log_posterior_estimate(
                    v2_obs["basal"],
                    prior_value=POPULATION_BASAL,
                    prior_log_std=0.5,
                    min_value=0.05,
                    max_value=5,
                    observed_floor_std=0.25,
                ),
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="gluca_modular_trust",
            estimates={
                "isf": modular_isf_value,
                "cr": modular_cr.value,
                "basal": modular_basal.value,
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="gluca_adaptive_trust",
            estimates={
                "isf": adaptive_isf_value,
                "cr": adaptive_cr.value,
                "basal": adaptive_basal.value,
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
        metric_row(
            patient=patient,
            seed=seed,
            estimator="normal_bayes_from_v2_obs",
            estimates={
                "isf": normal_posterior_estimate(
                    v2_obs["isf"],
                    prior_value=POPULATION_ISF,
                    prior_std=25,
                    min_value=10,
                    max_value=200,
                    observed_floor_std=8,
                ),
                "cr": normal_posterior_estimate(
                    v2_obs["cr"],
                    prior_value=POPULATION_CR,
                    prior_std=5,
                    min_value=2,
                    max_value=50,
                    observed_floor_std=2,
                ),
                "basal": normal_posterior_estimate(
                    v2_obs["basal"],
                    prior_value=POPULATION_BASAL,
                    prior_std=0.5,
                    min_value=0.05,
                    max_value=5,
                    observed_floor_std=0.1,
                ),
            },
            truths=truths,
            counts={key: len(value) for key, value in v2_obs.items()},
            done_ever=done_ever,
            first_done_step=first_done_step,
        ),
    ]
    return run_rows


def aggregate_rl4bg_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["estimator"]].append(row)
    population_rows = grouped.get("population_prior", [])
    population_errors: dict[str, float] = {}
    if population_rows:
        for key in ("isf_abs_pct_error", "cr_abs_pct_error", "basal_abs_pct_error", "mean_abs_pct_error"):
            population_errors[key] = statistics.fmean(float(row[key]) for row in population_rows)

    aggregate: list[dict[str, Any]] = []
    for estimator, estimator_rows in sorted(grouped.items()):
        row: dict[str, Any] = {
            "protocol": "rl4bg_free_living_replay",
            "estimator": estimator,
            "n_runs": len(estimator_rows),
            "done_rate": sum(1 for item in estimator_rows if item.get("done_ever")) / max(len(estimator_rows), 1),
        }
        for key in (
            "isf_abs_pct_error",
            "cr_abs_pct_error",
            "basal_abs_pct_error",
            "mean_abs_pct_error",
            "n_isf_observations",
            "n_cr_observations",
            "n_basal_observations",
        ):
            stats = summarize([float(item[key]) for item in estimator_rows if item.get(key) is not None])
            for stat, value in stats.items():
                row[f"{key}_{stat}"] = value
        for key in ("isf_abs_pct_error", "cr_abs_pct_error", "basal_abs_pct_error", "mean_abs_pct_error"):
            row[f"{key}_improvement_vs_population_mean_pct"] = pct_improvement(
                population_errors.get(key),
                row.get(f"{key}_mean"),
            )
        aggregate.append(row)
    return aggregate


def write_report(
    path: Path,
    *,
    clean_summary: list[dict[str, Any]],
    rl4bg_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    def best_clean(cohort: str) -> list[dict[str, Any]]:
        rows = [row for row in clean_summary if row["cohort"] == cohort]
        return sorted(rows, key=lambda row: float(row["net_mae"]))[:8]

    def best_rl4bg() -> list[dict[str, Any]]:
        rows = [
            row for row in rl4bg_summary
            if row["estimator"] != "oracle_therapy_profile_upper_bound"
        ]
        return sorted(rows, key=lambda row: float(row["mean_abs_pct_error_median"] or 999))[:10]

    lines = [
        "# Estimator Baseline Benchmark",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "## What Was Tested",
        "",
        "- Clean correction holdout: future correction-response prediction from isolated Padova correction events.",
        "- RL4BG free-living replay: final recovery of known adolescent CF/CR/basal from meal/insulin/CGM logs.",
        "- Baselines include fixed population, clinical prior, empirical mean/median, robust mean/Huber, EWMA, original Bayes, V2 clean-gated Bayes, app-style log posterior, modular trust, and adaptive trust estimates.",
        "",
        "## Modularized Failure Modes",
        "",
        "- Prior anchoring: a strong population prior can hurt after several clean patient-specific events.",
        "- Confounded ISF observations: meal residuals can corrupt ISF if carbs, CR, or timing are wrong.",
        "- Sparse clean corrections: if clean ISF events are absent, the estimator should shrink toward prior instead of inventing confidence.",
        "- Outlier or multimodal evidence: correction observations can form split clusters; use Huber center rather than simple averaging.",
        "- Stable evidence with single-point influence: winsorized mean reduces the impact of one odd event without discarding the data.",
        "- Estimator-selection risk: adaptive trust chooses a robust center with chronological one-step validation on training observations only.",
        "",
        "## Clean Correction Holdout",
        "",
    ]
    for cohort in sorted({row["cohort"] for row in clean_summary}):
        lines += [
            f"### {cohort.title()}",
            "",
            "| Estimator | Holdout n | Net MAE | Improvement vs fixed 50 | Improvement vs empirical mean |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in best_clean(cohort):
            lines.append(
                "| {estimator} | {n} | {mae:.2f} | {imp_fixed:.2f}% | {imp_emp:.2f}% |".format(
                    estimator=row["estimator"],
                    n=row["n_holdout_events"],
                    mae=float(row["net_mae"]),
                    imp_fixed=float(row["net_improvement_vs_fixed_pct"] or 0.0),
                    imp_emp=float(row["net_improvement_vs_empirical_mean_pct"] or 0.0),
                )
            )
        lines.append("")

    lines += [
        "## RL4BG Free-Living Parameter Recovery",
        "",
        "| Estimator | Runs | Median ISF err | Median CR err | Median basal err | Median mean err | Mean improvement vs population |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rl4bg():
        lines.append(
            "| {estimator} | {n} | {isf:.2f}% | {cr:.2f}% | {basal:.2f}% | {mean:.2f}% | {imp:.2f}% |".format(
                estimator=row["estimator"],
                n=row["n_runs"],
                isf=float(row["isf_abs_pct_error_median"] or 0.0),
                cr=float(row["cr_abs_pct_error_median"] or 0.0),
                basal=float(row["basal_abs_pct_error_median"] or 0.0),
                mean=float(row["mean_abs_pct_error_median"] or 0.0),
                imp=float(row["mean_abs_pct_error_improvement_vs_population_mean_pct"] or 0.0),
            )
        )

    lines += [
        "",
        "## Critical Interpretation",
        "",
        "- Clean correction events are the valid ISF-identification setting. Strong performance there supports the data-layer + Bayesian mechanism.",
        "- RL4BG free-living replay tests whether the estimator safely handles confounded logs. If V2 beats V1 there, that supports the stricter trust/gating changes.",
        "- The oracle therapy-profile row is an upper bound, not a usable learned estimator.",
        "- Be careful claiming superiority over empirical personalized baselines unless the result beats empirical mean or median, not just fixed population values.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_outputs(outdir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    include_paths = [
        outdir / "estimator_baseline_summary.json",
        outdir / "estimator_baseline_report.md",
        outdir / "clean_isf_estimator_summary.csv",
        outdir / "clean_isf_holdout_predictions.csv",
        outdir / "rl4bg_parameter_estimator_summary.csv",
        outdir / "rl4bg_parameter_estimator_run_metrics.csv",
        SCRIPT_DIR / "run_estimator_baseline_benchmark.py",
        REPO_ROOT / "bayesian_mpc" / "bayesian_estimator.py",
        REPO_ROOT / "bayesian_mpc" / "bayesian_v2.py",
        REPO_ROOT / "bayesian_mpc" / "estimator_failure_modes.py",
        REPO_ROOT / "Gluca_ios" / "Gluca" / "Models" / "ParameterLearner.swift",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include_paths:
            if path.exists():
                resolved = path.resolve()
                try:
                    arcname = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    arcname = Path("external_outputs") / resolved.name
                zf.write(resolved, arcname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl4bg-root", type=Path, default=DEFAULT_RL4BG_ROOT)
    parser.add_argument("--rl4bg-patients", nargs="+", default=ADOLESCENTS)
    parser.add_argument("--rl4bg-seeds", type=int, nargs="+", default=[6101, 6102, 6103])
    parser.add_argument("--rl4bg-days", type=int, default=10)
    parser.add_argument("--target", type=float, default=140.0)
    parser.add_argument("--bg-source", choices=["cgm", "true_bg"], default="cgm")
    parser.add_argument("--update-interval-min", type=float, default=30.0)
    parser.add_argument("--skip-rl4bg", action="store_true")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "estimator_baseline_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    clean_sources = [
        (
            "adults",
            REPO_ROOT / "validation" / "real_parameter_learning_validation" / "padova_clean_correction_events.csv",
        ),
        (
            "adolescents",
            REPO_ROOT / "validation" / "real_parameter_learning_validation" / "adolescent_clean_correction_rerun" / "padova_clean_correction_events.csv",
        ),
    ]
    clean_events: list[CleanCorrectionEvent] = []
    for cohort, path in clean_sources:
        loaded = load_clean_events(path, cohort)
        print(f"Loaded {len(loaded)} clean correction events for {cohort} from {path}", flush=True)
        clean_events.extend(loaded)

    clean_summary, clean_predictions = benchmark_clean_correction_events(clean_events)

    rl4bg_rows: list[dict[str, Any]] = []
    rl4bg_summary: list[dict[str, Any]] = []
    if not args.skip_rl4bg:
        args.rl4bg_root = ensure_rl4bg(args.rl4bg_root)
        configure_paths(args.rl4bg_root)
        total = len(args.rl4bg_patients) * len(args.rl4bg_seeds)
        idx = 0
        for seed in args.rl4bg_seeds:
            for patient in args.rl4bg_patients:
                idx += 1
                print(f"[{idx}/{total}] RL4BG estimator replay {patient} seed={seed}", flush=True)
                rl4bg_rows.extend(
                    run_rl4bg_single(
                        patient=patient,
                        seed=seed,
                        days=args.rl4bg_days,
                        target=args.target,
                        rl4bg_root=args.rl4bg_root,
                        bg_source=args.bg_source,
                        update_interval_min=args.update_interval_min,
                    )
                )
        rl4bg_summary = aggregate_rl4bg_rows(rl4bg_rows)

    write_csv(args.outdir / "clean_isf_estimator_summary.csv", [rounded_json(row) for row in clean_summary])
    write_csv(args.outdir / "clean_isf_holdout_predictions.csv", [rounded_json(row) for row in clean_predictions])
    write_csv(args.outdir / "rl4bg_parameter_estimator_run_metrics.csv", [rounded_json(row) for row in rl4bg_rows])
    write_csv(args.outdir / "rl4bg_parameter_estimator_summary.csv", [rounded_json(row) for row in rl4bg_summary])

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "rl4bg_root": str(args.rl4bg_root) if not args.skip_rl4bg else None,
        "rl4bg_patients": args.rl4bg_patients,
        "rl4bg_seeds": args.rl4bg_seeds,
        "rl4bg_days": args.rl4bg_days,
        "target": args.target,
        "bg_source": args.bg_source,
        "update_interval_min": args.update_interval_min,
        "runtime_seconds": time.time() - start,
        "notes": [
            "Estimator benchmark only; no controller performance is inferred from this file.",
            "Clean correction holdout is the valid ISF-identification setting.",
            "RL4BG replay tests estimator robustness under confounded meal/insulin logs.",
            "Oracle therapy-profile upper bound is included only to anchor the error scale.",
        ],
    }
    summary = {
        "metadata": metadata,
        "clean_correction_summary": clean_summary,
        "rl4bg_parameter_recovery_summary": rl4bg_summary,
        "files": {
            "clean_summary_csv": str(args.outdir / "clean_isf_estimator_summary.csv"),
            "clean_predictions_csv": str(args.outdir / "clean_isf_holdout_predictions.csv"),
            "rl4bg_summary_csv": str(args.outdir / "rl4bg_parameter_estimator_summary.csv"),
            "rl4bg_run_metrics_csv": str(args.outdir / "rl4bg_parameter_estimator_run_metrics.csv"),
            "report_md": str(args.outdir / "estimator_baseline_report.md"),
            "zip": str(SCRIPT_DIR / "estimator_baseline_claude_package.zip"),
        },
    }
    summary_path = args.outdir / "estimator_baseline_summary.json"
    summary_path.write_text(json.dumps(rounded_json(summary), indent=2) + "\n", encoding="utf-8")
    write_report(args.outdir / "estimator_baseline_report.md", clean_summary=clean_summary, rl4bg_summary=rl4bg_summary, metadata=metadata)
    package_outputs(args.outdir, SCRIPT_DIR / "estimator_baseline_claude_package.zip")

    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {args.outdir / 'estimator_baseline_report.md'}", flush=True)
    print(f"Wrote {SCRIPT_DIR / 'estimator_baseline_claude_package.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
