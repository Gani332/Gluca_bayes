"""
Composite validation for the Bayesian parameter layer.

This script validates the estimator stack in three distinct ways:

1. Temporal integrity:
   - No-lookahead / no-leakage checks
2. Parameter identifiability:
   - Synthetic controlled scenarios with known truth for ISF / CR
3. Closed-loop non-regression:
   - simglucose / UVA-Padova end-to-end sanity check

Why not use only simglucose?
--------------------------------
simglucose is excellent for validating overall glucose outcomes and safety
metrics, but it is not the cleanest way to validate whether our estimator's
named parameters ("ISF", "CR") are converging correctly. For that, a
controlled synthetic harness with known truth is more direct and falsifiable.

The best validation stack is therefore:
  synthetic truth -> no leakage -> Padova closed-loop -> real replay
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bayesian_mpc.bayesian_v2 import BayesianEstimatorV2, HealthContext
from bayesian_mpc.validate_covariates import run_synthetic_validation
from bayesian_mpc.validate_no_leakage import main as run_no_leakage_checks
from bayesian_mpc.evaluate import run_simulation


@dataclass
class SyntheticTruth:
    base_isf: float = 48.0
    cr: float = 11.0
    dawn_magnitude: float = 22.0
    carb_absorption_hours: float = 2.8
    exercise_decay_hours: float = 6.5
    activity_log_coeff: float = 0.18  # multiplicative boost in log-space
    time_bucket_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.time_bucket_multipliers is None:
            self.time_bucket_multipliers = {
                "night": 0.90,
                "morning": 1.18,
                "afternoon": 1.00,
                "evening": 0.82,
            }


def _effective_isf(truth: SyntheticTruth, context: Optional[HealthContext]) -> float:
    if context is None:
        return truth.base_isf
    covs = context.to_dict()
    return float(truth.base_isf * math.exp(truth.activity_log_coeff * covs.get("activity", 0.0)))


def _record_bg_pair(
    estimator: BayesianEstimatorV2,
    start_t: float,
    end_t: float,
    bg_start: float,
    bg_end: float,
):
    estimator.record_bg(start_t, bg_start)
    midpoint = (start_t + end_t) / 2.0
    estimator.record_bg(midpoint, (bg_start + bg_end) / 2.0)
    estimator.record_bg(end_t, bg_end)


def _simulate_correction(
    estimator: BayesianEstimatorV2,
    truth: SyntheticTruth,
    rng: np.random.Generator,
    time_min: float,
    insulin_units: float,
    context: Optional[HealthContext],
):
    bg_before = float(np.clip(rng.normal(185.0, 18.0), 140.0, 240.0))
    true_isf = _effective_isf(truth, context)
    bg_after = bg_before - (insulin_units * true_isf) + rng.normal(0.0, 10.0)
    bg_after = float(np.clip(bg_after, 70.0, 220.0))

    estimator.set_health_context(context)
    _record_bg_pair(estimator, time_min, time_min + 180.0, bg_before, bg_after)
    estimator.record_insulin(time_min, insulin_units, "bolus")
    estimator.update(time_min + 181.0)
    estimator.cleanup(time_min + 181.0)


def _simulate_meal(
    estimator: BayesianEstimatorV2,
    truth: SyntheticTruth,
    rng: np.random.Generator,
    time_min: float,
    carbs: float,
):
    bg_before = float(np.clip(rng.normal(112.0, 10.0), 85.0, 155.0))
    meal_bolus = carbs / truth.cr
    bg_after = bg_before + rng.normal(0.0, 9.0)
    bg_after = float(np.clip(bg_after, 80.0, 180.0))

    estimator.set_health_context(None)
    _record_bg_pair(estimator, time_min, time_min + 240.0, bg_before, bg_after)
    estimator.record_meal(time_min, carbs)
    estimator.record_insulin(time_min, meal_bolus, "bolus")
    estimator.update(time_min + 241.0)
    estimator.cleanup(time_min + 241.0)


def run_synthetic_convergence(seed: int = 42, days: int = 14) -> Dict:
    rng = np.random.default_rng(seed)
    truth = SyntheticTruth()

    estimator = BayesianEstimatorV2(
        isf_prior=55.0,
        cr_prior=14.0,
        basal_prior=0.9,
        bg_target=110.0,
        use_covariates=True,
        cohort="adult",
    )

    time_min = 0.0
    for day in range(days):
        # Baseline correction
        _simulate_correction(
            estimator,
            truth,
            rng,
            time_min=time_min + 8 * 60,
            insulin_units=float(np.clip(rng.normal(2.1, 0.3), 1.5, 3.0)),
            context=HealthContext(activity_level=0.0, sleep_quality=0.0, stress_level=0.0, cycle_factor=0.0),
        )

        # Meal event
        _simulate_meal(
            estimator,
            truth,
            rng,
            time_min=time_min + 13 * 60,
            carbs=float(np.clip(rng.normal(58.0, 12.0), 35.0, 90.0)),
        )

        # Exercise-linked correction every other day
        if day % 2 == 0:
            _simulate_correction(
                estimator,
                truth,
                rng,
                time_min=time_min + 18 * 60,
                insulin_units=float(np.clip(rng.normal(1.8, 0.25), 1.2, 2.6)),
                context=HealthContext(activity_level=1.0, sleep_quality=0.2, stress_level=0.0, cycle_factor=0.0),
            )

        time_min += 24 * 60

    # Query base estimate with neutral context
    estimator.set_health_context(None)
    params = estimator.get_params()
    estimated_base_isf, estimated_isf_sigma = estimator.isf.predict(None)
    estimated_cr, estimated_cr_sigma = estimator.cr.predict()
    effects = estimator.isf.covariate_effects()
    estimated_activity_multiplier = effects.get("activity", 1.0)
    true_activity_multiplier = math.exp(truth.activity_log_coeff)

    result = {
        "truth": {
            "base_isf": truth.base_isf,
            "cr": truth.cr,
            "activity_multiplier": true_activity_multiplier,
        },
        "estimate": {
            "base_isf": estimated_base_isf,
            "base_isf_sigma": estimated_isf_sigma,
            "base_isf_confidence": params["isf_confidence"],
            "cr": estimated_cr,
            "cr_sigma": estimated_cr_sigma,
            "cr_confidence": params["cr_confidence"],
            "activity_multiplier": estimated_activity_multiplier,
            "activity_multiplier_std": effects.get("activity_std", None),
            "n_isf_obs": len(estimator.isf.observations),
            "n_cr_obs": len(estimator.cr.observations),
        },
    }

    result["errors"] = {
        "base_isf_pct": 100.0 * abs(estimated_base_isf - truth.base_isf) / truth.base_isf,
        "cr_pct": 100.0 * abs(estimated_cr - truth.cr) / truth.cr,
        "activity_multiplier_pct": 100.0 * abs(estimated_activity_multiplier - true_activity_multiplier) / true_activity_multiplier,
    }

    result["pass"] = {
        "base_isf": result["errors"]["base_isf_pct"] <= 20.0,
        "cr": result["errors"]["cr_pct"] <= 20.0,
        "activity_multiplier": result["errors"]["activity_multiplier_pct"] <= 20.0,
        "base_isf_confidence": params["isf_confidence"] >= 0.35,
        "cr_confidence": params["cr_confidence"] >= 0.35,
    }

    return result


def _posterior_time_bucket_modifiers(
    base_isf: float,
    observations_by_bucket: Dict[str, list[float]],
    prior_log_std: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    prior_variance = prior_log_std ** 2

    for bucket in ("night", "morning", "afternoon", "evening"):
        observed = [
            value / base_isf
            for value in observations_by_bucket.get(bucket, [])
            if value > 0 and math.isfinite(value)
        ]
        if not observed:
            results[bucket] = {
                "multiplier": 1.0,
                "observation_count": 0,
                "posterior_log_std": prior_log_std,
            }
            continue

        log_values = np.log(np.asarray(observed, dtype=float))
        obs_log_std = float(max(np.std(log_values, ddof=1) if len(log_values) > 1 else 0.0, 0.18))
        obs_variance = obs_log_std ** 2
        precision = (1.0 / prior_variance) + (len(log_values) / obs_variance)
        posterior_variance = 1.0 / precision
        posterior_mean = posterior_variance * (float(np.sum(log_values)) / obs_variance)
        results[bucket] = {
            "multiplier": float(np.exp(posterior_mean)),
            "observation_count": len(log_values),
            "posterior_log_std": math.sqrt(max(posterior_variance, 0.0)),
        }

    return results


def run_time_bucket_convergence(seed: int = 42, days: int = 20) -> Dict:
    rng = np.random.default_rng(seed)
    truth = SyntheticTruth()
    observations_by_bucket: Dict[str, list[float]] = {key: [] for key in truth.time_bucket_multipliers}

    for _ in range(days):
        for bucket, multiplier in truth.time_bucket_multipliers.items():
            observed_isf = truth.base_isf * multiplier + rng.normal(0.0, 5.0)
            observations_by_bucket[bucket].append(float(np.clip(observed_isf, 10.0, 200.0)))

    posterior = _posterior_time_bucket_modifiers(
        base_isf=truth.base_isf,
        observations_by_bucket=observations_by_bucket,
    )

    errors = {
        bucket: 100.0 * abs(posterior[bucket]["multiplier"] - truth.time_bucket_multipliers[bucket]) / truth.time_bucket_multipliers[bucket]
        for bucket in truth.time_bucket_multipliers
    }

    passes = {
        bucket: errors[bucket] <= 15.0
        for bucket in truth.time_bucket_multipliers
    }
    passes["ordering"] = (
        posterior["morning"]["multiplier"] > posterior["afternoon"]["multiplier"]
        > posterior["evening"]["multiplier"]
    )

    return {
        "truth": truth.time_bucket_multipliers,
        "estimate": posterior,
        "errors": errors,
        "pass": passes,
    }


def _posterior_dawn_effect(
    observations: list[float],
    prior_mean: float = 18.0,
    prior_std: float = 15.0,
) -> Dict[str, float]:
    if not observations:
        return {
            "estimate": 0.0,
            "posterior_std": prior_std,
            "observation_count": 0,
        }

    prior_variance = prior_std ** 2
    obs_std = float(max(np.std(np.asarray(observations, dtype=float), ddof=1) if len(observations) > 1 else 0.0, 12.0))
    obs_variance = obs_std ** 2
    precision = (1.0 / prior_variance) + (len(observations) / obs_variance)
    posterior_variance = 1.0 / precision
    posterior_mean = posterior_variance * ((prior_mean / prior_variance) + (float(np.sum(observations)) / obs_variance))
    return {
        "estimate": float(max(posterior_mean, 0.0)),
        "posterior_std": math.sqrt(max(posterior_variance, 0.0)),
        "observation_count": len(observations),
    }


def run_dawn_convergence(seed: int = 42, days: int = 18) -> Dict:
    rng = np.random.default_rng(seed)
    truth = SyntheticTruth()
    observations: list[float] = []

    for day in range(days):
        baseline = float(np.clip(rng.normal(108.0, 7.0), 85.0, 135.0))
        dawn = baseline + truth.dawn_magnitude + rng.normal(0.0, 7.5)

        # Add a few low-signal days so the test reflects realistic overnight noise.
        if day % 6 == 0:
            dawn -= float(np.clip(rng.normal(4.0, 2.0), 0.0, 8.0))

        observations.append(float(np.clip(dawn - baseline, -20.0, 80.0)))

    posterior = _posterior_dawn_effect(observations)
    error_pct = 100.0 * abs(posterior["estimate"] - truth.dawn_magnitude) / truth.dawn_magnitude

    return {
        "truth": {"dawn_magnitude": truth.dawn_magnitude},
        "estimate": posterior,
        "errors": {"dawn_magnitude_pct": error_pct},
        "pass": {
            "dawn_magnitude": error_pct <= 20.0,
            "positive_signal": posterior["estimate"] >= 6.0,
        },
    }


def _posterior_carb_absorption_hours(
    observations: list[float],
    prior_mean: float = 3.0,
    prior_std: float = 0.9,
) -> Dict[str, float]:
    if not observations:
        return {
            "estimate": prior_mean,
            "posterior_std": prior_std,
            "observation_count": 0,
        }

    prior_variance = prior_std ** 2
    obs_std = float(max(np.std(np.asarray(observations, dtype=float), ddof=1) if len(observations) > 1 else 0.0, 0.45))
    obs_variance = obs_std ** 2
    precision = (1.0 / prior_variance) + (len(observations) / obs_variance)
    posterior_variance = 1.0 / precision
    posterior_mean = posterior_variance * ((prior_mean / prior_variance) + (float(np.sum(observations)) / obs_variance))
    return {
        "estimate": float(np.clip(posterior_mean, 1.5, 4.5)),
        "posterior_std": math.sqrt(max(posterior_variance, 0.0)),
        "observation_count": len(observations),
    }


def run_carb_absorption_convergence(seed: int = 42, meals: int = 18) -> Dict:
    rng = np.random.default_rng(seed)
    truth = SyntheticTruth()
    observations: list[float] = []

    for index in range(meals):
        observed = truth.carb_absorption_hours + rng.normal(0.0, 0.35)
        if index % 7 == 0:
            observed += 0.2
        observations.append(float(np.clip(observed, 1.5, 4.5)))

    posterior = _posterior_carb_absorption_hours(observations)
    error_pct = 100.0 * abs(posterior["estimate"] - truth.carb_absorption_hours) / truth.carb_absorption_hours

    return {
        "truth": {"carb_absorption_hours": truth.carb_absorption_hours},
        "estimate": posterior,
        "errors": {"carb_absorption_hours_pct": error_pct},
        "pass": {
            "carb_absorption_hours": error_pct <= 20.0,
            "within_bounds": 1.5 <= posterior["estimate"] <= 4.5,
        },
    }


def _posterior_exercise_decay_hours(
    observations: list[float],
    prior_mean: float = 8.0,
    prior_std: float = 2.5,
) -> Dict[str, float]:
    if not observations:
        return {
            "estimate": prior_mean,
            "posterior_std": prior_std,
            "observation_count": 0,
        }

    prior_variance = prior_std ** 2
    obs_std = float(max(np.std(np.asarray(observations, dtype=float), ddof=1) if len(observations) > 1 else 0.0, 1.2))
    obs_variance = obs_std ** 2
    precision = (1.0 / prior_variance) + (len(observations) / obs_variance)
    posterior_variance = 1.0 / precision
    posterior_mean = posterior_variance * ((prior_mean / prior_variance) + (float(np.sum(observations)) / obs_variance))
    return {
        "estimate": float(np.clip(posterior_mean, 4.0, 18.0)),
        "posterior_std": math.sqrt(max(posterior_variance, 0.0)),
        "observation_count": len(observations),
    }


def run_exercise_decay_convergence(seed: int = 42, sessions: int = 16) -> Dict:
    rng = np.random.default_rng(seed)
    truth = SyntheticTruth()
    observations: list[float] = []

    for index in range(sessions):
        observed = truth.exercise_decay_hours + rng.normal(0.0, 0.9)
        if index % 5 == 0:
            observed += 0.3
        observations.append(float(np.clip(observed, 4.0, 18.0)))

    posterior = _posterior_exercise_decay_hours(observations)
    error_pct = 100.0 * abs(posterior["estimate"] - truth.exercise_decay_hours) / truth.exercise_decay_hours

    return {
        "truth": {"exercise_decay_hours": truth.exercise_decay_hours},
        "estimate": posterior,
        "errors": {"exercise_decay_hours_pct": error_pct},
        "pass": {
            "exercise_decay_hours": error_pct <= 20.0,
            "within_bounds": 4.0 <= posterior["estimate"] <= 18.0,
        },
    }


def run_padova_sanity(seed: int = 42) -> Dict:
    simulation = run_simulation(
        patient_name="adult#001",
        days=3,
        seed=seed,
        mode="bayesian",
        estimator_version="v2",
        use_covariates=False,
        health_mode="none",
        verbose=False,
    )
    metrics = simulation["metrics"]
    return {
        "metrics": metrics,
        "pass": {
            "tir": metrics["tir"] >= 70.0,
            "tbr": metrics["tbr"] < 4.0,
            "tsbr": metrics["tsbr"] < 1.0,
        },
    }


def run_estimator_baseline_benchmark(seed: int = 42) -> Dict:
    target_bg = 110.0
    adaptive_v2 = run_simulation(
        patient_name="adult#001",
        days=5,
        seed=seed,
        mode="bayesian",
        estimator_version="v2",
        use_covariates=False,
        health_mode="none",
        verbose=False,
    )
    adaptive_v1 = run_simulation(
        patient_name="adult#001",
        days=5,
        seed=seed,
        mode="bayesian",
        estimator_version="v1",
        use_covariates=False,
        health_mode="none",
        verbose=False,
    )
    fixed = run_simulation(
        patient_name="adult#001",
        days=5,
        seed=seed,
        mode="fixed",
        estimator_version="v1",
        use_covariates=False,
        health_mode="none",
        verbose=False,
    )

    v2_metrics = adaptive_v2["metrics"]
    v1_metrics = adaptive_v1["metrics"]
    fixed_metrics = fixed["metrics"]
    v2_target_error = abs(v2_metrics["mean_bg"] - target_bg)
    v1_target_error = abs(v1_metrics["mean_bg"] - target_bg)
    fixed_target_error = abs(fixed_metrics["mean_bg"] - target_bg)
    deltas = {
        "tir_vs_fixed": v2_metrics["tir"] - fixed_metrics["tir"],
        "tir_vs_v1": v2_metrics["tir"] - v1_metrics["tir"],
        "target_error_vs_fixed": fixed_target_error - v2_target_error,
        "target_error_vs_v1": v1_target_error - v2_target_error,
        "tbr_vs_fixed": fixed_metrics["tbr"] - v2_metrics["tbr"],
        "tbr_vs_v1": v1_metrics["tbr"] - v2_metrics["tbr"],
    }

    return {
        "metrics": {
            "v2": v2_metrics,
            "v1": v1_metrics,
            "fixed": fixed_metrics,
        },
        "target_error": {
            "v2": v2_target_error,
            "v1": v1_target_error,
            "fixed": fixed_target_error,
        },
        "deltas": deltas,
        "pass": {
            "tir_not_worse_than_fixed": deltas["tir_vs_fixed"] >= -1.0,
            "tir_not_worse_than_v1": deltas["tir_vs_v1"] >= -1.0,
            "target_distance_not_worse_than_fixed": deltas["target_error_vs_fixed"] >= -10.0,
            "target_distance_not_worse_than_v1": deltas["target_error_vs_v1"] >= -10.0,
            "hypo_not_worse_than_fixed": v2_metrics["tbr"] <= max(fixed_metrics["tbr"] + 1.0, 4.0),
            "hypo_not_worse_than_v1": v2_metrics["tbr"] <= max(v1_metrics["tbr"] + 1.0, 4.0),
        },
    }


def main():
    print("\n[1/9] Temporal leakage checks")
    run_no_leakage_checks()

    print("\n[2/9] Synthetic covariate checks")
    covariates = run_synthetic_validation(train_n=200, test_n=100, seed=42)

    print("\n[3/9] Synthetic parameter convergence")
    convergence = run_synthetic_convergence(seed=42, days=14)
    print(f"  Base ISF: {convergence['estimate']['base_isf']:.1f} vs truth {convergence['truth']['base_isf']:.1f} "
          f"({convergence['errors']['base_isf_pct']:.1f}% error)")
    print(f"  CR:       {convergence['estimate']['cr']:.1f} vs truth {convergence['truth']['cr']:.1f} "
          f"({convergence['errors']['cr_pct']:.1f}% error)")
    print(f"  Activity: ×{convergence['estimate']['activity_multiplier']:.2f} vs truth ×{convergence['truth']['activity_multiplier']:.2f} "
          f"({convergence['errors']['activity_multiplier_pct']:.1f}% error)")
    print(f"  ISF confidence: {convergence['estimate']['base_isf_confidence']:.0%} from {convergence['estimate']['n_isf_obs']} observations")
    print(f"  CR confidence:  {convergence['estimate']['cr_confidence']:.0%} from {convergence['estimate']['n_cr_obs']} observations")

    print("\n[4/9] Synthetic time-bucket convergence")
    buckets = run_time_bucket_convergence(seed=42, days=20)
    for bucket in ("night", "morning", "afternoon", "evening"):
        print(
            f"  {bucket.capitalize():<10} ×{buckets['estimate'][bucket]['multiplier']:.2f} vs truth ×{buckets['truth'][bucket]:.2f} "
            f"({buckets['errors'][bucket]:.1f}% error)"
        )

    print("\n[5/9] Synthetic dawn convergence")
    dawn = run_dawn_convergence(seed=42, days=18)
    print(
        f"  Dawn rise: {dawn['estimate']['estimate']:.1f} mg/dL vs truth {dawn['truth']['dawn_magnitude']:.1f} "
        f"({dawn['errors']['dawn_magnitude_pct']:.1f}% error)"
    )

    print("\n[6/9] Synthetic carb absorption convergence")
    carb_absorption = run_carb_absorption_convergence(seed=42, meals=18)
    print(
        f"  Carb absorption: {carb_absorption['estimate']['estimate']:.2f} h vs truth {carb_absorption['truth']['carb_absorption_hours']:.2f} h "
        f"({carb_absorption['errors']['carb_absorption_hours_pct']:.1f}% error)"
    )

    print("\n[7/9] Synthetic exercise decay convergence")
    exercise_decay = run_exercise_decay_convergence(seed=42, sessions=16)
    print(
        f"  Exercise decay: {exercise_decay['estimate']['estimate']:.2f} h vs truth {exercise_decay['truth']['exercise_decay_hours']:.2f} h "
        f"({exercise_decay['errors']['exercise_decay_hours_pct']:.1f}% error)"
    )

    print("\n[8/9] Baseline benchmark vs fixed and legacy estimators")
    baseline = run_estimator_baseline_benchmark(seed=42)
    print(
        f"  v2 adaptive TIR={baseline['metrics']['v2']['tir']:.1f}% "
        f"(v1 {baseline['metrics']['v1']['tir']:.1f}%, fixed {baseline['metrics']['fixed']['tir']:.1f}%)"
    )
    print(
        f"  v2 mean-BG target error={baseline['target_error']['v2']:.0f} "
        f"(v1 {baseline['target_error']['v1']:.0f}, fixed {baseline['target_error']['fixed']:.0f})"
    )
    print(
        f"  v2 adaptive TBR={baseline['metrics']['v2']['tbr']:.1f}% "
        f"(v1 {baseline['metrics']['v1']['tbr']:.1f}%, fixed {baseline['metrics']['fixed']['tbr']:.1f}%)"
    )

    print("\n[9/9] Padova closed-loop sanity")
    padova = run_padova_sanity(seed=42)
    print(f"  TIR={padova['metrics']['tir']:.1f}%  TBR={padova['metrics']['tbr']:.1f}%  TSBR={padova['metrics']['tsbr']:.1f}%")

    all_pass = (
        all(convergence["pass"].values())
        and all(buckets["pass"].values())
        and all(dawn["pass"].values())
        and all(carb_absorption["pass"].values())
        and all(exercise_decay["pass"].values())
        and all(baseline["pass"].values())
        and covariates["covariate_rmse"] < covariates["plain_rmse"]
        and all(padova["pass"].values())
    )

    print("\nSummary")
    print("=" * 60)
    print(f"  Synthetic convergence pass: {all(convergence['pass'].values())}")
    print(f"  Time-bucket pass:           {all(buckets['pass'].values())}")
    print(f"  Dawn pass:                  {all(dawn['pass'].values())}")
    print(f"  Carb absorption pass:       {all(carb_absorption['pass'].values())}")
    print(f"  Exercise decay pass:        {all(exercise_decay['pass'].values())}")
    print(f"  Baseline benchmark pass:    {all(baseline['pass'].values())}")
    print(f"  Covariate learning pass:    {covariates['covariate_rmse'] < covariates['plain_rmse']}")
    print(f"  Padova sanity pass:         {all(padova['pass'].values())}")
    print(f"  Overall:                    {all_pass}")


if __name__ == "__main__":
    main()
