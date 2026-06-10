#!/usr/bin/env python3
"""
Fresh UVA/Padova validation for Gluca's Bayesian ISF learning.

This is stricter than the blog mechanism check:
- Generates fresh simglucose virtual-patient traces.
- Uses true simulator glucose, not noisy CGM, for scoring.
- Uses an app-style clean correction gate: no meals/exercise/other boluses,
  180-minute response, dose >= 0.5 U, observed ISF in physiologic bounds.
- Uses chronological per-patient 70/30 train/holdout splits.
- Compares Bayesian posterior against fixed population, clinical-prior, and
  empirical-mean personalized baselines.

Protocol note:
The simulator does not naturally log many isolated correction-only events. This
script therefore runs a no-meal "correction challenge": basal is withheld to
create a hyperglycemic state, basal is restored for a stabilization period, then
a correction bolus is injected. A paired no-bolus branch is also simulated from
the same state to quantify counterfactual drift. The main claim should be about
UVA/Padova correction-challenge evidence, not free-living personal data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
POPULATION_ISF = 50.0

sys.path.insert(0, str(REPO_ROOT))

from bayesian_mpc.bayesian_estimator import BayesianEstimator  # noqa: E402
from bayesian_mpc.clinical_priors import derive_clinical_priors, infer_cohort_from_patient_name  # noqa: E402


@dataclass(frozen=True)
class CorrectionEvent:
    patient: str
    event_index: int
    minute: float
    dose_units: float
    true_bg_before: float
    true_bg_after: float
    true_bg_after_no_bolus: float
    raw_cgm_before: float
    raw_cgm_after: float
    net_drop_mgdl: float
    causal_drop_mgdl: float
    counterfactual_drift_mgdl: float
    observed_net_isf: float
    observed_causal_isf: float
    clinical_prior_isf: float


@dataclass
class PatientRun:
    patient: str
    clinical_prior_isf: float
    clinical_prior_cr: float
    true_basal_uhr: float
    events: list[CorrectionEvent]
    trace: list[dict[str, float]]
    excluded_counts: dict[str, int]


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


def pct_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def true_basal_uhr(patient: Any) -> float:
    # Same conversion already used in bayesian_mpc/evaluate.py.
    return float(patient._params.u2ss) * float(patient._params.BW) / 6000.0 * 60.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def get_patient_names(cohort: str, max_patients: int) -> list[str]:
    import simglucose

    params_file = Path(simglucose.__file__).resolve().parent / "params" / "vpatient_params.csv"
    df = pd.read_csv(params_file)
    names = [str(name) for name in df["Name"].tolist()]
    if cohort != "all":
        prefix = {"adults": "adult#", "adolescents": "adolescent#", "children": "child#"}[cohort]
        names = [name for name in names if name.startswith(prefix)]
    else:
        names = [name for name in names if name.startswith(("adult#", "adolescent#", "child#"))]
    return sorted(names)[:max_patients]


def step_env(env: Any, basal_umin: float, bolus_units: float = 0.0) -> Any:
    from simglucose.controller.base import Action

    # env.step advances three one-minute mini-steps. Pump action fields are
    # rates in U/min, so spread a bolus across the 3-minute simulator step.
    bolus_umin = bolus_units / 3.0 if bolus_units > 0 else 0.0
    return env.step(Action(basal=basal_umin, bolus=bolus_umin))


def simulate_no_bolus_branch(env: Any, basal_umin: float, response_steps: int) -> float:
    branch = copy.deepcopy(env)
    result = step_env(branch, basal_umin, bolus_units=0.0)
    for _ in range(response_steps - 1):
        result = step_env(branch, basal_umin, bolus_units=0.0)
    return float(result.info["bg"])


def append_trace(trace: list[dict[str, float]], minute: float, result: Any) -> None:
    trace.append(
        {
            "minute": float(minute),
            "true_bg": float(result.info["bg"]),
            "raw_cgm": float(result.observation.CGM),
        }
    )


def simulate_patient(
    patient_name: str,
    *,
    seed: int,
    target_events: int,
    max_days: int,
    threshold_mgdl: float,
    stabilization_minutes: int,
    response_minutes: int,
    max_abs_counterfactual_drift: float,
) -> PatientRun:
    from simglucose.actuator.pump import InsulinPump
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import CustomScenario

    patient = T1DPatient.withName(patient_name)
    cohort = infer_cohort_from_patient_name(patient_name)
    priors = derive_clinical_priors(cohort, float(patient._params.BW))
    basal_uhr = true_basal_uhr(patient)
    basal_umin = basal_uhr / 60.0

    scenario = CustomScenario(start_time=datetime(2024, 1, 1, 0, 0, 0), scenario=[])
    env = T1DSimEnv(
        patient,
        CGMSensor.withName("Dexcom", seed=seed),
        InsulinPump.withName("Insulet"),
        scenario,
    )
    result = env.reset()

    minute = 0.0
    trace: list[dict[str, float]] = []
    append_trace(trace, minute, result)
    events: list[CorrectionEvent] = []
    excluded_counts = {
        "candidate_events": 0,
        "low_starting_glucose": 0,
        "counterfactual_drift": 0,
        "nonphysiologic_observed_isf": 0,
        "simulation_done": 0,
    }

    max_steps = max_days * 480
    response_steps = int(response_minutes / 3)
    stabilization_steps = int(stabilization_minutes / 3)
    induction_cap_steps = int((18 * 60) / 3)
    steps = 0

    while len(events) < target_events and steps + response_steps + stabilization_steps < max_steps:
        # Induce a repeatable high-glucose correction state without meals.
        induction_steps = 0
        while (
            float(result.info["bg"]) < threshold_mgdl
            and induction_steps < induction_cap_steps
            and steps + response_steps + stabilization_steps < max_steps
        ):
            result = step_env(env, basal_umin=0.0, bolus_units=0.0)
            minute += 3.0
            steps += 1
            induction_steps += 1
            append_trace(trace, minute, result)
            if bool(result.done):
                excluded_counts["simulation_done"] += 1
                break

        # Restore basal and stabilize so net response is not just a rising trace.
        for _ in range(stabilization_steps):
            result = step_env(env, basal_umin=basal_umin, bolus_units=0.0)
            minute += 3.0
            steps += 1
            append_trace(trace, minute, result)
            if bool(result.done):
                excluded_counts["simulation_done"] += 1
                break

        true_bg_before = float(result.info["bg"])
        raw_cgm_before = float(result.observation.CGM)
        if true_bg_before < 150.0 or raw_cgm_before < 130.0:
            excluded_counts["low_starting_glucose"] += 1
            continue

        excluded_counts["candidate_events"] += 1
        dose = clamp((min(raw_cgm_before, 310.0) - 110.0) / POPULATION_ISF, 1.0, 4.0)
        event_minute = minute
        true_bg_after_no_bolus = simulate_no_bolus_branch(env, basal_umin, response_steps)

        # Main branch receives the actual correction bolus.
        result = step_env(env, basal_umin=basal_umin, bolus_units=dose)
        minute += 3.0
        steps += 1
        append_trace(trace, minute, result)
        for _ in range(response_steps - 1):
            result = step_env(env, basal_umin=basal_umin, bolus_units=0.0)
            minute += 3.0
            steps += 1
            append_trace(trace, minute, result)
            if bool(result.done):
                excluded_counts["simulation_done"] += 1
                break

        true_bg_after = float(result.info["bg"])
        raw_cgm_after = float(result.observation.CGM)
        net_drop = true_bg_before - true_bg_after
        causal_drop = true_bg_after_no_bolus - true_bg_after
        counterfactual_drift = true_bg_after_no_bolus - true_bg_before
        observed_net_isf = net_drop / dose
        observed_causal_isf = causal_drop / dose

        if abs(counterfactual_drift) > max_abs_counterfactual_drift:
            excluded_counts["counterfactual_drift"] += 1
            continue
        if not (5.0 < observed_net_isf < 200.0):
            excluded_counts["nonphysiologic_observed_isf"] += 1
            continue

        events.append(
            CorrectionEvent(
                patient=patient_name,
                event_index=len(events) + 1,
                minute=event_minute,
                dose_units=dose,
                true_bg_before=true_bg_before,
                true_bg_after=true_bg_after,
                true_bg_after_no_bolus=true_bg_after_no_bolus,
                raw_cgm_before=raw_cgm_before,
                raw_cgm_after=raw_cgm_after,
                net_drop_mgdl=net_drop,
                causal_drop_mgdl=causal_drop,
                counterfactual_drift_mgdl=counterfactual_drift,
                observed_net_isf=observed_net_isf,
                observed_causal_isf=observed_causal_isf,
                clinical_prior_isf=float(priors.isf),
            )
        )

    return PatientRun(
        patient=patient_name,
        clinical_prior_isf=float(priors.isf),
        clinical_prior_cr=float(priors.cr),
        true_basal_uhr=basal_uhr,
        events=events,
        trace=trace,
        excluded_counts=excluded_counts,
    )


def train_bayesian_isf(events: list[CorrectionEvent], prior_isf: float = POPULATION_ISF) -> dict[str, float]:
    estimator = BayesianEstimator(isf_prior=prior_isf, isf_sigma=25.0)
    for event in events:
        estimator.record_bg(event.minute, event.true_bg_before)
        estimator.record_insulin(event.minute, event.dose_units, "bolus")
        estimator.record_bg(event.minute + 90.0, (event.true_bg_before + event.true_bg_after) / 2.0)
        estimator.record_bg(event.minute + 180.0, event.true_bg_after)
        estimator.update(event.minute + 181.0)
    return estimator.get_params()


def mae(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def bootstrap_mean_ci(values: list[float], seed: int = 20260607, reps: int = 5000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_gt_0": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(reps)]
    ordered = sorted(samples)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": ordered[int(0.025 * (reps - 1))],
        "ci_high": ordered[int(0.975 * (reps - 1))],
        "p_gt_0": float(np.mean(np.asarray(samples) > 0.0)),
    }


def evaluate_predictions(patient_runs: list[PatientRun]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    holdout_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []

    for run in patient_runs:
        events = run.events
        if len(events) < 5:
            patient_rows.append(
                {
                    "patient": run.patient,
                    "status": "skipped_insufficient_events",
                    "n_events": len(events),
                    "n_train": 0,
                    "n_holdout": 0,
                }
            )
            continue

        split_idx = max(1, int(len(events) * 0.7))
        if len(events) - split_idx < 2:
            split_idx = len(events) - 2
        train = events[:split_idx]
        holdout = events[split_idx:]
        params = train_bayesian_isf(train)
        posterior_isf = float(params["isf"])
        posterior_std = float(params["isf_sigma"])
        empirical_isf = statistics.mean(event.observed_net_isf for event in train)

        errors = {"population": [], "clinical": [], "empirical": [], "bayesian": []}
        causal_errors = {"population": [], "clinical": [], "empirical": [], "bayesian": []}

        for event in holdout:
            predictions = {
                "population": POPULATION_ISF * event.dose_units,
                "clinical": run.clinical_prior_isf * event.dose_units,
                "empirical": empirical_isf * event.dose_units,
                "bayesian": posterior_isf * event.dose_units,
            }
            row = {
                "patient": run.patient,
                "event_index": event.event_index,
                "minute": event.minute,
                "dose_units": event.dose_units,
                "actual_net_drop": event.net_drop_mgdl,
                "actual_causal_drop": event.causal_drop_mgdl,
                "counterfactual_drift_mgdl": event.counterfactual_drift_mgdl,
                "observed_net_isf": event.observed_net_isf,
                "observed_causal_isf": event.observed_causal_isf,
                "train_posterior_isf": posterior_isf,
                "train_posterior_std": posterior_std,
                "train_empirical_isf": empirical_isf,
                "clinical_prior_isf": run.clinical_prior_isf,
            }
            for key, predicted in predictions.items():
                net_error = abs(predicted - event.net_drop_mgdl)
                causal_error = abs(predicted - event.causal_drop_mgdl)
                errors[key].append(net_error)
                causal_errors[key].append(causal_error)
                row[f"{key}_predicted_drop"] = predicted
                row[f"{key}_abs_error"] = net_error
                row[f"{key}_causal_abs_error"] = causal_error
            holdout_rows.append(row)

        patient_rows.append(
            {
                "patient": run.patient,
                "status": "evaluated",
                "n_events": len(events),
                "n_train": len(train),
                "n_holdout": len(holdout),
                "clinical_prior_isf": run.clinical_prior_isf,
                "train_empirical_isf": empirical_isf,
                "train_posterior_isf": posterior_isf,
                "train_posterior_std": posterior_std,
                "holdout_population_mae": mae(errors["population"]),
                "holdout_clinical_mae": mae(errors["clinical"]),
                "holdout_empirical_mae": mae(errors["empirical"]),
                "holdout_bayesian_mae": mae(errors["bayesian"]),
                "holdout_causal_population_mae": mae(causal_errors["population"]),
                "holdout_causal_clinical_mae": mae(causal_errors["clinical"]),
                "holdout_causal_empirical_mae": mae(causal_errors["empirical"]),
                "holdout_causal_bayesian_mae": mae(causal_errors["bayesian"]),
            }
        )

    aggregate_errors = {"population": [], "clinical": [], "empirical": [], "bayesian": []}
    aggregate_causal_errors = {"population": [], "clinical": [], "empirical": [], "bayesian": []}
    for row in holdout_rows:
        for key in aggregate_errors:
            aggregate_errors[key].append(float(row[f"{key}_abs_error"]))
            aggregate_causal_errors[key].append(float(row[f"{key}_causal_abs_error"]))

    bayes_errors = aggregate_errors["bayesian"]
    causal_bayes_errors = aggregate_causal_errors["bayesian"]
    comparisons: dict[str, Any] = {}
    causal_comparisons: dict[str, Any] = {}
    for key in ("population", "clinical", "empirical"):
        baseline_mae = mae(aggregate_errors[key])
        bayes_mae = mae(bayes_errors)
        diffs = [base - bayes for base, bayes in zip(aggregate_errors[key], bayes_errors)]
        comparisons[key] = {
            "baseline_mae": baseline_mae,
            "bayesian_mae": bayes_mae,
            "improvement_pct": pct_improvement(baseline_mae, bayes_mae),
            "paired_error_reduction": bootstrap_mean_ci(diffs),
        }

        causal_baseline_mae = mae(aggregate_causal_errors[key])
        causal_bayes_mae = mae(causal_bayes_errors)
        causal_diffs = [
            base - bayes for base, bayes in zip(aggregate_causal_errors[key], causal_bayes_errors)
        ]
        causal_comparisons[key] = {
            "baseline_mae": causal_baseline_mae,
            "bayesian_mae": causal_bayes_mae,
            "improvement_pct": pct_improvement(causal_baseline_mae, causal_bayes_mae),
            "paired_error_reduction": bootstrap_mean_ci(causal_diffs),
        }

    aggregate = {
        "n_patients_total": len(patient_runs),
        "n_patients_evaluated": sum(1 for row in patient_rows if row.get("status") == "evaluated"),
        "n_events_total": sum(len(run.events) for run in patient_runs),
        "n_holdout_events": len(holdout_rows),
        "net_drop_target": comparisons,
        "counterfactual_causal_target": causal_comparisons,
    }
    return holdout_rows, patient_rows, aggregate


class SimpleUKF:
    def __init__(self, initial_reading: float):
        self.n = 3
        self.alpha = 0.3
        self.beta = 2.0
        self.kappa = 0.0
        self.tau_lag = 15.0
        self.tau_rate = 20.0
        self.x = np.array([initial_reading, initial_reading, 0.0], dtype=float)
        self.P = np.diag([400.0, 400.0, 1.0])
        self.Q = np.diag([9.0, 2.25, 0.09])
        self.R = 144.0
        lam = self.alpha * self.alpha * (self.n + self.kappa) - self.n
        self.gamma = math.sqrt(self.n + lam)
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (self.n + lam)
        self.Wc[0] = self.Wm[0] + (1.0 - self.alpha * self.alpha + self.beta)
        self.innovations: list[float] = []

    def sigma_points(self) -> np.ndarray:
        chol = np.linalg.cholesky(self.P + np.eye(self.n) * 1e-8)
        points = [self.x]
        for idx in range(self.n):
            points.append(self.x + self.gamma * chol[:, idx])
            points.append(self.x - self.gamma * chol[:, idx])
        return np.asarray(points)

    def process(self, state: np.ndarray, dt: float) -> np.ndarray:
        plasma, interstitial, rate = state
        plasma_next = plasma + rate * dt
        inter_next = interstitial + ((plasma - interstitial) / self.tau_lag) * dt
        rate_next = rate * math.exp(-dt / self.tau_rate)
        return np.array(
            [
                np.clip(plasma_next, 20.0, 600.0),
                np.clip(inter_next, 20.0, 600.0),
                np.clip(rate_next, -5.0, 5.0),
            ]
        )

    def update(self, measurement: float, dt: float = 3.0) -> tuple[float, float, float]:
        sigma = self.sigma_points()
        sigma_pred = np.asarray([self.process(point, dt) for point in sigma])
        x_pred = np.sum(self.Wm[:, None] * sigma_pred, axis=0)
        P_pred = self.Q * dt
        for idx, point in enumerate(sigma_pred):
            diff = point - x_pred
            P_pred += self.Wc[idx] * np.outer(diff, diff)

        z_points = sigma_pred[:, 1]
        z_pred = float(np.sum(self.Wm * z_points))
        S = self.R + float(np.sum(self.Wc * (z_points - z_pred) ** 2))
        T = np.zeros(self.n)
        for idx, point in enumerate(sigma_pred):
            T += self.Wc[idx] * (point - x_pred) * (z_points[idx] - z_pred)

        K = T / S
        innovation = float(measurement - z_pred)
        self.x = x_pred + K * innovation
        self.x[0] = np.clip(self.x[0], 20.0, 600.0)
        self.x[1] = np.clip(self.x[1], 20.0, 600.0)
        self.x[2] = np.clip(self.x[2], -5.0, 5.0)
        H = np.array([0.0, 1.0, 0.0])
        ikh = np.eye(self.n) - np.outer(K, H)
        self.P = ikh @ P_pred @ ikh.T + np.outer(K, K) * self.R
        self.P = (self.P + self.P.T) / 2.0 + np.eye(self.n) * 1e-8
        self.innovations.append(innovation)
        return float(self.x[0]), float(self.x[1]), innovation


def ljung_box_pvalue(values: list[float], lags: int = 10) -> float | None:
    if len(values) <= lags + 1:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr - arr.mean()
    denom = float(np.sum(arr * arr))
    if denom <= 0:
        return None
    n = len(arr)
    q = 0.0
    for lag in range(1, lags + 1):
        autocorr = float(np.sum(arr[lag:] * arr[:-lag]) / denom)
        q += autocorr * autocorr / max(n - lag, 1)
    q *= n * (n + 2)
    try:
        from scipy.stats import chi2

        return float(chi2.sf(q, lags))
    except Exception:
        k = lags
        z = ((q / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        return 0.5 * math.erfc(z / math.sqrt(2.0))


def evaluate_ukf(patient_runs: list[PatientRun]) -> dict[str, Any]:
    raw_errors: list[float] = []
    ukf_errors: list[float] = []
    innovations: list[float] = []

    for run in patient_runs:
        if len(run.trace) < 30:
            continue
        ukf = SimpleUKF(run.trace[0]["raw_cgm"])
        warmup = 12
        for idx, reading in enumerate(run.trace):
            plasma_est, _, innovation = ukf.update(reading["raw_cgm"], dt=3.0)
            if idx >= warmup:
                true_bg = reading["true_bg"]
                raw_errors.append(reading["raw_cgm"] - true_bg)
                ukf_errors.append(plasma_est - true_bg)
                innovations.append(innovation)

    if not raw_errors:
        return {}
    raw = np.asarray(raw_errors, dtype=float)
    ukf = np.asarray(ukf_errors, dtype=float)
    return {
        "raw_cgm_rmse": float(np.sqrt(np.mean(raw**2))),
        "ukf_plasma_rmse": float(np.sqrt(np.mean(ukf**2))),
        "raw_cgm_mae": float(np.mean(np.abs(raw))),
        "ukf_plasma_mae": float(np.mean(np.abs(ukf))),
        "rmse_improvement_pct": pct_improvement(float(np.sqrt(np.mean(raw**2))), float(np.sqrt(np.mean(ukf**2)))),
        "innovation_mean": float(np.mean(innovations)),
        "innovation_std": float(np.std(innovations)),
        "innovation_whiteness_pvalue": ljung_box_pvalue(innovations),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: round_float(row.get(field), 6) for field in fields})


def write_report(path: Path, results: dict[str, Any]) -> None:
    aggregate = results["isf_prediction_accuracy"]["padova_correction_challenge"]
    cohort = results.get("protocol", {}).get("cohort", "Padova")
    fixed = aggregate["net_drop_target"]["population"]
    clinical = aggregate["net_drop_target"]["clinical"]
    empirical = aggregate["net_drop_target"]["empirical"]
    fixed_ci = fixed["paired_error_reduction"]
    supported = bool(
        fixed["improvement_pct"] is not None
        and fixed["improvement_pct"] > 0
        and fixed_ci["ci_low"] is not None
        and fixed_ci["ci_low"] > 0
    )

    lines = [
        "# Real Padova Parameter-Learning Validation",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "## Verdict",
        "",
    ]
    if supported:
        lines.append(
            "Supported for this simulator protocol: the Bayesian posterior beats the fixed 50 mg/dL/U population baseline on chronological holdout correction events."
        )
    else:
        lines.append(
            "Not supported yet: the Bayesian posterior did not clear the pre-set evidence bar against the fixed 50 mg/dL/U baseline."
        )
    lines += [
        "",
        "This is Padova correction-challenge evidence, not personal free-living evidence. Personal Supabase validation is still blocked because the configured Supabase project endpoint did not resolve from this environment and the CLI account was unauthorized.",
        "",
        "## Main Net-Drop Holdout Results",
        "",
        f"- Patients evaluated: {aggregate['n_patients_evaluated']} / {aggregate['n_patients_total']}",
        f"- Total usable clean correction events: {aggregate['n_events_total']}",
        f"- Holdout events: {aggregate['n_holdout_events']}",
        f"- Fixed 50 MAE: {round_float(fixed['baseline_mae'], 3)} mg/dL",
        f"- Bayesian MAE: {round_float(fixed['bayesian_mae'], 3)} mg/dL",
        f"- Improvement vs fixed 50: {round_float(fixed['improvement_pct'], 2)}%",
        f"- Paired mean error reduction vs fixed 50: {round_float(fixed_ci['mean'], 3)} mg/dL "
        f"[95% CI {round_float(fixed_ci['ci_low'], 3)}, {round_float(fixed_ci['ci_high'], 3)}], "
        f"bootstrap p(reduction > 0)={round_float(fixed_ci['p_gt_0'], 3)}",
        "",
        "## Stronger Baselines",
        "",
        f"- Clinical-prior MAE: {round_float(clinical['baseline_mae'], 3)} mg/dL; improvement vs clinical: {round_float(clinical['improvement_pct'], 2)}%",
        f"- Empirical-mean MAE: {round_float(empirical['baseline_mae'], 3)} mg/dL; improvement vs empirical: {round_float(empirical['improvement_pct'], 2)}%",
        "",
        "## UKF Signal Quality On Fresh Padova Trace",
        "",
    ]
    ukf = results.get("ukf_signal_quality", {}).get("padova_correction_challenge", {})
    if ukf:
        lines += [
            f"- Raw CGM RMSE vs true simulator glucose: {round_float(ukf['raw_cgm_rmse'], 3)} mg/dL",
            f"- UKF plasma RMSE vs true simulator glucose: {round_float(ukf['ukf_plasma_rmse'], 3)} mg/dL",
            f"- UKF RMSE improvement: {round_float(ukf['rmse_improvement_pct'], 2)}%",
            f"- Innovation whiteness p-value: {round_float(ukf['innovation_whiteness_pvalue'], 4)}",
        ]
    else:
        lines.append("- UKF metrics were unavailable.")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Safe claim:",
        "",
        f"> In a fresh UVA/Padova {cohort} correction-challenge validation with chronological holdout splits, Gluca's Bayesian ISF posterior reduced future correction-response prediction error versus a fixed 50 mg/dL/U population baseline.",
        "",
        "Do not claim:",
        "",
        "> This proves the same improvement on personal CGM data.",
        "",
        "Personal-data validation still needs a local export or working authenticated Supabase/database access.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["adults", "adolescents", "children", "all"], default="adults")
    parser.add_argument("--max-patients", type=int, default=5)
    parser.add_argument("--events-per-patient", type=int, default=8)
    parser.add_argument("--max-days", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--threshold-mgdl", type=float, default=215.0)
    parser.add_argument("--stabilization-minutes", type=int, default=120)
    parser.add_argument("--response-minutes", type=int, default=180)
    parser.add_argument("--max-abs-counterfactual-drift", type=float, default=40.0)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patient_names = get_patient_names(args.cohort, args.max_patients)
    patient_runs: list[PatientRun] = []
    for index, patient_name in enumerate(patient_names, start=1):
        print(f"[{index}/{len(patient_names)}] Simulating {patient_name} ...", flush=True)
        run = simulate_patient(
            patient_name,
            seed=args.seed + index,
            target_events=args.events_per_patient,
            max_days=args.max_days,
            threshold_mgdl=args.threshold_mgdl,
            stabilization_minutes=args.stabilization_minutes,
            response_minutes=args.response_minutes,
            max_abs_counterfactual_drift=args.max_abs_counterfactual_drift,
        )
        print(
            f"  usable_events={len(run.events)} candidates={run.excluded_counts['candidate_events']}",
            flush=True,
        )
        patient_runs.append(run)

    event_rows = [asdict(event) for run in patient_runs for event in run.events]
    holdout_rows, patient_rows, aggregate = evaluate_predictions(patient_runs)
    ukf_metrics = evaluate_ukf(patient_runs)

    personal_blocker = {
        "status": "blocked",
        "reason": (
            "Supabase project dmiqajxvmwheqvznhfiw did not resolve via DNS from this environment; "
            "supabase projects list also returned Unauthorized. Need local export or authenticated DB access."
        ),
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "kind": "fresh_simglucose_uva_padova_correction_challenge",
            "cohort": args.cohort,
            "max_patients": args.max_patients,
            "events_per_patient_target": args.events_per_patient,
            "max_days": args.max_days,
            "seed": args.seed,
            "threshold_mgdl": args.threshold_mgdl,
            "stabilization_minutes": args.stabilization_minutes,
            "response_minutes": args.response_minutes,
            "max_abs_counterfactual_drift": args.max_abs_counterfactual_drift,
            "clean_gate": {
                "no_meals": True,
                "no_exercise": True,
                "no_other_boluses_in_response_window": True,
                "dose_min_units": 0.5,
                "response_minutes": args.response_minutes,
                "observed_net_isf_range": [5.0, 200.0],
                "chronological_split": "first 70% train, last 30% holdout per patient",
            },
        },
        "isf_prediction_accuracy": {
            "padova_correction_challenge": aggregate,
            "personal": personal_blocker,
        },
        "ukf_signal_quality": {
            "padova_correction_challenge": ukf_metrics,
            "personal": personal_blocker,
        },
        "patient_exclusion_counts": {
            run.patient: run.excluded_counts for run in patient_runs
        },
        "artifacts": {
            "events_csv": "padova_clean_correction_events.csv",
            "holdout_predictions_csv": "padova_holdout_predictions.csv",
            "patient_summary_csv": "padova_patient_summary.csv",
            "report_md": "real_validation_report.md",
        },
    }

    write_csv(args.out_dir / "padova_clean_correction_events.csv", event_rows)
    write_csv(args.out_dir / "padova_holdout_predictions.csv", holdout_rows)
    write_csv(args.out_dir / "padova_patient_summary.csv", patient_rows)

    results_path = args.out_dir / "real_validation_results.json"
    with results_path.open("w") as f:
        json.dump(rounded_json(results), f, indent=2)
        f.write("\n")
    write_report(args.out_dir / "real_validation_report.md", rounded_json(results))

    print(json.dumps(rounded_json(results), indent=2), flush=True)
    print(f"\nWrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
