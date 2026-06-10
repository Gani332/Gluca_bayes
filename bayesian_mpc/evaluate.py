"""
Validate the Bayesian dosing engine on simglucose.

Runs the engine in a closed-loop simulation:
  1. CGM readings feed into the engine
  2. Engine recommends bolus at mealtimes, adjusts basal
  3. Measure TIR, TBR, TAR against Battelino consensus

Compares against:
  - Fixed basal-bolus (doctor-prescribed, no adaptation)
  - Oracle (perfect knowledge of patient params)
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

from bayesian_mpc.dosing_engine import DosingEngine
from bayesian_mpc.bayesian_v2 import HealthContext
from bayesian_mpc.clinical_priors import derive_clinical_priors, infer_cohort_from_patient_name


STEPS_PER_DAY = 480  # 3-min steps


def compute_metrics(bg_trace: list) -> Dict:
    """Compute Battelino consensus metrics."""
    bg = np.array(bg_trace)
    n = len(bg)
    if n == 0:
        return {}

    return {
        "mean_bg": float(bg.mean()),
        "std_bg": float(bg.std()),
        "cv": float(bg.std() / bg.mean() * 100) if bg.mean() > 0 else 0,
        "tir": float(np.mean((bg >= 70) & (bg <= 180)) * 100),
        "tbr": float(np.mean(bg < 70) * 100),
        "tsbr": float(np.mean(bg < 54) * 100),
        "tar": float(np.mean(bg > 180) * 100),
        "tsar": float(np.mean(bg > 250) * 100),
        "min_bg": float(bg.min()),
        "max_bg": float(bg.max()),
    }


def build_health_context(step: int, health_mode: str) -> Optional[HealthContext]:
    """
    Create optional Apple Health context for estimator v2.

    Notes:
      - `none`: no health data at all
      - `neutral`: health data available, but all effects centered at zero
      - `synthetic`: deterministic pseudo-health data to exercise the plumbing

    simglucose does not model exercise/sleep/cycle effects, so `synthetic`
    is a software integration test, not a proof of clinical improvement.
    """
    if health_mode == "none":
        return None
    if health_mode == "neutral":
        return HealthContext()

    minute_of_day = (step * 3) % (24 * 60)
    day = (step * 3) // (24 * 60)

    workout_min_today = 45.0 if 7 * 60 <= minute_of_day < 8 * 60 else 0.0
    steps_2h = 5500 if 7 * 60 <= minute_of_day < 10 * 60 else 2200
    steps_daily = min(12000, 4000 + day * 500 + int(steps_2h * 1.2))
    sleep_hours = 6.0 if day % 3 == 0 else 8.0
    hrv_ms = 42.0 if day % 2 == 0 else 58.0
    cycle_phase = "luteal" if day % 28 in range(18, 25) else None

    return HealthContext.from_apple_health(
        steps_2h=steps_2h,
        steps_daily=steps_daily,
        workout_min_today=workout_min_today,
        sleep_hours=sleep_hours,
        hrv_ms=hrv_ms,
        cycle_phase=cycle_phase,
    )


def run_simulation(
    patient_name: str = "adult#001",
    days: int = 10,
    seed: int = 42,
    mode: str = "bayesian",
    initial_isf: Optional[float] = None,
    initial_cr: Optional[float] = None,
    initial_basal: Optional[float] = None,
    initial_bg_target: Optional[float] = None,
    estimator_version: str = "v1",
    use_covariates: bool = False,
    health_mode: str = "none",
    verbose: bool = True,
) -> Dict:
    """
    Run closed-loop simulation.

    Args:
        patient_name: UVA/Padova patient
        days: Simulation length
        seed: Random seed
        mode: "bayesian" (adaptive) or "fixed" (static params)
        initial_isf/cr/basal: Starting parameter guesses
        estimator_version: "v1" (Gaussian) or "v2" (log-space)
        use_covariates: Whether to enable optional Apple Health covariates in v2
        health_mode: none | neutral | synthetic
        verbose: Print progress
    """
    # Create simglucose environment
    patient = T1DPatient.withName(patient_name)
    cohort = infer_cohort_from_patient_name(patient_name)
    body_weight = float(patient._params.BW)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = RandomScenario(start_time=datetime(2024, 1, 1, 0, 0, 0), seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    priors = derive_clinical_priors(
        cohort,
        body_weight,
        isf=initial_isf,
        cr=initial_cr,
        basal_rate=initial_basal,
        bg_target=initial_bg_target,
    )

    # Create dosing engine
    engine = DosingEngine(
        isf=priors.isf,
        cr=priors.cr,
        basal_rate=priors.basal_rate,
        bg_target=priors.bg_target,
        body_weight=body_weight,
        cohort=cohort,
        estimator_version=estimator_version,
        use_covariates=use_covariates,
    )

    # Reset
    step_result = env.reset()
    obs = step_result.observation
    bg = obs.CGM

    bg_trace = [bg]
    insulin_trace = []
    total_steps = STEPS_PER_DAY * days
    time_min = 0.0
    last_meal_info = {}

    for step in range(total_steps):
        time_min = step * 3.0  # 3-min steps

        # Feed CGM reading to engine
        engine.record_glucose(bg, time_min)

        if estimator_version == "v2" and use_covariates:
            engine.set_health_context(build_health_context(step, health_mode))

        # Get meal info from simulator (available after previous step)
        meal_carbs = 0.0
        if isinstance(last_meal_info, dict):
            meal_carbs = last_meal_info.get("meal", 0.0)
        elif hasattr(last_meal_info, 'get'):
            meal_carbs = last_meal_info.get("meal", 0.0)

        # ── Compute insulin action ───────────────────────────────────
        if mode == "fixed":
            # Fixed mode: use clinically derived starting priors, no adaptation
            basal_per_step = priors.basal_rate * (3.0 / 60.0)  # U/hr → U per 3 min
            bolus = 0.0
            if meal_carbs > 0:
                iob = engine.model.get_iob(time_min)
                correction = 0.0
                if bg > priors.bg_target + priors.safety.correction_margin:
                    correction = (bg - priors.bg_target) / priors.isf
                    correction *= priors.safety.correction_scale_floor
                    correction = min(correction, priors.safety.max_correction_units)

                bolus = max(0.0, meal_carbs / priors.cr
                            + correction
                            - iob)
                if bg < priors.safety.low_reduce_bg:
                    bolus = min(bolus, meal_carbs / priors.cr)
                if bg < priors.safety.low_suspend_bg:
                    bolus = 0.0
                bolus = min(bolus, priors.safety.max_bolus_units)
                engine.model.record_meal(time_min, meal_carbs)
                engine.model.record_insulin(time_min, bolus)
        else:
            # Bayesian adaptive mode
            basal_rec = engine.recommend_basal(bg)
            basal_per_step = basal_rec["basal_rate"] * (3.0 / 60.0)
            bolus = 0.0
            if meal_carbs > 0:
                engine.record_meal(meal_carbs, time_min)
                bolus_rec = engine.recommend_bolus(bg, meal_carbs)
                bolus = bolus_rec["recommended_dose"]

        total_insulin = basal_per_step + bolus

        # Record insulin for IOB tracking
        if basal_per_step > 0:
            engine.record_insulin(basal_per_step, "basal", time_min)
        if bolus > 0:
            engine.record_insulin(bolus, "bolus", time_min)

        # ── Step environment ─────────────────────────────────────────
        action = Action(basal=total_insulin, bolus=0.0)
        try:
            obs, _, done, last_meal_info = env.step(action)
        except Exception:
            break

        bg = obs.CGM
        bg_trace.append(bg)
        insulin_trace.append(total_insulin)

        engine.tick(3.0)
        engine.model.cleanup(time_min)
        engine.estimator.cleanup(time_min)

        if done or bg < 10 or bg > 600:
            break

        # Periodic logging
        if verbose and step > 0 and step % (STEPS_PER_DAY) == 0:
            day = step // STEPS_PER_DAY
            day_bg = bg_trace[-STEPS_PER_DAY:]
            day_metrics = compute_metrics(day_bg)
            params = engine.estimator.get_params()
            extra = ""
            if "isf_context_multiplier" in params:
                extra = f", ctx×={params['isf_context_multiplier']:.2f}"
            print(f"  Day {day}: TIR={day_metrics['tir']:.1f}%, "
                  f"mean={day_metrics['mean_bg']:.0f}, "
                  f"ISF={params['isf']:.1f}(conf={params['isf_confidence']:.0%}), "
                  f"CR={params['cr']:.1f}(conf={params['cr_confidence']:.0%})"
                  f"{extra}")

    metrics = compute_metrics(bg_trace)

    if verbose:
        label = f"{mode}/{estimator_version}"
        if estimator_version == "v2" and use_covariates:
            label += f"/health={health_mode}"
        print(f"\n  {patient_name} ({label}, {days} days):")
        print(f"  TIR: {metrics['tir']:.1f}%  TBR: {metrics['tbr']:.1f}%  "
              f"TAR: {metrics['tar']:.1f}%  Mean: {metrics['mean_bg']:.0f}")
        print(f"  Final params: {engine.estimator.summary()}")

    return {
        "patient": patient_name,
        "mode": mode,
        "days": days,
        "metrics": metrics,
        "bg_trace": bg_trace,
        "estimator_version": estimator_version,
        "use_covariates": use_covariates,
        "health_mode": health_mode,
        "engine_status": engine.get_status(),
    }


def compare_approaches(
    patient_name: str = "adult#001",
    days: int = 10,
    seed: int = 42,
):
    """Compare Bayesian adaptive vs fixed-param dosing."""

    print(f"\n{'='*60}")
    print(f"  COMPARISON: {patient_name}, {days} days")
    print(f"{'='*60}")

    # Get patient's true params (for oracle comparison)
    patient = T1DPatient.withName(patient_name)
    u2ss = patient._params.u2ss
    bw = patient._params.BW
    true_basal = u2ss * bw / 6000 * 60  # U/hr

    # Bayesian adaptive from clinically derived priors
    print("\n[1] Bayesian Adaptive (clinical priors):")
    bayesian = run_simulation(
        patient_name, days, seed, mode="bayesian",
    )

    # Fixed (same clinical priors, no adaptation)
    print("\n[2] Fixed Clinical Priors:")
    fixed = run_simulation(
        patient_name, days, seed, mode="fixed",
    )

    # Oracle (uses approximately correct params)
    print("\n[3] Oracle (near-correct params):")
    oracle = run_simulation(
        patient_name, days, seed, mode="bayesian",
        initial_isf=35.0, initial_cr=10.0, initial_basal=true_basal,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'TIR':>8} {'TBR':>8} {'TAR':>8} {'Mean BG':>8}")
    print(f"{'-'*60}")
    for name, result in [("Bayesian Adaptive", bayesian),
                          ("Fixed Clinical", fixed),
                          ("Oracle", oracle)]:
        m = result["metrics"]
        print(f"{name:<25} {m['tir']:>7.1f}% {m['tbr']:>7.1f}% "
              f"{m['tar']:>7.1f}% {m['mean_bg']:>7.0f}")

    return {"bayesian": bayesian, "fixed": fixed, "oracle": oracle}


def compare_estimators(
    patient_name: str = "adult#001",
    days: int = 10,
    seed: int = 42,
    health_mode: str = "synthetic",
):
    """
    Compare v1 Gaussian estimator vs v2 log-space estimator.

    `health_mode` only tests the software path for optional Apple Health
    covariates. simglucose itself does not encode exercise/sleep/cycle effects,
    so any gain here should be interpreted cautiously.
    """
    print(f"\n{'='*68}")
    print(f"  ESTIMATOR COMPARISON: {patient_name}, {days} days")
    print(f"{'='*68}")
    if health_mode != "none":
        print(f"  Note: health_mode={health_mode} exercises optional covariate plumbing.")
        print("  simglucose does not model Apple Health covariates, so this is a non-regression check.")

    v1 = run_simulation(
        patient_name, days, seed,
        mode="bayesian",
        estimator_version="v1",
        use_covariates=False,
    )

    v2_no_health = run_simulation(
        patient_name, days, seed,
        mode="bayesian",
        estimator_version="v2",
        use_covariates=False,
    )

    v2_with_health = run_simulation(
        patient_name, days, seed,
        mode="bayesian",
        estimator_version="v2",
        use_covariates=True,
        health_mode=health_mode,
    )

    fixed = run_simulation(
        patient_name, days, seed,
        mode="fixed",
        estimator_version="v1",
        use_covariates=False,
    )

    print(f"\n{'='*68}")
    print(f"{'Method':<32} {'TIR':>8} {'TBR':>8} {'TAR':>8} {'Mean BG':>10}")
    print(f"{'-'*68}")
    rows = [
        ("v1 Gaussian", v1),
        ("v2 Log-space", v2_no_health),
        (f"v2 Log-space + {health_mode}", v2_with_health),
        ("Fixed Clinical", fixed),
    ]
    for name, result in rows:
        m = result["metrics"]
        print(f"{name:<32} {m['tir']:>7.1f}% {m['tbr']:>7.1f}% "
              f"{m['tar']:>7.1f}% {m['mean_bg']:>9.0f}")

    return {
        "v1": v1,
        "v2_no_health": v2_no_health,
        "v2_with_health": v2_with_health,
        "fixed": fixed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#001")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparison",
        choices=["approaches", "estimators"],
        default="approaches",
    )
    parser.add_argument(
        "--health-mode",
        choices=["none", "neutral", "synthetic"],
        default="synthetic",
    )
    args = parser.parse_args()

    if args.comparison == "estimators":
        compare_estimators(args.patient, args.days, args.seed, args.health_mode)
    else:
        compare_approaches(args.patient, args.days, args.seed)
