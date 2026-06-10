"""
Compare factor graph vs independent Bayesian estimator vs fixed params.

Uses identical simulation setup as evaluate.py for fair comparison.
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

from bayesian_mpc.patient_model import GlucoseModel
from bayesian_mpc.factor_graph import PatientFactorGraph


STEPS_PER_DAY = 480


def compute_metrics(bg_trace: list) -> Dict:
    bg = np.array(bg_trace)
    if len(bg) == 0:
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


def run_fg_simulation(
    patient_name: str = "adult#001",
    days: int = 10,
    seed: int = 42,
    initial_isf: float = 50.0,
    initial_cr: float = 10.0,
    initial_basal: float = 1.0,
    bg_target: float = 110.0,
    verbose: bool = True,
) -> Dict:
    """Run simulation with factor graph estimator."""

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = RandomScenario(start_time=datetime(2024, 1, 1, 0, 0, 0), seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Factor graph estimator
    fg = PatientFactorGraph(
        isf_prior=initial_isf, cr_prior=initial_cr,
        basal_prior=initial_basal, bg_target=bg_target,
    )

    # Physiological model for IOB/COB tracking
    model = GlucoseModel(
        isf=initial_isf, cr=initial_cr, basal_rate=initial_basal,
        bg_target=bg_target,
    )

    step_result = env.reset()
    obs = step_result.observation
    bg = obs.CGM

    bg_trace = [bg]
    total_steps = STEPS_PER_DAY * days
    last_meal_info = {}

    for step in range(total_steps):
        time_min = step * 3.0
        hour_of_day = (time_min / 60.0) % 24.0

        fg.record_bg(time_min, bg)

        # Get meal from previous step
        meal_carbs = 0.0
        if isinstance(last_meal_info, dict):
            meal_carbs = last_meal_info.get("meal", 0.0)

        # ── Get current params from factor graph ─────────────────────
        fg.try_update(time_min)
        params = fg.get_params(hour_of_day)

        isf = params["isf_circadian"]  # Time-of-day adjusted!
        cr = params["cr"]
        basal = params["basal"]

        # Update model with latest estimates
        model.update_params(isf=isf, cr=cr, basal_rate=basal)

        # ── Basal ────────────────────────────────────────────────────
        basal_per_step = basal * (3.0 / 60.0)

        # Low glucose suspend
        if bg < 70:
            basal_per_step = 0.0
        elif bg < 90:
            basal_per_step *= 0.5

        # High glucose increase
        if bg > 180:
            trajectory = model.predict(bg, time_min, horizon_min=60.0, dt=5.0)
            predicted_1h = trajectory[-1][1] if trajectory else bg
            if predicted_1h > 180:
                basal_per_step = min(basal_per_step * 1.5, basal * 3.0 * (3.0 / 60.0))

        # ── Bolus ────────────────────────────────────────────────────
        bolus = 0.0
        if meal_carbs > 0:
            iob = model.get_iob(time_min)
            meal_bolus = meal_carbs / cr
            correction = max(0.0, (bg - bg_target) / isf) if bg > bg_target + 20 else 0.0
            bolus = max(0.0, meal_bolus + correction - iob)
            bolus = min(bolus, 15.0)

            # Safety: check prediction
            trajectory = model.predict(
                bg, time_min, horizon_min=180.0, dt=5.0,
                future_insulin=[(0.0, bolus)],
            )
            pred_min = min(v for _, v in trajectory)
            if pred_min < 70 and bolus > 0:
                headroom = bg - 70
                if headroom > 0 and bg > pred_min:
                    bolus *= min(1.0, headroom / (bg - pred_min))
                else:
                    bolus = 0.0

            model.record_meal(time_min, meal_carbs)
            model.record_insulin(time_min, bolus)
            fg.record_meal(time_min, meal_carbs)
            fg.record_insulin(time_min, bolus, "bolus")

        total_insulin = basal_per_step + bolus

        # Record basal
        if basal_per_step > 0:
            fg.record_insulin(time_min, basal_per_step, "basal")
            model.record_insulin(time_min, basal_per_step)

        # ── Step ─────────────────────────────────────────────────────
        action = Action(basal=total_insulin, bolus=0.0)
        try:
            obs, _, done, last_meal_info = env.step(action)
        except Exception:
            break

        bg = obs.CGM
        bg_trace.append(bg)

        model.cleanup(time_min)
        fg.cleanup(time_min)

        if done or bg < 10 or bg > 600:
            break

        if verbose and step > 0 and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            day_bg = bg_trace[-STEPS_PER_DAY:]
            day_m = compute_metrics(day_bg)
            p = fg.get_params(hour_of_day)
            print(f"  Day {day}: TIR={day_m['tir']:.1f}%, mean={day_m['mean_bg']:.0f}, "
                  f"ISF={p['isf']:.1f}(conf={p['isf_confidence']:.0%}), "
                  f"CR={p['cr']:.1f}(conf={p['cr_confidence']:.0%}), "
                  f"corr={p['isf_cr_correlation']:.2f}")

    metrics = compute_metrics(bg_trace)
    final_params = fg.get_params()

    if verbose:
        print(f"\n  {patient_name} (factor graph, {days} days):")
        print(f"  TIR: {metrics['tir']:.1f}%  TBR: {metrics['tbr']:.1f}%  "
              f"TAR: {metrics['tar']:.1f}%  Mean: {metrics['mean_bg']:.0f}")
        print(f"  Final: {fg.summary()}")

    return {"metrics": metrics, "params": final_params, "bg_trace": bg_trace}


def compare_all(
    patient_name: str = "adult#001",
    days: int = 10,
    seed: int = 42,
):
    """Compare factor graph vs independent Bayes vs fixed."""

    print(f"\n{'='*65}")
    print(f"  COMPARISON: {patient_name}, {days} days")
    print(f"{'='*65}")

    # Get patient true basal for oracle
    patient = T1DPatient.withName(patient_name)
    true_basal = patient._params.u2ss * patient._params.BW / 6000 * 60

    # 1. Factor graph (wrong starting params)
    print("\n[1] Factor Graph (population defaults):")
    fg_result = run_fg_simulation(
        patient_name, days, seed,
        initial_isf=50.0, initial_cr=10.0, initial_basal=1.0,
    )

    # 2. Factor graph with better starting params (oracle-ish)
    print("\n[2] Factor Graph (near-correct params):")
    fg_oracle = run_fg_simulation(
        patient_name, days, seed,
        initial_isf=35.0, initial_cr=10.0, initial_basal=true_basal,
    )

    # 3. Independent Bayesian (from evaluate.py for reference)
    from bayesian_mpc.evaluate import run_simulation
    print("\n[3] Independent Bayesian (population defaults):")
    indep_result = run_simulation(
        patient_name, days, seed, mode="bayesian",
        initial_isf=50.0, initial_cr=10.0, initial_basal=1.0,
    )

    # 4. Fixed params
    print("\n[4] Fixed (population defaults):")
    fixed_result = run_simulation(
        patient_name, days, seed, mode="fixed",
        initial_isf=50.0, initial_cr=10.0, initial_basal=1.0,
    )

    print(f"\n{'='*65}")
    print(f"{'Method':<30} {'TIR':>8} {'TBR':>8} {'TAR':>8} {'Mean':>8}")
    print(f"{'-'*65}")
    for name, r in [
        ("Factor Graph (pop defaults)", fg_result),
        ("Factor Graph (oracle start)", fg_oracle),
        ("Independent Bayes (pop)", indep_result),
        ("Fixed (pop defaults)", fixed_result),
    ]:
        m = r["metrics"]
        print(f"{name:<30} {m['tir']:>7.1f}% {m['tbr']:>7.1f}% "
              f"{m['tar']:>7.1f}% {m['mean_bg']:>7.0f}")

    return {
        "factor_graph": fg_result,
        "factor_graph_oracle": fg_oracle,
        "independent_bayes": indep_result,
        "fixed": fixed_result,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#001")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compare_all(args.patient, args.days, args.seed)
