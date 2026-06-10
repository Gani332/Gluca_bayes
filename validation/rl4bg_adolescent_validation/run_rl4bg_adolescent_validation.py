#!/usr/bin/env python3
"""
Adolescent-only validation in the Fox et al. RL4BG environment.

This intentionally does not use stock simglucose. It loads the public
MLD3/RL4BG code release and evaluates:
- Fox-style basal-bolus (ManualBBController)
- Fox-style PID
- Fox-style PID-MA (PID action plus ManualBB meal bolus)
- Gluca fixed-profile formula sanity controller
- Gluca tuned feedback controller (profiled meal/correction plus scaled PID-MA feedback)
- Gluca Bayesian controllers emitting the same single U/min action

Metrics are computed after dropping the first simulated day, matching the
released RL4BG baseline scripts (`show_history()[288:]`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RL4BG_ROOT = Path("/tmp/RL4BG")
RL4BG_URL = "https://github.com/MLD3/RL4BG.git"

ADOLESCENTS = [f"adolescent#{i:03d}" for i in range(1, 11)]

PID_GAINS = {
    "adolescent#001": (-1.74e-04, -1.00e-07, -1.00e-02),
    "adolescent#002": (-1.00e-04, -1.00e-07, -6.31e-03),
    "adolescent#003": (-1.00e-04, -1.00e-07, -3.98e-03),
    "adolescent#004": (-1.00e-04, -1.00e-07, -4.79e-03),
    "adolescent#005": (-6.31e-05, -1.00e-07, -6.31e-03),
    "adolescent#006": (-4.54e-10, -1.58e-11, -1.00e-02),
    "adolescent#007": (-1.07e-07, -6.07e-08, -6.31e-03),
    "adolescent#008": (-4.54e-10, -4.54e-12, -1.00e-02),
    "adolescent#009": (-6.31e-05, -1.00e-07, -3.98e-03),
    "adolescent#010": (-4.54e-10, -4.54e-12, -1.00e-02),
}

PID_MA_GAINS = {
    "adolescent#001": (-1.00e-04, -4.72e-08, -6.31e-03),
    "adolescent#002": (-1.00e-05, -1.00e-07, -3.49e-03),
    "adolescent#003": (-6.31e-05, -1.00e-07, -2.09e-03),
    "adolescent#004": (-6.31e-05, -1.00e-07, -2.51e-03),
    "adolescent#005": (-4.79e-05, -1.00e-07, -3.98e-03),
    "adolescent#006": (-1.00e-04, -1.00e-07, -2.75e-03),
    "adolescent#007": (-1.00e-05, -1.00e-07, -3.02e-03),
    "adolescent#008": (-1.58e-09, -1.00e-07, -2.75e-03),
    "adolescent#009": (-3.98e-05, -1.00e-07, -1.91e-03),
    "adolescent#010": (-1.00e-04, -1.00e-07, -4.37e-03),
}

DEFAULT_CONTROLLERS = [
    "fox_bb",
    "fox_pid",
    "fox_pid_ma",
    "gluca_feedback_profiled",
    "gluca_fixed_profile_formula",
    "gluca_bayesian_profiled",
    "gluca_bayesian_clinical",
]

ENGINE_CONTROLLERS = {
    "gluca_bayesian_profiled",
    "gluca_bayesian_clinical",
}

GLUCA_FEEDBACK_SCALES = {
    "pid_p": 0.75,
    "pid_i": 0.50,
    "pid_d": 1.00,
    "meal": 1.00,
    "hypo": 0.75,
    "correction": 1.00,
}


def ensure_rl4bg(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "bgp" / "simglucose" / "envs" / "simglucose_gym_env.py").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", RL4BG_URL, str(path)], check=True)
    return path


def configure_paths(rl4bg_root: Path) -> None:
    os.environ.setdefault("GLUCA_ENABLE_RESEARCH_DOSING", "1")
    sys.path.insert(0, str(rl4bg_root))
    sys.path.insert(0, str(REPO_ROOT))


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


def summarize(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"mean": None, "median": None, "std": None, "q1": None, "q3": None}
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
    }


def pct_change(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference == 0:
        return None
    return 100.0 * (reference - candidate) / reference


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_env(patient: str, seed: int, rl4bg_root: Path, time_std: float | None):
    from bgp.rl.reward_functions import risk_diff
    from bgp.simglucose.envs.simglucose_gym_env import DeepSACT1DEnv

    return DeepSACT1DEnv(
        reward_fun=risk_diff,
        patient_name=patient,
        seeds={"numpy": seed, "sensor": seed, "scenario": seed},
        reset_lim={"lower_lim": 10, "upper_lim": 1000},
        time=False,
        meal=False,
        bw_meals=True,
        load=False,
        gt=False,
        n_hours=4,
        norm=False,
        time_std=time_std,
        action_cap=None,
        action_bias=0,
        action_scale=1,
        meal_announce=None,
        residual_basal=False,
        residual_bolus=False,
        residual_PID=False,
        fake_gt=False,
        fake_real=False,
        suppress_carbs=False,
        limited_gt=False,
        termination_penalty=None,
        use_pid_load=False,
        hist_init=True,
        harrison_benedict=True,
        meal_duration=5,
        carb_error_std=0,
        carb_miss_prob=0,
        source_dir=str(rl4bg_root),
    )


def build_manual_bb(env: Any, target: float):
    from bgp.simglucose.controller.basal_bolus_ctrller import ManualBBController

    return ManualBBController(
        target=target,
        cr=float(env.CR),
        cf=float(env.CF),
        basal=float(env.ideal_basal),
        sample_rate=int(env.sample_time),
        use_cf=True,
        use_bol=True,
        cooldown=180,
        corrected=True,
        use_low_lim=True,
        low_lim=target,
    )


def make_engine(controller: str, env: Any, target: float):
    from bayesian_mpc.dosing_engine import DosingEngine

    kwargs: dict[str, Any] = {
        "bg_target": target,
        "body_weight": float(env.bw),
        "cohort": "adolescent",
        "estimator_version": "v2",
        "use_covariates": False,
    }
    if controller == "gluca_bayesian_profiled":
        kwargs.update(
            {
                "isf": float(env.CF),
                "cr": float(env.CR),
                "basal_rate": float(env.ideal_basal) * 60.0,
            }
        )
    return DosingEngine(**kwargs)


def fixed_profile_formula(
    env: Any,
    current_bg: float,
    carbs_grams: float,
    target: float,
) -> dict[str, float]:
    """DosingEngine-style fixed-profile meal/correction formula."""
    cr = float(env.CR)
    isf = float(env.CF)
    meal_component = carbs_grams / cr if carbs_grams > 0 else 0.0
    correction = 0.0
    if current_bg > target + 25.0:
        correction = (current_bg - target) / isf
    dose = max(0.0, meal_component + correction)
    dose = min(dose, 12.0)
    if current_bg < 75.0:
        dose = 0.0
    elif current_bg < 95.0:
        dose = min(dose, meal_component)
    return {
        "recommended_dose": dose,
        "meal_component": meal_component,
        "correction_component": correction,
        "iob": 0.0,
        "total_iob": 0.0,
        "cr_used": cr,
        "isf_used": isf,
    }


def feedback_meal_bolus_umin(
    env: Any,
    meal_grams: float,
    glucose: float,
    target: float,
    sample_time: float,
    last_correction_min: float,
) -> tuple[float, float]:
    """ManualBB-style meal bolus with locked Gluca feedback tuning."""
    if meal_grams <= 0:
        return 0.0, last_correction_min + sample_time

    carb = (meal_grams / float(env.CR)) * GLUCA_FEEDBACK_SCALES["meal"]
    hyper = 0.0
    if glucose > target:
        hyper = ((glucose - target) / float(env.CF)) * GLUCA_FEEDBACK_SCALES["correction"]
    hypo = 0.0
    if glucose < target:
        hypo = ((target - glucose) / float(env.CF)) * GLUCA_FEEDBACK_SCALES["hypo"]

    next_last_correction = last_correction_min + sample_time
    if next_last_correction <= 180.0:
        hyper = 0.0
    elif hyper > 0.0:
        next_last_correction = 0.0

    bolus_units = max(0.0, carb + hyper - hypo)
    return bolus_units / sample_time, next_last_correction


def run_rollout(
    patient: str,
    seed: int,
    controller: str,
    days: int,
    target: float,
    rl4bg_root: Path,
    time_std: float | None,
    trace_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from bgp.rl.pid import PID

    env = build_env(patient, seed, rl4bg_root, time_std)
    env.reset()
    sample_time = float(env.sample_time)
    total_steps = int(days * env.day)
    pending_carbs = 0.0
    done_ever = False
    first_done_step: int | None = None
    trace_rows: list[dict[str, Any]] = []

    manual_bb = None
    pid = None
    engine = None
    feedback_last_correction_min = float("inf")

    if controller == "fox_bb":
        manual_bb = build_manual_bb(env, target)
        action_umin = float(manual_bb.manual_bb_policy(carbs=0.0, glucose=target).basal)
    elif controller == "fox_pid":
        kp, ki, kd = PID_GAINS[patient]
        pid = PID(target, kp, ki, kd, basal=float(env.ideal_basal))
        action_umin = float(pid.step(env.env.CGM_hist[-1]))
    elif controller == "fox_pid_ma":
        kp, ki, kd = PID_MA_GAINS[patient]
        pid = PID(target, kp, ki, kd, basal=float(env.ideal_basal))
        manual_bb = build_manual_bb(env, target)
        action_umin = float(pid.step(env.env.CGM_hist[-1]))
    elif controller == "gluca_feedback_profiled":
        kp, ki, kd = PID_MA_GAINS[patient]
        pid = PID(
            target,
            kp * GLUCA_FEEDBACK_SCALES["pid_p"],
            ki * GLUCA_FEEDBACK_SCALES["pid_i"],
            kd * GLUCA_FEEDBACK_SCALES["pid_d"],
            basal=float(env.ideal_basal),
        )
        action_umin = float(pid.step(env.env.CGM_hist[-1]))
    elif controller == "gluca_fixed_profile_formula":
        action_umin = float(env.ideal_basal)
    elif controller in ENGINE_CONTROLLERS:
        engine = make_engine(controller, env, target)
        action_umin = float(env.ideal_basal)
    else:
        raise ValueError(f"Unknown controller: {controller}")

    basal_units = 0.0
    bolus_units = 0.0
    total_meal_grams = 0.0
    meal_events = 0

    for step in range(total_steps):
        cgm_before = float(env.env.CGM_hist[-1])
        bg_before = float(env.env.BG_hist[-1])
        action_umin = max(0.0, float(action_umin))
        state, reward, done, info = env.step(action=action_umin)
        done_ever = done_ever or bool(done)
        if done and first_done_step is None:
            first_done_step = step

        meal_grams = float(info["meal"]) * sample_time
        if meal_grams > 0:
            meal_events += 1
            total_meal_grams += meal_grams

        basal_part = 0.0
        bolus_part = action_umin
        if controller == "fox_bb":
            # In the RL4BG wrapper BB is passed as one total insulin-rate action,
            # but ManualBB exposes the basal component.
            basal_part = float(env.ideal_basal)
            bolus_part = max(0.0, action_umin - basal_part)
        basal_units += basal_part * sample_time
        bolus_units += bolus_part * sample_time

        if trace_stride > 0 and step % trace_stride == 0:
            trace_rows.append(
                {
                    "patient": patient,
                    "seed": seed,
                    "controller": controller,
                    "step": step,
                    "time_min": step * sample_time,
                    "bg_before": bg_before,
                    "cgm_before": cgm_before,
                    "meal_grams": meal_grams,
                    "action_umin": action_umin,
                    "reward": float(reward),
                    "done": bool(done),
                }
            )

        cgm_now = float(env.env.CGM_hist[-1])
        time_min = (step + 1) * sample_time

        if controller == "fox_bb":
            assert manual_bb is not None
            action = manual_bb.manual_bb_policy(carbs=meal_grams, glucose=cgm_now)
            action_umin = float(action.basal + action.bolus)

        elif controller == "fox_pid":
            assert pid is not None
            action_umin = float(pid.step(cgm_now))

        elif controller == "fox_pid_ma":
            assert pid is not None and manual_bb is not None
            pid_umin = float(pid.step(cgm_now))
            meal_action = manual_bb.manual_bb_policy(carbs=meal_grams, glucose=cgm_now)
            action_umin = max(0.0, pid_umin + float(meal_action.bolus))

        elif controller == "gluca_feedback_profiled":
            assert pid is not None
            pid_umin = max(0.0, float(pid.step(cgm_now)))
            meal_umin, feedback_last_correction_min = feedback_meal_bolus_umin(
                env=env,
                meal_grams=meal_grams,
                glucose=cgm_now,
                target=target,
                sample_time=sample_time,
                last_correction_min=feedback_last_correction_min,
            )
            action_umin = max(0.0, pid_umin + meal_umin)

        elif controller == "gluca_fixed_profile_formula":
            basal_umin = float(env.ideal_basal)
            bolus_units_now = 0.0
            bolus_rec = None
            if pending_carbs > 0:
                bolus_rec = fixed_profile_formula(env, cgm_now, pending_carbs, target)
                bolus_units_now = max(0.0, float(bolus_rec["recommended_dose"]))
            action_umin = basal_umin + bolus_units_now / sample_time
            pending_carbs = meal_grams

        else:
            assert engine is not None
            engine.record_glucose(cgm_now, time_min)
            if pending_carbs > 0:
                engine.record_meal(pending_carbs, time_min)
            basal_rec = engine.recommend_basal(cgm_now)
            basal_umin = max(0.0, float(basal_rec["basal_rate"]) / 60.0)
            bolus_units_now = 0.0
            if pending_carbs > 0:
                bolus_rec = engine.recommend_bolus(cgm_now, pending_carbs)
                bolus_units_now = max(0.0, float(bolus_rec["recommended_dose"]))
            if basal_umin > 0:
                engine.record_insulin(basal_umin * sample_time, "basal", time_min)
            if bolus_units_now > 0:
                engine.record_insulin(bolus_units_now, "bolus", time_min)
            engine.tick(sample_time)
            engine.model.cleanup(time_min)
            # Retain estimator history for the rollout. Cleaning event history
            # mid-run can delete confounders before delayed CR/ISF checks run,
            # making old split meals look cleaner than they were.
            action_umin = basal_umin + bolus_units_now / sample_time
            pending_carbs = meal_grams

    hist = env.env.show_history()[env.day:]
    bg = hist["BG"].to_numpy(dtype=float)
    risk = hist["Risk"].to_numpy(dtype=float)
    magni_risk = hist["Magni_Risk"].to_numpy(dtype=float) if "Magni_Risk" in hist else risk
    insulin = hist["insulin"].to_numpy(dtype=float)
    cho = hist["CHO"].to_numpy(dtype=float)

    metrics = {
        "patient": patient,
        "seed": seed,
        "controller": controller,
        "days_requested": days,
        "eval_days_after_warmup": max(0.0, (len(hist) * sample_time) / (24 * 60)),
        "sample_time_min": sample_time,
        "n_eval_samples": int(len(bg)),
        "done_ever": bool(done_ever),
        "first_done_step": first_done_step,
        "mean_bg": float(np.mean(bg)),
        "median_bg": float(np.median(bg)),
        "min_bg": float(np.min(bg)),
        "max_bg": float(np.max(bg)),
        "tir_70_180_pct": float(np.mean((bg >= 70) & (bg <= 180)) * 100.0),
        "tbr_lt70_pct": float(np.mean(bg < 70) * 100.0),
        "tsbr_lt54_pct": float(np.mean(bg < 54) * 100.0),
        "tar_gt180_pct": float(np.mean(bg > 180) * 100.0),
        "tsar_gt250_pct": float(np.mean(bg > 250) * 100.0),
        "risk_mean": float(np.mean(risk)),
        "magni_risk_mean": float(np.mean(magni_risk)),
        "total_insulin_units_eval_window": float(np.sum(insulin) * sample_time),
        "total_cho_grams_eval_window": float(np.sum(cho) * sample_time),
        "total_action_units_full_rollout": float((basal_units + bolus_units)),
        "basal_units_full_rollout": float(basal_units),
        "bolus_units_full_rollout": float(bolus_units),
        "meal_events_full_rollout": meal_events,
        "meal_grams_full_rollout": total_meal_grams,
        "cr": float(env.CR),
        "cf": float(env.CF),
        "ideal_basal_uhr": float(env.ideal_basal) * 60.0,
        "body_weight_kg": float(env.bw),
    }
    return metrics, trace_rows


def aggregate(rows: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    metric_names = [
        "risk_mean",
        "magni_risk_mean",
        "tir_70_180_pct",
        "tbr_lt70_pct",
        "tsbr_lt54_pct",
        "tar_gt180_pct",
        "tsar_gt250_pct",
        "mean_bg",
        "total_insulin_units_eval_window",
        "total_cho_grams_eval_window",
    ]
    output: list[dict[str, Any]] = []
    for controller in sorted({row["controller"] for row in rows}):
        subset = [row for row in rows if row["controller"] == controller]
        item: dict[str, Any] = {
            "controller": controller,
            "n_runs": len(subset),
            "n_done_ever": sum(1 for row in subset if row["done_ever"]),
            "done_rate_pct": 100.0 * sum(1 for row in subset if row["done_ever"]) / len(subset),
        }
        for metric in metric_names:
            stats = summarize([row[metric] for row in subset])
            for stat, value in stats.items():
                item[f"{metric}_{stat}"] = value
        output.append(item)

    ref = next((row for row in output if row["controller"] == reference), None)
    if ref:
        for row in output:
            row["risk_improvement_vs_reference_pct"] = pct_change(
                ref.get("risk_mean_median"), row.get("risk_mean_median")
            )
            row["tir_delta_vs_reference_pct_points"] = (
                row.get("tir_70_180_pct_median") - ref.get("tir_70_180_pct_median")
                if row.get("tir_70_180_pct_median") is not None and ref.get("tir_70_180_pct_median") is not None
                else None
            )
    return output


def write_report(path: Path, args: argparse.Namespace, aggregates: list[dict[str, Any]]) -> None:
    def finite_or(value: Any, fallback: float) -> float:
        if value is None:
            return fallback
        value = float(value)
        return value if math.isfinite(value) else fallback

    best = min(aggregates, key=lambda row: finite_or(row.get("risk_mean_median"), float("inf")))
    lines = [
        "# RL4BG Adolescent Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Protocol",
        "",
        f"- RL4BG root: `{args.rl4bg_root}`",
        f"- Patients: adolescents only ({', '.join(args.patients)})",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- Days per rollout: {args.days}; first day dropped from metrics.",
        f"- Target: {args.target} mg/dL.",
        f"- Controllers: {', '.join(args.controllers)}",
        "- Environment: public MLD3/RL4BG `DeepSACT1DEnv`, Harrison-Benedict meals, 5-minute meal duration, historical init.",
        "",
        "## Aggregate Results",
        "",
        "| Controller | n | done rate | median risk | median TIR | median TBR | median TAR | median mean BG | risk improvement vs BB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregates, key=lambda item: finite_or(item.get("risk_mean_median"), float("inf"))):
        lines.append(
            "| {controller} | {n_runs} | {done:.1f}% | {risk:.2f} | {tir:.2f}% | {tbr:.2f}% | {tar:.2f}% | {mean_bg:.1f} | {imp:.2f}% |".format(
                controller=row["controller"],
                n_runs=row["n_runs"],
                done=row.get("done_rate_pct") or 0.0,
                risk=row.get("risk_mean_median") or 0.0,
                tir=row.get("tir_70_180_pct_median") or 0.0,
                tbr=row.get("tbr_lt70_pct_median") or 0.0,
                tar=row.get("tar_gt180_pct_median") or 0.0,
                mean_bg=row.get("mean_bg_median") or 0.0,
                imp=row.get("risk_improvement_vs_reference_pct") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Lowest median risk in this run: `{best['controller']}`.",
            "- Treat this as a same-environment adolescent comparison, not a full paper reproduction unless run with the paper-scale 100 seeds x 10 days protocol.",
            "- Fox/PID mechanics use the public RL4BG `PID` class and `ManualBBController`; Gluca emits the same single U/min action into the same environment wrapper.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_sources(rl4bg_root: Path, outdir: Path) -> None:
    snapshot = outdir / "rl4bg_source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for relative in [
        "bgp/rl/pid.py",
        "bgp/manual_bb.py",
        "bgp/pid_data_collection.py",
        "bgp/simglucose/envs/simglucose_gym_env.py",
        "bgp/simglucose/controller/basal_bolus_ctrller.py",
        "bgp/simglucose/params/Quest2.csv",
        "bgp/simglucose/params/vpatient_params.csv",
    ]:
        src = rl4bg_root / relative
        if src.exists():
            dst = snapshot / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def package_outputs(outdir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in [
            outdir / "rl4bg_adolescent_results.json",
            outdir / "rl4bg_adolescent_aggregate_metrics.csv",
            outdir / "rl4bg_adolescent_patient_metrics.csv",
            outdir / "rl4bg_adolescent_step_traces.csv",
            outdir / "rl4bg_adolescent_report.md",
            SCRIPT_DIR / "run_rl4bg_adolescent_validation.py",
        ]:
            if file_path.exists():
                zf.write(file_path, file_path.relative_to(SCRIPT_DIR))
        snapshot = outdir / "rl4bg_source_snapshot"
        if snapshot.exists():
            for file_path in snapshot.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(SCRIPT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl4bg-root", type=Path, default=DEFAULT_RL4BG_ROOT)
    parser.add_argument("--patients", nargs="+", default=ADOLESCENTS)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--target", type=float, default=140.0)
    parser.add_argument("--time-std", type=float, default=None)
    parser.add_argument("--controllers", nargs="+", choices=DEFAULT_CONTROLLERS, default=DEFAULT_CONTROLLERS)
    parser.add_argument("--trace-stride", type=int, default=12)
    parser.add_argument("--reference-controller", default="fox_bb")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "rl4bg_adolescent_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.rl4bg_root = ensure_rl4bg(args.rl4bg_root)
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configure_paths(args.rl4bg_root)

    start = time.time()
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    total = len(args.patients) * len(args.seeds) * len(args.controllers)
    idx = 0
    for seed in args.seeds:
        for patient in args.patients:
            for controller in args.controllers:
                idx += 1
                print(f"[{idx}/{total}] {controller} {patient} seed={seed}", flush=True)
                metrics, trace_rows = run_rollout(
                    patient=patient,
                    seed=seed,
                    controller=controller,
                    days=args.days,
                    target=args.target,
                    rl4bg_root=args.rl4bg_root,
                    time_std=args.time_std,
                    trace_stride=args.trace_stride,
                )
                rows.append(metrics)
                traces.extend(trace_rows)

    aggregates = aggregate(rows, args.reference_controller)
    aggregate_path = args.outdir / "rl4bg_adolescent_aggregate_metrics.csv"
    patient_path = args.outdir / "rl4bg_adolescent_patient_metrics.csv"
    trace_path = args.outdir / "rl4bg_adolescent_step_traces.csv"
    report_path = args.outdir / "rl4bg_adolescent_report.md"
    results_path = args.outdir / "rl4bg_adolescent_results.json"
    zip_path = SCRIPT_DIR / "rl4bg_adolescent_validation_package.zip"

    write_csv(aggregate_path, [rounded_json(row) for row in aggregates])
    write_csv(patient_path, [rounded_json(row) for row in rows])
    write_csv(trace_path, [rounded_json(row) for row in traces])
    write_report(report_path, args, aggregates)
    snapshot_sources(args.rl4bg_root, args.outdir)

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "rl4bg_root": str(args.rl4bg_root),
            "rl4bg_url": RL4BG_URL,
            "patients": args.patients,
            "seeds": args.seeds,
            "days": args.days,
            "target": args.target,
            "time_std": args.time_std,
            "controllers": args.controllers,
            "trace_stride": args.trace_stride,
            "runtime_seconds": time.time() - start,
            "notes": [
                "Uses public MLD3/RL4BG DeepSACT1DEnv, not stock simglucose.",
                "Metrics drop the first day with show_history()[env.day:], matching RL4BG baseline scripts.",
                "Fox BB/PID/PID-MA use public RL4BG controller mechanics with appendix adolescent gains.",
                "Gluca feedback profiled uses the same profile data and PID-MA gain table shape, with scales selected on disjoint training seeds.",
                "Gluca Bayesian controllers emit the same single U/min action into the RL4BG wrapper.",
                "Estimator history is retained across each rollout so confounder gates cannot be altered by cleanup.",
            ],
        },
        "aggregate_metrics": aggregates,
        "patient_metrics": rows,
        "files": {
            "aggregate_csv": str(aggregate_path),
            "patient_csv": str(patient_path),
            "trace_csv": str(trace_path),
            "report_md": str(report_path),
            "zip": str(zip_path),
        },
    }
    results_path.write_text(json.dumps(rounded_json(results), indent=2) + "\n", encoding="utf-8")
    package_outputs(args.outdir, zip_path)

    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
