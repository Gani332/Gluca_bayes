"""
Full-cohort validation for the Bayesian dosing engine.

Runs the UVA/Padova 30-patient cohort through the closed-loop simulator and
summarizes cohort-level safety and efficacy metrics.

This is intended as a release-gating script for the Bayesian path, not a unit test.
"""

import argparse
import json
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List

import pandas as pd
import simglucose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bayesian_mpc.evaluate import compute_metrics, run_simulation


def get_all_patient_names() -> List[str]:
    params_file = os.path.join(os.path.dirname(simglucose.__file__), "params", "vpatient_params.csv")
    df = pd.read_csv(params_file)
    names = [
        str(name) for name in df["Name"].tolist()
        if str(name).startswith(("adult#", "adolescent#", "child#"))
    ]
    return sorted(names)


def get_cohort_names(cohort: str) -> List[str]:
    names = get_all_patient_names()
    if cohort == "all":
        return names
    return [name for name in names if name.startswith(f"{cohort[:-1]}#")]


def battelino_pass(metrics: Dict) -> Dict[str, bool]:
    return {
        "tir": metrics["tir"] >= 70.0,
        "tbr": metrics["tbr"] < 4.0,
        "tsbr": metrics["tsbr"] < 1.0,
        "tar": metrics["tar"] < 25.0,
        "tsar": metrics["tsar"] < 5.0,
        "cv": metrics["cv"] < 36.0,
    }


def summarize_results(results: List[Dict]) -> Dict:
    metric_keys = ["tir", "tbr", "tsbr", "tar", "tsar", "mean_bg", "cv", "min_bg", "max_bg"]
    summary = {
        "n_patients": len(results),
        "metrics": {},
        "consensus": {},
        "per_patient": results,
    }

    for key in metric_keys:
        values = [result["metrics"][key] for result in results]
        summary["metrics"][key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    pass_maps = [result["consensus"] for result in results]
    for key in ["tir", "tbr", "tsbr", "tar", "tsar", "cv"]:
        passed = sum(1 for mapping in pass_maps if mapping[key])
        summary["consensus"][key] = {
            "passed": passed,
            "total": len(results),
            "rate": passed / max(1, len(results)),
        }

    summary["consensus"]["all_targets"] = {
        "passed": sum(1 for mapping in pass_maps if all(mapping.values())),
        "total": len(results),
        "rate": sum(1 for mapping in pass_maps if all(mapping.values())) / max(1, len(results)),
    }

    return summary


def validate_configuration(
    patient_names: List[str],
    *,
    days: int,
    estimator_version: str,
    mode: str,
    use_covariates: bool,
    health_mode: str,
    seed: int,
) -> Dict:
    results = []

    for idx, patient_name in enumerate(patient_names, start=1):
        print(f"[{idx:02d}/{len(patient_names)}] {patient_name}  mode={mode} estimator={estimator_version}")
        simulation = run_simulation(
            patient_name=patient_name,
            days=days,
            seed=seed,
            mode=mode,
            estimator_version=estimator_version,
            use_covariates=use_covariates,
            health_mode=health_mode,
            verbose=False,
        )
        metrics = simulation["metrics"]
        consensus = battelino_pass(metrics)
        results.append({
            "patient": patient_name,
            "metrics": metrics,
            "consensus": consensus,
            "engine_status": simulation.get("engine_status", {}),
        })

    return summarize_results(results)


def print_summary(label: str, summary: Dict):
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print(f"{'=' * 78}")
    print(
        f"TIR={summary['metrics']['tir']['mean']:.1f}% ±{summary['metrics']['tir']['std']:.1f} | "
        f"TBR={summary['metrics']['tbr']['mean']:.1f}% ±{summary['metrics']['tbr']['std']:.1f} | "
        f"TSBR={summary['metrics']['tsbr']['mean']:.1f}% | "
        f"TAR={summary['metrics']['tar']['mean']:.1f}% | "
        f"Mean BG={summary['metrics']['mean_bg']['mean']:.0f} | "
        f"CV={summary['metrics']['cv']['mean']:.1f}%"
    )
    print(
        "Consensus pass rates: "
        f"TIR {summary['consensus']['tir']['passed']}/{summary['consensus']['tir']['total']}, "
        f"TBR {summary['consensus']['tbr']['passed']}/{summary['consensus']['tbr']['total']}, "
        f"TSBR {summary['consensus']['tsbr']['passed']}/{summary['consensus']['tsbr']['total']}, "
        f"TAR {summary['consensus']['tar']['passed']}/{summary['consensus']['tar']['total']}, "
        f"TSAR {summary['consensus']['tsar']['passed']}/{summary['consensus']['tsar']['total']}, "
        f"CV {summary['consensus']['cv']['passed']}/{summary['consensus']['cv']['total']}"
    )
    print(
        f"All Battelino targets met: {summary['consensus']['all_targets']['passed']}/"
        f"{summary['consensus']['all_targets']['total']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["all", "adults", "adolescents", "children"], default="all")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["v2", "v1", "fixed"],
        choices=["v2", "v2-health", "v1", "fixed"],
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    patient_names = get_cohort_names(args.cohort)
    all_summaries = {}

    config_map = {
        "v2": dict(mode="bayesian", estimator_version="v2", use_covariates=False, health_mode="none"),
        "v2-health": dict(mode="bayesian", estimator_version="v2", use_covariates=True, health_mode="synthetic"),
        "v1": dict(mode="bayesian", estimator_version="v1", use_covariates=False, health_mode="none"),
        "fixed": dict(mode="fixed", estimator_version="v1", use_covariates=False, health_mode="none"),
    }

    for config_name in args.configs:
        summary = validate_configuration(
            patient_names,
            days=args.days,
            seed=args.seed,
            **config_map[config_name],
        )
        all_summaries[config_name] = summary
        print_summary(f"{config_name} | cohort={args.cohort} | days={args.days}", summary)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
