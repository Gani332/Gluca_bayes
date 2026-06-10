#!/usr/bin/env python3
"""
Locked ISF estimator tuning.

This script keeps tuning and claiming separate:

1. Split patients into development and locked-test groups.
2. Tune only on development patients.
3. Evaluate the selected estimator once on locked-test patients.
4. Report whether it beats profile/pump-style proxies and stronger
   personalized baselines.

This is still estimator validation, not controller or clinical validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bayesian_mpc.estimator_failure_modes import (  # noqa: E402
    adaptive_robust_positive_estimate,
    huber_location,
    modular_positive_estimate,
    trimmed_mean,
    winsorized_mean,
)


POPULATION_ISF = 50.0


@dataclass(frozen=True)
class Event:
    cohort: str
    patient: str
    event_index: int
    dose_units: float
    net_drop_mgdl: float
    observed_net_isf: float
    clinical_prior_isf: float


@dataclass(frozen=True)
class Candidate:
    family: str
    min_observations: int
    data_dominance_observations: int
    outlier_iqr_threshold: float

    @property
    def name(self) -> str:
        return (
            f"gluca_tuned_{self.family}"
            f"_min{self.min_observations}"
            f"_dom{self.data_dominance_observations}"
            f"_iqr{self.outlier_iqr_threshold:g}"
        )


def read_events(path: Path, cohort: str) -> list[Event]:
    rows: list[Event] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                Event(
                    cohort=cohort,
                    patient=row["patient"],
                    event_index=int(float(row["event_index"])),
                    dose_units=float(row["dose_units"]),
                    net_drop_mgdl=float(row["net_drop_mgdl"]),
                    observed_net_isf=float(row["observed_net_isf"]),
                    clinical_prior_isf=float(row.get("clinical_prior_isf") or POPULATION_ISF),
                )
            )
    return rows


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


def round_value(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: round_value(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_value(item, digits) for item in value]
    return value


def pct_improvement(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def ewma(values: list[float], initial: float = POPULATION_ISF, alpha: float = 0.35) -> float:
    estimate = initial
    for value in values:
        estimate = alpha * value + (1.0 - alpha) * estimate
    return estimate


def split_patient_events(events: list[Event]) -> tuple[list[Event], list[Event]]:
    ordered = sorted(events, key=lambda event: event.event_index)
    if len(ordered) < 5:
        return [], []
    split_idx = max(1, int(len(ordered) * 0.7))
    if len(ordered) - split_idx < 2:
        split_idx = len(ordered) - 2
    return ordered[:split_idx], ordered[split_idx:]


def patient_groups(events: list[Event]) -> dict[tuple[str, str], list[Event]]:
    grouped: dict[tuple[str, str], list[Event]] = {}
    for event in events:
        grouped.setdefault((event.cohort, event.patient), []).append(event)
    return grouped


def split_dev_test_patients(events: list[Event], test_patients_per_cohort: int) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    grouped = patient_groups(events)
    dev: set[tuple[str, str]] = set()
    test: set[tuple[str, str]] = set()
    cohorts = sorted({cohort for cohort, _patient in grouped})
    for cohort in cohorts:
        patients = sorted(patient for item_cohort, patient in grouped if item_cohort == cohort)
        test_patients = set(patients[-test_patients_per_cohort:])
        for patient in patients:
            key = (cohort, patient)
            if patient in test_patients:
                test.add(key)
            else:
                dev.add(key)
    return dev, test


def baseline_estimates(train: list[Event]) -> dict[str, float]:
    observed = [event.observed_net_isf for event in train]
    recent_count = min(3, len(observed))
    return {
        "fixed_population_50": POPULATION_ISF,
        "clinical_prior": train[0].clinical_prior_isf if train else POPULATION_ISF,
        "empirical_mean": statistics.fmean(observed),
        "empirical_median": statistics.median(observed),
        "empirical_trimmed_mean": trimmed_mean(observed),
        "robust_winsorized_mean": winsorized_mean(observed),
        "robust_huber_center": huber_location(observed),
        "recent_3_mean": statistics.fmean(observed[-recent_count:]),
        "ewma_alpha_0_35": ewma(observed),
    }


def candidate_estimate(candidate: Candidate, train: list[Event]) -> tuple[float, str, float]:
    observed = [event.observed_net_isf for event in train]
    kwargs = {
        "prior_value": POPULATION_ISF,
        "min_value": 5.0,
        "max_value": 200.0,
        "min_observations": candidate.min_observations,
        "data_dominance_observations": candidate.data_dominance_observations,
        "outlier_iqr_threshold": candidate.outlier_iqr_threshold,
    }
    if candidate.family == "modular":
        estimate = modular_positive_estimate(observed, **kwargs)
    elif candidate.family == "adaptive":
        estimate = adaptive_robust_positive_estimate(observed, **kwargs)
    else:
        raise ValueError(f"Unknown candidate family: {candidate.family}")
    return estimate.value, estimate.method, estimate.data_weight


def score_estimates(
    *,
    events: list[Event],
    patient_keys: set[tuple[str, str]],
    estimates_by_patient: dict[tuple[str, str], dict[str, tuple[float, str, float]]],
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = patient_groups(events)
    for key in sorted(patient_keys):
        train, holdout = split_patient_events(grouped[key])
        if not train or not holdout:
            continue
        for estimator, (isf, method, data_weight) in estimates_by_patient[key].items():
            for event in holdout:
                predicted_drop = event.dose_units * isf
                rows.append(
                    {
                        "split": split_name,
                        "cohort": event.cohort,
                        "patient": event.patient,
                        "event_index": event.event_index,
                        "estimator": estimator,
                        "method": method,
                        "n_train": len(train),
                        "n_holdout": len(holdout),
                        "isf_estimate": isf,
                        "data_weight": data_weight,
                        "dose_units": event.dose_units,
                        "actual_net_drop": event.net_drop_mgdl,
                        "predicted_drop": predicted_drop,
                        "net_abs_error": abs(predicted_drop - event.net_drop_mgdl),
                    }
                )
    return rows


def summarize_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["estimator"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (split, estimator), items in sorted(grouped.items()):
        errors = [float(item["net_abs_error"]) for item in items]
        output.append(
            {
                "split": split,
                "estimator": estimator,
                "n_events": len(items),
                "net_mae": statistics.fmean(errors),
                "median_abs_error": statistics.median(errors),
                "mean_data_weight": statistics.fmean(float(item["data_weight"]) for item in items),
            }
        )
    return output


def candidate_grid() -> list[Candidate]:
    candidates: list[Candidate] = []
    for family in ["modular", "adaptive"]:
        for min_observations in [2, 3, 4, 5]:
            for data_dominance in [3, 4, 5, 6, 8, 10]:
                for iqr in [0.05, 0.10, 0.15, 0.20, 0.30]:
                    candidates.append(
                        Candidate(
                            family=family,
                            min_observations=min_observations,
                            data_dominance_observations=data_dominance,
                            outlier_iqr_threshold=iqr,
                        )
                    )
    return candidates


def estimates_for_candidate(events: list[Event], patient_keys: set[tuple[str, str]], candidate: Candidate | None) -> dict[tuple[str, str], dict[str, tuple[float, str, float]]]:
    grouped = patient_groups(events)
    output: dict[tuple[str, str], dict[str, tuple[float, str, float]]] = {}
    for key in sorted(patient_keys):
        train, _holdout = split_patient_events(grouped[key])
        if not train:
            continue
        if candidate is None:
            estimates = {
                name: (value, name, 1.0 if name not in {"fixed_population_50", "clinical_prior"} else 0.0)
                for name, value in baseline_estimates(train).items()
            }
        else:
            value, method, data_weight = candidate_estimate(candidate, train)
            estimates = {candidate.name: (value, method, data_weight)}
        output[key] = estimates
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-patients-per-cohort", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "locked_isf_latest")
    args = parser.parse_args()

    events = [
        *read_events(REPO_ROOT / "validation" / "real_parameter_learning_validation" / "padova_clean_correction_events.csv", "adults"),
        *read_events(REPO_ROOT / "validation" / "real_parameter_learning_validation" / "adolescent_clean_correction_rerun" / "padova_clean_correction_events.csv", "adolescents"),
    ]
    dev_patients, test_patients = split_dev_test_patients(events, args.test_patients_per_cohort)
    args.outdir.mkdir(parents=True, exist_ok=True)

    baseline_rows = [
        *score_estimates(
            events=events,
            patient_keys=dev_patients,
            estimates_by_patient=estimates_for_candidate(events, dev_patients, None),
            split_name="dev",
        ),
        *score_estimates(
            events=events,
            patient_keys=test_patients,
            estimates_by_patient=estimates_for_candidate(events, test_patients, None),
            split_name="locked_test",
        ),
    ]

    candidate_summaries: list[dict[str, Any]] = []
    best_candidate: Candidate | None = None
    best_dev_mae = float("inf")
    best_candidate_rows: list[dict[str, Any]] = []

    for candidate in candidate_grid():
        rows = score_estimates(
            events=events,
            patient_keys=dev_patients,
            estimates_by_patient=estimates_for_candidate(events, dev_patients, candidate),
            split_name="dev",
        )
        if not rows:
            continue
        dev_mae = statistics.fmean(float(row["net_abs_error"]) for row in rows)
        candidate_summaries.append(
            {
                **asdict(candidate),
                "candidate": candidate.name,
                "dev_net_mae": dev_mae,
                "dev_n_events": len(rows),
                "dev_mean_data_weight": statistics.fmean(float(row["data_weight"]) for row in rows),
            }
        )
        if dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            best_candidate = candidate
            best_candidate_rows = rows

    if best_candidate is None:
        raise RuntimeError("No candidate could be scored")

    locked_rows = score_estimates(
        events=events,
        patient_keys=test_patients,
        estimates_by_patient=estimates_for_candidate(events, test_patients, best_candidate),
        split_name="locked_test",
    )
    all_rows = [*baseline_rows, *best_candidate_rows, *locked_rows]
    summary_rows = summarize_predictions(all_rows)

    def summary_value(split: str, estimator: str) -> float:
        return next(row["net_mae"] for row in summary_rows if row["split"] == split and row["estimator"] == estimator)

    locked_tuned_mae = summary_value("locked_test", best_candidate.name)
    locked_profile_mae = summary_value("locked_test", "clinical_prior")
    strong_baselines = [
        "empirical_mean",
        "empirical_median",
        "empirical_trimmed_mean",
        "robust_winsorized_mean",
        "robust_huber_center",
        "recent_3_mean",
        "ewma_alpha_0_35",
    ]
    locked_strong_scores = {
        name: summary_value("locked_test", name)
        for name in strong_baselines
    }
    best_strong_name, best_strong_mae = min(locked_strong_scores.items(), key=lambda item: item[1])

    result = {
        "protocol": "Tune on development patients only; evaluate once on locked-test patients.",
        "dev_patients": sorted([f"{cohort}/{patient}" for cohort, patient in dev_patients]),
        "locked_test_patients": sorted([f"{cohort}/{patient}" for cohort, patient in test_patients]),
        "selected_candidate": {
            **asdict(best_candidate),
            "name": best_candidate.name,
            "dev_net_mae": best_dev_mae,
        },
        "locked_test": {
            "tuned_net_mae": locked_tuned_mae,
            "clinical_profile_proxy_mae": locked_profile_mae,
            "best_strong_baseline": best_strong_name,
            "best_strong_baseline_mae": best_strong_mae,
            "improvement_vs_profile_proxy_pct": pct_improvement(locked_profile_mae, locked_tuned_mae),
            "improvement_vs_best_strong_baseline_pct": pct_improvement(best_strong_mae, locked_tuned_mae),
            "beats_profile_proxy": locked_tuned_mae < locked_profile_mae,
            "beats_best_strong_baseline": locked_tuned_mae < best_strong_mae,
        },
        "interpretation": {
            "can_claim_beats_profile_proxy": locked_tuned_mae < locked_profile_mae,
            "can_claim_beats_robust_personalized_baseline": locked_tuned_mae < best_strong_mae,
            "can_claim_beats_current_pumps": False,
            "why_not_current_pumps": "This is ISF estimator validation against profile/proxy and statistical baselines, not a closed-loop commercial pump controller trial.",
        },
    }

    write_csv(args.outdir / "candidate_dev_scores.csv", [round_value(row) for row in candidate_summaries])
    write_csv(args.outdir / "locked_isf_predictions.csv", [round_value(row) for row in all_rows])
    write_csv(args.outdir / "locked_isf_summary.csv", [round_value(row) for row in summary_rows])
    (args.outdir / "locked_isf_tuning_result.json").write_text(
        json.dumps(round_value(result), indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(round_value(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
