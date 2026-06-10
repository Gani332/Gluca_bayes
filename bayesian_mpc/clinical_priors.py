"""
Clinically grounded cohort priors and simple safety profiles.

These are starting priors, not hard treatment rules. They are used when a
patient-specific prescription or learned settings are unavailable.

This module is intentionally kept in parity with the shipped iOS app for the
startup ISF/CR priors so offline evaluation does not drift from production
behavior.

Design principles:
  - Keep the math simple and interpretable.
  - Use standard clinical starting heuristics rather than simulator-specific
    hidden physiology.
  - Bias defaults toward hypo-avoidance, especially for younger cohorts.

Sources used to define these defaults:
  - iOS startup priors: pediatric ISF/ICR medians and adult sex-weight insulin
    estimates converted through 500/1800-style rules.
  - Endotext safety framing: pediatric/adult targets and conservative basal
    fractions.

The code uses conservative points within those ranges because these values act
as priors for a shadow-mode adaptive system, not as final prescribed doses.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


VALID_COHORTS = {"adult", "adolescent", "child"}
VALID_CLINICAL_SEX = {"female", "male", "unspecified"}

ADULT_NEUTRAL_INSULIN_PER_KG = 0.617
ADULT_MALE_INSULIN_PER_KG = 0.665
ADULT_FEMALE_INSULIN_PER_KG = 0.584


@dataclass(frozen=True)
class SafetyProfile:
    cohort: str
    bg_target: float
    correction_margin: float
    low_suspend_bg: float
    low_reduce_bg: float
    predicted_suspend_bg: float
    max_bolus_units: float
    max_correction_units: float
    max_basal_multiplier: float
    max_basal_step_up: float
    correction_scale_floor: float
    basal_scale_floor: float


@dataclass(frozen=True)
class ClinicalPriors:
    cohort: str
    weight_kg: float
    tdd_units_per_day: float
    tdd_units_per_kg: float
    basal_fraction: float
    isf: float
    cr: float
    basal_rate: float
    bg_target: float
    safety: SafetyProfile


def normalize_cohort(cohort: Optional[str]) -> str:
    raw = (cohort or "adult").strip().lower()
    mapping = {
        "adults": "adult",
        "adult": "adult",
        "adolescents": "adolescent",
        "adolescent": "adolescent",
        "children": "child",
        "child": "child",
    }
    return mapping.get(raw, "adult")


def infer_cohort_from_patient_name(patient_name: Optional[str]) -> str:
    raw = (patient_name or "").lower()
    if raw.startswith("adult#"):
        return "adult"
    if raw.startswith("adolescent#"):
        return "adolescent"
    if raw.startswith("child#"):
        return "child"
    return "adult"


def normalize_clinical_sex(clinical_sex: Optional[str]) -> str:
    raw = (clinical_sex or "unspecified").strip().lower()
    mapping = {
        "female": "female",
        "f": "female",
        "male": "male",
        "m": "male",
        "unspecified": "unspecified",
        "unknown": "unspecified",
        "prefer not to say": "unspecified",
    }
    return mapping.get(raw, "unspecified")


def _cohort_tdd_per_kg(cohort: str) -> float:
    # Conservative points within published clinical starting ranges.
    if cohort == "adolescent":
        return 0.9
    if cohort == "child":
        return 0.7
    return 0.5


def _adult_insulin_per_kg(clinical_sex: str) -> float:
    if clinical_sex == "female":
        return ADULT_FEMALE_INSULIN_PER_KG
    if clinical_sex == "male":
        return ADULT_MALE_INSULIN_PER_KG
    return ADULT_NEUTRAL_INSULIN_PER_KG


def _pediatric_carb_ratio(base: float, clinical_sex: str) -> float:
    if clinical_sex == "female":
        factor = 0.96
    elif clinical_sex == "male":
        factor = 1.04
    else:
        factor = 1.0
    return float(np.clip(base * factor, 4.0, 30.0))


def _startup_isf_cr(cohort: str, weight_kg: float, clinical_sex: str) -> tuple[float, float]:
    if cohort == "child":
        return 120.0, _pediatric_carb_ratio(10.0, clinical_sex)

    if cohort == "adolescent":
        return 50.0, _pediatric_carb_ratio(6.1, clinical_sex)

    tdd = weight_kg * _adult_insulin_per_kg(clinical_sex)
    isf = float(np.clip(1800.0 / max(tdd, 1e-6), 20.0, 120.0))
    cr = float(np.clip(500.0 / max(tdd, 1e-6), 4.0, 30.0))
    return isf, cr


def _cohort_basal_fraction(cohort: str) -> float:
    # Lower basal fractions are safer for pump-style control and align with the
    # lower end of published basal/TDD ratios.
    if cohort == "child":
        return 0.35
    if cohort == "adolescent":
        return 0.40
    return 0.45


def get_safety_profile(cohort: Optional[str], bg_target: Optional[float] = None) -> SafetyProfile:
    cohort_name = normalize_cohort(cohort)

    if cohort_name == "child":
        target = 150.0 if bg_target is None else float(bg_target)
        return SafetyProfile(
            cohort=cohort_name,
            bg_target=target,
            correction_margin=30.0,
            low_suspend_bg=80.0,
            low_reduce_bg=110.0,
            predicted_suspend_bg=100.0,
            max_bolus_units=8.0,
            max_correction_units=2.5,
            max_basal_multiplier=1.15,
            max_basal_step_up=0.20,
            correction_scale_floor=0.25,
            basal_scale_floor=0.25,
        )

    if cohort_name == "adolescent":
        target = 120.0 if bg_target is None else float(bg_target)
        return SafetyProfile(
            cohort=cohort_name,
            bg_target=target,
            correction_margin=25.0,
            low_suspend_bg=75.0,
            low_reduce_bg=95.0,
            predicted_suspend_bg=90.0,
            max_bolus_units=12.0,
            max_correction_units=4.0,
            max_basal_multiplier=1.25,
            max_basal_step_up=0.35,
            correction_scale_floor=0.35,
            basal_scale_floor=0.35,
        )

    target = 110.0 if bg_target is None else float(bg_target)
    return SafetyProfile(
        cohort="adult",
        bg_target=target,
        correction_margin=20.0,
        low_suspend_bg=70.0,
        low_reduce_bg=90.0,
        predicted_suspend_bg=80.0,
        max_bolus_units=15.0,
        max_correction_units=6.0,
        max_basal_multiplier=1.50,
        max_basal_step_up=0.50,
        correction_scale_floor=0.50,
        basal_scale_floor=0.50,
    )


def derive_clinical_priors(
    cohort: Optional[str],
    weight_kg: Optional[float],
    *,
    clinical_sex: Optional[str] = None,
    isf: Optional[float] = None,
    cr: Optional[float] = None,
    basal_rate: Optional[float] = None,
    bg_target: Optional[float] = None,
) -> ClinicalPriors:
    cohort_name = normalize_cohort(cohort)
    sex_name = normalize_clinical_sex(clinical_sex)
    weight = float(weight_kg or 70.0)
    weight = float(np.clip(weight, 15.0, 180.0))

    safety = get_safety_profile(cohort_name, bg_target)

    tdd_per_kg = _adult_insulin_per_kg(sex_name) if cohort_name == "adult" else _cohort_tdd_per_kg(cohort_name)
    basal_fraction = _cohort_basal_fraction(cohort_name)
    tdd = weight * tdd_per_kg

    startup_isf, startup_cr = _startup_isf_cr(cohort_name, weight, sex_name)
    derived_basal = (tdd * basal_fraction) / 24.0

    return ClinicalPriors(
        cohort=cohort_name,
        weight_kg=weight,
        tdd_units_per_day=float(tdd),
        tdd_units_per_kg=float(tdd_per_kg),
        basal_fraction=float(basal_fraction),
        isf=float(np.clip(isf if isf is not None else startup_isf, 10.0, 200.0)),
        cr=float(np.clip(cr if cr is not None else startup_cr, 2.0, 50.0)),
        basal_rate=float(np.clip(basal_rate if basal_rate is not None else derived_basal, 0.05, 5.0)),
        bg_target=float(safety.bg_target),
        safety=safety,
    )
