"""
Research-only dosing engine for offline simulation.

Do not import this module from the Gluca iOS app, Supabase functions, or any
user-facing product path. Instantiating DosingEngine requires the explicit
GLUCA_ENABLE_RESEARCH_DOSING=1 environment flag.

Mathematically this is simple:

  bolus = carbs / CR + max(0, (BG - target) / ISF - dosing_IOB)

  basal_adjustment = (predicted_BG - target) / ISF * adjustment_factor

That's it. The complexity is in getting good estimates of:
  - BG (from UKF — denoised, lag-corrected)
  - CR and ISF (from Bayesian estimator — personalized, uncertainty-aware)
  - IOB (from insulin PK model — proper absorption curve, not just exponential)

Scheduled basal is tracked for prediction, but dosing IOB excludes basal and
only offsets correction insulin. Meal coverage is not erased by basal IOB.

The engine doesn't optimize anything. It computes the formula,
applies safety limits, and reports its confidence.

Two estimator backends are supported:
  - v1: independent Gaussian posteriors (legacy baseline)
  - v2: log-space Bayesian regression with optional health covariates

Shadow mode here means offline evaluation only. It is not a product safety
boundary and does not make dose recommendations appropriate for users.
"""

import os
import numpy as np
from typing import Dict, Optional

from .patient_model import GlucoseModel, InsulinPK
from .bayesian_estimator import BayesianEstimator
from .bayesian_v2 import BayesianEstimatorV2, HealthContext
from .clinical_priors import derive_clinical_priors, normalize_cohort


def _require_research_mode() -> None:
    if os.environ.get("GLUCA_ENABLE_RESEARCH_DOSING") != "1":
        raise RuntimeError(
            "DosingEngine is research-only. Set GLUCA_ENABLE_RESEARCH_DOSING=1 "
            "only for offline simulation or validation scripts."
        )


class DosingEngine:
    """
    Clean insulin dosing engine.

    Inputs (from user): carbs eaten, insulin taken
    Inputs (from CGM): glucose readings every 3-5 min
    Outputs: bolus recommendation, basal adjustment, predictions, confidence
    """

    def __init__(
        self,
        # Initial params (from doctor's prescription or population defaults)
        isf: Optional[float] = None,
        cr: Optional[float] = None,
        basal_rate: Optional[float] = None,
        bg_target: Optional[float] = None,
        body_weight: float = 70.0,
        cohort: str = "adult",
        clinical_sex: str = "unspecified",
        estimator_version: str = "v1",
        use_covariates: bool = False,
    ):
        _require_research_mode()

        self.cohort = normalize_cohort(cohort)
        self.clinical_sex = clinical_sex
        self.clinical_priors = derive_clinical_priors(
            self.cohort,
            body_weight,
            clinical_sex=clinical_sex,
            isf=isf,
            cr=cr,
            basal_rate=basal_rate,
            bg_target=bg_target,
        )
        self.safety = self.clinical_priors.safety

        # Physiological model (for prediction + IOB/COB tracking)
        self.model = GlucoseModel(
            isf=self.clinical_priors.isf,
            cr=self.clinical_priors.cr,
            basal_rate=self.clinical_priors.basal_rate,
            bg_target=self.clinical_priors.bg_target,
            body_weight=body_weight,
        )

        if estimator_version not in {"v1", "v2"}:
            raise ValueError("estimator_version must be 'v1' or 'v2'")

        self.estimator_version = estimator_version
        self.use_covariates = bool(use_covariates and estimator_version == "v2")

        # Bayesian estimator (learns ISF, CR, basal from data)
        if self.estimator_version == "v2":
            self.estimator = BayesianEstimatorV2(
                isf_prior=self.clinical_priors.isf,
                cr_prior=self.clinical_priors.cr,
                basal_prior=self.clinical_priors.basal_rate,
                bg_target=self.clinical_priors.bg_target,
                use_covariates=self.use_covariates,
                cohort=self.cohort,
            )
        else:
            self.estimator = BayesianEstimator(
                isf_prior=self.clinical_priors.isf,
                cr_prior=self.clinical_priors.cr,
                basal_prior=self.clinical_priors.basal_rate,
                bg_target=self.clinical_priors.bg_target,
            )

        # Current time tracking (minutes from arbitrary epoch)
        self.current_time = 0.0
        self.bg_target = self.clinical_priors.bg_target
        self.shadow_mode = True
        self.health_context: Optional[HealthContext] = None
        # IOB used for bolus arithmetic should represent discretionary insulin
        # that can offset correction insulin. Scheduled basal is tracked in the
        # physiological model for prediction, but should not erase meal coverage.
        self.dosing_iob = InsulinPK()

    # ── User inputs ──────────────────────────────────────────────────────

    def record_glucose(self, bg: float, time_min: Optional[float] = None):
        """Record a CGM/UKF glucose reading."""
        if time_min is not None:
            self.current_time = time_min
        self.estimator.record_bg(self.current_time, bg)

    def record_insulin(self, units: float, insulin_type: str = "bolus",
                       time_min: Optional[float] = None):
        """Record insulin taken by the user."""
        t = time_min if time_min is not None else self.current_time
        self.model.record_insulin(t, units)
        self.estimator.record_insulin(t, units, insulin_type)
        if insulin_type != "basal":
            self.dosing_iob.add_dose(t, units)

    def record_meal(self, carbs_grams: float,
                    time_min: Optional[float] = None):
        """Record carbs eaten by the user."""
        t = time_min if time_min is not None else self.current_time
        self.model.record_meal(t, carbs_grams)
        self.estimator.record_meal(t, carbs_grams)

    def tick(self, dt: float = 3.0):
        """Advance time (call this each CGM reading interval)."""
        self.current_time += dt
        self.dosing_iob.cleanup(self.current_time)

    def set_health_context(self, context: Optional[HealthContext]):
        """
        Set Apple Health-derived covariates for the current decision window.

        This is optional. If the active estimator doesn't support covariates,
        the call is a no-op.
        """
        self.health_context = context
        if hasattr(self.estimator, "set_health_context"):
            self.estimator.set_health_context(context)

    def set_apple_health_context(
        self,
        steps_2h: int = 0,
        steps_daily: int = 0,
        workout_min_today: float = 0.0,
        sleep_hours: float = 7.5,
        hrv_ms: float = 50.0,
        cycle_phase: Optional[str] = None,
    ) -> Optional[HealthContext]:
        """
        Convenience wrapper around HealthContext.from_apple_health().

        Returns the normalized context so callers can log it if needed.
        """
        if not self.use_covariates:
            self.set_health_context(None)
            return None

        context = HealthContext.from_apple_health(
            steps_2h=steps_2h,
            steps_daily=steps_daily,
            workout_min_today=workout_min_today,
            sleep_hours=sleep_hours,
            hrv_ms=hrv_ms,
            cycle_phase=cycle_phase,
        )
        self.set_health_context(context)
        return context

    # ── Recommendations ──────────────────────────────────────────────────

    def recommend_bolus(self, current_bg: float,
                        carbs_grams: float = 0.0) -> Dict:
        """
        Recommend a bolus dose.

        The formula:
          meal_bolus = carbs / CR
          correction  = (BG - target) / ISF   (only if BG > target + 20)
          dose = meal_bolus + correction - IOB
          dose = clamp(dose, 0, safety_max)

        Args:
            current_bg: Current glucose (mg/dL), ideally from UKF
            carbs_grams: Meal carbs (0 for correction-only)

        Returns:
            Dict with dose, breakdown, confidence, predictions
        """
        # Try to update parameters from recent data
        self.estimator.update(self.current_time)

        params = self.estimator.get_params()
        isf = params["isf"]
        cr = params["cr"]
        isf_confidence = float(params.get("isf_confidence", 0.0))

        # Update model with latest params
        self.model.update_params(isf=isf, cr=cr)

        # Current IOB. Total IOB stays available for prediction/status, while
        # dosing IOB excludes scheduled basal and is only applied to correction.
        total_iob = self.model.get_iob(self.current_time)
        dosing_iob = self.dosing_iob.get_iob(self.current_time)
        cob = self.model.get_cob(self.current_time)

        # ── The formula ──────────────────────────────────────────────
        meal_bolus = carbs_grams / cr if carbs_grams > 0 else 0.0

        gross_correction = 0.0
        if current_bg > self.bg_target + self.safety.correction_margin:
            gross_correction = (current_bg - self.bg_target) / isf
            correction_gate = (
                self.safety.correction_scale_floor
                + (1.0 - self.safety.correction_scale_floor) * isf_confidence
            )
            gross_correction *= correction_gate
            gross_correction = min(gross_correction, self.safety.max_correction_units)

        correction = max(0.0, gross_correction - dosing_iob)

        raw_dose = meal_bolus + correction
        dose = max(0.0, raw_dose)

        # Safety limits
        max_dose = self.safety.max_bolus_units
        if current_bg < self.safety.low_suspend_bg:
            dose = 0.0  # Never give insulin during hypo
        elif current_bg < self.safety.low_reduce_bg:
            dose = min(dose, meal_bolus)  # Only cover meal, no correction

        dose = min(dose, max_dose)

        # ── Prediction ───────────────────────────────────────────────
        trajectory = self.model.predict(
            current_bg, self.current_time,
            horizon_min=180.0,  # 3 hours
            dt=5.0,
            future_insulin=[(0.0, dose)] if dose > 0 else None,
        )
        predicted_bg_1h = next(
            (bg for t, bg in trajectory if abs(t - 60) < 3), current_bg
        )
        predicted_bg_2h = next(
            (bg for t, bg in trajectory if abs(t - 120) < 3), current_bg
        )
        predicted_min = min(bg for _, bg in trajectory)

        # If prediction shows hypo, reduce dose
        if predicted_min < self.safety.predicted_suspend_bg and dose > 0:
            # Scale down dose proportionally
            headroom = current_bg - self.safety.predicted_suspend_bg
            if headroom > 0:
                dose = dose * min(1.0, headroom / (current_bg - predicted_min))
            else:
                dose = 0.0

        # ── Confidence ───────────────────────────────────────────────
        param_info = self.estimator.get_params()
        confidence = min(
            param_info["isf_confidence"],
            param_info["cr_confidence"] if carbs_grams > 0 else 1.0,
        )

        return {
            "recommended_dose": round(float(dose), 2),
            "meal_component": round(float(meal_bolus), 2),
            "correction_component": round(float(correction), 2),
            "gross_correction_component": round(float(gross_correction), 2),
            "iob": round(float(dosing_iob), 2),
            "total_iob": round(float(total_iob), 2),
            "cob": round(float(cob), 1),
            "current_bg": float(current_bg),
            "isf_used": round(float(isf), 1),
            "cr_used": round(float(cr), 1),
            "predicted_bg_1h": round(float(predicted_bg_1h), 0),
            "predicted_bg_2h": round(float(predicted_bg_2h), 0),
            "predicted_min_bg": round(float(predicted_min), 0),
            "confidence": round(float(confidence), 2),
            "shadow_mode": self.shadow_mode,
            "cohort": self.cohort,
        }

    def recommend_basal(self, current_bg: float) -> Dict:
        """
        Recommend basal rate adjustment (pump users).

        Simple proportional adjustment:
          If predicted BG > target: increase basal slightly
          If predicted BG < target: decrease basal (or suspend)

        Args:
            current_bg: Current glucose (mg/dL)

        Returns:
            Dict with recommended basal rate and reason
        """
        self.estimator.update(self.current_time)
        params = self.estimator.get_params()
        base_basal = params["basal"]  # U/hr
        basal_confidence = float(params.get("basal_confidence", 0.0))

        # Predict 1 hour ahead with current basal
        trajectory = self.model.predict(
            current_bg, self.current_time,
            horizon_min=60.0, dt=5.0,
        )
        predicted_bg_1h = trajectory[-1][1] if trajectory else current_bg

        # Simple adjustment
        if current_bg < self.safety.low_suspend_bg or predicted_bg_1h < self.safety.predicted_suspend_bg:
            # Low glucose suspend
            rate = 0.0
            reason = "low_glucose_suspend"
        elif current_bg < self.safety.low_reduce_bg or predicted_bg_1h < self.safety.low_reduce_bg:
            # Reduce to 50%
            rate = base_basal * 0.5
            reason = "reduced_low_trend"
        elif current_bg > self.bg_target + 70 and predicted_bg_1h > self.bg_target + 70:
            # Increase by up to 50%
            upward_gate = (
                self.safety.basal_scale_floor
                + (1.0 - self.safety.basal_scale_floor) * basal_confidence
            )
            candidate = base_basal * (1.0 + (self.safety.max_basal_multiplier - 1.0) * upward_gate)
            rate = min(candidate, base_basal + self.safety.max_basal_step_up)
            reason = "increased_high"
        else:
            rate = base_basal
            reason = "normal"

        return {
            "basal_rate": round(float(rate), 2),
            "base_rate": round(float(base_basal), 2),
            "adjustment_reason": reason,
            "current_bg": float(current_bg),
            "predicted_bg_1h": round(float(predicted_bg_1h), 0),
            "shadow_mode": self.shadow_mode,
            "cohort": self.cohort,
        }

    def get_status(self) -> Dict:
        """Get current engine status for display."""
        params = self.estimator.get_params()
        status = {
            "isf": round(params["isf"], 1),
            "cr": round(params["cr"], 1),
            "basal": round(params["basal"], 2),
            "isf_confidence": round(params["isf_confidence"], 2),
            "cr_confidence": round(params["cr_confidence"], 2),
            "basal_confidence": round(params["basal_confidence"], 2),
            "iob": round(self.dosing_iob.get_iob(self.current_time), 2),
            "total_iob": round(self.model.get_iob(self.current_time), 2),
            "cob": round(self.model.get_cob(self.current_time), 1),
            "shadow_mode": self.shadow_mode,
            "estimator_version": self.estimator_version,
            "use_covariates": self.use_covariates,
            "cohort": self.cohort,
            "bg_target": round(self.bg_target, 0),
            "weight_kg": round(self.clinical_priors.weight_kg, 1),
            "param_summary": self.estimator.summary(),
        }

        if self.health_context is not None:
            status["health_context"] = self.health_context.to_dict()

        if "isf_base" in params:
            status["isf_base"] = round(params["isf_base"], 1)
        if "isf_context_multiplier" in params:
            status["isf_context_multiplier"] = round(params["isf_context_multiplier"], 2)
        if "isf_covariate_effects" in params:
            status["isf_covariate_effects"] = params["isf_covariate_effects"]

        return status
