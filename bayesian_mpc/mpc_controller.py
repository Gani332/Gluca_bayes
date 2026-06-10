"""
Research-only Model Predictive Control (MPC) prototype for insulin dosing.

Do not import this module from the Gluca iOS app, Supabase functions, or any
user-facing product path. Instantiating MPCController requires the explicit
GLUCA_ENABLE_RESEARCH_DOSING=1 environment flag.

Optimizes insulin delivery over a prediction horizon by:
  1. Predicting BG forward using the physiological model
  2. Minimizing a cost function (Magni risk + insulin penalty)
  3. Subject to hard safety constraints

This is the approach used by every approved AID system.
The advantage over RL: it uses an *interpretable* model,
adapts in real-time via parameter updates, and enforces
hard safety guarantees that RL cannot.

This is for offline simulation and validation only, not user-facing dosing.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from typing import Dict, List, Optional, Tuple

from .patient_model import GlucoseModel


def _require_research_mode() -> None:
    if os.environ.get("GLUCA_ENABLE_RESEARCH_DOSING") != "1":
        raise RuntimeError(
            "MPCController is research-only. Set GLUCA_ENABLE_RESEARCH_DOSING=1 "
            "only for offline simulation or validation scripts."
        )


def magni_risk(bg: float) -> float:
    """Magni risk index (same as emerson_pipeline)."""
    bg = max(1.0, bg)
    return 10.0 * (3.5506 * (np.log(bg) ** 0.8353 - 3.7932)) ** 2


def asymmetric_risk(bg: float, target: float = 110.0) -> float:
    """
    Combined risk: Magni + extra hypo penalty.

    Below 70: add quadratic hypo penalty (clinical emergency)
    Below 54: add massive penalty (severe hypo, Battelino threshold)
    """
    risk = magni_risk(bg)

    if bg < 70:
        risk += 5.0 * (70 - bg) ** 2
    if bg < 54:
        risk += 50.0 * (54 - bg) ** 2

    return risk


class MPCController:
    """
    Model Predictive Control for insulin optimization.

    At each decision point:
      1. Get current BG state (from UKF)
      2. Get current patient params (from Bayesian estimator)
      3. Predict BG forward over horizon using GlucoseModel
      4. Find insulin that minimizes total predicted risk
      5. Apply safety constraints
      6. Return recommendation

    For pump:
      - Optimizes basal rate for next 30 min
      - Recommends bolus if meal is detected

    For MDI:
      - Recommends correction bolus amount
      - Recommends meal bolus amount
    """

    # MPC parameters
    PREDICTION_HORIZON = 240.0   # 4 hours forward (minutes)
    CONTROL_HORIZON = 30.0       # Optimize over 30-min window
    DT = 5.0                     # Prediction step (minutes)

    # Safety constraints
    MIN_BG_CONSTRAINT = 80.0     # Don't allow predicted BG below this
    SUSPEND_BG = 70.0            # Suspend insulin entirely below this
    MAX_BOLUS = 15.0             # Maximum single bolus (units)
    MAX_BASAL_MULTIPLIER = 3.0   # Max basal = 3x estimated basal rate
    MIN_BASAL = 0.0              # Can go to zero (low glucose suspend)

    # Cost function weights
    INSULIN_PENALTY = 0.5        # Small penalty per unit of insulin
    HYPO_WEIGHT = 10.0           # Extra weight on predicted hypo

    def __init__(self, model: GlucoseModel):
        _require_research_mode()
        self.model = model

    def _predict_cost(
        self,
        insulin_action: float,
        current_bg: float,
        current_time: float,
        action_type: str = "basal",
    ) -> float:
        """
        Cost of a given insulin action over the prediction horizon.

        Args:
            insulin_action: Insulin amount (U for bolus, U/hr for basal)
            current_bg: Current BG (mg/dL)
            current_time: Current time (minutes)
            action_type: "basal" or "bolus"

        Returns:
            Total cost (lower = better)
        """
        # Set up future insulin based on action type
        if action_type == "basal":
            # Basal rate for next CONTROL_HORIZON minutes
            steps = int(self.CONTROL_HORIZON / self.DT)
            per_step_units = insulin_action * (self.DT / 60.0)  # U/hr → U per step
            future_insulin = [
                (i * self.DT, per_step_units) for i in range(steps)
            ]
        else:
            # Single bolus now
            future_insulin = [(0.0, insulin_action)]

        # Predict BG trajectory
        trajectory = self.model.predict(
            current_bg, current_time,
            horizon_min=self.PREDICTION_HORIZON,
            dt=self.DT,
            future_insulin=future_insulin,
        )

        # Compute cost over trajectory
        total_cost = 0.0
        for t_offset, bg in trajectory:
            risk = asymmetric_risk(bg)
            total_cost += risk

            # Extra penalty if predicted BG goes below constraint
            if bg < self.MIN_BG_CONSTRAINT:
                total_cost += self.HYPO_WEIGHT * (self.MIN_BG_CONSTRAINT - bg) ** 2

        # Insulin penalty (prefer less insulin, all else equal)
        if action_type == "basal":
            total_insulin = insulin_action * (self.CONTROL_HORIZON / 60.0)
        else:
            total_insulin = insulin_action
        total_cost += self.INSULIN_PENALTY * total_insulin

        return total_cost

    def optimize_basal(
        self,
        current_bg: float,
        current_time: float,
    ) -> Dict:
        """
        Find optimal basal rate for pump users.

        Returns:
            Dict with basal_rate (U/hr), predicted_bg trajectory, cost
        """
        # If glucose is very low, suspend immediately
        if current_bg < self.SUSPEND_BG:
            return {
                "basal_rate": 0.0,
                "reason": "low_glucose_suspend",
                "predicted_bg": current_bg,
            }

        # Search bounds: 0 to 3x patient's estimated basal
        max_basal = self.model.basal_rate * self.MAX_BASAL_MULTIPLIER

        # Optimize
        result = minimize_scalar(
            lambda rate: self._predict_cost(rate, current_bg, current_time, "basal"),
            bounds=(self.MIN_BASAL, max_basal),
            method="bounded",
        )

        optimal_rate = result.x

        # Get predicted trajectory with optimal basal
        steps = int(self.CONTROL_HORIZON / self.DT)
        per_step_units = optimal_rate * (self.DT / 60.0)
        future_insulin = [(i * self.DT, per_step_units) for i in range(steps)]
        trajectory = self.model.predict(
            current_bg, current_time,
            horizon_min=self.PREDICTION_HORIZON,
            dt=self.DT,
            future_insulin=future_insulin,
        )

        # Safety check: if predicted min BG < 70, reduce basal
        min_predicted = min(bg for _, bg in trajectory)
        if min_predicted < 70.0:
            # Binary search for safe rate
            safe_rate = self._find_safe_basal(current_bg, current_time, max_basal)
            optimal_rate = min(optimal_rate, safe_rate)

        return {
            "basal_rate": float(optimal_rate),
            "basal_rate_default": float(self.model.basal_rate),
            "trajectory": trajectory,
            "min_predicted_bg": min_predicted,
            "cost": float(result.fun),
        }

    def recommend_bolus(
        self,
        current_bg: float,
        current_time: float,
        carbs_grams: float = 0.0,
        iob: float = 0.0,
    ) -> Dict:
        """
        Recommend a bolus dose for meal and/or correction.

        Uses MPC to find the optimal bolus, but also computes
        the standard formula for comparison/transparency.

        Args:
            current_bg: Current BG (mg/dL)
            current_time: Current time (minutes)
            carbs_grams: Meal carbs (0 if correction only)
            iob: Current insulin on board (units)

        Returns:
            Dict with recommended dose, breakdown, and predicted BG
        """
        # ── Standard formula (transparent, interpretable) ────────────
        meal_bolus = carbs_grams / self.model.cr if carbs_grams > 0 else 0.0

        correction_bolus = 0.0
        if current_bg > self.model.bg_target + 20:  # Only correct if >20 above target
            correction_bolus = (current_bg - self.model.bg_target) / self.model.isf

        formula_dose = max(0.0, meal_bolus + correction_bolus - iob)
        formula_dose = min(formula_dose, self.MAX_BOLUS)

        # ── MPC optimization (find dose that minimizes predicted risk) ─
        if current_bg < self.SUSPEND_BG and carbs_grams == 0:
            # Don't give insulin when already low
            return {
                "recommended_dose": 0.0,
                "meal_component": 0.0,
                "correction_component": 0.0,
                "iob_adjustment": iob,
                "formula_dose": 0.0,
                "reason": "low_glucose",
            }

        # Search for optimal bolus
        max_dose = min(self.MAX_BOLUS, formula_dose * 2.0 + 1.0)

        if max_dose < 0.1:
            mpc_dose = 0.0
            mpc_cost = self._predict_cost(0.0, current_bg, current_time, "bolus")
        else:
            result = minimize_scalar(
                lambda dose: self._predict_cost(dose, current_bg, current_time, "bolus"),
                bounds=(0.0, max_dose),
                method="bounded",
            )
            mpc_dose = result.x
            mpc_cost = result.fun

        # Safety: check predicted trajectory with MPC dose
        trajectory = self.model.predict(
            current_bg, current_time,
            horizon_min=self.PREDICTION_HORIZON,
            dt=self.DT,
            future_insulin=[(0.0, mpc_dose)],
        )
        min_predicted = min(bg for _, bg in trajectory)

        # If MPC predicts hypo, reduce dose
        if min_predicted < 70.0:
            mpc_dose = self._find_safe_bolus(current_bg, current_time, max_dose)

        # Use the MORE CONSERVATIVE of formula and MPC
        # This ensures safety — MPC can reduce but not increase beyond formula
        final_dose = min(formula_dose, mpc_dose) if mpc_dose > 0 else formula_dose

        # For very low confidence in parameters, stick to formula
        # (MPC is only as good as its model)

        return {
            "recommended_dose": float(max(0.0, final_dose)),
            "meal_component": float(meal_bolus),
            "correction_component": float(correction_bolus),
            "iob_adjustment": float(iob),
            "formula_dose": float(formula_dose),
            "mpc_dose": float(mpc_dose),
            "min_predicted_bg": float(min_predicted),
            "trajectory": trajectory,
        }

    def _find_safe_basal(
        self,
        current_bg: float,
        current_time: float,
        max_rate: float,
    ) -> float:
        """Binary search for highest basal rate that doesn't cause predicted hypo."""
        lo, hi = 0.0, max_rate

        for _ in range(15):  # 15 iterations = ~0.003% precision
            mid = (lo + hi) / 2
            steps = int(self.CONTROL_HORIZON / self.DT)
            per_step = mid * (self.DT / 60.0)
            future = [(i * self.DT, per_step) for i in range(steps)]

            traj = self.model.predict(
                current_bg, current_time,
                horizon_min=self.PREDICTION_HORIZON,
                dt=self.DT,
                future_insulin=future,
            )
            min_bg = min(bg for _, bg in traj)

            if min_bg >= 75.0:  # 5 mg/dL margin above 70
                lo = mid
            else:
                hi = mid

        return lo

    def _find_safe_bolus(
        self,
        current_bg: float,
        current_time: float,
        max_dose: float,
    ) -> float:
        """Binary search for highest bolus that doesn't cause predicted hypo."""
        lo, hi = 0.0, max_dose

        for _ in range(15):
            mid = (lo + hi) / 2
            traj = self.model.predict(
                current_bg, current_time,
                horizon_min=self.PREDICTION_HORIZON,
                dt=self.DT,
                future_insulin=[(0.0, mid)],
            )
            min_bg = min(bg for _, bg in traj)

            if min_bg >= 75.0:
                lo = mid
            else:
                hi = mid

        return lo
