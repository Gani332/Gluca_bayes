"""
Physiological glucose model for MPC prediction.

Uses a simplified Bergman minimal model with:
  - Subcutaneous insulin pharmacokinetics (2-compartment)
  - Gut carbohydrate absorption (2-compartment)
  - Glucose dynamics driven by insulin action and meal absorption

This model is what MPC uses to predict BG 2-4 hours ahead.
Patient-specific parameters (ISF, CR, basal) are estimated
online by the Bayesian estimator.

The model is deliberately simple — complex models need more
parameters and are harder to identify from limited data.
What matters is that it captures the right *shape* of insulin
and meal responses so MPC can make reasonable decisions.

References:
  Bergman et al. (1981) "Minimal Model of Glucose Kinetics"
  Dalla Man et al. (2007) meal absorption model
  Hovorka et al. (2004) subcutaneous insulin kinetics
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════════════════════
# Insulin Pharmacokinetics (Subcutaneous → Plasma → Action)
# ═══════════════════════════════════════════════════════════════════════════════

class InsulinPK:
    """
    2-compartment subcutaneous insulin model.

    Models the delay between injection and blood glucose effect:
      SC1 → SC2 → Plasma → Action on glucose

    Rapid-acting (lispro/aspart): peak ~60 min, duration ~4 hr
    This gives the "insulin on board" (IOB) and "insulin activity" curves.
    """

    # Rapid-acting insulin parameters (lispro/aspart)
    TAU_SC = 55.0          # SC absorption time constant (min)
    TAU_ACTION = 70.0      # Insulin action time constant (min)
    PEAK_TIME = 60.0       # Approximate peak activity (min)
    DURATION = 300.0       # Total duration of action (min, ~5 hr)

    def __init__(self):
        self.doses = []  # list of (time_min, units)

    def add_dose(self, time_min: float, units: float):
        """Record an insulin dose (bolus or basal increment)."""
        if units > 0:
            self.doses.append((time_min, units))

    def _single_dose_activity(self, dt: float, units: float) -> float:
        """
        Insulin activity rate (U/min) at time dt after injection.

        Uses a two-exponential model:
          activity(t) = units * (exp(-t/tau_sc) - exp(-t/tau_action)) * scale
        """
        if dt < 0 or dt > self.DURATION:
            return 0.0

        # Normalized activity curve (area under curve = units)
        a = np.exp(-dt / self.TAU_SC)
        b = np.exp(-dt / self.TAU_ACTION)
        # Scale so integral over 0..inf = units
        scale = 1.0 / (self.TAU_SC - self.TAU_ACTION)
        return units * (a - b) * scale

    def _single_dose_iob(self, dt: float, units: float) -> float:
        """Insulin on board at time dt after injection."""
        if dt < 0:
            return units
        if dt > self.DURATION:
            return 0.0

        # IOB = integral of activity from dt to infinity
        remaining_sc = np.exp(-dt / self.TAU_SC) * self.TAU_SC
        remaining_action = np.exp(-dt / self.TAU_ACTION) * self.TAU_ACTION
        scale = 1.0 / (self.TAU_SC - self.TAU_ACTION)
        return units * (remaining_sc - remaining_action) * scale

    def get_activity(self, current_time: float) -> float:
        """Total insulin activity rate (U/min) from all active doses."""
        total = 0.0
        for dose_time, units in self.doses:
            dt = current_time - dose_time
            total += self._single_dose_activity(dt, units)
        return total

    def get_iob(self, current_time: float) -> float:
        """Total insulin on board (U) from all active doses."""
        total = 0.0
        for dose_time, units in self.doses:
            dt = current_time - dose_time
            total += self._single_dose_iob(dt, units)
        return total

    def cleanup(self, current_time: float):
        """Remove expired doses to prevent memory growth."""
        self.doses = [(t, u) for t, u in self.doses
                      if current_time - t < self.DURATION]

    def reset(self):
        self.doses = []


# ═══════════════════════════════════════════════════════════════════════════════
# Carbohydrate Absorption (Gut → Glucose Appearance)
# ═══════════════════════════════════════════════════════════════════════════════

class CarbAbsorption:
    """
    2-compartment carbohydrate absorption model.

    Models delay between eating and glucose appearance in blood:
      Stomach → Gut → Plasma glucose

    Based on Dalla Man et al. (2007).
    """

    TAU_STOMACH = 20.0     # Gastric emptying time constant (min)
    TAU_GUT = 30.0         # Intestinal absorption time constant (min)
    BIO_AVAILABILITY = 0.8  # Fraction of carbs that become glucose
    DURATION = 300.0        # Total absorption duration (min, ~5 hr)

    def __init__(self):
        self.meals = []  # list of (time_min, grams_carbs)

    def add_meal(self, time_min: float, carbs_grams: float):
        if carbs_grams > 0:
            self.meals.append((time_min, carbs_grams))

    def _single_meal_rate(self, dt: float, carbs: float) -> float:
        """
        Glucose appearance rate (mg/min) at time dt after meal.

        Rate = carbs * bio * scale * (exp(-dt/tau_s) - exp(-dt/tau_g))
        """
        if dt < 0 or dt > self.DURATION:
            return 0.0

        a = np.exp(-dt / self.TAU_STOMACH)
        b = np.exp(-dt / self.TAU_GUT)
        # Convert grams carbs → mg glucose (1g carb ≈ 1000mg glucose)
        # Scale so integral = carbs * bio * 1000 mg
        scale = 1000.0 * self.BIO_AVAILABILITY / (self.TAU_STOMACH - self.TAU_GUT)
        return carbs * (a - b) * scale

    def _single_meal_cob(self, dt: float, carbs: float) -> float:
        """Carbs on board at time dt after meal."""
        if dt < 0:
            return carbs
        if dt > self.DURATION:
            return 0.0

        remaining_s = np.exp(-dt / self.TAU_STOMACH) * self.TAU_STOMACH
        remaining_g = np.exp(-dt / self.TAU_GUT) * self.TAU_GUT
        scale = self.BIO_AVAILABILITY / (self.TAU_STOMACH - self.TAU_GUT)
        return carbs * (remaining_s - remaining_g) * scale

    def get_rate(self, current_time: float) -> float:
        """Total glucose appearance rate (mg/min) from all active meals."""
        total = 0.0
        for meal_time, carbs in self.meals:
            dt = current_time - meal_time
            total += self._single_meal_rate(dt, carbs)
        return total

    def get_cob(self, current_time: float) -> float:
        """Total carbs on board (grams) from all active meals."""
        total = 0.0
        for meal_time, carbs in self.meals:
            dt = current_time - meal_time
            total += self._single_meal_cob(dt, carbs)
        return total

    def cleanup(self, current_time: float):
        self.meals = [(t, c) for t, c in self.meals
                      if current_time - t < self.DURATION]

    def reset(self):
        self.meals = []


# ═══════════════════════════════════════════════════════════════════════════════
# Glucose Dynamics Model
# ═══════════════════════════════════════════════════════════════════════════════

class GlucoseModel:
    """
    Simplified glucose dynamics for MPC prediction.

    dBG/dt = -ISF * insulin_activity + carb_absorption/Vd - p1*(BG - BG_basal)

    Where:
      - ISF: insulin sensitivity factor (mg/dL per U of activity)
      - insulin_activity: from InsulinPK (U/min currently active)
      - carb_absorption: from CarbAbsorption (mg/min glucose appearing)
      - Vd: volume of distribution (~body_weight * 0.2 L, for concentration→mg/dL)
      - p1: glucose effectiveness (rate BG returns to basal without insulin)
      - BG_basal: fasting glucose target (what BG settles to with correct basal)

    Patient-specific parameters are updated by the Bayesian estimator:
      - isf (mg/dL per U)
      - cr (grams carb per U)
      - basal_rate (U/hr)
      - bg_target (mg/dL)
    """

    # Population defaults
    DEFAULT_ISF = 50.0        # mg/dL per unit (typical adult)
    DEFAULT_CR = 10.0         # grams per unit (typical adult)
    DEFAULT_BASAL = 1.0       # U/hr (typical adult)
    DEFAULT_BG_TARGET = 110.0 # mg/dL
    DEFAULT_BODY_WEIGHT = 70.0  # kg

    # Glucose effectiveness: BG returns to target at ~1% per minute
    P1 = 0.01  # 1/min

    def __init__(
        self,
        isf: float = DEFAULT_ISF,
        cr: float = DEFAULT_CR,
        basal_rate: float = DEFAULT_BASAL,
        bg_target: float = DEFAULT_BG_TARGET,
        body_weight: float = DEFAULT_BODY_WEIGHT,
    ):
        self.isf = isf
        self.cr = cr
        self.basal_rate = basal_rate
        self.bg_target = bg_target
        self.body_weight = body_weight

        # Volume of distribution for glucose (dL)
        # ~16% of body weight in liters, convert to dL
        self.vd = body_weight * 0.16 * 10  # dL

        self.insulin_pk = InsulinPK()
        self.carb_abs = CarbAbsorption()

    def update_params(self, isf: float = None, cr: float = None,
                      basal_rate: float = None, bg_target: float = None):
        """Update patient-specific parameters from Bayesian estimator."""
        if isf is not None:
            self.isf = isf
        if cr is not None:
            self.cr = cr
        if basal_rate is not None:
            self.basal_rate = basal_rate
        if bg_target is not None:
            self.bg_target = bg_target

    def record_insulin(self, time_min: float, units: float):
        """Record an insulin dose."""
        self.insulin_pk.add_dose(time_min, units)

    def record_meal(self, time_min: float, carbs_grams: float):
        """Record a meal."""
        self.carb_abs.add_meal(time_min, carbs_grams)

    def get_iob(self, time_min: float) -> float:
        return self.insulin_pk.get_iob(time_min)

    def get_cob(self, time_min: float) -> float:
        return self.carb_abs.get_cob(time_min)

    def dBG_dt(self, bg: float, time_min: float) -> float:
        """
        Rate of BG change at given time.

        dBG/dt = meal_effect - insulin_effect - glucose_effectiveness
        """
        # Insulin effect: activity (U/min) * ISF (mg/dL per U)
        insulin_activity = self.insulin_pk.get_activity(time_min)
        insulin_effect = self.isf * insulin_activity  # mg/dL per min

        # Meal effect: glucose appearance (mg/min) / volume (dL) → mg/dL per min
        meal_rate = self.carb_abs.get_rate(time_min)
        meal_effect = meal_rate / self.vd  # mg/dL per min

        # Glucose effectiveness: mean-reversion to target
        effectiveness = self.P1 * (bg - self.bg_target)

        return meal_effect - insulin_effect - effectiveness

    def predict(
        self,
        current_bg: float,
        current_time: float,
        horizon_min: float = 240.0,
        dt: float = 5.0,
        future_insulin: list = None,
        future_meals: list = None,
    ) -> list:
        """
        Predict BG forward from current state.

        Args:
            current_bg: Current blood glucose (mg/dL)
            current_time: Current time (minutes)
            horizon_min: Prediction horizon (minutes)
            dt: Step size (minutes)
            future_insulin: List of (time_offset_min, units) for planned insulin
            future_meals: List of (time_offset_min, carbs_grams) for expected meals

        Returns:
            List of (time_offset, predicted_bg) tuples
        """
        # Add planned future events (temporary)
        if future_insulin:
            for offset, units in future_insulin:
                self.insulin_pk.add_dose(current_time + offset, units)
        if future_meals:
            for offset, carbs in future_meals:
                self.carb_abs.add_meal(current_time + offset, carbs)

        # Forward Euler integration
        bg = current_bg
        trajectory = [(0.0, bg)]

        t = current_time
        for step in range(int(horizon_min / dt)):
            dbg = self.dBG_dt(bg, t)
            bg += dbg * dt
            bg = np.clip(bg, 20.0, 600.0)
            t += dt
            trajectory.append(((step + 1) * dt, bg))

        # Remove temporary future events
        if future_insulin:
            for offset, units in future_insulin:
                ft = current_time + offset
                self.insulin_pk.doses = [
                    (t, u) for t, u in self.insulin_pk.doses
                    if not (abs(t - ft) < 0.1 and abs(u - units) < 0.01)
                ]
        if future_meals:
            for offset, carbs in future_meals:
                ft = current_time + offset
                self.carb_abs.meals = [
                    (t, c) for t, c in self.carb_abs.meals
                    if not (abs(t - ft) < 0.1 and abs(c - carbs) < 0.01)
                ]

        return trajectory

    def cleanup(self, current_time: float):
        """Remove expired doses/meals."""
        self.insulin_pk.cleanup(current_time)
        self.carb_abs.cleanup(current_time)

    def reset(self):
        self.insulin_pk.reset()
        self.carb_abs.reset()
