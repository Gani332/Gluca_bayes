"""
Log-linear Bayesian parameter estimator with optional covariates.

Upgrade from bayesian_estimator.py:
  - Parameters modeled in log-space (ISF, CR, basal are always positive)
  - Covariates (exercise, sleep, cycle) are multiplicative in natural space
  - Apple Health data is optional — system degrades gracefully without it

Math:

  log(ISF_t) = theta_0 + theta_1 * x_steps + theta_2 * x_sleep + ...
  ISF_t = exp(log_ISF_t)

  Priors: theta ~ Normal(mu_prior, Sigma_prior)
  Observation: y = log(observed_ISF) ~ Normal(X @ theta, sigma_obs^2)
  Update: conjugate Normal-Normal (Bayesian linear regression)

  Without covariates: just theta_0 (reduces to the original estimator)
  With covariates: learns patient-specific sensitivity to each factor

The key property: this is STILL exact Bayesian inference.
No approximations, no samplers, no optimizers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .clinical_priors import normalize_cohort


# ═══════════════════════════════════════════════════════════════════════════════
# Log-Linear Parameter with Bayesian Regression
# ═══════════════════════════════════════════════════════════════════════════════

class LogLinearParam:
    """
    A positive parameter estimated via Bayesian linear regression in log-space.

    log(param) = theta @ features
    param = exp(theta @ features)

    Features:
      [0]: intercept (always 1.0) — the base parameter value
      [1:]: optional covariates (exercise, sleep, etc.)

    Without covariates, features = [1.0] and this reduces to a
    scalar log-normal estimate — mathematically equivalent to the
    original BayesianParam but with guaranteed positivity.
    """

    def __init__(
        self,
        name: str,
        prior_value: float,
        prior_log_std: float = 0.5,
        covariate_names: List[str] = None,
        covariate_prior_std: float = 0.15,
        min_val: float = 1.0,
        max_val: float = 500.0,
    ):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

        # Feature names: [intercept, ...covariates]
        self.covariate_names = covariate_names or []
        self.dim = 1 + len(self.covariate_names)

        # Prior on theta (log-space coefficients)
        self.theta_mean = np.zeros(self.dim)
        self.theta_mean[0] = np.log(max(1e-6, prior_value))  # intercept = log(prior)

        self.theta_cov = np.eye(self.dim)
        self.theta_cov[0, 0] = prior_log_std ** 2  # intercept uncertainty
        for i in range(1, self.dim):
            self.theta_cov[i, i] = covariate_prior_std ** 2  # covariate effect uncertainty

        # Store prior for confidence computation
        self._prior_theta_cov = self.theta_cov.copy()

        self.observations = []  # (time, observed_value, features)

    def _build_features(self, covariates: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Build feature vector [1.0, cov1, cov2, ...]."""
        x = np.zeros(self.dim)
        x[0] = 1.0  # intercept
        if covariates and self.covariate_names:
            for i, name in enumerate(self.covariate_names):
                if name in covariates:
                    x[i + 1] = covariates[name]
        return x

    def predict(self, covariates: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Predict parameter value with uncertainty.

        Returns:
            (point_estimate, std_value) in natural space
        """
        x = self._build_features(covariates)

        # Predictive mean and variance in log-space
        log_mean = x @ self.theta_mean
        log_var = x @ self.theta_cov @ x

        # For dosing/control we use the posterior median exp(log_mean), not
        # the arithmetic mean exp(log_mean + 0.5 * log_var). The arithmetic
        # mean is upward-biased under high uncertainty and causes systematic
        # under-dosing when the parameter is an insulin divisor (ISF, CR).
        point = np.exp(log_mean)

        # Delta-method approximation of natural-space standard deviation.
        std = point * np.sqrt(max(log_var, 0.0))

        point = np.clip(point, self.min_val, self.max_val)
        return float(point), float(std)

    def update(self, observed_value: float, obs_log_std: float = 0.3,
               covariates: Optional[Dict[str, float]] = None,
               timestamp: float = 0.0):
        """
        Bayesian update with a new observation.

        Uses conjugate Normal-Normal update in log-space:
          Prior: theta ~ N(mu, Sigma)
          Observation: log(y) ~ N(x @ theta, sigma_obs^2)
          Posterior: theta ~ N(mu_new, Sigma_new)
        """
        if observed_value <= 0 or not np.isfinite(observed_value):
            return
        if not (self.min_val <= observed_value <= self.max_val):
            return

        y = np.log(observed_value)
        x = self._build_features(covariates)
        sigma2 = obs_log_std ** 2

        # Kalman-style update for Bayesian linear regression
        # S = x' Sigma x + sigma_obs^2
        S = x @ self.theta_cov @ x + sigma2
        # K = Sigma x / S (Kalman gain)
        K = (self.theta_cov @ x) / S
        # Innovation
        innovation = y - x @ self.theta_mean

        # Update
        self.theta_mean = self.theta_mean + K * innovation
        self.theta_cov = self.theta_cov - np.outer(K, K) * S

        # Ensure covariance stays symmetric and positive definite
        self.theta_cov = (self.theta_cov + self.theta_cov.T) / 2
        self.theta_cov += np.eye(self.dim) * 1e-8

        self.observations.append((timestamp, observed_value, x.copy()))

    def decay_toward_prior(self, rate: float = 0.001):
        """Widen posterior slightly to allow re-adaptation over time."""
        self.theta_cov = (1 + rate) * self.theta_cov
        # Cap at prior uncertainty
        for i in range(self.dim):
            self.theta_cov[i, i] = min(
                self.theta_cov[i, i],
                self._prior_theta_cov[i, i],
            )

    @property
    def confidence(self) -> float:
        """0-1 confidence. Based on reduction in intercept uncertainty."""
        prior_var = self._prior_theta_cov[0, 0]
        post_var = self.theta_cov[0, 0]
        return float(max(0, min(1, 1 - post_var / prior_var)))

    def covariate_effects(self) -> Dict[str, float]:
        """Return learned covariate multipliers (exp of coefficients)."""
        effects = {}
        for i, name in enumerate(self.covariate_names):
            # Coefficient in log-space → multiplier in natural space
            effects[name] = float(np.exp(self.theta_mean[i + 1]))
            effects[f"{name}_std"] = float(np.sqrt(self.theta_cov[i + 1, i + 1]))
        return effects


# ═══════════════════════════════════════════════════════════════════════════════
# Apple Health Context (Optional)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthContext:
    """
    Normalized health covariates from Apple Health.

    All values are centered around 0 (no effect) so that
    missing data = 0 = no adjustment. This makes the system
    degrade gracefully when Apple Health data is unavailable.
    """
    # Activity: normalized so 0 = average day, +1 = very active, -1 = sedentary
    activity_level: float = 0.0
    # Sleep: 0 = normal, +1 = great sleep, -1 = poor sleep
    sleep_quality: float = 0.0
    # Stress proxy (from HRV): 0 = normal, +1 = relaxed, -1 = stressed
    stress_level: float = 0.0
    # Cycle phase: 0 = not tracked or follicular, -0.3 = luteal (typical ISF reduction)
    cycle_factor: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "activity": float(self.activity_level),
            "sleep": float(self.sleep_quality),
            "stress": float(self.stress_level),
            "cycle": float(self.cycle_factor),
        }

    @staticmethod
    def from_apple_health(
        steps_2h: int = 0,
        steps_daily: int = 0,
        workout_min_today: float = 0,
        sleep_hours: float = 7.5,
        hrv_ms: float = 50,
        cycle_phase: str = None,
    ) -> 'HealthContext':
        """
        Convert raw Apple Health data to normalized covariates.

        All outputs centered at 0 so missing data has no effect.
        """
        # Activity: based on recent steps and workouts
        # Population average ~3000 steps in 2 hours, ~8000 daily
        activity = 0.0
        if steps_2h > 0:
            activity += (steps_2h - 3000) / 3000  # 6000 steps → +1.0
        if workout_min_today > 0:
            activity += min(workout_min_today / 60, 1.5)  # 60 min → +1.0
        activity = np.clip(activity, -1.5, 2.0)

        # Sleep: centered on 7.5 hours
        sleep = 0.0
        if sleep_hours > 0:
            sleep = (sleep_hours - 7.5) / 2.0  # 9.5h → +1.0, 5.5h → -1.0
            sleep = np.clip(sleep, -1.5, 1.0)

        # Stress from HRV (higher HRV = less stress)
        # Population average ~50ms, more is better
        stress = 0.0
        if hrv_ms > 0:
            stress = (hrv_ms - 50) / 30  # 80ms → +1.0 (relaxed), 20ms → -1.0 (stressed)
            stress = np.clip(stress, -1.5, 1.5)

        # Cycle phase
        cycle = 0.0
        if cycle_phase == "luteal":
            cycle = -0.3  # Typical ISF reduction
        elif cycle_phase == "menstrual":
            cycle = -0.15

        return HealthContext(
            activity_level=float(activity),
            sleep_quality=float(sleep),
            stress_level=float(stress),
            cycle_factor=float(cycle),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Bayesian Estimator
# ═══════════════════════════════════════════════════════════════════════════════

COVARIATE_NAMES = ["activity", "sleep", "stress", "cycle"]


class BayesianEstimatorV2:
    """
    Log-linear Bayesian estimator with optional Apple Health covariates.

    Drop-in replacement for BayesianEstimator. When no covariates are
    provided (health_context=None), it behaves identically to v1 but
    with log-normal parameter distributions (guaranteed positive).

    When Apple Health data is available, it learns patient-specific
    sensitivity to exercise, sleep, stress, and menstrual cycle.
    """

    def __init__(
        self,
        isf_prior: float = 50.0,
        cr_prior: float = 10.0,
        basal_prior: float = 1.0,
        bg_target: float = 110.0,
        use_covariates: bool = True,
        cohort: str = "adult",
    ):
        self.bg_target = bg_target
        self.use_covariates = use_covariates
        self.cohort = normalize_cohort(cohort)

        covs = COVARIATE_NAMES if use_covariates else []

        self.isf = LogLinearParam(
            "ISF", isf_prior, prior_log_std=0.7,
            covariate_names=covs, covariate_prior_std=0.1,
            min_val=10.0, max_val=200.0,
        )
        self.cr = LogLinearParam(
            "CR", cr_prior, prior_log_std=0.6,
            covariate_names=[],  # CR less affected by activity/sleep
            min_val=2.0, max_val=50.0,
        )
        self.basal = LogLinearParam(
            "Basal", basal_prior, prior_log_std=0.5,
            covariate_names=[],  # Basal changes slowly
            min_val=0.05, max_val=5.0,
        )

        # Event history
        self._bg_readings = []
        self._insulin_events = []
        self._meal_events = []

        # Current health context (optional)
        self._health_context = None

    def _basal_config(self) -> Dict[str, float]:
        if self.cohort == "child":
            return {
                "fasting_window_min": 360.0,
                "meal_exclusion_min": 90.0,
                "min_gap_min": 18.0 * 60.0,
                "max_bg_change": 100.0,
                "max_adjust_fraction": 0.15,
                "obs_log_std": 0.65,
            }
        if self.cohort == "adolescent":
            return {
                "fasting_window_min": 300.0,
                "meal_exclusion_min": 75.0,
                "min_gap_min": 12.0 * 60.0,
                "max_bg_change": 110.0,
                "max_adjust_fraction": 0.20,
                "obs_log_std": 0.55,
            }
        return {
            "fasting_window_min": 240.0,
            "meal_exclusion_min": 60.0,
            "min_gap_min": 8.0 * 60.0,
            "max_bg_change": 120.0,
            "max_adjust_fraction": 0.30,
            "obs_log_std": 0.45,
        }

    def set_health_context(self, context: Optional[HealthContext]):
        """Set current Apple Health context. None = no data available."""
        self._health_context = context

    def _get_covariates(self) -> Optional[Dict[str, float]]:
        """Get current covariate dict, or None if unavailable."""
        if self._health_context and self.use_covariates:
            return self._health_context.to_dict()
        return None

    # ── Recording events (same API as v1) ────────────────────────────

    def record_bg(self, time_min: float, bg: float):
        self._bg_readings.append((time_min, bg))

    def record_insulin(self, time_min: float, units: float,
                       insulin_type: str = "bolus"):
        self._insulin_events.append((time_min, units, insulin_type))

    def record_meal(self, time_min: float, carbs: float):
        self._meal_events.append((time_min, carbs))

    def _meal_cluster(self, meal_time: float, max_gap_min: float = 30.0) -> Tuple[float, float, float]:
        """
        Return the connected meal cluster containing meal_time.

        Simulator meals and real app logs can arrive as several adjacent carb
        entries. Treating each fragment as an independent CR observation lets a
        small trailing fragment inherit the whole bolus and corrupts learning.
        """
        meals = sorted((t, c) for t, c in self._meal_events if c > 0)
        if not meals:
            return meal_time, meal_time, 0.0

        idx = min(range(len(meals)), key=lambda i: abs(meals[i][0] - meal_time))
        if abs(meals[idx][0] - meal_time) > 1.0:
            return meal_time, meal_time, 0.0

        start = idx
        while start > 0 and meals[start][0] - meals[start - 1][0] <= max_gap_min:
            start -= 1

        end = idx
        while end + 1 < len(meals) and meals[end + 1][0] - meals[end][0] <= max_gap_min:
            end += 1

        cluster = meals[start:end + 1]
        return cluster[0][0], cluster[-1][0], sum(c for _, c in cluster)

    def _has_meal_between(
        self,
        start_min: float,
        end_min: float,
        min_carbs: float = 5.0,
        exclude_start: Optional[float] = None,
        exclude_end: Optional[float] = None,
    ) -> bool:
        for t, carbs in self._meal_events:
            if carbs <= min_carbs or not (start_min <= t <= end_min):
                continue
            if exclude_start is not None and exclude_end is not None and exclude_start <= t <= exclude_end:
                continue
            return True
        return False

    def _has_bolus_between(
        self,
        start_min: float,
        end_min: float,
        min_units: float = 0.3,
        exclude_start: Optional[float] = None,
        exclude_end: Optional[float] = None,
    ) -> bool:
        for t, units, insulin_type in self._insulin_events:
            if insulin_type != "bolus" or units < min_units or not (start_min <= t <= end_min):
                continue
            if exclude_start is not None and exclude_end is not None and exclude_start <= t <= exclude_end:
                continue
            return True
        return False

    # ── Parameter updates ────────────────────────────────────────────

    def try_update_isf(self, current_time: float) -> Optional[float]:
        """Estimate ISF from correction and meal events."""
        RESPONSE_WINDOW = 180.0
        covariates = self._get_covariates()
        latest = None

        for ins_time, ins_units, ins_type in reversed(self._insulin_events):
            if ins_type != "bolus" or ins_units < 0.5:
                continue
            if current_time - ins_time < RESPONSE_WINDOW:
                continue
            if any(abs(t - ins_time) < 1.0 for t, _, _ in self.isf.observations):
                continue

            bg_before = self._get_bg_near(ins_time, 15.0)
            bg_after = self._get_bg_near(ins_time + RESPONSE_WINDOW, 15.0)
            if bg_before is None or bg_after is None:
                continue

            # Only learn ISF from clean correction windows. Meal-residual ISF
            # estimates are too confounded for closed-loop controller tuning.
            if self._has_meal_between(ins_time - 60.0, ins_time + RESPONSE_WINDOW):
                continue
            if self._has_bolus_between(
                ins_time - 60.0,
                ins_time + RESPONSE_WINDOW,
                exclude_start=ins_time - 1.0,
                exclude_end=ins_time + 1.0,
            ):
                continue

            bg_drop = bg_before - bg_after
            observed_isf = bg_drop / ins_units
            obs_log_std = 0.45  # Still noisy in the wild

            if 5.0 < observed_isf < 200.0:
                self.isf.update(observed_isf, obs_log_std, covariates, ins_time)
                latest = observed_isf

        return latest

    def try_update_cr(self, current_time: float) -> Optional[float]:
        """Estimate CR from meal events."""
        RESPONSE_WINDOW = 240.0
        BG_RETURN_THRESHOLD = 30.0
        latest = None

        for meal_time, carbs in reversed(self._meal_events):
            cluster_start, cluster_end, cluster_carbs = self._meal_cluster(meal_time)
            if abs(meal_time - cluster_start) > 1.0:
                continue
            if cluster_carbs < 10 or current_time - cluster_start < RESPONSE_WINDOW:
                continue
            if any(abs(t - cluster_start) < 1.0 for t, _, _ in self.cr.observations):
                continue

            bolus_start = cluster_start - 15.0
            bolus_end = cluster_end + 30.0
            window_end = cluster_start + RESPONSE_WINDOW
            if self._has_meal_between(
                cluster_start - 60.0,
                window_end,
                exclude_start=cluster_start,
                exclude_end=cluster_end,
            ):
                continue
            if self._has_bolus_between(
                cluster_start - 60.0,
                window_end,
                exclude_start=bolus_start,
                exclude_end=bolus_end,
            ):
                continue

            bolus = sum(u for it, u, tp in self._insulin_events
                        if tp == "bolus" and bolus_start <= it <= bolus_end)
            if bolus < 0.5:
                continue

            bg_before = self._get_bg_near(cluster_start, 15.0)
            bg_after = self._get_bg_near(cluster_start + RESPONSE_WINDOW, 15.0)
            if bg_before is None or bg_after is None:
                continue

            isf_est, _ = self.isf.predict(self._get_covariates())
            correction_component = max(0, (bg_before - self.bg_target) / isf_est) \
                                   if bg_before > self.bg_target + 20 else 0
            meal_component = bolus - correction_component

            if meal_component < 0.3:
                continue

            observed_cr = cluster_carbs / meal_component

            # Adjust for outcome relative to target, not relative to pre-meal
            # BG. A high pre-meal glucose that returns to target should not
            # teach the learner that the carb ratio is weaker.
            outcome_error = bg_after - self.bg_target
            if abs(outcome_error) > 10:
                insulin_error = outcome_error / isf_est
                corrected_meal = meal_component + insulin_error
                if corrected_meal > 0.3:
                    observed_cr = cluster_carbs / corrected_meal

            if 2.0 < observed_cr < 50.0:
                self.cr.update(observed_cr, 0.45, None, cluster_start)
                latest = observed_cr

        return latest

    def try_update_basal(self, current_time: float) -> Optional[float]:
        """Estimate basal from fasting periods."""
        config = self._basal_config()
        FASTING_WINDOW = config["fasting_window_min"]
        MEAL_EXCLUSION = config["meal_exclusion_min"]
        MIN_GAP = config["min_gap_min"]
        latest = None

        for i in range(len(self._bg_readings) - 1):
            t_start, bg_start = self._bg_readings[i]
            t_end = t_start + FASTING_WINDOW
            if t_end > current_time:
                continue
            if any(abs(t - t_start) < 1.0 for t, _, _ in self.basal.observations):
                continue
            if self.basal.observations:
                last_basal_obs_time = self.basal.observations[-1][0]
                if t_start - last_basal_obs_time < MIN_GAP:
                    continue

            active_meal_lookback = max(MEAL_EXCLUSION, 180.0)
            active_bolus_lookback = 300.0
            has_meal = any(t_start - active_meal_lookback < mt < t_end for mt, _ in self._meal_events)
            has_bolus = any(t_start - active_bolus_lookback < it < t_end and tp == "bolus"
                           for it, _, tp in self._insulin_events)
            if has_meal or has_bolus:
                continue

            bg_end = self._get_bg_near(t_end, 15.0)
            if bg_end is None:
                continue

            bg_change = bg_end - bg_start
            if abs(bg_change) > config["max_bg_change"]:
                continue

            basal_total = sum(u for it, u, tp in self._insulin_events
                              if t_start <= it < t_end and tp == "basal")
            basal_rate = basal_total / (FASTING_WINDOW / 60.0)
            if basal_rate < 0.01:
                continue

            isf_est, _ = self.isf.predict(self._get_covariates())
            drift_rate = (bg_end - bg_start) / (FASTING_WINDOW / 60.0)
            basal_correction = drift_rate / isf_est
            max_adjust = max(0.05, basal_rate * config["max_adjust_fraction"])
            basal_correction = float(np.clip(basal_correction, -max_adjust, max_adjust))
            observed_basal = basal_rate + basal_correction

            if 0.05 < observed_basal < 5.0:
                self.basal.update(observed_basal, config["obs_log_std"], None, t_start)
                latest = observed_basal

        return latest

    def update(self, current_time: float) -> Dict[str, Optional[float]]:
        """Run all parameter updates."""
        self.isf.decay_toward_prior(0.0005)
        self.cr.decay_toward_prior(0.0005)
        self.basal.decay_toward_prior(0.0005)

        return {
            "isf": self.try_update_isf(current_time),
            "cr": self.try_update_cr(current_time),
            "basal": self.try_update_basal(current_time),
        }

    # ── Query interface (compatible with v1) ─────────────────────────

    def get_params(self) -> Dict:
        """Get current parameter estimates."""
        covs = self._get_covariates()
        isf_mean, isf_std = self.isf.predict(covs)
        cr_mean, cr_std = self.cr.predict()
        basal_mean, basal_std = self.basal.predict()

        result = {
            "isf": isf_mean,
            "isf_sigma": isf_std,
            "isf_confidence": self.isf.confidence,
            "cr": cr_mean,
            "cr_sigma": cr_std,
            "cr_confidence": self.cr.confidence,
            "basal": basal_mean,
            "basal_sigma": basal_std,
            "basal_confidence": self.basal.confidence,
        }

        # Add covariate effects if available
        if self.use_covariates and self.isf.covariate_names:
            effects = self.isf.covariate_effects()
            result["isf_covariate_effects"] = effects
            if covs:
                isf_base, _ = self.isf.predict(None)
                result["isf_base"] = isf_base
                result["isf_context_multiplier"] = isf_mean / isf_base if isf_base > 0 else 1.0

        return result

    def _get_bg_near(self, target_time: float, tolerance: float) -> Optional[float]:
        best, best_dist = None, tolerance + 1
        for t, bg in self._bg_readings:
            dist = abs(t - target_time)
            if dist < best_dist:
                best, best_dist = bg, dist
        return best if best_dist <= tolerance else None

    def cleanup(self, current_time: float, keep_hours: float = 48.0):
        cutoff = current_time - keep_hours * 60
        self._bg_readings = [(t, bg) for t, bg in self._bg_readings if t > cutoff]
        self._insulin_events = [(t, u, tp) for t, u, tp in self._insulin_events if t > cutoff]
        self._meal_events = [(t, c) for t, c in self._meal_events if t > cutoff]

    def summary(self) -> str:
        p = self.get_params()
        lines = [
            f"ISF: {p['isf']:.1f} ±{p['isf_sigma']:.1f} mg/dL/U "
            f"(conf={p['isf_confidence']:.0%}, {len(self.isf.observations)} obs)",
            f"CR:  {p['cr']:.1f} ±{p['cr_sigma']:.1f} g/U "
            f"(conf={p['cr_confidence']:.0%}, {len(self.cr.observations)} obs)",
            f"Basal: {p['basal']:.2f} ±{p['basal_sigma']:.2f} U/hr "
            f"(conf={p['basal_confidence']:.0%}, {len(self.basal.observations)} obs)",
        ]
        if "isf_covariate_effects" in p:
            effects = p["isf_covariate_effects"]
            lines.append("ISF covariate effects:")
            for name in COVARIATE_NAMES:
                if name in effects:
                    mult = effects[name]
                    std = effects.get(f"{name}_std", 0)
                    lines.append(f"  {name}: ×{mult:.2f} (±{std:.2f})")
            if "isf_context_multiplier" in p:
                lines.append(f"  Current context multiplier: ×{p['isf_context_multiplier']:.2f}")
        return "\n".join(lines)
