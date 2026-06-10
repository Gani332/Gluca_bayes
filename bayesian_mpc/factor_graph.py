"""
Gaussian Factor Graph for joint patient parameter estimation.

Upgrade from independent Bayesian estimators to a proper graphical model
that captures correlations and temporal dynamics.

Graph structure:

  [ISF_t] ---- (temporal) ---- [ISF_{t+1}]
     |                              |
     |                              |
  (TDD correlation)            (TDD correlation)
     |                              |
     |                              |
  [CR_t]  ---- (temporal) ----  [CR_{t+1}]
     |                              |
     |                              |
  [Basal_t] -- (temporal) -- [Basal_{t+1}]

           \       |       /
            \      |      /
         (glucose observation factor)
                   |
                [BG_t]

Factors:
  1. Prior factors: population priors on each parameter
  2. Temporal factors: parameters evolve slowly (random walk + circadian)
  3. Correlation factor: ISF and CR are coupled via total daily dose
  4. Observation factors: each meal/correction event constrains params jointly

Since all distributions are Gaussian, belief propagation gives exact
posterior inference. This is equivalent to a Kalman filter over the
parameter space, but the factor graph structure makes the model explicit.

Key improvement over independent estimation:
  - A meal event updates BOTH CR and ISF simultaneously
  - Circadian ISF variation is modeled (morning ≠ evening)
  - Uncertainty correctly propagates between correlated parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian Message Primitives
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GaussianMessage:
    """A Gaussian message in canonical (information) form: N^{-1}(eta, Lambda)."""
    eta: np.ndarray    # Information vector (Lambda @ mu)
    Lambda: np.ndarray  # Precision matrix (inverse covariance)

    @property
    def dim(self) -> int:
        return len(self.eta)

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.Lambda, self.eta)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.Lambda)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @staticmethod
    def from_mean_cov(mu: np.ndarray, cov: np.ndarray) -> 'GaussianMessage':
        Lambda = np.linalg.inv(cov)
        eta = Lambda @ mu
        return GaussianMessage(eta=eta, Lambda=Lambda)

    @staticmethod
    def uninformative(dim: int) -> 'GaussianMessage':
        return GaussianMessage(
            eta=np.zeros(dim),
            Lambda=np.eye(dim) * 1e-10,
        )

    def multiply(self, other: 'GaussianMessage') -> 'GaussianMessage':
        """Product of two Gaussians (sum in information form)."""
        return GaussianMessage(
            eta=self.eta + other.eta,
            Lambda=self.Lambda + other.Lambda,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factor Graph
# ═══════════════════════════════════════════════════════════════════════════════

# Parameter indices in the joint state vector
ISF = 0   # Insulin Sensitivity Factor (mg/dL per U)
CR = 1    # Carb Ratio (g per U)
BASAL = 2 # Basal rate (U/hr)
DIM = 3   # Total parameter dimension


class PatientFactorGraph:
    """
    Gaussian factor graph for online patient parameter estimation.

    State vector: [ISF, CR, Basal]

    The posterior is maintained as a joint Gaussian over all three
    parameters, capturing their correlations.
    """

    def __init__(
        self,
        isf_prior: float = 50.0,
        cr_prior: float = 10.0,
        basal_prior: float = 1.0,
        bg_target: float = 110.0,
    ):
        self.bg_target = bg_target

        # ── Prior ────────────────────────────────────────────────────
        prior_mean = np.array([isf_prior, cr_prior, basal_prior])

        # Prior covariance: moderate uncertainty, with ISF-CR correlation
        # ISF and CR are positively correlated (both scale with insulin sensitivity)
        # Correlation coefficient ~ 0.4
        isf_var = 25.0 ** 2   # std = 25
        cr_var = 5.0 ** 2     # std = 5
        basal_var = 0.5 ** 2  # std = 0.5
        isf_cr_cov = 0.4 * 25.0 * 5.0  # correlation = 0.4

        prior_cov = np.array([
            [isf_var,     isf_cr_cov, 0.0],
            [isf_cr_cov,  cr_var,     0.0],
            [0.0,         0.0,        basal_var],
        ])

        self.prior = GaussianMessage.from_mean_cov(prior_mean, prior_cov)
        self.belief = GaussianMessage.from_mean_cov(prior_mean, prior_cov)

        # ── Temporal dynamics ────────────────────────────────────────
        # Process noise: how much parameters drift per hour
        # ISF: ~5 mg/dL/U std per hour (circadian + activity)
        # CR: ~1.5 g/U std per hour
        # Basal: ~0.05 U/hr std per hour
        self._process_noise_per_hour = np.diag([
            5.0 ** 2,    # ISF variance per hour
            1.5 ** 2,    # CR variance per hour
            0.05 ** 2,   # Basal variance per hour
        ])

        # ── Circadian model ──────────────────────────────────────────
        # ISF multiplier by time of day (approximate):
        # Lower in morning (dawn phenomenon), higher at night
        # Index by hour of day (0-23)
        self._circadian_isf = np.array([
            1.0, 1.0, 1.0, 1.0, 0.95, 0.85,   # 00-05: overnight, dawn drop
            0.75, 0.70, 0.75, 0.85, 0.90, 0.95, # 06-11: morning low
            1.0, 1.05, 1.10, 1.10, 1.05, 1.0,   # 12-17: afternoon peak
            0.95, 0.95, 1.0, 1.0, 1.0, 1.0,     # 18-23: evening
        ])

        # ── History for observation factors ──────────────────────────
        self._bg_readings = []
        self._insulin_events = []
        self._meal_events = []
        self._observation_count = 0

    # ── Recording events ─────────────────────────────────────────────

    def record_bg(self, time_min: float, bg: float):
        self._bg_readings.append((time_min, bg))

    def record_insulin(self, time_min: float, units: float,
                       insulin_type: str = "bolus"):
        self._insulin_events.append((time_min, units, insulin_type))

    def record_meal(self, time_min: float, carbs: float):
        self._meal_events.append((time_min, carbs))

    # ── Temporal prediction (process update) ─────────────────────────

    def predict(self, dt_hours: float):
        """
        Propagate belief forward in time.

        Adds process noise proportional to elapsed time.
        This widens the posterior, allowing re-adaptation.
        """
        Q = self._process_noise_per_hour * dt_hours

        # Add process noise: increase covariance (decrease precision)
        cov = self.belief.cov + Q
        self.belief = GaussianMessage.from_mean_cov(self.belief.mean, cov)

    # ── Observation factors ──────────────────────────────────────────

    def observe_meal_response(
        self,
        carbs: float,
        bolus_units: float,
        bg_before: float,
        bg_after: float,
        time_min: float,
    ):
        """
        Update belief from a meal + bolus observation.

        The glucose model says:
          bg_after = bg_before + carbs * absorption_factor / CR_effective
                     - bolus * ISF - correction_from_basal

        Rearranging, this gives a linear constraint on [ISF, CR]:
          bg_change = carbs * k / CR - bolus * ISF

        where k = glucose rise per gram of carb (≈ 3-5 mg/dL/g depending
        on body weight, roughly 1000 * bioavailability / Vd).

        This is a LINEAR observation in ISF and CR, so the Gaussian
        update is exact.
        """
        mu = self.belief.mean
        isf_est = mu[ISF]
        cr_est = mu[CR]
        bg_change = bg_after - bg_before

        # ── CR observation ───────────────────────────────────────────
        # Direct: CR ≈ carbs / meal_bolus_component, adjusted for BG outcome
        if bolus_units > 0.3 and carbs > 5:
            correction_component = 0.0
            if bg_before > self.bg_target + 20:
                correction_component = (bg_before - self.bg_target) / isf_est
            meal_component = bolus_units - correction_component

            if meal_component > 0.3:
                observed_cr = carbs / meal_component

                # Adjust for BG outcome: if BG rose, need more insulin → lower CR
                if abs(bg_change) > 10:
                    insulin_error = bg_change / isf_est
                    corrected_meal = meal_component + insulin_error
                    if corrected_meal > 0.3:
                        observed_cr = carbs / corrected_meal

                if 2.0 < observed_cr < 50.0:
                    H = np.zeros((1, DIM))
                    H[0, CR] = 1.0
                    z = np.array([observed_cr - cr_est])
                    R = np.array([[4.0 ** 2]])
                    self._gaussian_update(H, z, R)

        # ── ISF observation from residual ────────────────────────────
        # After meal coverage (carbs/CR), remaining bolus is correction
        if bolus_units > 0.5:
            expected_meal = carbs / cr_est if carbs > 0 else 0
            correction_units = bolus_units - expected_meal

            if abs(correction_units) > 0.3:
                observed_isf = -bg_change / correction_units
                if 5.0 < observed_isf < 200.0:
                    H = np.zeros((1, DIM))
                    H[0, ISF] = 1.0
                    z = np.array([observed_isf - isf_est])
                    R = np.array([[15.0 ** 2]])
                    self._gaussian_update(H, z, R)

        self._observation_count += 1

    def observe_correction_response(
        self,
        bolus_units: float,
        bg_before: float,
        bg_after: float,
    ):
        """
        Update belief from a correction-only event (no meal).

        ISF = (bg_before - bg_after) / bolus_units

        This is the cleanest signal for ISF estimation.
        """
        if bolus_units < 0.3:
            return

        observed_isf = (bg_before - bg_after) / bolus_units
        if not (5.0 < observed_isf < 200.0):
            return

        mu = self.belief.mean

        H = np.zeros((1, DIM))
        H[0, ISF] = 1.0

        # Observation: ISF = observed_isf
        z = np.array([observed_isf - mu[ISF]])

        # Low noise for clean correction events
        R = np.array([[8.0 ** 2]])

        self._gaussian_update(H, z, R)
        self._observation_count += 1

    def observe_fasting_drift(
        self,
        bg_start: float,
        bg_end: float,
        hours: float,
        basal_rate_used: float,
    ):
        """
        Update basal estimate from fasting BG drift.

        If BG rises during fasting: basal too low
        If BG falls: basal too high

        drift_rate = (bg_end - bg_start) / hours  (mg/dL per hour)
        basal_correction = drift_rate / ISF  (U/hr)
        optimal_basal = basal_rate_used + basal_correction
        """
        mu = self.belief.mean
        isf_est = mu[ISF]

        drift_rate = (bg_end - bg_start) / hours
        basal_correction = drift_rate / isf_est
        observed_basal = basal_rate_used + basal_correction

        if not (0.05 < observed_basal < 5.0):
            return

        H = np.zeros((1, DIM))
        H[0, BASAL] = 1.0

        z = np.array([observed_basal - mu[BASAL]])
        R = np.array([[0.15 ** 2]])

        self._gaussian_update(H, z, R)
        self._observation_count += 1

    # ── Core Gaussian update ─────────────────────────────────────────

    def _gaussian_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray):
        """
        Kalman-style update in information form.

        H: observation matrix (m x DIM)
        z: innovation (m,)
        R: observation noise covariance (m x m)
        """
        R_inv = np.linalg.inv(R)

        # Information update
        Lambda_new = self.belief.Lambda + H.T @ R_inv @ H
        eta_new = self.belief.eta + H.T @ R_inv @ (z + H @ self.belief.mean)

        self.belief = GaussianMessage(eta=eta_new, Lambda=Lambda_new)

        # Clamp means to physiological range
        mu = self.belief.mean
        mu[ISF] = np.clip(mu[ISF], 10.0, 200.0)
        mu[CR] = np.clip(mu[CR], 2.0, 50.0)
        mu[BASAL] = np.clip(mu[BASAL], 0.05, 5.0)
        self.belief = GaussianMessage.from_mean_cov(mu, self.belief.cov)

    # ── Circadian adjustment ─────────────────────────────────────────

    def get_circadian_isf(self, hour_of_day: float) -> float:
        """Get ISF adjusted for time of day."""
        hour = int(hour_of_day) % 24
        frac = hour_of_day - int(hour_of_day)
        next_hour = (hour + 1) % 24
        multiplier = (1 - frac) * self._circadian_isf[hour] + frac * self._circadian_isf[next_hour]
        return self.belief.mean[ISF] * multiplier

    # ── Query interface ──────────────────────────────────────────────

    def get_params(self, hour_of_day: float = 12.0) -> Dict:
        """Get current parameter estimates."""
        mu = self.belief.mean
        std = self.belief.std
        cov = self.belief.cov

        # Correlation between ISF and CR
        isf_cr_corr = cov[ISF, CR] / (std[ISF] * std[CR]) if std[ISF] > 0 and std[CR] > 0 else 0

        prior_std = np.sqrt(np.diag(self.prior.cov))
        confidence = np.clip(1.0 - std / prior_std, 0.0, 1.0)

        return {
            "isf": float(mu[ISF]),
            "isf_circadian": float(self.get_circadian_isf(hour_of_day)),
            "cr": float(mu[CR]),
            "basal": float(mu[BASAL]),
            "isf_std": float(std[ISF]),
            "cr_std": float(std[CR]),
            "basal_std": float(std[BASAL]),
            "isf_confidence": float(confidence[ISF]),
            "cr_confidence": float(confidence[CR]),
            "basal_confidence": float(confidence[BASAL]),
            "isf_cr_correlation": float(isf_cr_corr),
            "observations": self._observation_count,
        }

    def get_joint_covariance(self) -> np.ndarray:
        """Return the full 3x3 posterior covariance."""
        return self.belief.cov

    # ── Event processing (matches BayesianEstimator API) ─────────────

    def try_update(self, current_time: float) -> Dict[str, Optional[float]]:
        """
        Scan event history for unprocessed observation opportunities.

        Returns dict of any newly observed values.
        """
        RESPONSE_WINDOW = 180.0  # 3 hours
        results = {"isf": None, "cr": None, "basal": None}

        # Process meal+bolus events
        for meal_time, carbs in reversed(self._meal_events):
            if carbs < 5 or current_time - meal_time < RESPONSE_WINDOW:
                continue

            # Check already processed
            if hasattr(self, '_processed_meals') and meal_time in self._processed_meals:
                continue

            # Find bolus near meal
            bolus = 0.0
            for ins_time, units, itype in self._insulin_events:
                if itype == "bolus" and abs(ins_time - meal_time) < 30:
                    bolus += units

            if bolus < 0.3:
                continue

            bg_before = self._get_bg_near(meal_time, 15.0)
            bg_after = self._get_bg_near(meal_time + RESPONSE_WINDOW, 15.0)
            if bg_before is None or bg_after is None:
                continue

            self.observe_meal_response(carbs, bolus, bg_before, bg_after, meal_time)

            if not hasattr(self, '_processed_meals'):
                self._processed_meals = set()
            self._processed_meals.add(meal_time)

            mu = self.belief.mean
            results["isf"] = float(mu[ISF])
            results["cr"] = float(mu[CR])

        # Process correction-only boluses
        for ins_time, units, itype in reversed(self._insulin_events):
            if itype != "bolus" or units < 0.5:
                continue
            if current_time - ins_time < RESPONSE_WINDOW:
                continue
            if hasattr(self, '_processed_corrections') and ins_time in self._processed_corrections:
                continue

            # No meal nearby
            has_meal = any(abs(mt - ins_time) < 60 for mt, _ in self._meal_events)
            if has_meal:
                continue

            bg_before = self._get_bg_near(ins_time, 15.0)
            bg_after = self._get_bg_near(ins_time + RESPONSE_WINDOW, 15.0)
            if bg_before is None or bg_after is None:
                continue

            self.observe_correction_response(units, bg_before, bg_after)

            if not hasattr(self, '_processed_corrections'):
                self._processed_corrections = set()
            self._processed_corrections.add(ins_time)

            results["isf"] = float(self.belief.mean[ISF])

        # Process fasting periods for basal estimation
        for i in range(len(self._bg_readings) - 1):
            t_start, bg_start = self._bg_readings[i]
            t_end = t_start + 240.0  # 4 hours
            if t_end > current_time:
                continue
            if hasattr(self, '_processed_fasting') and t_start in self._processed_fasting:
                continue

            # No meals or boluses in window
            has_meal = any(t_start - 60 < mt < t_end for mt, _ in self._meal_events)
            has_bolus = any(t_start < it < t_end and tp == "bolus"
                           for it, _, tp in self._insulin_events)
            if has_meal or has_bolus:
                continue

            bg_end = self._get_bg_near(t_end, 15.0)
            if bg_end is None:
                continue

            # Compute basal used during window
            basal_total = sum(u for it, u, tp in self._insulin_events
                              if t_start <= it < t_end and tp == "basal")
            basal_rate = basal_total / 4.0  # U/hr

            if basal_rate < 0.01:
                continue

            self.observe_fasting_drift(bg_start, bg_end, 4.0, basal_rate)

            if not hasattr(self, '_processed_fasting'):
                self._processed_fasting = set()
            self._processed_fasting.add(t_start)

            results["basal"] = float(self.belief.mean[BASAL])

        # Temporal drift (small process noise)
        self.predict(dt_hours=0.05)  # ~3 min

        return results

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
        cov = self.belief.cov
        return (
            f"ISF: {p['isf']:.1f} ±{p['isf_std']:.1f} mg/dL/U "
            f"(conf={p['isf_confidence']:.0%})\n"
            f"CR:  {p['cr']:.1f} ±{p['cr_std']:.1f} g/U "
            f"(conf={p['cr_confidence']:.0%})\n"
            f"Basal: {p['basal']:.2f} ±{p['basal_std']:.2f} U/hr "
            f"(conf={p['basal_confidence']:.0%})\n"
            f"ISF-CR correlation: {p['isf_cr_correlation']:.2f}\n"
            f"Observations: {p['observations']}"
        )
