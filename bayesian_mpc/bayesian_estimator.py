"""
Online Bayesian estimation of patient-specific insulin parameters.

Estimates three critical parameters that vary between patients
and change over time within a single patient:

  1. ISF (Insulin Sensitivity Factor): mg/dL drop per unit of insulin
     - Varies 2-3x within a day (lower in morning, higher at night)
     - Estimated from correction events (insulin given, BG response observed)

  2. CR (Carb Ratio): grams of carbs covered by one unit of insulin
     - Varies with meal type, time of day, activity
     - Estimated from meal events (carbs eaten, insulin given, BG response)

  3. Basal Rate: background insulin needed to keep BG stable during fasting
     - Varies with circadian rhythm, activity, stress
     - Estimated from fasting periods (no meals/boluses, observe BG drift)

Uses conjugate Normal-Normal Bayesian updates:
  Prior:     theta ~ Normal(mu_prior, sigma_prior^2)
  Likelihood: observation ~ Normal(theta, sigma_obs^2)
  Posterior:  theta ~ Normal(mu_post, sigma_post^2)

  mu_post = (mu_prior/sigma_prior^2 + obs/sigma_obs^2) /
            (1/sigma_prior^2 + 1/sigma_obs^2)

This gives us:
  - Point estimate (posterior mean) for dosing
  - Uncertainty (posterior std) for safety — wider uncertainty → more conservative

The key insight: we don't need thousands of data points.
Each insulin-glucose response pair is a direct observation of these
parameters. With ~2 weeks of data (20-30 meals, 5-10 corrections),
we can get useful personalized estimates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BayesianParam:
    """A single parameter with Bayesian posterior tracking."""
    name: str
    mu: float              # Posterior mean
    sigma: float           # Posterior std
    mu_prior: float        # Population prior mean
    sigma_prior: float     # Population prior std
    min_val: float         # Physiological minimum
    max_val: float         # Physiological maximum
    observations: list = field(default_factory=list)  # (timestamp, observed_value, obs_uncertainty)

    @property
    def ci_95(self) -> Tuple[float, float]:
        """95% credible interval."""
        return (
            max(self.min_val, self.mu - 1.96 * self.sigma),
            min(self.max_val, self.mu + 1.96 * self.sigma),
        )

    @property
    def confidence(self) -> float:
        """0-1 confidence score. 1 = very certain, 0 = still at prior."""
        # Ratio of prior width to posterior width
        return max(0.0, min(1.0, 1.0 - self.sigma / self.sigma_prior))

    def update(self, observation: float, obs_sigma: float,
               timestamp: float = 0.0):
        """
        Bayesian update with a new observation.

        Args:
            observation: Observed parameter value
            obs_sigma: Uncertainty of this observation
            timestamp: When this was observed (for recency weighting)
        """
        # Clamp observation to physiological range
        observation = np.clip(observation, self.min_val, self.max_val)
        obs_sigma = max(obs_sigma, 1e-6)

        # Conjugate Normal-Normal update
        prior_precision = 1.0 / (self.sigma ** 2)
        obs_precision = 1.0 / (obs_sigma ** 2)

        post_precision = prior_precision + obs_precision
        post_sigma = np.sqrt(1.0 / post_precision)
        post_mu = (self.mu * prior_precision + observation * obs_precision) / post_precision

        self.mu = np.clip(post_mu, self.min_val, self.max_val)
        self.sigma = post_sigma

        self.observations.append((timestamp, observation, obs_sigma))

    def decay_toward_prior(self, rate: float = 0.001):
        """
        Slowly widen posterior toward prior between observations.

        This accounts for the fact that parameters change over time
        (circadian variation, lifestyle changes). Without decay,
        the posterior would become arbitrarily narrow and stop adapting.

        Args:
            rate: How much to widen per call (0.001 = very slow)
        """
        self.sigma = min(
            self.sigma_prior,
            self.sigma * (1.0 + rate)
        )


class BayesianEstimator:
    """
    Online Bayesian estimator for patient insulin parameters.

    Maintains posterior distributions over ISF, CR, and basal rate.
    Updates from observed insulin-glucose response pairs.
    Provides point estimates + uncertainty for the MPC controller.
    """

    def __init__(
        self,
        # Population priors (can be set from doctor's prescription)
        isf_prior: float = 50.0,
        cr_prior: float = 10.0,
        basal_prior: float = 1.0,
        bg_target: float = 110.0,
        # Prior uncertainty (wide = less informative)
        isf_sigma: float = 25.0,
        cr_sigma: float = 5.0,
        basal_sigma: float = 0.5,
    ):
        self.bg_target = bg_target

        self.isf = BayesianParam(
            name="ISF",
            mu=isf_prior, sigma=isf_sigma,
            mu_prior=isf_prior, sigma_prior=isf_sigma,
            min_val=10.0, max_val=200.0,
        )

        self.cr = BayesianParam(
            name="CR",
            mu=cr_prior, sigma=cr_sigma,
            mu_prior=cr_prior, sigma_prior=cr_sigma,
            min_val=2.0, max_val=50.0,
        )

        self.basal = BayesianParam(
            name="Basal",
            mu=basal_prior, sigma=basal_sigma,
            mu_prior=basal_prior, sigma_prior=basal_sigma,
            min_val=0.1, max_val=5.0,
        )

        # Event history for computing parameter observations
        self._insulin_events = []   # (time_min, units, type='bolus'|'basal')
        self._meal_events = []      # (time_min, carbs_grams)
        self._bg_readings = []      # (time_min, bg_mg_dl)

    def record_bg(self, time_min: float, bg: float):
        """Record a BG reading (from CGM or UKF)."""
        self._bg_readings.append((time_min, bg))

    def record_insulin(self, time_min: float, units: float,
                       insulin_type: str = "bolus"):
        """Record an insulin dose."""
        self._insulin_events.append((time_min, units, insulin_type))

    def record_meal(self, time_min: float, carbs: float):
        """Record a meal."""
        self._meal_events.append((time_min, carbs))

    def try_update_isf(self, current_time: float) -> Optional[float]:
        """
        Estimate ISF from any bolus event (correction or meal).

        Method 1: Correction bolus (no meal) → ISF = BG_drop / units
        Method 2: Meal bolus → ISF from residual after accounting for carbs/CR

        Method 2 is noisier but fires much more often (every meal),
        giving the estimator data to work with from day 1.
        """
        RESPONSE_WINDOW = 180.0    # 3 hours
        latest_isf = None

        for ins_time, ins_units, ins_type in reversed(self._insulin_events):
            if ins_type != "bolus" or ins_units < 0.5:
                continue
            if current_time - ins_time < RESPONSE_WINDOW:
                continue
            if any(abs(t - ins_time) < 1.0
                   for t, _, _ in self.isf.observations):
                continue

            bg_before = self._get_bg_near(ins_time, tolerance=15.0)
            bg_after = self._get_bg_near(ins_time + RESPONSE_WINDOW, tolerance=15.0)
            if bg_before is None or bg_after is None:
                continue

            # Check if there's a meal near this bolus
            nearby_meal = None
            for meal_time, carbs in self._meal_events:
                if abs(meal_time - ins_time) < 30.0 and carbs > 5:
                    nearby_meal = carbs
                    break

            if nearby_meal is None:
                # Method 1: pure correction — clean ISF signal
                bg_drop = bg_before - bg_after
                observed_isf = bg_drop / ins_units
                obs_sigma = max(8.0, 20.0 / ins_units)
            else:
                # Method 2: meal event — ISF from residual
                # Expected BG change from meal = carbs * 1000 * bio / Vd
                # Simplified: expected BG rise ≈ carbs * 5 (rough mg/dL per gram)
                # Then: ISF = (expected_rise - actual_rise) / insulin_units
                #
                # More precisely: if CR is correct, meal bolus = carbs/CR
                # covers the meal exactly. Any BG change is from the
                # correction component → ISF = (bg_before - bg_after) / correction_units
                meal_coverage_units = nearby_meal / self.cr.mu
                correction_units = ins_units - meal_coverage_units
                if abs(correction_units) < 0.3:
                    # Meal was well-covered, BG change ≈ 0 expected
                    # Actual BG change tells us about residual insulin effect
                    # ISF ≈ (bg_before - bg_after) / small_correction, but very noisy
                    continue
                bg_drop = bg_before - bg_after
                observed_isf = bg_drop / correction_units
                obs_sigma = max(12.0, 30.0 / abs(correction_units))

            if 5.0 < observed_isf < 200.0:
                self.isf.update(observed_isf, obs_sigma, ins_time)
                latest_isf = observed_isf

        return latest_isf

    def try_update_cr(self, current_time: float) -> Optional[float]:
        """
        Try to compute a CR observation from recent meal events.

        Looks for: meal with corresponding bolus, then checks if
        BG returned to near-starting level (indicating good coverage).

        CR = carbs / insulin_that_covered_them
        """
        RESPONSE_WINDOW = 240.0    # 4 hours for meal absorption
        BG_RETURN_THRESHOLD = 30.0  # BG within 30 mg/dL of pre-meal = good coverage

        for meal_time, carbs in reversed(self._meal_events):
            if carbs < 10.0:  # Skip snacks
                continue

            if current_time - meal_time < RESPONSE_WINDOW:
                continue

            # Already processed?
            if any(abs(t - meal_time) < 1.0
                   for t, _, _ in self.cr.observations):
                continue

            # Find bolus near meal time (within 30 min)
            bolus_units = 0.0
            for ins_time, ins_units, ins_type in self._insulin_events:
                if ins_type == "bolus" and abs(ins_time - meal_time) < 30.0:
                    bolus_units += ins_units

            if bolus_units < 0.5:
                continue

            # Check BG before and after
            bg_before = self._get_bg_near(meal_time, tolerance=15.0)
            bg_after = self._get_bg_near(meal_time + RESPONSE_WINDOW, tolerance=15.0)

            if bg_before is None or bg_after is None:
                continue

            # If BG returned to near baseline, the bolus was about right
            # → CR ≈ carbs / bolus_units
            bg_diff = abs(bg_after - bg_before)
            if bg_diff < BG_RETURN_THRESHOLD:
                observed_cr = carbs / bolus_units
                if 2.0 < observed_cr < 50.0:
                    obs_sigma = max(2.0, 5.0 / np.sqrt(carbs / 30.0))
                    self.cr.update(observed_cr, obs_sigma, meal_time)
                    return observed_cr
            else:
                # BG didn't return — estimate what CR should have been
                # If BG rose, CR should be lower (need more insulin)
                # If BG dropped, CR should be higher (gave too much)
                bg_rise = bg_after - bg_before
                # Correction needed: bg_rise / ISF units
                correction_units = bg_rise / self.isf.mu
                effective_units = bolus_units + correction_units
                if effective_units > 0.5:
                    observed_cr = carbs / effective_units
                    if 2.0 < observed_cr < 50.0:
                        # Higher uncertainty since BG didn't return
                        obs_sigma = max(3.0, 8.0 / np.sqrt(carbs / 30.0))
                        self.cr.update(observed_cr, obs_sigma, meal_time)
                        return observed_cr

        return None

    def try_update_basal(self, current_time: float) -> Optional[float]:
        """
        Try to estimate basal rate from fasting periods.

        Looks for: 4+ hour windows with no meals and no boluses,
        then checks BG drift. If BG rises → need more basal.
        If drops → need less.
        """
        FASTING_WINDOW = 240.0   # 4 hours
        MEAL_EXCLUSION = 60.0    # No meals within 1 hr before or during

        # Find fasting windows
        for i in range(len(self._bg_readings) - 1):
            t_start, bg_start = self._bg_readings[i]
            t_end = t_start + FASTING_WINDOW

            if t_end > current_time:
                continue

            # Already processed?
            if any(abs(t - t_start) < 1.0
                   for t, _, _ in self.basal.observations):
                continue

            # Check no meals in window
            has_meal = any(
                t_start - MEAL_EXCLUSION < mt < t_end
                for mt, _ in self._meal_events
            )
            if has_meal:
                continue

            # Check no boluses in window
            has_bolus = any(
                t_start < it < t_end and itype == "bolus"
                for it, _, itype in self._insulin_events
            )
            if has_bolus:
                continue

            bg_end = self._get_bg_near(t_end, tolerance=15.0)
            if bg_end is None:
                continue

            # BG drift rate (mg/dL per hour)
            drift_rate = (bg_end - bg_start) / (FASTING_WINDOW / 60.0)

            # Current basal (sum of basal insulin during window)
            basal_total = sum(
                units for it, units, itype in self._insulin_events
                if t_start <= it < t_end and itype == "basal"
            )
            current_basal_rate = basal_total / (FASTING_WINDOW / 60.0)  # U/hr

            if current_basal_rate < 0.01:
                # No basal recorded — can't estimate
                continue

            # Adjust: if BG rising, need more basal; if falling, need less
            # Additional basal needed = drift_rate / ISF (per hour)
            basal_adjustment = drift_rate / self.isf.mu
            observed_basal = current_basal_rate + basal_adjustment

            if 0.1 < observed_basal < 5.0:
                obs_sigma = max(0.1, 0.3)
                self.basal.update(observed_basal, obs_sigma, t_start)
                return observed_basal

        return None

    def update(self, current_time: float) -> Dict[str, Optional[float]]:
        """
        Run all parameter updates. Call periodically (every 15-30 min).

        Returns dict of any newly observed parameter values.
        """
        # Decay posteriors slightly (allows adaptation over time)
        self.isf.decay_toward_prior(rate=0.0005)
        self.cr.decay_toward_prior(rate=0.0005)
        self.basal.decay_toward_prior(rate=0.0005)

        return {
            "isf": self.try_update_isf(current_time),
            "cr": self.try_update_cr(current_time),
            "basal": self.try_update_basal(current_time),
        }

    def get_params(self) -> Dict:
        """Get current parameter estimates for MPC."""
        return {
            "isf": self.isf.mu,
            "isf_sigma": self.isf.sigma,
            "isf_confidence": self.isf.confidence,
            "cr": self.cr.mu,
            "cr_sigma": self.cr.sigma,
            "cr_confidence": self.cr.confidence,
            "basal": self.basal.mu,
            "basal_sigma": self.basal.sigma,
            "basal_confidence": self.basal.confidence,
        }

    def get_conservative_params(self) -> Dict:
        """
        Get conservative parameter estimates (for safety).

        When uncertain, use values that result in LESS insulin:
          - Higher ISF → less correction insulin
          - Higher CR → less meal insulin
          - Lower basal → less background insulin
        """
        return {
            "isf": self.isf.mu + 0.5 * self.isf.sigma,   # lean high → less insulin
            "cr": self.cr.mu + 0.5 * self.cr.sigma,       # lean high → less insulin
            "basal": self.basal.mu - 0.5 * self.basal.sigma,  # lean low → less insulin
        }

    def _get_bg_near(self, target_time: float,
                     tolerance: float = 15.0) -> Optional[float]:
        """Find BG reading closest to target_time within tolerance."""
        best = None
        best_dist = tolerance + 1

        for t, bg in self._bg_readings:
            dist = abs(t - target_time)
            if dist < best_dist:
                best = bg
                best_dist = dist

        return best if best_dist <= tolerance else None

    def cleanup(self, current_time: float, keep_hours: float = 48.0):
        """Remove old events beyond keep window."""
        cutoff = current_time - keep_hours * 60
        self._bg_readings = [(t, bg) for t, bg in self._bg_readings if t > cutoff]
        self._insulin_events = [(t, u, tp) for t, u, tp in self._insulin_events if t > cutoff]
        self._meal_events = [(t, c) for t, c in self._meal_events if t > cutoff]

    def summary(self) -> str:
        """Human-readable summary of current estimates."""
        p = self.get_params()
        lines = [
            f"ISF: {p['isf']:.1f} mg/dL/U (±{p['isf_sigma']:.1f}, "
            f"confidence: {p['isf_confidence']:.0%}, "
            f"{len(self.isf.observations)} obs)",

            f"CR:  {p['cr']:.1f} g/U (±{p['cr_sigma']:.1f}, "
            f"confidence: {p['cr_confidence']:.0%}, "
            f"{len(self.cr.observations)} obs)",

            f"Basal: {p['basal']:.2f} U/hr (±{p['basal_sigma']:.2f}, "
            f"confidence: {p['basal_confidence']:.0%}, "
            f"{len(self.basal.observations)} obs)",
        ]
        return "\n".join(lines)
