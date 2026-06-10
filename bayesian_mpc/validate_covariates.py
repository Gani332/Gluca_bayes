"""
Synthetic validation for Apple Health covariates in bayesian_v2.

simglucose does not model exercise, sleep, HRV, or cycle effects, so it
cannot tell us whether optional Apple Health inputs improve parameter
estimation. This script validates that the log-linear estimator learns
those effects when they truly exist.

It compares two ISF models on synthetic data:
  1. No covariates: estimates a single global ISF
  2. With covariates: learns multiplicative effects from health context
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bayesian_mpc.bayesian_v2 import COVARIATE_NAMES, HealthContext, LogLinearParam


TRUE_BASE_ISF = 45.0
TRUE_LOG_COEFFS = {
    "activity": 0.20,   # more activity -> higher ISF
    "sleep": 0.10,      # better sleep -> slightly higher ISF
    "stress": -0.12,    # more stress -> lower ISF
    "cycle": -0.25,     # luteal -> lower ISF
}
OBS_LOG_STD = 0.18


def _sample_context(rng: np.random.Generator) -> HealthContext:
    """Generate plausible Apple Health-like raw values, then normalize them."""
    steps_2h = int(np.clip(rng.normal(3200, 1600), 0, 9000))
    steps_daily = int(np.clip(rng.normal(8500, 2500), 1500, 20000))
    workout_min_today = float(np.clip(rng.normal(25, 25), 0, 120))
    sleep_hours = float(np.clip(rng.normal(7.4, 1.0), 4.5, 10.0))
    hrv_ms = float(np.clip(rng.normal(50, 12), 20, 90))
    cycle_phase = rng.choice([None, None, None, "menstrual", "luteal"])

    return HealthContext.from_apple_health(
        steps_2h=steps_2h,
        steps_daily=steps_daily,
        workout_min_today=workout_min_today,
        sleep_hours=sleep_hours,
        hrv_ms=hrv_ms,
        cycle_phase=cycle_phase,
    )


def _true_isf(context: HealthContext, rng: np.random.Generator) -> float:
    """Generate an observed ISF from the true latent log-linear model."""
    covs = context.to_dict()
    log_isf = np.log(TRUE_BASE_ISF)
    for name in COVARIATE_NAMES:
        log_isf += TRUE_LOG_COEFFS[name] * covs.get(name, 0.0)
    log_isf += rng.normal(0.0, OBS_LOG_STD)
    return float(np.exp(log_isf))


def run_synthetic_validation(train_n: int = 200, test_n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)

    plain = LogLinearParam(
        "ISF",
        prior_value=50.0,
        prior_log_std=0.7,
        covariate_names=[],
        min_val=10.0,
        max_val=200.0,
    )
    with_covariates = LogLinearParam(
        "ISF",
        prior_value=50.0,
        prior_log_std=0.7,
        covariate_names=COVARIATE_NAMES,
        covariate_prior_std=0.12,
        min_val=10.0,
        max_val=200.0,
    )

    train_rows = []
    for i in range(train_n):
        ctx = _sample_context(rng)
        observed_isf = _true_isf(ctx, rng)
        covs = ctx.to_dict()
        plain.update(observed_isf, OBS_LOG_STD, None, float(i))
        with_covariates.update(observed_isf, OBS_LOG_STD, covs, float(i))
        train_rows.append((ctx, observed_isf))

    plain_errors = []
    cov_errors = []
    for _ in range(test_n):
        ctx = _sample_context(rng)
        observed_isf = _true_isf(ctx, rng)
        covs = ctx.to_dict()

        pred_plain, _ = plain.predict()
        pred_cov, _ = with_covariates.predict(covs)

        plain_errors.append((pred_plain - observed_isf) ** 2)
        cov_errors.append((pred_cov - observed_isf) ** 2)

    plain_rmse = float(np.sqrt(np.mean(plain_errors)))
    cov_rmse = float(np.sqrt(np.mean(cov_errors)))

    print("\nSynthetic Covariate Validation")
    print("=" * 40)
    print(f"Train samples: {train_n}")
    print(f"Test samples:  {test_n}")
    print(f"True base ISF: {TRUE_BASE_ISF:.1f} mg/dL/U")
    print("\nTrue log-coefficients:")
    for name in COVARIATE_NAMES:
        print(f"  {name:<8} {TRUE_LOG_COEFFS[name]:>6.2f}")

    print("\nLearned multiplicative effects:")
    effects = with_covariates.covariate_effects()
    for name in COVARIATE_NAMES:
        print(f"  {name:<8} ×{effects[name]:.2f} (std={effects[f'{name}_std']:.2f})")

    print("\nHeld-out RMSE:")
    print(f"  No covariates:   {plain_rmse:.2f} mg/dL/U")
    print(f"  With covariates: {cov_rmse:.2f} mg/dL/U")
    if cov_rmse < plain_rmse:
        improvement = 100.0 * (plain_rmse - cov_rmse) / plain_rmse
        print(f"  Improvement:     {improvement:.1f}%")
    else:
        regression = 100.0 * (cov_rmse - plain_rmse) / plain_rmse
        print(f"  Regression:      {regression:.1f}%")

    return {
        "plain_rmse": plain_rmse,
        "covariate_rmse": cov_rmse,
        "effects": effects,
    }


if __name__ == "__main__":
    run_synthetic_validation()
