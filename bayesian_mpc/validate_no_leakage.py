"""
Temporal leakage validation for Bayesian estimators.

Checks that parameter updates only happen once the required response window
has elapsed, i.e. no lookahead into future glucose outcomes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bayesian_mpc.bayesian_estimator import BayesianEstimator
from bayesian_mpc.bayesian_v2 import BayesianEstimatorV2


def _seed_bg(estimator, start_bg=140.0, minutes=360, step=5):
    for t in range(0, minutes + step, step):
        estimator.record_bg(float(t), start_bg)


def _assert_no_recent_isf_update(estimator_cls, label: str):
    estimator = estimator_cls(isf_prior=50.0, cr_prior=10.0, basal_prior=1.0, bg_target=110.0)
    _seed_bg(estimator, minutes=180)
    estimator.record_insulin(100.0, 2.0, "bolus")
    estimator.update(120.0)  # less than 180 min after bolus
    assert len(estimator.isf.observations) == 0, f"{label}: recent bolus leaked into ISF update"


def _assert_no_recent_cr_update(estimator_cls, label: str):
    estimator = estimator_cls(isf_prior=50.0, cr_prior=10.0, basal_prior=1.0, bg_target=110.0)
    _seed_bg(estimator, minutes=240)
    estimator.record_meal(60.0, 60.0)
    estimator.record_insulin(60.0, 6.0, "bolus")
    estimator.update(180.0)  # less than 240 min after meal
    assert len(estimator.cr.observations) == 0, f"{label}: recent meal leaked into CR update"


def _assert_positive_control(estimator_cls, label: str):
    estimator = estimator_cls(isf_prior=50.0, cr_prior=10.0, basal_prior=1.0, bg_target=110.0)

    # Simulate a meal/bolus with a completed 4h response window.
    for t in range(0, 301, 5):
        bg = 140.0 if t < 240 else 145.0
        estimator.record_bg(float(t), bg)
    estimator.record_meal(0.0, 60.0)
    estimator.record_insulin(0.0, 6.0, "bolus")
    estimator.update(300.0)
    assert len(estimator.cr.observations) >= 1, f"{label}: positive control failed to produce CR update"


def main():
    checks = [
        (BayesianEstimator, "v1"),
        (BayesianEstimatorV2, "v2"),
    ]

    for estimator_cls, label in checks:
        _assert_no_recent_isf_update(estimator_cls, label)
        _assert_no_recent_cr_update(estimator_cls, label)
        _assert_positive_control(estimator_cls, label)

    print("[No Leakage] All temporal leakage checks passed for v1 and v2.")


if __name__ == "__main__":
    main()
