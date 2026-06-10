"""
Failure-mode aware parameter-estimation helpers.

These helpers are deliberately train-only: they inspect only observed parameter
values and priors. They never see simulator truth or holdout outcomes.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class EstimatorFailureMode(str, Enum):
    """Explicit failure modes seen in parameter-learning validation."""

    SPARSE_EVIDENCE = "sparse_evidence"
    PRIOR_ANCHORING_RISK = "prior_anchoring_risk"
    OUTLIER_OR_MULTIMODAL_EVIDENCE = "outlier_or_multimodal_evidence"
    HIGH_DISPERSION_EVIDENCE = "high_dispersion_evidence"
    NONPHYSIOLOGIC_EVIDENCE = "nonphysiologic_evidence"


@dataclass(frozen=True)
class FailureModeAssessment:
    n_observations: int
    median: float | None
    mean: float | None
    relative_iqr: float | None
    relative_mean_median_gap: float | None
    modes: tuple[EstimatorFailureMode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModularEstimate:
    value: float
    method: str
    assessment: FailureModeAssessment
    data_weight: float


def clean_values(values: Iterable[float], min_value: float, max_value: float) -> list[float]:
    return [
        float(value)
        for value in values
        if math.isfinite(float(value)) and min_value <= float(value) <= max_value
    ]


def winsorized_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    if not values:
        raise ValueError("winsorized_mean requires at least one value")
    ordered = sorted(values)
    trim = int(len(ordered) * trim_fraction)
    lo = ordered[trim] if trim < len(ordered) else ordered[0]
    hi = ordered[len(ordered) - trim - 1] if trim < len(ordered) else ordered[-1]
    return statistics.fmean(min(max(value, lo), hi) for value in values)


def trimmed_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    if not values:
        raise ValueError("trimmed_mean requires at least one value")
    ordered = sorted(values)
    trim = int(len(ordered) * trim_fraction)
    if trim and len(ordered) - (2 * trim) > 0:
        ordered = ordered[trim:len(ordered) - trim]
    return statistics.fmean(ordered)


def huber_location(values: list[float], c: float = 1.345, max_iter: int = 20) -> float:
    if not values:
        raise ValueError("huber_location requires at least one value")
    estimate = statistics.median(values)
    mad = statistics.median(abs(value - estimate) for value in values) if len(values) > 1 else 0.0
    scale = max(1.4826 * mad, 1e-6)
    for _ in range(max_iter):
        weights = [
            min(1.0, c * scale / max(abs(value - estimate), 1e-9))
            for value in values
        ]
        updated = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)
        if abs(updated - estimate) < 1e-6:
            break
        estimate = updated
    return float(estimate)


def log_space_blend(prior_value: float, data_value: float, data_weight: float) -> float:
    data_weight = min(max(data_weight, 0.0), 1.0)
    if prior_value <= 0 or data_value <= 0:
        return (data_weight * data_value) + ((1.0 - data_weight) * prior_value)
    log_value = (
        data_weight * math.log(data_value)
        + (1.0 - data_weight) * math.log(prior_value)
    )
    return math.exp(log_value)


def _candidate_center(method: str, values: list[float]) -> float:
    if not values:
        raise ValueError("_candidate_center requires at least one value")
    if method == "mean":
        return statistics.fmean(values)
    if method == "median":
        return statistics.median(values)
    if method == "trimmed_mean":
        return trimmed_mean(values)
    if method == "winsorized_mean":
        return winsorized_mean(values)
    if method == "huber_center":
        return huber_location(values)
    if method == "recent_3_mean":
        return statistics.fmean(values[-min(3, len(values)):])
    raise ValueError(f"Unknown candidate method: {method}")


def adaptive_robust_positive_estimate(
    values: Iterable[float],
    *,
    prior_value: float,
    min_value: float,
    max_value: float,
    min_observations: int = 4,
    data_dominance_observations: int = 5,
    outlier_iqr_threshold: float = 0.10,
    validation_min_history: int = 3,
    candidate_methods: tuple[str, ...] = (
        "winsorized_mean",
        "huber_center",
        "median",
        "trimmed_mean",
        "recent_3_mean",
        "mean",
    ),
) -> ModularEstimate:
    """
    Choose a robust positive estimator from past observations only.

    This is a small meta-estimator: it walks forward through the training
    sequence, asks which robust center would have best predicted the next
    observed value, then applies that candidate to all available training data.
    The heldout/test observations are never used for selection.
    """
    raw_values = list(values)
    clean = clean_values(raw_values, min_value, max_value)
    assessment = assess_failure_modes(
        raw_values,
        prior_value=prior_value,
        min_value=min_value,
        max_value=max_value,
        min_observations=min_observations,
        outlier_iqr_threshold=outlier_iqr_threshold,
    )
    if not clean:
        return ModularEstimate(
            value=float(min(max(prior_value, min_value), max_value)),
            method="adaptive_prior_only_no_clean_evidence",
            assessment=assessment,
            data_weight=0.0,
        )

    if len(clean) <= validation_min_history:
        base = modular_positive_estimate(
            clean,
            prior_value=prior_value,
            min_value=min_value,
            max_value=max_value,
            min_observations=min_observations,
            data_dominance_observations=data_dominance_observations,
            outlier_iqr_threshold=outlier_iqr_threshold,
        )
        return ModularEstimate(
            value=base.value,
            method=f"adaptive_sparse_{base.method}",
            assessment=assessment,
            data_weight=base.data_weight,
        )

    scores: dict[str, list[float]] = {method: [] for method in candidate_methods}
    for idx in range(validation_min_history, len(clean)):
        prefix = clean[:idx]
        actual = clean[idx]
        for method in candidate_methods:
            try:
                predicted = _candidate_center(method, prefix)
            except ValueError:
                continue
            predicted = min(max(predicted, min_value), max_value)
            if predicted > 0 and actual > 0:
                error = abs(math.log(predicted) - math.log(actual))
            else:
                error = abs(predicted - actual)
            scores[method].append(error)

    valid_scores = [
        (statistics.fmean(method_scores), order, method)
        for order, (method, method_scores) in enumerate(scores.items())
        if method_scores
    ]
    if not valid_scores:
        return modular_positive_estimate(
            clean,
            prior_value=prior_value,
            min_value=min_value,
            max_value=max_value,
            min_observations=min_observations,
            data_dominance_observations=data_dominance_observations,
            outlier_iqr_threshold=outlier_iqr_threshold,
        )

    _score, _order, best_method = min(valid_scores, key=lambda row: (row[0], row[1]))
    center = _candidate_center(best_method, clean)
    data_weight = min(1.0, len(clean) / max(float(data_dominance_observations), 1.0))
    value = log_space_blend(prior_value, center, data_weight)
    return ModularEstimate(
        value=float(min(max(value, min_value), max_value)),
        method=f"adaptive_cv_{best_method}",
        assessment=assessment,
        data_weight=data_weight,
    )


def assess_failure_modes(
    values: Iterable[float],
    *,
    prior_value: float,
    min_value: float,
    max_value: float,
    min_observations: int = 4,
    outlier_iqr_threshold: float = 0.10,
    high_dispersion_threshold: float = 0.18,
    prior_gap_threshold: float = 0.15,
) -> FailureModeAssessment:
    raw_values = list(values)
    clean = clean_values(raw_values, min_value, max_value)
    rejected = len(raw_values) - len(clean)
    modes: list[EstimatorFailureMode] = []
    if rejected > 0:
        modes.append(EstimatorFailureMode.NONPHYSIOLOGIC_EVIDENCE)
    if len(clean) < min_observations:
        modes.append(EstimatorFailureMode.SPARSE_EVIDENCE)
    if not clean:
        return FailureModeAssessment(
            n_observations=0,
            median=None,
            mean=None,
            relative_iqr=None,
            relative_mean_median_gap=None,
            modes=tuple(modes),
        )

    ordered = sorted(clean)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    q1 = ordered[int(0.25 * (len(ordered) - 1))]
    q3 = ordered[int(0.75 * (len(ordered) - 1))]
    relative_iqr = (q3 - q1) / max(abs(median), 1e-9)
    relative_gap = abs(mean - median) / max(abs(median), 1e-9)
    relative_prior_gap = abs(median - prior_value) / max(abs(prior_value), 1e-9)

    if relative_iqr > outlier_iqr_threshold:
        modes.append(EstimatorFailureMode.OUTLIER_OR_MULTIMODAL_EVIDENCE)
    if relative_iqr > high_dispersion_threshold:
        modes.append(EstimatorFailureMode.HIGH_DISPERSION_EVIDENCE)
    if relative_prior_gap > prior_gap_threshold and len(clean) >= min_observations:
        modes.append(EstimatorFailureMode.PRIOR_ANCHORING_RISK)

    return FailureModeAssessment(
        n_observations=len(clean),
        median=median,
        mean=mean,
        relative_iqr=relative_iqr,
        relative_mean_median_gap=relative_gap,
        modes=tuple(dict.fromkeys(modes)),
    )


def modular_positive_estimate(
    values: Iterable[float],
    *,
    prior_value: float,
    min_value: float,
    max_value: float,
    min_observations: int = 4,
    data_dominance_observations: int = 5,
    outlier_iqr_threshold: float = 0.10,
) -> ModularEstimate:
    """
    Estimate a positive parameter using explicit failure-mode handling.

    Rules:
    - Sparse evidence is shrunk toward the prior in log-space.
    - Stable evidence uses a winsorized mean to reduce single-point influence.
    - Outlier/multimodal evidence uses a Huber center.

    All decisions are made from training observations only.
    """
    raw_values = list(values)
    clean = clean_values(raw_values, min_value, max_value)
    assessment = assess_failure_modes(
        raw_values,
        prior_value=prior_value,
        min_value=min_value,
        max_value=max_value,
        min_observations=min_observations,
        outlier_iqr_threshold=outlier_iqr_threshold,
    )
    if not clean:
        return ModularEstimate(
            value=float(min(max(prior_value, min_value), max_value)),
            method="prior_only_no_clean_evidence",
            assessment=assessment,
            data_weight=0.0,
        )

    if EstimatorFailureMode.OUTLIER_OR_MULTIMODAL_EVIDENCE in assessment.modes:
        center = huber_location(clean)
        method = "huber_center_for_outlier_or_multimodal_evidence"
    else:
        center = winsorized_mean(clean)
        method = "winsorized_mean_for_stable_evidence"

    data_weight = min(1.0, len(clean) / max(float(data_dominance_observations), 1.0))
    value = log_space_blend(prior_value, center, data_weight)
    return ModularEstimate(
        value=float(min(max(value, min_value), max_value)),
        method=method,
        assessment=assessment,
        data_weight=data_weight,
    )
