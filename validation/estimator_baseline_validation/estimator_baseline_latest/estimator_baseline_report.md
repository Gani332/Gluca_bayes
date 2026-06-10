# Estimator Baseline Benchmark

Generated: 2026-06-08T11:12:30.757816+00:00

## What Was Tested

- Clean correction holdout: future correction-response prediction from isolated Padova correction events.
- RL4BG free-living replay: final recovery of known adolescent CF/CR/basal from meal/insulin/CGM logs.
- Baselines include fixed population, clinical prior, empirical mean/median, robust mean/Huber, EWMA, original Bayes, V2 clean-gated Bayes, app-style log posterior, modular trust, and adaptive trust estimates.

## Modularized Failure Modes

- Prior anchoring: a strong population prior can hurt after several clean patient-specific events.
- Confounded ISF observations: meal residuals can corrupt ISF if carbs, CR, or timing are wrong.
- Sparse clean corrections: if clean ISF events are absent, the estimator should shrink toward prior instead of inventing confidence.
- Outlier or multimodal evidence: correction observations can form split clusters; use Huber center rather than simple averaging.
- Stable evidence with single-point influence: winsorized mean reduces the impact of one odd event without discarding the data.
- Estimator-selection risk: adaptive trust chooses a robust center with chronological one-step validation on training observations only.

## Clean Correction Holdout

### Adolescents

| Estimator | Holdout n | Net MAE | Improvement vs fixed 50 | Improvement vs empirical mean |
|---|---:|---:|---:|---:|
| gluca_modular_trust | 24 | 3.09 | 91.20% | 17.81% |
| robust_winsorized_mean | 24 | 3.09 | 91.20% | 17.81% |
| empirical_trimmed_mean | 24 | 3.11 | 91.16% | 17.41% |
| recent_3_mean | 24 | 3.17 | 90.97% | 15.67% |
| empirical_median | 24 | 3.18 | 90.93% | 15.31% |
| robust_huber_center | 24 | 3.30 | 90.62% | 12.36% |
| original_bayes_v1 | 24 | 3.36 | 90.45% | 10.77% |
| gluca_adaptive_trust | 24 | 3.56 | 89.87% | 5.43% |

### Adults

| Estimator | Holdout n | Net MAE | Improvement vs fixed 50 | Improvement vs empirical mean |
|---|---:|---:|---:|---:|
| robust_huber_center | 27 | 4.94 | 91.47% | 8.95% |
| gluca_modular_trust | 27 | 4.99 | 91.39% | 8.07% |
| empirical_median | 27 | 5.04 | 91.30% | 7.12% |
| empirical_trimmed_mean | 27 | 5.11 | 91.19% | 5.90% |
| robust_winsorized_mean | 27 | 5.12 | 91.16% | 5.62% |
| recent_3_mean | 27 | 5.13 | 91.16% | 5.58% |
| gluca_adaptive_trust | 27 | 5.23 | 90.98% | 3.73% |
| empirical_mean | 27 | 5.43 | 90.63% | 0.00% |

## RL4BG Free-Living Parameter Recovery

| Estimator | Runs | Median ISF err | Median CR err | Median basal err | Median mean err | Mean improvement vs population |
|---|---:|---:|---:|---:|---:|---:|
| gluca_modular_trust | 30 | 27.11% | 20.15% | 3.65% | 16.02% | 45.51% |
| gluca_adaptive_trust | 30 | 27.11% | 18.44% | 4.82% | 16.35% | 45.16% |
| gluca_v2_clean_gate | 30 | 27.11% | 20.76% | 3.65% | 16.68% | 43.75% |
| normal_bayes_from_v2_obs | 30 | 23.72% | 23.20% | 4.46% | 17.03% | 42.39% |
| app_style_log_posterior_from_clean_obs | 30 | 24.92% | 22.25% | 4.78% | 17.24% | 42.24% |
| v2_observation_mean | 30 | 30.33% | 24.36% | 4.34% | 19.63% | 36.18% |
| original_bayes_v1 | 30 | 32.58% | 22.29% | 11.51% | 21.38% | 24.83% |
| population_prior | 30 | 26.94% | 28.13% | 27.92% | 24.09% | 0.00% |
| v1_observation_median | 30 | 39.34% | 36.48% | 7.71% | 27.52% | -12.58% |
| v1_observation_mean | 30 | 38.57% | 34.02% | 11.46% | 30.19% | -24.44% |

## Critical Interpretation

- Clean correction events are the valid ISF-identification setting. Strong performance there supports the data-layer + Bayesian mechanism.
- RL4BG free-living replay tests whether the estimator safely handles confounded logs. If V2 beats V1 there, that supports the stricter trust/gating changes.
- The oracle therapy-profile row is an upper bound, not a usable learned estimator.
- Be careful claiming superiority over empirical personalized baselines unless the result beats empirical mean or median, not just fixed population values.
