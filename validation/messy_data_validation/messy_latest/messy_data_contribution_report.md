# Messy-Data Contribution Analysis

## Protocol

- Source: existing RL4BG adolescent free-living replay outputs.
- Simulator parameters are not changed for this analysis.
- Data are messy by design: meals, basal, boluses and CGM are mixed.
- This is estimator validation, not a clinical or closed-loop pump superiority test.

## Main Result

| Estimator | Overall mean absolute % error | Overall median absolute % error |
|---|---:|---:|
| Messy empirical mean | 38.58% | 30.19% |
| Clean-gated empirical mean | 19.79% | 19.63% |
| Clean-gated normal Bayes | 17.86% | 17.03% |
| Gluca Bayesian/trust | 16.89% | 16.02% |

## What Improved

- Clean gating reduced overall mean error from 38.58% to 19.79% (48.71% improvement versus messy empirical averaging).
- Gluca Bayesian/trust reduced overall mean error from 19.79% to 16.89% (14.63% improvement versus clean-gated empirical averaging).
- The full stack reduced overall mean error from 38.58% to 16.89% (56.21% improvement versus messy empirical averaging).

## Interpretation

This supports a real contribution, but the contribution is not simply 'Bayes beats
averaging' in isolation. The bigger contribution is the full pipeline: selecting less-
confounded observations, using uncertainty and trust to avoid overreacting, and then
updating patient-specific parameters. On perfectly clean ISF events, robust averaging is
hard to beat. On messy free-living replay, naive averaging gets hurt because it treats
confounded observations as equally trustworthy.

## Claim Boundary

- Supported: Gluca beats naive empirical averaging on messy simulator replay.
- Supported: Gluca beats clean-gated empirical averaging on overall parameter recovery in this replay.
- Not supported: Gluca clinically beats current commercial pumps.
- Not supported: Bayesian posterior mean alone is always better than a robust personalised average.

See `messy_data_comparisons.csv` for parameter-level comparisons.
