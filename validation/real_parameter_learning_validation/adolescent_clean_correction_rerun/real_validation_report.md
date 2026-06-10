# Real Padova Parameter-Learning Validation

Generated: 2026-06-08T07:58:55.091715+00:00

## Verdict

Supported for this simulator protocol: the Bayesian posterior beats the fixed 50 mg/dL/U population baseline on chronological holdout correction events.

This is Padova correction-challenge evidence, not personal free-living evidence. Personal Supabase validation is still blocked because the configured Supabase project endpoint did not resolve from this environment and the CLI account was unauthorized.

## Main Net-Drop Holdout Results

- Patients evaluated: 8 / 10
- Total usable clean correction events: 64
- Holdout events: 24
- Fixed 50 MAE: 35.123 mg/dL
- Bayesian MAE: 3.355 mg/dL
- Improvement vs fixed 50: 90.45%
- Paired mean error reduction vs fixed 50: 31.767 mg/dL [95% CI 23.285, 40.599], bootstrap p(reduction > 0)=1.0

## Stronger Baselines

- Clinical-prior MAE: 35.123 mg/dL; improvement vs clinical: 90.45%
- Empirical-mean MAE: 3.76 mg/dL; improvement vs empirical: 10.77%

## UKF Signal Quality On Fresh Padova Trace

- Raw CGM RMSE vs true simulator glucose: 11.681 mg/dL
- UKF plasma RMSE vs true simulator glucose: 14.207 mg/dL
- UKF RMSE improvement: -21.63%
- Innovation whiteness p-value: 0.0

## Claim Boundary

Safe claim:

> In a fresh UVA/Padova adolescents correction-challenge validation with chronological holdout splits, Gluca's Bayesian ISF posterior reduced future correction-response prediction error versus a fixed 50 mg/dL/U population baseline.

Do not claim:

> This proves the same improvement on personal CGM data.

Personal-data validation still needs a local export or working authenticated Supabase/database access.
