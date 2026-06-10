# Comprehensive Gluca Estimator Validation

Generated: 2026-06-08T12:49:48.660598+00:00

## Primary Pump-Proxy Comparison

| Domain | Metric | Gluca | Baseline | Baseline value | Improvement vs baseline | Supported | Value-add |
|---|---|---:|---|---:|---:|---|---|
| ISF correction response (adolescents) | net drop MAE mg/dL | 3.091 | clinical_prior | 35.12 | 91.20% | yes | yes |
| ISF correction response (adults) | net drop MAE mg/dL | 4.99 | clinical_prior | 38.06 | 86.89% | yes | yes |
| CR recovery | median abs pct error | 20.15 | configured_profile_proxy | 28.13 | 28.37% | yes | yes |
| Basal recovery | median abs pct error | 3.654 | configured_profile_proxy | 27.92 | 86.91% | yes | yes |
| Overall parameter recovery | median abs pct error | 16.02 | configured_profile_proxy | 24.09 | 33.49% | yes | yes |
| Meal inference | event F1 | 0.8841 | bolus_context_only | 0.8841 | 0.00% | yes | no |
| Carb absorption timing | MAE hours | 0.7495 | fixed_3h | 1.279 | 41.42% | yes | yes |
| Dawn/morning rise prediction | MAE mg/dL | 6.739 | no_explicit_dawn_personalization | 6.194 | -8.81% | no | no |

## Strong Non-Bayesian Sanity Baselines

| Domain | Metric | Gluca | Baseline | Baseline value | Improvement vs baseline | Supported | Value-add |
|---|---|---:|---|---:|---:|---|---|
| ISF correction response (adolescents) | net drop MAE mg/dL | 3.091 | robust_winsorized_mean | 3.091 | 0.00% | yes | no |
| ISF correction response (adults) | net drop MAE mg/dL | 4.99 | robust_huber_center | 4.943 | -0.97% | no | no |
| CR recovery | median abs pct error | 20.15 | normal_bayes_from_v2_obs | 23.2 | 13.13% | yes | yes |
| Basal recovery | median abs pct error | 3.654 | v2_observation_mean | 4.337 | 15.76% | yes | yes |
| Overall parameter recovery | median abs pct error | 16.02 | normal_bayes_from_v2_obs | 17.03 | 5.90% | yes | yes |
| Meal inference | event F1 | 0.8841 | bolus_only | 0.8841 | 0.00% | yes | no |
| Carb absorption timing | MAE hours | 0.7495 | fixed_3h | 1.279 | 41.42% | yes | yes |
| Dawn/morning rise prediction | MAE mg/dL | 6.739 | no_dawn | 6.194 | -8.81% | no | no |

## Interpretation

- Pump-proxy comparisons supported: 7/8; positive value-add rows: 6/8.
- Strong non-Bayesian sanity comparisons supported: 6/8; positive value-add rows: 4/8.
- Primary pump-proxy rows compare against public-current pump behavior proxies, not proprietary closed-loop reimplementations.
- Tandem Control-IQ uses active Personal Profile settings for delivery decisions such as correction factor; Omnipod 5 adapts automated delivery from recent TDI; Medtronic 780G SmartGuard/Bolus Wizard uses configured bolus settings and automated correction logic.
- Bayesian/normal posterior rows are kept only as internal ablations in detailed CSVs, not as the primary comparison.
- The valid broad claim is multi-task: Gluca improves several personalization/event-state tasks versus current pump-style proxies, not one universal hidden-parameter score.
- ISF/CR/basal have simulator parameter or response ground truth.
- Meal inference has event ground truth in synthetic/simulator traces and can also be tested against personal logged meals.
- Carb absorption and dawn are predictive-outcome validations unless a simulator exposes explicit parameter truth.
- Unsupported or tie-only rows should stay out of a strong value-add claim until they beat the relevant baseline.

## Files

- headline_claim_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/headline_claim_rows.csv`
- pump_proxy_claim_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/pump_proxy_claim_rows.csv`
- strong_baseline_claim_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/strong_baseline_claim_rows.csv`
- parameter_task_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/parameter_task_rows.csv`
- meal_inference_detector_summary: `validation/comprehensive_estimator_validation/comprehensive_latest/meal_inference_detector_summary.csv`
- meal_inference_trace_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/meal_inference_trace_rows.csv`
- carb_absorption_summary: `validation/comprehensive_estimator_validation/comprehensive_latest/carb_absorption_summary.csv`
- carb_absorption_event_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/carb_absorption_event_rows.csv`
- dawn_summary: `validation/comprehensive_estimator_validation/comprehensive_latest/dawn_summary.csv`
- dawn_event_rows: `validation/comprehensive_estimator_validation/comprehensive_latest/dawn_event_rows.csv`
- paired_significance_checks: `validation/comprehensive_estimator_validation/comprehensive_latest/paired_significance_checks.csv`
- report: `validation/comprehensive_estimator_validation/comprehensive_latest/comprehensive_validation_report.md`
- zip: `validation/comprehensive_estimator_validation/comprehensive_validation_claude_package.zip`
