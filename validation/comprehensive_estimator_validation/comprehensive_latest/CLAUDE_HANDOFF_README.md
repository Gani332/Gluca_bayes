# Gluca Validation Handoff

Generated: 2026-06-08

This package contains the validation outputs, source code, and blog-ready plots for the current Gluca claim:

> Gluca adds value as a Bayesian personalization layer over current pump-style configured settings/proxies, especially for learning ISF response, carb ratio, basal needs, overall therapy parameters, and carb absorption timing.

## Primary Claim Boundary

Safe claim:

> In simulator validation with chronological splits and known ground truth where available, Gluca's Bayesian/trust-gated personalization improves several therapy-parameter and event-state estimates versus pump-style configured-profile proxies.

Do not claim:

> Gluca has clinically proven superiority over commercial pumps as a full dosing/controller system.

Do not claim:

> Gluca beats every possible robust personalized estimator.

The validation supports value-add in 6/8 primary pump-proxy rows, supports parity without added value for meal inference, and does not support the current dawn/morning-rise estimator.

## Pump-Proxy Framing

The primary comparator is not "another Bayesian baseline." The comparator is current pump-style behavior:

- Tandem Control-IQ uses active Personal Profile settings for delivery decisions such as correction factor and carb ratio.
- Omnipod 5 adapts automated delivery from recent total daily insulin, while SmartBolus settings still include target glucose, insulin-to-carb ratio, correction factor, insulin duration, and reverse correction.
- Medtronic 780G SmartGuard/Bolus Wizard uses configured bolus settings plus automated correction logic.

Official source links used for this framing:

- Tandem: https://www.tandemdiabetes.com/support-center/pumps-and-supplies/automated-insulin-delivery/article/using-personal-profiles-with-control-iq
- Omnipod: https://www.omnipod.com/current-podders/resources/omnipod-5/managing-glucose-levels
- Medtronic: https://www.medtronicdiabetes.com/customer-support/minimed-780g-system-support/entering-bolus-wizard-settings

## Primary Results

| Domain | Gluca | Pump-style baseline | Improvement | Value-add |
|---|---:|---:|---:|---|
| ISF correction response, adolescents | 3.09 mg/dL MAE | 35.12 mg/dL MAE | 91.20% | yes |
| ISF correction response, adults | 4.99 mg/dL MAE | 38.06 mg/dL MAE | 86.89% | yes |
| CR recovery | 20.15% error | 28.13% error | 28.37% | yes |
| Basal recovery | 3.65% error | 27.92% error | 86.91% | yes |
| Overall parameter recovery | 16.02% error | 24.09% error | 33.49% | yes |
| Meal inference | 0.884 F1 | 0.884 F1 | 0.00% | no, parity only |
| Carb absorption timing | 0.75 h MAE | 1.28 h MAE | 41.42% | yes |
| Dawn/morning rise | 6.74 mg/dL MAE | 6.19 mg/dL MAE | -8.81% | no |

## Statistical Checks

Paired bootstrap checks are in `paired_significance_checks.csv`.

Key rows:

- Adolescent ISF vs clinical/profile proxy: 91.20% improvement, 95% CI for mean error reduction `[23.94, 40.56]`, bootstrap P(reduction > 0) = 1.0.
- Adult ISF vs clinical/profile proxy: 86.89% improvement, 95% CI `[20.64, 45.88]`, bootstrap P(reduction > 0) = 1.0.
- CR recovery vs population/profile proxy: 35.17% mean-error improvement, 95% CI `[4.80, 15.30]`, bootstrap P(reduction > 0) = 0.9998.
- Overall parameter recovery vs population/profile proxy: 45.51% mean-error improvement, 95% CI `[9.40, 19.07]`, bootstrap P(reduction > 0) = 1.0.
- Carb absorption timing vs fixed 3h: 41.27% improvement, 95% CI `[0.42, 0.64]`, bootstrap P(reduction > 0) = 1.0.
- Dawn vs no-dawn: -8.81%; unsupported.

## Blog-Ready Plots

Plots are in `plots/`:

- `01_pump_proxy_value_add.png` - best headline figure.
- `02_isf_clean_correction_mae.png` - ISF correction-response error vs profile and robust baselines.
- `03_parameter_recovery_vs_profile_proxy.png` - CR, basal, and overall parameter recovery.
- `04_paired_error_reduction_forest.png` - paired bootstrap error-reduction checks.
- `05_absorption_and_meal_inference.png` - absorption timing win and meal inference parity.
- `06_strong_sanity_baselines.png` - sanity check against stronger non-Gluca baselines.

Recommended blog order:

1. `01_pump_proxy_value_add.png`
2. `03_parameter_recovery_vs_profile_proxy.png`
3. `02_isf_clean_correction_mae.png`
4. `04_paired_error_reduction_forest.png`
5. `05_absorption_and_meal_inference.png`

## Included Evidence Files

- `pump_proxy_claim_rows.csv` - primary headline table against current pump-style proxies.
- `strong_baseline_claim_rows.csv` - sanity table against strong non-Gluca baselines.
- `paired_significance_checks.csv` and `.json` - paired bootstrap checks.
- `parameter_task_rows.csv` - task-level parameter result rows.
- `meal_inference_detector_summary.csv` - meal inference detector comparisons.
- `carb_absorption_summary.csv` and `carb_absorption_event_rows.csv`.
- `dawn_summary.csv` and `dawn_event_rows.csv`.
- `comprehensive_validation_summary.json`.
- `comprehensive_validation_report.md`.

## Reproducible Code

Core estimator code:

- `bayesian_mpc/estimator_failure_modes.py`
- `bayesian_mpc/bayesian_v2.py`
- `bayesian_mpc/bayesian_estimator.py`

Validation and plotting code:

- `validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py`
- `validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py`
- `validation/comprehensive_estimator_validation/run_significance_checks.py`
- `validation/comprehensive_estimator_validation/make_blog_plots.py`

Commands used:

```bash
python validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py \
  --outdir validation/estimator_baseline_validation/estimator_baseline_latest

python validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py

python validation/comprehensive_estimator_validation/run_significance_checks.py

python validation/comprehensive_estimator_validation/make_blog_plots.py
```

Compile check:

```bash
python -m py_compile \
  bayesian_mpc/estimator_failure_modes.py \
  validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py \
  validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py \
  validation/comprehensive_estimator_validation/run_significance_checks.py \
  validation/comprehensive_estimator_validation/make_blog_plots.py
```

## Caveats

- This is simulator/offline validation, not clinical proof.
- The pump baselines are public-behavior proxies, not proprietary reimplementations of Tandem, Omnipod, or Medtronic algorithms.
- Clean ISF validation is strongest when correction events are isolated and identifiable.
- Meal inference currently reaches parity with bolus-context detection but does not add value over it.
- Dawn/morning-rise prediction is not supported and should be excluded from the strong claim.
- Strong robust personalized baselines remain important sanity checks. Gluca does not beat every robust baseline in every subgroup.
