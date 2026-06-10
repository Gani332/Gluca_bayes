# Real Parameter-Learning Validation

This folder contains fresh UVA/Padova simulator validation for Gluca's Bayesian ISF learner.

## What Was Run

- Fresh `simglucose` UVA/Padova adult virtual-patient correction challenge.
- App-style clean correction gate: no meals, no exercise, no overlapping boluses, 180-minute response, observed ISF in physiologic range.
- Per-patient chronological 70/30 train/holdout split.
- Baselines:
  - fixed population ISF = 50 mg/dL/U
  - clinical-prior ISF derived from cohort and weight
  - empirical mean ISF from train events
  - Bayesian posterior ISF from train events

## Main Result

In the 10-adult run, 9 patients produced usable clean events:

- Usable clean correction events: 72
- Holdout events: 27
- Fixed 50 MAE: 57.962 mg/dL
- Bayesian MAE: 5.831 mg/dL
- Improvement vs fixed 50: 89.94%
- Paired bootstrap error reduction vs fixed 50: 52.131 mg/dL, 95% CI [40.576, 63.095]

The Bayesian posterior also beat the clinical-prior baseline, but did not beat the simple empirical-mean personalized baseline.

## Claim Boundary

Safe claim:

> In a fresh UVA/Padova adult correction-challenge validation with chronological holdout splits, Gluca's Bayesian ISF posterior reduced future correction-response prediction error versus a fixed 50 mg/dL/U population baseline.

Do not claim this proves the same result on personal free-living CGM data. Personal-data validation is still blocked until there is a local export or working authenticated Supabase/database access.

## Files

- `run_real_padova_parameter_validation.py`: reproducible validation script.
- `real_validation_results.json`: machine-readable results.
- `real_validation_report.md`: short human-readable verdict.
- `padova_clean_correction_events.csv`: all usable clean correction events.
- `padova_holdout_predictions.csv`: holdout predictions and errors.
- `padova_patient_summary.csv`: per-patient train/holdout summaries.

## Reproduce

From the repo root:

```bash
./.venv/bin/python validation/real_parameter_learning_validation/run_real_padova_parameter_validation.py --max-patients 10 --events-per-patient 8 --max-days 35
```
