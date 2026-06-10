# Locked Estimator Tuning

This folder is for honest tuning experiments.

The rule is simple:

1. Tune on development patients only.
2. Freeze the selected estimator.
3. Evaluate once on locked-test patients.
4. Report whether the method beats profile proxies and strong personalised baselines.

This is deliberately separate from the main blog validation path so we do not accidentally tune the published evidence until it says what we want.

Run:

```bash
python validation/locked_estimator_tuning/run_locked_isf_tuning.py
```

Outputs:

- `locked_isf_latest/candidate_dev_scores.csv`
- `locked_isf_latest/locked_isf_predictions.csv`
- `locked_isf_latest/locked_isf_summary.csv`
- `locked_isf_latest/locked_isf_tuning_result.json`

The JSON contains explicit booleans for:

- `beats_profile_proxy`
- `beats_best_strong_baseline`
- `can_claim_beats_current_pumps`

The last value should remain `false` unless the validation is changed to a real closed-loop controller comparison against a credible pump/controller baseline.
