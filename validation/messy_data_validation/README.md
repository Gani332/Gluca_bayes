# Messy Data Validation

This validation asks a narrower and more realistic question than clean ISF holdout:

> When the data are messy, does Gluca add value over naive empirical averaging?

It uses existing RL4BG free-living replay outputs, where meals, boluses, basal delivery and CGM are mixed. This is closer to real app data than isolated clean correction events.

Run:

```bash
python validation/messy_data_validation/run_messy_data_contribution.py
```

Outputs:

- `messy_latest/messy_data_aggregate.csv`
- `messy_latest/messy_data_comparisons.csv`
- `messy_latest/messy_data_contribution_result.json`
- `messy_latest/messy_data_contribution_report.md`
- `messy_latest/messy_data_contribution.png`
- `messy_latest/01_messy_value_add.png`
- `messy_latest/02_messy_parameter_recovery.png`
- `messy_latest/03_messy_stack_decomposition.png`
- `messy_latest/04_messy_paired_improvement.png`
- `messy_latest/05_messy_run_distribution.png`
- `messy_latest/06_messy_observation_sanity.png`

Claim boundary:

- Supported: Gluca beats naive empirical averaging on messy simulator replay.
- Supported: Gluca beats clean-gated empirical averaging on overall parameter recovery in this replay.
- Not supported: Gluca clinically beats current commercial pumps.
- Not supported: Bayesian posterior mean alone always beats robust personalised averaging.
