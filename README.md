# Gluca Bayes

Evidence repo for Gluca's Bayesian parameter-personalisation validation.

This repository contains the estimator code, validation scripts, CSV outputs, and blog-ready plots used to support the Gluca blog post. The short version: in messy simulator/offline replay, the gated Bayesian/trust personalisation layer improves overall therapy-parameter recovery versus naive empirical averaging and clean-gated averaging.

This is research code. It is not a medical device, not clinical proof, and must not be used to make insulin dosing decisions.

## Safe Claim

Supported claim:

> In messy simulator/offline replay with known ground truth, Gluca's clean event selection plus Bayesian/trust personalisation improves overall ISF, carb ratio, and basal parameter recovery versus naive empirical averaging and clean-gated empirical averaging.

Do not claim:

- Gluca is clinically superior to Tandem, Omnipod, Medtronic, or any other commercial pump.
- Gluca replaces a closed-loop controller.
- Gluca should be used for dosing.
- Gluca beats every robust personalised estimator in every subgroup.
- Bayesian posterior means alone always beat robust personalised averaging.
- Meal inference is a clear value-add yet.
- Dawn/morning-rise prediction is solved.

## Headline Results

Lower is better.

| Estimator | Overall mean absolute % error | Overall median absolute % error |
|---|---:|---:|
| Messy empirical mean | 38.58% | 30.19% |
| Clean-gated empirical mean | 19.79% | 19.63% |
| Clean-gated normal Bayes | 17.86% | 17.03% |
| Gluca Bayesian/trust | 16.89% | 16.02% |

Primary outputs:

```text
validation/messy_data_validation/messy_latest/messy_data_contribution_report.md
validation/messy_data_validation/messy_latest/messy_data_aggregate.csv
validation/messy_data_validation/messy_latest/messy_data_comparisons.csv
```

Older configured-profile proxy outputs are still included for transparency under:

```text
validation/comprehensive_estimator_validation/comprehensive_latest/
```

## Locked Tuning Sanity Check

I also ran a stricter tuning check where estimator hyperparameters are selected on development patients only, then evaluated once on locked-test patients.

Output:

```text
validation/locked_estimator_tuning/locked_isf_latest/locked_isf_tuning_result.json
```

Result:

| Locked-test comparison | Result |
|---|---:|
| Tuned Gluca ISF MAE | 1.93 mg/dL |
| Clinical/profile proxy MAE | 34.44 mg/dL |
| Best strong personalised baseline | empirical median |
| Best strong baseline MAE | 1.85 mg/dL |
| Improvement vs profile proxy | 94.40% |
| Improvement vs best strong baseline | -4.08% |

Interpretation: tuning improves strongly over the profile proxy, but does not beat the best robust personalised baseline on locked-test patients. This is why the claim stays focused on configured-profile proxies and explainable personalisation, not superiority over all personalised estimators or current commercial pump systems.

## Messy-Data Contribution Check

The clean ISF task is almost too clean: once the event is isolated, robust averaging is hard to beat. The more realistic test is messy free-living replay, where meals, boluses, basal delivery and CGM are mixed.

Output:

```text
validation/messy_data_validation/messy_latest/messy_data_contribution_report.md
```

Main result on RL4BG messy replay:

| Estimator | Overall mean absolute % error | Overall median absolute % error |
|---|---:|---:|
| Messy empirical mean | 38.58% | 30.19% |
| Clean-gated empirical mean | 19.79% | 19.63% |
| Clean-gated normal Bayes | 17.86% | 17.03% |
| Gluca Bayesian/trust | 16.89% | 16.02% |

Interpretation: the genuine contribution is the full personalisation stack, not Bayes as a magic word. Clean event selection cuts error sharply, and Gluca's Bayesian/trust layer adds a further overall improvement versus clean-gated averaging in this messy replay.

## What Is Being Compared

The baseline is not the private controller inside a commercial pump. Those algorithms are proprietary and are not reimplemented here.

The fair comparison is the part this approach is trying to improve: configured therapy parameters and profile assumptions such as correction factor/ISF, carb ratio, basal needs, targets, and insulin action/absorption timing.

Official pump-setting references used for this framing:

- Tandem Control-IQ personal profiles: https://www.tandemdiabetes.com/support-center/pumps-and-supplies/automated-insulin-delivery/article/using-personal-profiles-with-control-iq
- Omnipod 5 bolus settings: https://www.omnipod.com/current-podders/resources/omnipod-5/managing-glucose-levels
- Medtronic 780G Bolus Wizard settings: https://www.medtronicdiabetes.com/customer-support/minimed-780g-system-support/entering-bolus-wizard-settings

## Repository Structure

```text
bayesian_mpc/
  Bayesian estimators, trust-gated parameter learner, and supporting model code.

Gluca_ios/Gluca/Models/ParameterLearner.swift
  App-side Swift reference for the parameter learner.

validation/real_parameter_learning_validation/
  Padova clean correction-event validation data and outputs.

validation/estimator_baseline_validation/
  Estimator baseline benchmark code and cached outputs.

validation/comprehensive_estimator_validation/
  Final comprehensive validation, significance checks, CSVs, reports, and blog plots.
```

## Blog-Ready Plots

The current blog-ready messy-validation plots are in:

```text
validation/messy_data_validation/messy_latest/
```

Recommended order:

1. `01_messy_value_add.png`
2. `02_messy_parameter_recovery.png`
3. `03_messy_stack_decomposition.png`
4. `04_messy_paired_improvement.png`
5. `05_messy_run_distribution.png`
6. `06_messy_observation_sanity.png`

## Reproduce

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile check:

```bash
python -m py_compile \
  bayesian_mpc/*.py \
  validation/meal_inference_eval.py \
  validation/rl4bg_adolescent_validation/run_rl4bg_adolescent_validation.py \
  validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py \
  validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py \
  validation/comprehensive_estimator_validation/run_significance_checks.py \
  validation/comprehensive_estimator_validation/make_blog_plots.py
```

Fast smoke path:

```bash
python validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py \
  --skip-rl4bg \
  --outdir /tmp/gluca_estimator_smoke

python validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py \
  --outdir /tmp/gluca_comprehensive_smoke
```

Regenerate significance checks and blog plots:

```bash
python validation/comprehensive_estimator_validation/run_significance_checks.py
python validation/comprehensive_estimator_validation/make_blog_plots.py
```

Regenerate the messy-data validation and current blog plots:

```bash
python validation/messy_data_validation/run_messy_data_contribution.py
```

The full estimator benchmark without `--skip-rl4bg` uses the public `MLD3/RL4BG` repository. The helper script will clone it to `/tmp/RL4BG` if it is not already present.

## Important Caveats

- This is simulator/offline validation, not clinical validation.
- Chronological splits are used where holdout evaluation is needed.
- Ground truth is used where the simulator exposes it.
- Pump baselines are public-behaviour/configured-profile proxies, not proprietary controller clones.
- The strongest current blog claim is messy estimator validation, not commercial pump superiority.
- Meal inference currently matches the bolus-context baseline rather than beating it.
- Dawn/morning-rise prediction is currently unsupported.
- Strong robust personalised baselines remain important sanity checks.

## Main Evidence Files

- `validation/comprehensive_estimator_validation/comprehensive_latest/CLAUDE_HANDOFF_README.md`
- `validation/comprehensive_estimator_validation/comprehensive_latest/comprehensive_validation_report.md`
- `validation/comprehensive_estimator_validation/comprehensive_latest/pump_proxy_claim_rows.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/strong_baseline_claim_rows.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/paired_significance_checks.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/parameter_task_rows.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/meal_inference_detector_summary.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/carb_absorption_summary.csv`
- `validation/comprehensive_estimator_validation/comprehensive_latest/dawn_summary.csv`
- `validation/messy_data_validation/messy_latest/messy_data_contribution_report.md`
- `validation/messy_data_validation/messy_latest/messy_data_aggregate.csv`
- `validation/messy_data_validation/messy_latest/messy_data_comparisons.csv`
- `validation/messy_data_validation/messy_latest/messy_data_contribution_result.json`

## Contributing

Pull requests are welcome for bug fixes, clearer docs, stronger baselines, reproducibility improvements, and better validation design.

If you think a claim is too strong, the preferred contribution is a failing test, stronger comparator, or cleaner validation split that makes the boundary obvious.
