# Gluca Blog Supporting Package

This package contains the code, CSV outputs, plots, and source snippets needed to audit the Gluca blog validation claim.

Safe claim:

> In simulator/offline validation with chronological splits and known ground truth where available, Gluca's Bayesian/trust-gated personalization improves several therapy-parameter and event-state estimates versus pump-style configured-profile proxies.

Do not claim clinical superiority over commercial pumps as full closed-loop controllers. Meal inference is parity only, dawn/morning-rise prediction is unsupported, and robust personalized baselines remain important sanity checks.

## Contents

- `validation/comprehensive_estimator_validation/comprehensive_latest/`: final JSON/CSV/plot outputs used by the blog.
- `validation/comprehensive_estimator_validation/*.py`: comprehensive validation, significance checks, and blog plot generation.
- `validation/estimator_baseline_validation/estimator_baseline_latest/`: estimator baseline outputs consumed by the comprehensive validation.
- `validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py`: estimator benchmark script.
- `validation/real_parameter_learning_validation/`: clean Padova correction-event source CSVs.
- `validation/meal_inference_eval.py`: meal inference evaluation utilities used by the comprehensive validation.
- `validation/rl4bg_adolescent_validation/run_rl4bg_adolescent_validation.py`: RL4BG helper used by the estimator benchmark.
- `bayesian_mpc/`: Bayesian estimator and supporting research modules.
- `Gluca_ios/Gluca/Models/ParameterLearner.swift`: app-side parameter learner reference.

## Smoke-Tested Commands

From the unzipped package root:

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

```bash
python validation/estimator_baseline_validation/run_estimator_baseline_benchmark.py \
  --skip-rl4bg \
  --outdir /tmp/gluca_estimator_smoke
```

```bash
python validation/comprehensive_estimator_validation/run_comprehensive_estimator_validation.py \
  --outdir /tmp/gluca_comprehensive_smoke
```

```bash
python validation/comprehensive_estimator_validation/run_significance_checks.py
python validation/comprehensive_estimator_validation/make_blog_plots.py
```

The full estimator benchmark without `--skip-rl4bg` uses the public MLD3/RL4BG repository. The helper will clone it to `/tmp/RL4BG` if it is not already present.

## Python Dependencies

Core smoke path:

- `numpy`
- `pandas`
- `matplotlib`

Optional/full simulator reruns:

- `simglucose`
- `scipy`
- the public `MLD3/RL4BG` repo

