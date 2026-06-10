# Claim Boundaries

This repo supports a narrow, evidence-backed claim:

> Gluca is an explainable Bayesian personalisation layer that improves several therapy-parameter estimates versus pump-style configured-profile proxies in simulator/offline validation.

## Supported

- ISF correction-response estimation improves versus fixed/profile proxy baselines.
- Carb ratio recovery improves versus configured-profile proxy baselines.
- Basal recovery improves versus configured-profile proxy baselines.
- Overall parameter recovery improves versus configured-profile proxy baselines.
- Carb absorption timing improves versus a fixed 3-hour assumption.
- The method is explainable: posterior mean, posterior width, confidence, and event gating are inspectable.

## Not Supported

- Clinical superiority over commercial pump systems.
- Replacement of a closed-loop controller.
- Direct dosing use.
- Value-add over every robust personalised estimator.
- Meal inference superiority over bolus-context detection.
- Dawn/morning-rise prediction.

## Why This Boundary Matters

Commercial AID systems are full controller systems. Gluca is not being validated here as a full controller. The validation target is the parameter-learning layer: can we learn better personalised therapy parameters from observed events than fixed/configured profile proxies?

That is useful, but it is not the same as proving clinical pump superiority.
