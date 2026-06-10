# Bayesian Parameter Estimation + Model Predictive Control
# for T1D insulin dosing.
#
# This is the approach used by real-world AID systems
# (Medtronic 780G, Omnipod 5, CamAPS FX, Loop).
#
# Core idea:
#   1. Physiological model predicts glucose forward 2-4 hours
#   2. MPC optimizes insulin to minimize predicted risk
#   3. Bayesian estimation adapts model to each patient online
#   4. UKF provides denoised glucose state from CGM
