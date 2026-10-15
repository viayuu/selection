# R78 Necessary Execution-Path Checks

Scope: executor checks of r78_zero_selection.py and test_r78.py, not Astra
research review and not an independent external code-review certification.
Requested execution/check routing: Solmax. No multiple broad review rounds.

Command in easynco with CUDA_VISIBLE_DEVICES empty:
`python -m unittest code.V4.test_r78`.
Observed: 9 tests, exit0, 2026-10-15.

Covered: fixed-prior finite-difference gradient; exactly1024 updates with
permutation-tail carry; rejecting VAL optimization or batch changes; real CE
optimization and sum-zero; identity preservation; candidate name/global-ID
alignment; masks and canonical score ties; valid-candidate oracle regret;
unrounded mean-cost/regret gate without substituting macro VS; negative gate
never opens TEST; repeated formal TEST cannot rewrite its frozen rule.

Execution-path corrections made before launch: paired regret uses the legal
mask, and the exclusive TEST-start marker is acquired before writing its lock.
The final focused suite passed after these corrections. This certifies those
deterministic checks only, not convergence, calibration, generalization, or an
R78 performance improvement. Full numerical convergence is reported from TRAIN.
