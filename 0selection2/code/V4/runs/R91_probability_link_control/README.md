# R91 Probability-Link Control

Two full18 fresh seed2 experiments reuse the frozen NEW90k data, R84 preparation,
and R88B known TSP execution-role input. They do not change the architecture,
native winner labels, legal pools, natural sampling, or inference ranking.

| Arm | CE | Risk probabilities | R-Drop probabilities |
| --- | --- | --- | --- |
| A_utility_ce | masked utility | original maximin | original maximin |
| B_utility_probabilities | masked utility | masked utility | masked utility |

Both retain the original within-instance utility-difference comparison loss. Its
predicted Top3 focus is computed from the **two original maximin outputs** as
before, not from averaged raw utility. Coefficients remain CE0.35,
comparison0.35, risk0.02 and the original R-Drop ramp to0.35. Final predictions
still use maximin; there is no extra head, calibration, sampling quota or bias.

## Fixed Budget

- Logical batch1024, microcap352, evaluation batch256; actual AdamW1e-4,
  weight decay1e-4, dropout0.1, norm clipping1.
- Warmup4320, maximum57600 successful updates, VAL every1440, cosine end LR1e-6.
- Same fresh trainables and batch/task streams, 58,982,400 row exposures per arm.
  Stream wrap means this is not exactly40 full dataset epochs.
- Choose the first strict minimum full18 VAL macro_vs_sbs_pct checkpoint using
  TRAIN-fitted SBS. One locked checkpoint/rule supplies all TEST and LIB results.
- Reuse two released gpu03/1465 allocation-owned3090s and the existing node cache.
  Do not interrupt R89 on gpu05 or rebuild the dataset/cache.

## Interpretation

This is a controlled probability-link study within the current full objective,
not a claim that utility CE has never been tried: R64/R65/R75 already used it
with different datasets and CE-plus-KL recipes. The same-role old-loss R88B and
main R84A results are historical controls, not newly matched simultaneous runs.
R90's Jacobian calculation alone neither proves harm nor predicts an improvement.

Use the prelocked VAL screen and report complete TEST results even on failure.
The fixed36 missed-winner target list measures true recovery, introduced errors
and cost changes, not whether methods merely get selected. Existing TEST/LIB
results have participated in prior research; they are not a new blind holdout.

See `protocol_lock.json` for the fixed decision rules and the launcher `LAUNCH.md`
for the actual allocation-bound commands. No solver execution, new labels,
candidate changes, R58 repairs, automatic restarts or partial Git commits.
