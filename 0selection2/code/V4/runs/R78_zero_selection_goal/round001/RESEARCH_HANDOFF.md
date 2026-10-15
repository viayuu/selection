# R78 Round001 TRAIN/VAL Handoff

Outcome: **inconclusive**. No coefficients or taskwise rules were searched.
R76 own-TRAIN frozen-score head fit; VAL already selected the backbone. Not independent confirmation.

| Split | Rule | Top1 % | Mean cost | Actual regret % | VS TRAIN SBS % |
| --- | --- | --- | --- | --- | --- |
| TRAIN | identity | 69.369646 | 11.373004840 | 0.460650092 | -0.143845417 |
| TRAIN | candidate | 69.425982 | 11.373099773 | 0.462689801 | -0.139222429 |
| VAL | identity | 69.236913 | 11.402474922 | 0.464503404 | -0.142069159 |
| VAL | candidate | 69.379546 | 11.402575925 | 0.467179029 | -0.137179498 |

Convergence, taskwise effects, changed decisions and all method counts are retained in the adjacent JSON/CSV files.
VAL taskwise adverse effects: TSP, ATSP, OVRPL.
A negative converged screen rejects this prescribed CE intercept fit, not all possible offsets or learnability.
If negative/inconclusive, keep identity and request the next Astra TRAIN/VAL-only research recommendation; do not select rules on TEST.

## Numerical Convergence Boundary

All 18 tasks used the fixed 30000-update budget, totaling 540000 optimizer
updates at exactly 1024 examples each. Only OVRPBTW passed the predeclared
two-successive-check convergence rule. The table shows final full-TRAIN checks;
one final gradient below 1e-5 alone does not satisfy that rule. The objective
change threshold is 1e-7. Unrounded values and the preceding checks are retained
in `fit/<problem>.json`; display precision below is not used for acceptance.

| Task | Updates | Final gradient infinity norm | Last objective change | Converged |
| --- | ---: | ---: | ---: | --- |
| ATSP | 30000 | 6.575809e-6 | 4.250445e-9 | no |
| CVRP | 30000 | 1.015904e-5 | 3.256932e-8 | no |
| OVRP | 30000 | 5.805616e-6 | 3.807869e-8 | no |
| OVRPB | 30000 | 9.782801e-6 | 1.030949e-8 | no |
| OVRPBL | 30000 | 2.401978e-6 | 1.324382e-9 | no |
| OVRPBLTW | 30000 | 1.044194e-6 | 2.207435e-8 | no |
| OVRPBTW | 30000 | 4.812209e-6 | 1.172548e-10 | yes |
| OVRPL | 30000 | 9.403813e-6 | 1.723789e-9 | no |
| OVRPLTW | 30000 | 1.237465e-5 | 1.267353e-10 | no |
| OVRPTW | 30000 | 1.200908e-5 | 1.491869e-9 | no |
| TSP | 30000 | 1.143289e-5 | 1.590297e-9 | no |
| VRPB | 30000 | 1.039863e-5 | 2.208089e-9 | no |
| VRPBL | 30000 | 2.115212e-5 | 8.863365e-8 | no |
| VRPBLTW | 30000 | 1.657241e-5 | 1.101234e-8 | no |
| VRPBTW | 30000 | 1.445879e-5 | 2.453646e-8 | no |
| VRPL | 30000 | 2.565004e-5 | 1.761150e-8 | no |
| VRPLTW | 30000 | 1.280425e-5 | 3.483842e-10 | no |
| VRPTW | 30000 | 2.381137e-6 | 2.170398e-9 | no |

This numerical candidate fails the observed zero-tolerance VAL mean-cost and
regret gates. That is distinct from showing that the fully converged optimum
of this fixed CE/prior objective would fail, or that all useful offsets fail.
No optimizer budget, convergence threshold, coefficient, or temperature was
changed after observing VAL. Identity remains selected and TEST was not opened.

## Coverage And Paired Uncertainty

The existing `methods_train.csv` and `methods_val.csv` give the number of
method/problem pairs with positive native winner support but zero selections:

| Split | Identity | Candidate |
| --- | ---: | ---: |
| TRAIN | 55 | 55 |
| VAL | 54 | 54 |

These counts are observations, not coverage targets or fitting weights.
Mean probabilities, precision/recall, margins and taskwise changes are in the
same CSV/JSON artifacts. No new backbone inference or baseline diagnosis was
performed to add this summary.

Paired VAL delta intervals from the predeclared 500 within-task instance
bootstrap replicates, candidate minus identity:

| Metric | 95% interval |
| --- | --- |
| Macro native Top1, percentage points | [0.046998, 0.239695] |
| Macro mean cost | [-0.000103564, 0.000314194] |
| Macro actual regret, percentage points | [-0.000302806, 0.005706569] |
| Macro VS TRAIN SBS, percentage points | [0.002239886, 0.007494755] |

Intervals do not replace the empirical zero-degradation gate, authorize a
cost tolerance, or establish non-inferiority. They assume independent instances
within each fixed task, do not account for unobserved generation groups, and do
not undo prior backbone selection on VAL. This remains an in-sample TRAIN
head-fit mechanism screen with an inconclusive optimization status.

Research-owner intake: use these TRAIN/VAL facts to select one next action.
R78-B is not automatically authorized by the Top1/cost trade-off; no TEST-based
rule selection, frequency balancing, or alpha/temperature sweep was performed.
