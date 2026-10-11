# R60: Expanded Partial Comparison Supervision

Seed2; original18 tasks and complete legal pools. Validation only; no test access.
R58 remains deferred. Legacy B feasible-domain discrepancies are not repaired.

## Validation-selected outcomes

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 11.168517 | -1.0700 | 0.926048 |
| R60A | ALL | 18000 | 48.7556 | 82.0389 | 92.5667 | 11.169491 | -1.0694 | 0.931175 |
| R60B | ALL | 18000 | 49.3611 | 82.2389 | 92.7222 | 11.168642 | -1.0735 | 0.922789 |

Both arms train the original R45A/R43A scoring architecture from scratch.
A adds old-TRAIN MOEL/MTL comparison replay; B uses additional TRAIN instances.
The original full-pool objective, native winner and R-Drop remain in both arms.
Extra data supervise only the observed utility difference with coefficient0.35.
No unobserved solver cost, full-pool winner, Oracle or risk target is invented.
No separate binary expert or candidate pruning is used at inference.

## Training and selection

| Arm | Best epoch | Stop epoch | Successful updates | Run wall hours | Last5 val Top1 % |
|---|---:|---:|---:|---:|---:|
| R60A | 32 | 40 | 56880 | 1.2959 | 48.8189 |
| R60B | 39 | 40 | 56880 | 1.4487 | 49.2544 |

Checkpoint selection: strict minimum original-validation macro_vs_sbs_pct.
Shared initialization, original batches and auxiliary task/scale plan. Replay A sets
the common LR trace and training horizon; B independently selects its best. Extra ties are
excluded from comparison loss, not relabeled as a full-pool winner.

## Selected-checkpoint train and validation

| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |
|---|---|---:|---:|---:|---:|
| R60A | train | 51.3961 | 1.040752 | 11.166574 | 0.859832 |
| R60A | val | 48.7556 | 1.107439 | 11.169491 | 0.931175 |
| R60B | train | 50.7706 | 1.041664 | 11.167328 | 0.872533 |
| R60B | val | 49.3611 | 1.092423 | 11.168642 | 0.922789 |

Train evaluation covers the original full-pool training set in eval mode.
Auxiliary comparison curves are online training diagnostics, not full-pool Top1.

## Decision changes

| Before | After | Category | N | Macro cost delta | Macro regret delta pp |
|---|---|---|---:|---:|---:|
| R45A | R60A | corrected | 1132 | -0.01503642 | -0.132347 |
| R45A | R60A | harmed | 1197 | +0.01581864 | +0.136905 |
| R45A | R60A | both_wrong_cheaper | 280 | -0.00311579 | -0.022996 |
| R45A | R60A | both_wrong_dearer | 264 | +0.00330754 | +0.023565 |
| R45A | R60A | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| R45A | R60A | unchanged | 15127 | +0.00000000 | +0.000000 |
| R45A | R60A | all | 18000 | +0.00097397 | +0.005127 |
| R45A | R60B | corrected | 897 | -0.01130418 | -0.102300 |
| R45A | R60B | harmed | 853 | +0.01110116 | +0.095132 |
| R45A | R60B | both_wrong_cheaper | 224 | -0.00230608 | -0.017241 |
| R45A | R60B | both_wrong_dearer | 255 | +0.00263483 | +0.021151 |
| R45A | R60B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| R45A | R60B | unchanged | 15771 | +0.00000000 | +0.000000 |
| R45A | R60B | all | 18000 | +0.00012573 | -0.003259 |
| R60A | R60B | corrected | 1030 | -0.01307978 | -0.117057 |
| R60A | R60B | harmed | 921 | +0.01212798 | +0.107346 |
| R60A | R60B | both_wrong_cheaper | 211 | -0.00246475 | -0.018160 |
| R60A | R60B | both_wrong_dearer | 257 | +0.00256831 | +0.019485 |
| R60A | R60B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| R60A | R60B | unchanged | 15581 | +0.00000000 | +0.000000 |
| R60A | R60B | all | 18000 | -0.00084823 | -0.008386 |

## Precommitted decision

Advancement screen passed: **False**.
B minus R45A Top1: +0.2444pp; required>=3.
B minus replay A: +0.6056pp; required>=2.
Regret no worse than R45A/A: True/True.

This screen is not proof of70% accuracy or performance on unseen test data.
A negative result rejects this data/optimization configuration, not a learning ceiling.

## Evidence limits

One seed. R45A is historical; A is the matched update/replay control.
Only MOEL/MTL receive expanded observations; do not generalize to all solver pairs.
Existing expansion provenance/calibration supports the selected columns; small
stratified replay is not proof of all-instance historical deployment equivalence.
No new solver runs or cost-table reconstruction occur in R60. Original prefix rows
are excluded from B auxiliary data. Source identities/counts are recorded in the
data manifest and protocol. Legacy task-semantics uncertainty remains explicit.

## Per problem

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | TSP | 1000 | 38.5000 | 61.1000 | 75.3000 | 7.914080 | -0.5494 | 0.561555 |
| R45A historical | CVRP | 1000 | 47.9000 | 69.2000 | 83.3000 | 18.202897 | -0.4998 | 0.676407 |
| R45A historical | ATSP | 1000 | 78.5000 | 96.8000 | 99.3000 | 1.634751 | -2.3169 | 0.313291 |
| R45A historical | OVRP | 1000 | 38.0000 | 65.8000 | 83.9000 | 7.451840 | -0.1316 | 0.662677 |
| R45A historical | VRPB | 1000 | 55.4000 | 91.7000 | 99.6000 | 9.332689 | -0.0811 | 0.695334 |
| R45A historical | VRPL | 1000 | 33.5000 | 59.2000 | 73.6000 | 12.399742 | -0.0602 | 0.451090 |
| R45A historical | VRPTW | 1000 | 47.0000 | 76.6000 | 92.9000 | 17.885999 | -4.1329 | 1.373462 |
| R45A historical | OVRPTW | 1000 | 51.0000 | 93.3000 | 100.0000 | 10.043002 | -0.0541 | 0.858606 |
| R45A historical | OVRPB | 1000 | 58.3000 | 96.8000 | 98.4000 | 6.316885 | -0.0806 | 0.906703 |
| R45A historical | OVRPL | 1000 | 40.2000 | 69.1000 | 86.2000 | 7.432119 | -0.0732 | 0.720632 |
| R45A historical | VRPBL | 1000 | 52.0000 | 91.6000 | 98.8000 | 9.359363 | +0.0010 | 0.815433 |
| R45A historical | VRPBTW | 1000 | 44.4000 | 81.0000 | 93.0000 | 19.040335 | -3.1999 | 1.635488 |
| R45A historical | VRPLTW | 1000 | 45.7000 | 81.7000 | 94.1000 | 17.617218 | -4.8189 | 1.236411 |
| R45A historical | OVRPBL | 1000 | 58.8000 | 95.5000 | 98.5000 | 6.300912 | -0.1481 | 0.903190 |
| R45A historical | OVRPBTW | 1000 | 50.0000 | 91.3000 | 99.7000 | 10.497878 | -0.0847 | 1.159206 |
| R45A historical | OVRPLTW | 1000 | 50.8000 | 92.7000 | 99.8000 | 10.127592 | +0.0038 | 0.885000 |
| R45A historical | VRPBLTW | 1000 | 44.0000 | 78.6000 | 93.4000 | 18.943402 | -2.9528 | 1.654986 |
| R45A historical | OVRPBLTW | 1000 | 50.1000 | 92.0000 | 99.9000 | 10.532592 | -0.0814 | 1.159401 |
| R60A | TSP | 1000 | 38.2000 | 60.4000 | 75.8000 | 7.913699 | -0.5542 | 0.562690 |
| R60A | CVRP | 1000 | 48.4000 | 68.5000 | 82.1000 | 18.202475 | -0.5021 | 0.683640 |
| R60A | ATSP | 1000 | 78.3000 | 96.9000 | 99.3000 | 1.634649 | -2.3230 | 0.309573 |
| R60A | OVRP | 1000 | 37.7000 | 65.9000 | 83.6000 | 7.453229 | -0.1130 | 0.688065 |
| R60A | VRPB | 1000 | 53.6000 | 91.7000 | 99.7000 | 9.330375 | -0.1059 | 0.688092 |
| R60A | VRPL | 1000 | 33.8000 | 58.6000 | 73.3000 | 12.399350 | -0.0634 | 0.445261 |
| R60A | VRPTW | 1000 | 44.9000 | 75.6000 | 92.2000 | 17.895790 | -4.0804 | 1.428869 |
| R60A | OVRPTW | 1000 | 50.0000 | 93.3000 | 100.0000 | 10.040605 | -0.0779 | 0.841905 |
| R60A | OVRPB | 1000 | 59.9000 | 96.9000 | 98.5000 | 6.312754 | -0.1459 | 0.851020 |
| R60A | OVRPL | 1000 | 40.1000 | 68.6000 | 86.1000 | 7.431056 | -0.0875 | 0.711232 |
| R60A | VRPBL | 1000 | 51.9000 | 91.0000 | 98.7000 | 9.359287 | +0.0002 | 0.812366 |
| R60A | VRPBTW | 1000 | 41.9000 | 80.3000 | 92.5000 | 19.048405 | -3.1588 | 1.685739 |
| R60A | VRPLTW | 1000 | 46.7000 | 80.6000 | 94.3000 | 17.625040 | -4.7766 | 1.258800 |
| R60A | OVRPBL | 1000 | 56.6000 | 95.3000 | 98.3000 | 6.303940 | -0.1002 | 0.956629 |
| R60A | OVRPBTW | 1000 | 50.7000 | 91.3000 | 99.7000 | 10.501678 | -0.0485 | 1.197726 |
| R60A | OVRPLTW | 1000 | 50.4000 | 92.7000 | 100.0000 | 10.123234 | -0.0392 | 0.843472 |
| R60A | VRPBLTW | 1000 | 42.2000 | 77.2000 | 92.3000 | 18.950649 | -2.9157 | 1.690141 |
| R60A | OVRPBLTW | 1000 | 52.3000 | 91.9000 | 99.8000 | 10.524614 | -0.1571 | 1.105937 |
| R60B | TSP | 1000 | 37.6000 | 60.2000 | 75.8000 | 7.918297 | -0.4964 | 0.624154 |
| R60B | CVRP | 1000 | 49.3000 | 69.3000 | 83.6000 | 18.199659 | -0.5175 | 0.654440 |
| R60B | ATSP | 1000 | 78.4000 | 96.8000 | 99.3000 | 1.634628 | -2.3242 | 0.306340 |
| R60B | OVRP | 1000 | 36.6000 | 65.4000 | 83.6000 | 7.454570 | -0.0950 | 0.704896 |
| R60B | VRPB | 1000 | 55.2000 | 91.8000 | 99.6000 | 9.333949 | -0.0677 | 0.719457 |
| R60B | VRPL | 1000 | 34.3000 | 58.6000 | 73.2000 | 12.398925 | -0.0668 | 0.443409 |
| R60B | VRPTW | 1000 | 46.7000 | 76.9000 | 93.0000 | 17.890947 | -4.1064 | 1.394335 |
| R60B | OVRPTW | 1000 | 51.5000 | 93.3000 | 100.0000 | 10.039644 | -0.0875 | 0.825358 |
| R60B | OVRPB | 1000 | 59.9000 | 96.8000 | 98.6000 | 6.312018 | -0.1576 | 0.834979 |
| R60B | OVRPL | 1000 | 40.7000 | 68.8000 | 85.9000 | 7.429528 | -0.1080 | 0.689648 |
| R60B | VRPBL | 1000 | 52.2000 | 91.3000 | 99.0000 | 9.359368 | +0.0011 | 0.822030 |
| R60B | VRPBTW | 1000 | 42.7000 | 80.2000 | 92.5000 | 19.043829 | -3.1821 | 1.659605 |
| R60B | VRPLTW | 1000 | 45.6000 | 81.0000 | 93.9000 | 17.622698 | -4.7893 | 1.255128 |
| R60B | OVRPBL | 1000 | 59.2000 | 95.5000 | 98.4000 | 6.301526 | -0.1384 | 0.900730 |
| R60B | OVRPBTW | 1000 | 50.8000 | 91.3000 | 99.8000 | 10.497273 | -0.0905 | 1.155130 |
| R60B | OVRPLTW | 1000 | 52.6000 | 92.7000 | 100.0000 | 10.122716 | -0.0443 | 0.837164 |
| R60B | VRPBLTW | 1000 | 42.8000 | 78.5000 | 92.9000 | 18.947743 | -2.9305 | 1.669868 |
| R60B | OVRPBLTW | 1000 | 52.4000 | 91.9000 | 99.9000 | 10.528245 | -0.1227 | 1.113532 |
