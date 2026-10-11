# R62: Source-Separated Portfolio-Performance Pretraining

Seed2; original18 tasks, complete legal pools and native winners. Validation only.
No test access, new solver execution, new annotations or R58 operations.

## Validation-selected outcomes

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 11.168517 | -1.0700 | 0.926048 |
| R60B incumbent | ALL | 18000 | 49.3611 | 82.2389 | 92.7222 | 11.168642 | -1.0735 | 0.922789 |
| R62A | ALL | 18000 | 49.0278 | 82.2111 | 92.7167 | 11.168751 | -1.0715 | 0.925225 |
| R62B | ALL | 18000 | 48.8500 | 82.3222 | 92.7778 | 11.169226 | -1.0653 | 0.931233 |

A and B pretrain on the same available auxiliary deployment source.
A replays a fixed nested new10k/task subset; B uses the complete available new-input bank.
Missing auxiliary observations restrict pair supervision, never the forward candidate pool.
Stage1 learns observed non-tied utility comparisons without constructing partial-pool winners.
The auxiliary score head is reset; all other original selector parameters transfer.
Stage2 uses only unchanged legacy full-pool labels and the original R43 objective/R-Drop.
Inference uses the original one-pass single-solver selector, not a pretraining branch.

## Matched training

| Arm | Pretrain updates | Finetune updates | Best epoch | Stop epoch | Total run hours |
|---|---:|---:|---:|---:|---:|
| R62A | 25344 | 56880 | 39 | 40 | 1.5997 |
| R62B | 25344 | 56880 | 24 | 40 | 1.5991 |

Both arms transfer the final fixed-budget pretraining state, not a validation-picked state.
A controls the common fine-tuning LR trace/horizon; each selects strict minimum macro_vs_sbs_pct.
Scheduled updates and actual input-size exposure are matched, not observed pair counts.
Actual source/observation coverage is recorded by ordinary conversion in data_cache/preparation.json.

## Selected-checkpoint train and validation

| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |
|---|---|---:|---:|---:|---:|
| R62A | train | 50.7061 | 1.040901 | 11.167878 | 0.877744 |
| R62A | val | 49.0278 | 1.094684 | 11.168751 | 0.925225 |
| R62B | train | 50.3511 | 1.049121 | 11.169086 | 0.887372 |
| R62B | val | 48.8500 | 1.089627 | 11.169226 | 0.931233 |

Train evaluation uses eval mode, original full-pool TRAIN and all tail batches.
Auxiliary training pair accuracy is not full-pool selection accuracy.

## Decision changes

| Before | After | Category | N | Macro cost delta | Macro regret delta pp |
|---|---|---|---:|---:|---:|
| R60B | R62A | corrected | 1117 | -0.01582737 | -0.134435 |
| R60B | R62A | harmed | 1177 | +0.01574024 | +0.134008 |
| R60B | R62A | both_wrong_cheaper | 318 | -0.00335111 | -0.025090 |
| R60B | R62A | both_wrong_dearer | 323 | +0.00354726 | +0.027953 |
| R60B | R62A | both_wrong_equal | 2 | +0.00000000 | +0.000000 |
| R60B | R62A | unchanged | 15063 | +0.00000000 | +0.000000 |
| R60B | R62A | all | 18000 | +0.00010902 | +0.002436 |
| R60B | R62B | corrected | 1247 | -0.01663817 | -0.143676 |
| R60B | R62B | harmed | 1339 | +0.01718675 | +0.152127 |
| R60B | R62B | both_wrong_cheaper | 369 | -0.00382132 | -0.028509 |
| R60B | R62B | both_wrong_dearer | 349 | +0.00385693 | +0.028503 |
| R60B | R62B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R60B | R62B | unchanged | 14695 | +0.00000000 | +0.000000 |
| R60B | R62B | all | 18000 | +0.00058419 | +0.008444 |
| R62A | R62B | corrected | 1028 | -0.01446643 | -0.119704 |
| R62A | R62B | harmed | 1060 | +0.01460348 | +0.126115 |
| R62A | R62B | both_wrong_cheaper | 290 | -0.00260910 | -0.020785 |
| R62A | R62B | both_wrong_dearer | 271 | +0.00294722 | +0.020381 |
| R62A | R62B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R62A | R62B | unchanged | 15350 | +0.00000000 | +0.000000 |
| R62A | R62B | all | 18000 | +0.00047517 | +0.006008 |

## Precommitted decision

Advancement screen passed: **False**.
B minus incumbent Top1: -0.5111pp; required>=3.
B minus subset A: -0.1778pp; required>=2.
Regret no worse than incumbent/A: False/False.

This is a direction screen, not unseen-test evidence or achievement of70% Top1.
Preserve partial gains even when the substantial-gain screen fails.
Failure rejects this transfer recipe, not the possibility of better selection or an accuracy ceiling.

## Evidence limits

One seed. R45A/R60B are historical references, not newly trained matched controls.
Expansion deployments can differ in weights, budgets, stochastic execution and constraint semantics.
They provide explicitly different-source auxiliary knowledge, not certified legacy labels.
Observed cells are never interpreted as a complete-pool winner or used to replace original costs.
Both matched arms use the same frozen auxiliary source; normal loading records actual coverage.
Existing source calibration and VRPLTW input-repair limitations remain disclosed.
The legacy B feasible-domain mismatch remains unresolved; this is not a repaired common-task benchmark.

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
| R60B incumbent | TSP | 1000 | 37.6000 | 60.2000 | 75.8000 | 7.918297 | -0.4964 | 0.624154 |
| R60B incumbent | CVRP | 1000 | 49.3000 | 69.3000 | 83.6000 | 18.199659 | -0.5175 | 0.654440 |
| R60B incumbent | ATSP | 1000 | 78.4000 | 96.8000 | 99.3000 | 1.634628 | -2.3242 | 0.306340 |
| R60B incumbent | OVRP | 1000 | 36.6000 | 65.4000 | 83.6000 | 7.454570 | -0.0950 | 0.704896 |
| R60B incumbent | VRPB | 1000 | 55.2000 | 91.8000 | 99.6000 | 9.333949 | -0.0677 | 0.719457 |
| R60B incumbent | VRPL | 1000 | 34.3000 | 58.6000 | 73.2000 | 12.398925 | -0.0668 | 0.443409 |
| R60B incumbent | VRPTW | 1000 | 46.7000 | 76.9000 | 93.0000 | 17.890947 | -4.1064 | 1.394335 |
| R60B incumbent | OVRPTW | 1000 | 51.5000 | 93.3000 | 100.0000 | 10.039644 | -0.0875 | 0.825358 |
| R60B incumbent | OVRPB | 1000 | 59.9000 | 96.8000 | 98.6000 | 6.312018 | -0.1576 | 0.834979 |
| R60B incumbent | OVRPL | 1000 | 40.7000 | 68.8000 | 85.9000 | 7.429528 | -0.1080 | 0.689648 |
| R60B incumbent | VRPBL | 1000 | 52.2000 | 91.3000 | 99.0000 | 9.359368 | +0.0011 | 0.822030 |
| R60B incumbent | VRPBTW | 1000 | 42.7000 | 80.2000 | 92.5000 | 19.043829 | -3.1821 | 1.659605 |
| R60B incumbent | VRPLTW | 1000 | 45.6000 | 81.0000 | 93.9000 | 17.622698 | -4.7893 | 1.255128 |
| R60B incumbent | OVRPBL | 1000 | 59.2000 | 95.5000 | 98.4000 | 6.301526 | -0.1384 | 0.900730 |
| R60B incumbent | OVRPBTW | 1000 | 50.8000 | 91.3000 | 99.8000 | 10.497273 | -0.0905 | 1.155130 |
| R60B incumbent | OVRPLTW | 1000 | 52.6000 | 92.7000 | 100.0000 | 10.122716 | -0.0443 | 0.837164 |
| R60B incumbent | VRPBLTW | 1000 | 42.8000 | 78.5000 | 92.9000 | 18.947743 | -2.9305 | 1.669868 |
| R60B incumbent | OVRPBLTW | 1000 | 52.4000 | 91.9000 | 99.9000 | 10.528245 | -0.1227 | 1.113532 |
| R62A | TSP | 1000 | 37.4000 | 61.6000 | 76.4000 | 7.916406 | -0.5202 | 0.600110 |
| R62A | CVRP | 1000 | 48.8000 | 70.7000 | 82.7000 | 18.201201 | -0.5091 | 0.658703 |
| R62A | ATSP | 1000 | 78.0000 | 96.9000 | 99.4000 | 1.634864 | -2.3101 | 0.322526 |
| R62A | OVRP | 1000 | 35.0000 | 65.5000 | 84.1000 | 7.455683 | -0.0801 | 0.719946 |
| R62A | VRPB | 1000 | 56.7000 | 91.8000 | 99.5000 | 9.329863 | -0.1114 | 0.668808 |
| R62A | VRPL | 1000 | 32.3000 | 56.4000 | 72.2000 | 12.404174 | -0.0245 | 0.485484 |
| R62A | VRPTW | 1000 | 45.6000 | 75.8000 | 93.3000 | 17.893616 | -4.0921 | 1.427731 |
| R62A | OVRPTW | 1000 | 50.9000 | 93.3000 | 100.0000 | 10.042699 | -0.0571 | 0.861734 |
| R62A | OVRPB | 1000 | 59.3000 | 96.8000 | 98.4000 | 6.313452 | -0.1349 | 0.860221 |
| R62A | OVRPL | 1000 | 38.8000 | 69.2000 | 86.4000 | 7.431322 | -0.0839 | 0.709964 |
| R62A | VRPBL | 1000 | 54.6000 | 91.4000 | 98.8000 | 9.353091 | -0.0660 | 0.747640 |
| R62A | VRPBTW | 1000 | 41.7000 | 80.5000 | 93.1000 | 19.051512 | -3.1430 | 1.703024 |
| R62A | VRPLTW | 1000 | 46.4000 | 79.9000 | 93.1000 | 17.616986 | -4.8201 | 1.209295 |
| R62A | OVRPBL | 1000 | 58.6000 | 95.5000 | 98.4000 | 6.302353 | -0.1253 | 0.914519 |
| R62A | OVRPBTW | 1000 | 52.0000 | 91.3000 | 99.7000 | 10.495216 | -0.1100 | 1.137742 |
| R62A | OVRPLTW | 1000 | 50.9000 | 92.7000 | 99.9000 | 10.124136 | -0.0303 | 0.871409 |
| R62A | VRPBLTW | 1000 | 45.7000 | 78.5000 | 93.6000 | 18.940439 | -2.9680 | 1.618564 |
| R62A | OVRPBLTW | 1000 | 49.8000 | 92.0000 | 99.9000 | 10.530509 | -0.1012 | 1.136628 |
| R62B | TSP | 1000 | 37.9000 | 61.4000 | 76.7000 | 7.914807 | -0.5403 | 0.575045 |
| R62B | CVRP | 1000 | 50.3000 | 70.6000 | 82.7000 | 18.200608 | -0.5123 | 0.655278 |
| R62B | ATSP | 1000 | 78.3000 | 96.9000 | 99.5000 | 1.634944 | -2.3053 | 0.325681 |
| R62B | OVRP | 1000 | 35.2000 | 66.7000 | 84.6000 | 7.455788 | -0.0787 | 0.714818 |
| R62B | VRPB | 1000 | 55.0000 | 91.8000 | 99.5000 | 9.334598 | -0.0607 | 0.715949 |
| R62B | VRPL | 1000 | 32.4000 | 57.0000 | 73.2000 | 12.403058 | -0.0335 | 0.475300 |
| R62B | VRPTW | 1000 | 45.7000 | 75.4000 | 92.4000 | 17.889208 | -4.1157 | 1.395379 |
| R62B | OVRPTW | 1000 | 49.9000 | 93.3000 | 100.0000 | 10.044773 | -0.0364 | 0.879604 |
| R62B | OVRPB | 1000 | 58.6000 | 96.8000 | 98.3000 | 6.317174 | -0.0760 | 0.914902 |
| R62B | OVRPL | 1000 | 40.9000 | 68.8000 | 86.1000 | 7.430540 | -0.0944 | 0.700032 |
| R62B | VRPBL | 1000 | 53.5000 | 91.4000 | 98.9000 | 9.354573 | -0.0502 | 0.759745 |
| R62B | VRPBTW | 1000 | 42.0000 | 80.8000 | 93.2000 | 19.057633 | -3.1119 | 1.738800 |
| R62B | VRPLTW | 1000 | 45.7000 | 80.4000 | 93.6000 | 17.626828 | -4.7669 | 1.262181 |
| R62B | OVRPBL | 1000 | 57.2000 | 95.5000 | 98.3000 | 6.303096 | -0.1135 | 0.929909 |
| R62B | OVRPBTW | 1000 | 50.3000 | 91.3000 | 99.7000 | 10.496327 | -0.0995 | 1.171116 |
| R62B | OVRPLTW | 1000 | 51.1000 | 92.7000 | 100.0000 | 10.124355 | -0.0281 | 0.864815 |
| R62B | VRPBLTW | 1000 | 45.4000 | 79.0000 | 93.4000 | 18.930777 | -3.0175 | 1.572229 |
| R62B | OVRPBLTW | 1000 | 49.9000 | 92.0000 | 99.9000 | 10.526988 | -0.1346 | 1.111411 |
