# R61: Dense Saved-Route Behavior Pretraining

Seed2; original18 tasks, complete legal pools and native winners. Validation only.
No test access, new solver execution, new annotations or R58 operations.

## Validation-selected outcomes

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 11.168517 | -1.0700 | 0.926048 |
| R60B incumbent | ALL | 18000 | 49.3611 | 82.2389 | 92.7222 | 11.168642 | -1.0735 | 0.922789 |
| R61A | ALL | 18000 | 48.8111 | 82.4000 | 92.7667 | 11.167871 | -1.0732 | 0.924975 |
| R61B | ALL | 18000 | 49.0778 | 82.3222 | 92.8667 | 11.168711 | -1.0680 | 0.929035 |

A pretrains on original-TRAIN route replay; B uses additional saved TRAIN routes.
Both transfer only the final instance encoder into fresh complete-pool selectors.
Fine-tuning uses the unchanged original R43 objective and R-Drop; route heads are discarded.
There is no route prediction, solver probing, binary expert or candidate gate at inference.

## Matched training

| Arm | Pretrain updates | Finetune updates | Best epoch | Stop epoch | Total run hours |
|---|---:|---:|---:|---:|---:|
| R61A | 21120 | 45504 | 24 | 32 | 1.0465 |
| R61B | 21120 | 45504 | 32 | 32 | 1.0601 |

Stage1 transfers the final encoder after two fixed complete passes; no validation selection.
Stage2 selects strict minimum full18 validation macro_vs_sbs_pct, not best Top1.
Replay A controls the common fine-tuning LR trace/horizon; B selects its own best checkpoint.
Initialization, source split, actual-size schedule and update budgets are matched.

## Selected-checkpoint train and validation

| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |
|---|---|---:|---:|---:|---:|
| R61A | train | 49.9856 | 1.052510 | 11.171298 | 0.902166 |
| R61A | val | 48.8111 | 1.088655 | 11.167871 | 0.924975 |
| R61B | train | 50.5289 | 1.046239 | 11.169085 | 0.883698 |
| R61B | val | 49.0778 | 1.091640 | 11.168711 | 0.929035 |

Train evaluation covers original full-pool TRAIN in eval mode.
Pretraining curves show online successor learning, not full-pool selection accuracy.

## Decision changes

| Before | After | Category | N | Macro cost delta | Macro regret delta pp |
|---|---|---|---:|---:|---:|
| R60B | R61A | corrected | 1445 | -0.02006193 | -0.164989 |
| R60B | R61A | harmed | 1544 | +0.01955230 | +0.172063 |
| R60B | R61A | both_wrong_cheaper | 432 | -0.00471832 | -0.035083 |
| R60B | R61A | both_wrong_dearer | 391 | +0.00445635 | +0.030195 |
| R60B | R61A | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R60B | R61A | unchanged | 14187 | +0.00000000 | +0.000000 |
| R60B | R61A | all | 18000 | -0.00077159 | +0.002186 |
| R60B | R61B | corrected | 1259 | -0.01735363 | -0.145567 |
| R60B | R61B | harmed | 1310 | +0.01734612 | +0.153064 |
| R60B | R61B | both_wrong_cheaper | 331 | -0.00379129 | -0.027872 |
| R60B | R61B | both_wrong_dearer | 364 | +0.00386770 | +0.026621 |
| R60B | R61B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R60B | R61B | unchanged | 14735 | +0.00000000 | +0.000000 |
| R60B | R61B | all | 18000 | +0.00006890 | +0.006246 |
| R61A | R61B | corrected | 1181 | -0.01452712 | -0.125444 |
| R61A | R61B | harmed | 1133 | +0.01539735 | +0.128072 |
| R61A | R61B | both_wrong_cheaper | 276 | -0.00291001 | -0.020567 |
| R61A | R61B | both_wrong_dearer | 340 | +0.00288028 | +0.021999 |
| R61A | R61B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| R61A | R61B | unchanged | 15070 | +0.00000000 | +0.000000 |
| R61A | R61B | all | 18000 | +0.00084050 | +0.004060 |

## Precommitted decision

Advancement screen passed: **False**.
B minus incumbent Top1: -0.2833pp; required>=3.
B minus replay A: +0.2667pp; required>=2.
Regret no worse than incumbent/A: False/False.

The screen is a direction decision, not proof of70% or unseen-test performance.
Better successor learning alone cannot establish better solver selection.
A negative result rejects this pretraining/transfer recipe, not an accuracy ceiling.

## Evidence limits

One seed. R45A and the locked R60B epoch39 incumbent are historical references.
Only the two archived ReLD route behaviors provide expanded structural targets.
No additional winner or complete-pool label is invented for expanded rows.
Actual route availability/counts and source pairing are recorded during target conversion.
Small existing calibration is not complete historical deployment certification.
Legacy B feasible-domain inconsistencies and the existing VRPLTW repair provenance
remain disclosed; this experiment does not claim a repaired common-task benchmark.

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
| R61A | TSP | 1000 | 38.3000 | 62.1000 | 76.6000 | 7.915435 | -0.5324 | 0.579517 |
| R61A | CVRP | 1000 | 48.1000 | 69.5000 | 83.2000 | 18.206621 | -0.4794 | 0.686366 |
| R61A | ATSP | 1000 | 79.1000 | 97.0000 | 99.1000 | 1.634773 | -2.3156 | 0.314553 |
| R61A | OVRP | 1000 | 34.4000 | 65.2000 | 83.7000 | 7.455627 | -0.0809 | 0.711360 |
| R61A | VRPB | 1000 | 54.8000 | 91.9000 | 99.4000 | 9.332972 | -0.0781 | 0.708272 |
| R61A | VRPL | 1000 | 33.8000 | 57.7000 | 74.0000 | 12.400079 | -0.0575 | 0.452646 |
| R61A | VRPTW | 1000 | 44.7000 | 76.2000 | 93.1000 | 17.885290 | -4.1367 | 1.377421 |
| R61A | OVRPTW | 1000 | 50.1000 | 93.3000 | 100.0000 | 10.044316 | -0.0410 | 0.882080 |
| R61A | OVRPB | 1000 | 57.6000 | 96.8000 | 98.3000 | 6.315438 | -0.1035 | 0.891542 |
| R61A | OVRPL | 1000 | 39.3000 | 69.5000 | 85.8000 | 7.429913 | -0.1028 | 0.698422 |
| R61A | VRPBL | 1000 | 51.7000 | 91.9000 | 99.2000 | 9.358075 | -0.0128 | 0.814697 |
| R61A | VRPBTW | 1000 | 44.7000 | 81.0000 | 93.1000 | 19.034325 | -3.2304 | 1.611696 |
| R61A | VRPLTW | 1000 | 48.9000 | 80.7000 | 93.0000 | 17.628592 | -4.7574 | 1.288366 |
| R61A | OVRPBL | 1000 | 57.8000 | 95.5000 | 98.6000 | 6.302612 | -0.1212 | 0.916084 |
| R61A | OVRPBTW | 1000 | 48.5000 | 91.3000 | 99.8000 | 10.500372 | -0.0610 | 1.194688 |
| R61A | OVRPLTW | 1000 | 52.1000 | 92.7000 | 100.0000 | 10.121782 | -0.0536 | 0.840717 |
| R61A | VRPBLTW | 1000 | 44.6000 | 78.9000 | 93.1000 | 18.925910 | -3.0424 | 1.538793 |
| R61A | OVRPBLTW | 1000 | 50.1000 | 92.0000 | 99.8000 | 10.529541 | -0.1104 | 1.142330 |
| R61B | TSP | 1000 | 38.1000 | 61.5000 | 76.2000 | 7.914884 | -0.5393 | 0.582393 |
| R61B | CVRP | 1000 | 47.5000 | 68.3000 | 82.5000 | 18.206374 | -0.4808 | 0.686990 |
| R61B | ATSP | 1000 | 78.2000 | 97.0000 | 99.3000 | 1.634855 | -2.3106 | 0.320559 |
| R61B | OVRP | 1000 | 36.2000 | 65.4000 | 84.1000 | 7.453730 | -0.1063 | 0.691467 |
| R61B | VRPB | 1000 | 55.8000 | 91.8000 | 99.6000 | 9.331731 | -0.0914 | 0.691884 |
| R61B | VRPL | 1000 | 33.5000 | 59.1000 | 74.1000 | 12.402267 | -0.0399 | 0.470318 |
| R61B | VRPTW | 1000 | 47.5000 | 76.8000 | 94.0000 | 17.878294 | -4.1742 | 1.318763 |
| R61B | OVRPTW | 1000 | 49.7000 | 93.3000 | 100.0000 | 10.046478 | -0.0195 | 0.902513 |
| R61B | OVRPB | 1000 | 58.7000 | 96.8000 | 98.6000 | 6.317358 | -0.0731 | 0.935383 |
| R61B | OVRPL | 1000 | 39.9000 | 69.3000 | 86.1000 | 7.431144 | -0.0863 | 0.715552 |
| R61B | VRPBL | 1000 | 52.5000 | 91.8000 | 99.0000 | 9.356510 | -0.0295 | 0.789296 |
| R61B | VRPBTW | 1000 | 43.8000 | 80.7000 | 93.0000 | 19.042681 | -3.1879 | 1.640781 |
| R61B | VRPLTW | 1000 | 46.4000 | 80.3000 | 93.6000 | 17.623317 | -4.7859 | 1.244693 |
| R61B | OVRPBL | 1000 | 58.7000 | 95.5000 | 98.6000 | 6.302422 | -0.1242 | 0.914409 |
| R61B | OVRPBTW | 1000 | 51.1000 | 91.3000 | 99.8000 | 10.496143 | -0.1012 | 1.142085 |
| R61B | OVRPLTW | 1000 | 51.8000 | 92.7000 | 99.9000 | 10.126209 | -0.0098 | 0.885866 |
| R61B | VRPBLTW | 1000 | 43.4000 | 78.2000 | 93.4000 | 18.942555 | -2.9571 | 1.633731 |
| R61B | OVRPBLTW | 1000 | 50.6000 | 92.0000 | 99.8000 | 10.529852 | -0.1074 | 1.155952 |
