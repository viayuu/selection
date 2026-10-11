# R59: Trainable Routing-Pretrained Instance Encoding

Original18 tasks and full candidate pools; seed2. Validation only: no R59 test access.
Legacy deployment scenario: the unresolved B feasible-domain discrepancy is NOT repaired.
R58 is deferred by user. No solver decoding, new cost tables, expansion data or relabeling.

## Validation-selected outcomes

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 11.168517 | -1.0700 | 0.926048 |
| R59A | ALL | 18000 | 48.5278 | 82.0556 | 92.7000 | 11.170771 | -1.0512 | 0.947082 |
| R59B | ALL | 18000 | 48.0333 | 81.7556 | 92.4833 | 11.171358 | -1.0425 | 0.962345 |

A/B have the same trainable architecture and non-transferred initial parameters.
A starts its native encoder randomly; B loads the existing ReLD-MTL encoder weights.
Both encoders train jointly with the selector. R45A is a historical, not matched, control.
The shared native path covers CVRP and15 MVRP tasks; TSP/ATSP retain the old encoder path.
Raw/local geometry and condition features preserve fields omitted by the native encoder.

## Training and selection

| Arm | Best epoch | Stop epoch | Successful updates | Final val Top1 % | Last5 val Top1 % |
|---|---:|---:|---:|---:|---:|
| R59A | 32 | 40 | 56880 | 48.7000 | 48.6711 |
| R59B | 16 | 20 | 28440 | 47.5944 | 47.7578 |

| Arm | Trainable parameters | Native parameters | Actual run wall hours | First epoch optimization seconds |
|---|---:|---:|---:|---:|
| R59A | 3077897 | 1185408 | 1.0834 | 87.08 |
| R59B | 3077897 | 1185408 | 0.5453 | 87.90 |

Actual per-arm wall includes training and evaluations, not data preparation/preflight.
First-epoch optimization time is measured; the prelaunch4--8GPUh estimate was not a cutoff.

## Selected-checkpoint train and validation

| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |
|---|---|---:|---:|---:|---:|
| R59A | train | 49.5311 | 1.067149 | 11.171277 | 0.905768 |
| R59A | val | 48.5278 | 1.094975 | 11.170771 | 0.947082 |
| R59B | train | 52.3022 | 1.014315 | 11.164233 | 0.838570 |
| R59B | val | 48.0333 | 1.138157 | 11.171358 | 0.962345 |

Selection is the strict minimum validation macro_vs_sbs_pct, as precommitted.
Native encoder LR3e-5, other parameters1e-4 in both arms; identical scheduling rules,
not necessarily identical realized LR trajectories or stopping epochs. Original R45A
comparison/risk/R-Drop objectives are reused without a coefficient search.

## Paired decision changes

| Before | After | Category | N | Macro mean-cost delta | Macro regret delta pp |
|---|---|---|---:|---:|---:|
| R45A | R59A | corrected | 1195 | -0.01536067 | -0.131082 |
| R45A | R59A | harmed | 1301 | +0.01797532 | +0.153210 |
| R45A | R59A | both_wrong_cheaper | 344 | -0.00384515 | -0.027366 |
| R45A | R59A | both_wrong_dearer | 326 | +0.00348515 | +0.026271 |
| R45A | R59A | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R45A | R59A | unchanged | 14833 | +0.00000000 | +0.000000 |
| R45A | R59A | all | 18000 | +0.00225465 | +0.021033 |
| R45A | R59B | corrected | 1996 | -0.02546942 | -0.225981 |
| R45A | R59B | harmed | 2191 | +0.02813285 | +0.257229 |
| R45A | R59B | both_wrong_cheaper | 526 | -0.00597145 | -0.041800 |
| R45A | R59B | both_wrong_dearer | 573 | +0.00614918 | +0.046848 |
| R45A | R59B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| R45A | R59B | unchanged | 12713 | +0.00000000 | +0.000000 |
| R45A | R59B | all | 18000 | +0.00284116 | +0.036297 |
| R59A | R59B | corrected | 2166 | -0.02910832 | -0.248619 |
| R59A | R59B | harmed | 2255 | +0.02913320 | +0.257416 |
| R59A | R59B | both_wrong_cheaper | 553 | -0.00574338 | -0.041399 |
| R59A | R59B | both_wrong_dearer | 613 | +0.00630501 | +0.047866 |
| R59A | R59B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| R59A | R59B | unchanged | 12413 | +0.00000000 | +0.000000 |
| R59A | R59B | all | 18000 | +0.00058651 | +0.015263 |

## Precommitted decision

Validation advancement screen passed: **False**.
B minus R45A Top1: -1.0833pp (required>=3).
B minus A Top1: -0.4944pp (required>=2).
Regret no worse than R45A/A: False/False.

Passing this screen is not achievement of70% or evidence from an unseen test.
A negative result rejects this transfer configuration, not a hard accuracy ceiling.
No extra epochs, fine-tuning variants, task gates or test-driven changes are licensed
by this report. Subsequent mechanisms require a new precommitted review round.

## Interpretation limits

One seed and external solver pretraining; total pretraining data are not matched.
Historical training-instance provenance of the native weights is incomplete; absence
of pretraining overlap cannot be certified. Existing B-domain and deployment gaps
remain open. This evaluates native-winner prediction in the legacy cost table only.
The historical test has been seen in prior rounds and is not described as pristine.

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
| R59A | TSP | 1000 | 37.6000 | 61.6000 | 76.9000 | 7.915904 | -0.5265 | 0.593253 |
| R59A | CVRP | 1000 | 47.4000 | 68.0000 | 81.8000 | 18.200915 | -0.5106 | 0.635602 |
| R59A | ATSP | 1000 | 77.0000 | 96.6000 | 99.2000 | 1.635229 | -2.2883 | 0.346919 |
| R59A | OVRP | 1000 | 37.0000 | 65.6000 | 84.0000 | 7.454182 | -0.1002 | 0.700355 |
| R59A | VRPB | 1000 | 55.6000 | 91.7000 | 99.3000 | 9.332876 | -0.0791 | 0.690824 |
| R59A | VRPL | 1000 | 32.1000 | 57.3000 | 72.1000 | 12.404705 | -0.0202 | 0.487932 |
| R59A | VRPTW | 1000 | 46.2000 | 76.0000 | 93.7000 | 17.893419 | -4.0931 | 1.417497 |
| R59A | OVRPTW | 1000 | 51.4000 | 93.3000 | 99.9000 | 10.041227 | -0.0717 | 0.847186 |
| R59A | OVRPB | 1000 | 57.5000 | 96.8000 | 98.3000 | 6.317308 | -0.0739 | 0.921279 |
| R59A | OVRPL | 1000 | 40.6000 | 68.3000 | 86.2000 | 7.430379 | -0.0966 | 0.697276 |
| R59A | VRPBL | 1000 | 53.5000 | 91.6000 | 98.8000 | 9.357128 | -0.0229 | 0.794735 |
| R59A | VRPBTW | 1000 | 42.0000 | 80.2000 | 93.9000 | 19.040305 | -3.2000 | 1.642094 |
| R59A | VRPLTW | 1000 | 47.1000 | 80.2000 | 93.6000 | 17.622197 | -4.7920 | 1.244238 |
| R59A | OVRPBL | 1000 | 57.2000 | 95.5000 | 98.3000 | 6.307268 | -0.0474 | 0.994854 |
| R59A | OVRPBTW | 1000 | 50.5000 | 91.3000 | 99.7000 | 10.501724 | -0.0481 | 1.218523 |
| R59A | OVRPLTW | 1000 | 50.6000 | 92.7000 | 99.9000 | 10.124040 | -0.0313 | 0.853400 |
| R59A | VRPBLTW | 1000 | 41.1000 | 78.3000 | 93.1000 | 18.958834 | -2.8737 | 1.752277 |
| R59A | OVRPBLTW | 1000 | 49.1000 | 92.0000 | 99.9000 | 10.536244 | -0.0468 | 1.209228 |
| R59B | TSP | 1000 | 36.8000 | 60.3000 | 74.1000 | 7.918291 | -0.4965 | 0.609073 |
| R59B | CVRP | 1000 | 46.9000 | 70.4000 | 82.6000 | 18.222593 | -0.3921 | 0.795575 |
| R59B | ATSP | 1000 | 77.6000 | 96.7000 | 99.2000 | 1.635183 | -2.2910 | 0.342123 |
| R59B | OVRP | 1000 | 33.5000 | 62.9000 | 84.0000 | 7.456974 | -0.0628 | 0.738575 |
| R59B | VRPB | 1000 | 54.5000 | 91.1000 | 99.3000 | 9.332456 | -0.0836 | 0.693017 |
| R59B | VRPL | 1000 | 31.3000 | 55.2000 | 74.2000 | 12.405124 | -0.0168 | 0.496693 |
| R59B | VRPTW | 1000 | 43.5000 | 75.4000 | 91.7000 | 17.892176 | -4.0998 | 1.427724 |
| R59B | OVRPTW | 1000 | 49.1000 | 93.3000 | 99.8000 | 10.046498 | -0.0193 | 0.906753 |
| R59B | OVRPB | 1000 | 60.6000 | 96.8000 | 98.5000 | 6.309164 | -0.2027 | 0.802427 |
| R59B | OVRPL | 1000 | 37.9000 | 68.8000 | 86.4000 | 7.434967 | -0.0349 | 0.764668 |
| R59B | VRPBL | 1000 | 54.2000 | 91.1000 | 99.2000 | 9.356490 | -0.0297 | 0.799519 |
| R59B | VRPBTW | 1000 | 45.4000 | 79.7000 | 92.2000 | 19.024603 | -3.2798 | 1.582293 |
| R59B | VRPLTW | 1000 | 45.6000 | 79.5000 | 92.3000 | 17.635981 | -4.7175 | 1.330793 |
| R59B | OVRPBL | 1000 | 56.4000 | 95.2000 | 98.5000 | 6.311215 | +0.0151 | 1.092929 |
| R59B | OVRPBTW | 1000 | 47.6000 | 91.1000 | 99.5000 | 10.509803 | +0.0288 | 1.296587 |
| R59B | OVRPLTW | 1000 | 52.3000 | 92.6000 | 99.8000 | 10.125426 | -0.0176 | 0.861663 |
| R59B | VRPBLTW | 1000 | 42.6000 | 79.5000 | 93.7000 | 18.931897 | -3.0117 | 1.581031 |
| R59B | OVRPBLTW | 1000 | 48.8000 | 92.0000 | 99.7000 | 10.535596 | -0.0529 | 1.200769 |
