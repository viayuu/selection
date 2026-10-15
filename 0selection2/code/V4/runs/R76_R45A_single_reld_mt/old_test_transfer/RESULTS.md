# R76: original TEST with current candidate pools

Locked checkpoint: epoch 18; `5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9`.
No retraining, checkpoint selection, solver execution, or changes to the formal R76 TEST.

| Dataset | Instances | Top1 | Top2 | Top3 | Actual regret |
|---|---:|---:|---:|---:|---:|
| Original TEST, aligned candidates | 18,000 | 63.7833% | 82.8889% | 91.3000% | 1.1137% |
| R76 new-scenario TEST | 70,110 | 69.4395% | 88.9203% | 95.1305% | 0.4641% |

## Label alignment
- B tasks: remove RouteFinder, MoSES_RF, and MoSES_CaDA.
- All MVRP tasks: remove RELD_MTL; retain RELD_MOEL at global ID16 and rename it reld_mt.
- Preserve every original instance, historical FP64 cost, and native row order.
- Recompute winner, cost ranking, Oracle, relative gaps, and post-hoc old TEST SBS.
- 6,068 original winner identities changed after filtering.
- Report vs_SBS against the R76 TRAIN-selected fixed deployment, evaluated on these historical costs.

## Interpretation Boundary
This aligns candidates, not all historical solver deployment protocols. No unknown cost is fabricated.
Historical/new cost-generation differences, including previously identified time-window semantics, remain.
The old/new TEST comparison is a cross-scenario transfer diagnostic, not a controlled data-size effect.
R45A original 47.9111% used the unfiltered candidate pools and is not a same-pool comparison.

## By Problem
| Problem | Pool size | Top1 | Actual regret |
|---|---:|---:|---:|
| TSP | 8 | 25.40% | 0.9164% |
| CVRP | 10 | 29.20% | 1.7762% |
| ATSP | 5 | 72.60% | 0.5537% |
| OVRP | 6 | 39.50% | 0.7391% |
| VRPB | 3 | 80.40% | 0.3650% |
| VRPL | 6 | 32.20% | 0.4806% |
| VRPTW | 6 | 40.50% | 5.8456% |
| OVRPTW | 6 | 82.30% | 0.2351% |
| OVRPB | 3 | 89.60% | 0.2368% |
| OVRPL | 6 | 36.40% | 0.7375% |
| VRPBL | 3 | 81.80% | 0.4190% |
| VRPBTW | 3 | 79.10% | 0.4678% |
| VRPLTW | 6 | 41.00% | 5.6701% |
| OVRPBL | 3 | 89.80% | 0.2152% |
| OVRPBTW | 3 | 81.40% | 0.3554% |
| OVRPLTW | 6 | 84.00% | 0.2677% |
| VRPBLTW | 3 | 79.00% | 0.4762% |
| OVRPBLTW | 3 | 83.90% | 0.2883% |
