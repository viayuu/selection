# R76: R45A on the New Single-ReLD Scenario

One from-scratch seed2 model. Model and loss are R45A; batches and execution are four-GPU distributed. No original trained selector weights or old candidate cache were reused.

Data: /public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014

18 tasks, 1,262,052 TRAIN / 70,110 validation / 70,110 TEST rows. reld_mt is the fixed MOEL deployment; no MTL column or best-of-two costs.

## Locked Results

| Split | N | Top1 % | Top2 % | Top3 % | CE | Mean cost | vs TRAIN SBS % | Actual regret % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1262052 | 69.3696 | 88.7389 | 95.1335 | 0.752703 | 11.373005 | -0.1438 | 0.460650 |
| val | 70110 | 69.2369 | 88.9089 | 95.1733 | 0.755865 | 11.402475 | -0.1421 | 0.464503 |
| test | 70110 | 69.4395 | 88.9203 | 95.1305 | 0.750073 | 11.353131 | -0.1404 | 0.464144 |

Selected epoch: 18. Completed epochs: 26. Successful updates: 3744. Stop: validation early stop.

Selection used validation macro_vs_sbs_pct only, with the SBS identity fitted on TRAIN. The selected checkpoint was locked before TEST cache construction and received one full TEST. There is no TEST-based epoch/rule/task-wise model selection.

## TRAIN-Fitted References on TEST

| Reference | Top1 % | Mean cost | Actual regret % |
|---|---:|---:|---:|
| train_sbs | 66.1746 | 11.356886 | 0.584830 |
| train_majority | 66.7280 | 11.364432 | 0.669031 |
| train_min_mean_regret | 66.1746 | 11.356886 | 0.584830 |

## Batches and Scope

| Problem | Rank-local microbatch caps (4090,4090,3090,3090) | Global microbatch capacity | Logical batch |
|---|---|---:|---:|
| TSP | 443,443,416,416 | 1718 | 9856 |
| CVRP | 352,352,384,384 | 1472 | 9856 |
| ATSP | 2496,2496,2464,2464 | 9920 | 9856 |
| OVRP | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPB | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPL | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPTW | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPTW | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPB | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPL | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPBL | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPBTW | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPLTW | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPBL | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPBTW | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPLTW | 2464,2464,2464,2464 | 9856 | 9856 |
| VRPBLTW | 2464,2464,2464,2464 | 9856 | 9856 |
| OVRPBLTW | 2464,2464,2464,2464 | 9856 | 9856 |

Large batches intentionally differ from historical R45A batch128 and change update counts. LR was not automatically scaled. Each epoch still retains every real TRAIN row and tail exactly once. The whole local objective, including KL, is weighted for uneven rank row counts.

The logical global batch is uniform across tasks. TSP/CVRP use accumulation without parameter updates between microbatches; this does not multiply optimizer updates or task sampling weight.

The two4090 and two3090 cards trained one synchronized model, not four seeds. Input preparation was parallel; node-local arrays and pinned prefetch reduced shared-filesystem waits. W&B mode and measured capacity/timing are recorded in the protocol and capacity receipts.

Old R45A/R75 used different splits, candidate pools and/or deployment labels; their TEST figures are not matched controls for this release. Compare this model to the references on this same scenario. Removing candidates does not certify complete remaining constraint equivalence; R58 stays deferred.

Saved predictions are in evaluation/<split>/<tag>/rank*.npz. Full per-task results are in comparison.csv; learning_curves.png and optimization_curves.png use successful optimizer updates.

method_metrics.csv and method_comparison.csv retain each single method, the selector's gains/losses, and the separately labelled post-hoc split-best method. Per-problem Top1, mean-cost and arm-distribution plots compare saved validation and TEST predictions. Fixed-method true_cost_top2/3 mean membership in the true cost ranking; selector Top2/3 instead mean winner coverage in its predicted ranking. Do not mix these definitions or use the post-hoc split-best identity for checkpoint/policy selection.
