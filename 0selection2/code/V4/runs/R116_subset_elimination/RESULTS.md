# R116 Complete Models

## What Completed?
Both independent full18 pipelines completed train, locked TEST, timing, LIB and reports at rc0.
One whole-model VAL-selected checkpoint per arm; saved predictions only, no model execution or TEST tuning.
| Arm | Selected update | Checkpoint SHA256 |
| --- | --- | --- |
| A_static | 28440 | c8a1288efe99aab41f56d97a409fdb69aee46c987438dfc660cf849a95d74f1a |
| B_subset | 25596 | d3e19c297fe94b31080393ab28b7171581bed449afd16cad35250d225dacba26 |
Data version: selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018.

## What Are The Results?
| Model | Split | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R116/A_static | VAL | 70.897531 | 11.417557 | 11.427978 | 11.371692 | -0.186796 | 0.400880 | 0.410795 |
| R116/B_subset | VAL | 70.933333 | 11.417765 | 11.427978 | 11.371692 | -0.186384 | 0.401308 | 0.412386 |
| R116/A_static | TEST | 71.134568 | 11.374935 | 11.385299 | 11.329932 | -0.186968 | 0.396260 | 0.404762 |
| R116/B_subset | TEST | 71.255556 | 11.374748 | 11.385299 | 11.329932 | -0.190016 | 0.393213 | 0.402782 |
| R94B | TEST | 71.577778 | 11.372234 | 11.385299 | 11.329932 | -0.211169 | 0.371939 | 0.379115 |

ALL gives every problem equal weight. Cross-problem raw costs are descriptive only.
SBS is frozen full-TRAIN fitted; Oracle is the per-instance legal-pool minimum. Native/FP64 ind, not FP32 argmin.

## What Changed?
| Comparison | Split | Corrected | Harmed | Wrong cheaper | Wrong dearer | dTop1 pp | dRegret pp | W % | G % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_subset vs A_static | VAL | 779 | 750 | 371 | 459 | 0.035802 | 0.001591 | 1.4198 | -0.002658 |
| B_subset vs A_static | TEST | 796 | 698 | 388 | 439 | 0.120988 | -0.001980 | 1.4617 | -0.001355 |
| A_static vs R94B | TEST | 1542 | 1901 | 752 | 785 | -0.443210 | 0.025646 | 2.8309 | -0.029081 |
| B_subset vs R94B | TEST | 1535 | 1796 | 678 | 782 | -0.322222 | 0.023666 | 2.7309 | -0.026918 |

100*mean(current_cost < baseline_cost), cost win rate.
100*mean((baseline_cost-current_cost)/baseline_cost), mean relative cost improvement.
Cost/regret deltas are current minus baseline; G is positive for improvement.
Both-wrong cost/regret contributions use all problem rows as denominator; gross gain/loss are separate CSV columns.

## Historical References
| Model | Status | Reason |
| --- | --- | --- |
| R77 | missing_or_incompatible | Incomplete or inconsistent full18 frozen model: /public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2 |
| R94B | complete |  |

## Scientific Boundary
Complete A/B fresh parameter hashes and shapes match; logical batch=1024.
The original full-reference hash is provenance, not the hash of the retained original-parameter subset.
same seeded task/shuffle protocol; actual per-update sample IDs not independently archived.
single seed2; no multi-seed significance or automatic promotion.
whole saved historical models; not fresh parameter/budget matched controls.
TEST participated in earlier R&D; not a pristine blind test; no finish-stage reselection.

Per-problem and native LIB metrics: metrics.csv. Pair decompositions: comparison.csv.
Instance-level events: events.csv. Input/output hashes: finish_complete.json.
Per-arm method-distribution, NSS and learning-curve PDFs remain in each full_model_statistics directory.

## Predeclared VAL Gate

B-A Top1: 0.035802 pp; actual regret: 0.001591 pp; vsSBS: 0.000412 pp.

Scientific gate: False; complete-DP p95 <=2x point selector: True. No TEST-based selection or new arms.

Both arms use champion CE plus full-ranking NLL. Historical R77 is not an isolated loss ablation.
