# R117 Complete Models

## What Completed?
Both independent full18 pipelines completed train, locked TEST, timing, LIB and reports at rc0.
One whole-model VAL-selected checkpoint per arm; saved predictions only, no model execution or TEST tuning.
| Arm | Selected update | Checkpoint SHA256 |
| --- | --- | --- |
| A | 27018 | 0027a21460463258f81a0183644b3fee4807d1121852f47d28201ef3502bc84a |
| B | 27018 | 7157ed93b116c038c0ae956588141893007996f671dfcf8de19c1c8c263929ae |
Data version: selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018.

## What Are The Results?
| Model | Split | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R117/A | VAL | 71.322222 | 11.416106 | 11.427978 | 11.371692 | -0.198173 | 0.389434 | 0.398842 |
| R117/B | VAL | 71.341975 | 11.416666 | 11.427978 | 11.371692 | -0.194558 | 0.393064 | 0.402931 |
| R117/A | TEST | 71.446914 | 11.372882 | 11.385299 | 11.329932 | -0.205613 | 0.377541 | 0.385552 |
| R117/B | TEST | 71.512346 | 11.372920 | 11.385299 | 11.329932 | -0.204291 | 0.378864 | 0.387063 |
| R94B | TEST | 71.577778 | 11.372234 | 11.385299 | 11.329932 | -0.211169 | 0.371939 | 0.379115 |

ALL gives every problem equal weight. Cross-problem raw costs are descriptive only.
SBS is frozen full-TRAIN fitted; Oracle is the per-instance legal-pool minimum. Native/FP64 ind, not FP32 argmin.

## What Changed?
| Comparison | Split | Corrected | Harmed | Wrong cheaper | Wrong dearer | dTop1 pp | dRegret pp | W % | G % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B vs A | VAL | 1139 | 1123 | 526 | 505 | 0.019753 | 0.004089 | 2.0556 | -0.005621 |
| B vs A | TEST | 1123 | 1070 | 509 | 516 | 0.065432 | 0.001511 | 2.0148 | -0.003544 |
| A vs R94B | TEST | 1237 | 1343 | 602 | 621 | -0.130864 | 0.006437 | 2.2679 | -0.008328 |
| B vs R94B | TEST | 1387 | 1440 | 648 | 675 | -0.065432 | 0.007948 | 2.5099 | -0.009913 |

100*mean(current_cost < baseline_cost), cost win rate.
100*mean((baseline_cost-current_cost)/baseline_cost), mean relative cost improvement.
Cost/regret deltas are current minus baseline; G is positive for improvement.
Both-wrong cost/regret contributions use all problem rows as denominator; gross gain/loss are separate CSV columns.

## Historical References
| Model | Status | Reason |
| --- | --- | --- |
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

B-A Top1=0.019753 pp; actual regret=0.004089 pp; vs_SBS=0.003615 pp. Gate passed: False.

Frozen R84 fixed36 true recovery, false picks and outside36/full18 partitions: ZERO_SELECTION.pdf and target_recovery*.csv.
Pretrained ReLD static base is identical in both arms; only added frozen native computation weights differ.
Cache-hit selector time is not deployed native-computation latency. No TEST/LIB-based reselection.
