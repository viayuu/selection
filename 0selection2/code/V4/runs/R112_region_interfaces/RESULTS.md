# R112 Complete Models

## What Completed?
Original dispatch failed only at CPU PDF rendering; independent CPU recovery completed both reports. Original train, locked TEST, GPU timing and LIB were reused at rc0; failed receipts are preserved.
One whole-model VAL-selected checkpoint per arm; saved predictions only, no model execution or TEST tuning.
| Arm | Selected update | Checkpoint SHA256 |
| --- | --- | --- |
| A_collapsed | 57600 | 4bf692f36fde2f3f7ce461ccc96a01b2bd254e606c9ea40c7f5e3aea9c44ae3e |
| B_interfaces | 43200 | 55ccef0ca150c28311cf9340d740a4f0cac80f3a921b1d8f3cbe07fd9017ecb5 |
Data version: selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018.

## What Are The Results?
| Model | Split | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R112/A_collapsed | VAL | 71.060494 | 11.417398 | 11.427978 | 11.371692 | -0.187797 | 0.399854 | 0.410621 |
| R112/B_interfaces | VAL | 71.024691 | 11.417900 | 11.427978 | 11.371692 | -0.183590 | 0.404089 | 0.415310 |
| R112/A_collapsed | TEST | 71.283951 | 11.374503 | 11.385299 | 11.329932 | -0.190906 | 0.392293 | 0.400308 |
| R112/B_interfaces | TEST | 71.333333 | 11.374755 | 11.385299 | 11.329932 | -0.190510 | 0.392669 | 0.401151 |
| R94B | TEST | 71.577778 | 11.372234 | 11.385299 | 11.329932 | -0.211169 | 0.371939 | 0.379115 |
| R89B | TEST | 71.419753 | 11.373470 | 11.385299 | 11.329932 | -0.201446 | 0.381702 | 0.389646 |

ALL gives every problem equal weight. Cross-problem raw costs are descriptive only.
SBS is frozen full-TRAIN fitted; Oracle is the per-instance legal-pool minimum. Native/FP64 ind, not FP32 argmin.

## What Changed?
| Comparison | Split | Corrected | Harmed | Wrong cheaper | Wrong dearer | dTop1 pp | dRegret pp | W % | G % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_interfaces vs A_collapsed | VAL | 1252 | 1281 | 563 | 571 | -0.035802 | 0.004689 | 2.2407 | -0.006481 |
| B_interfaces vs A_collapsed | TEST | 1262 | 1222 | 575 | 577 | 0.049383 | 0.000843 | 2.2679 | -0.003263 |
| A_collapsed vs R94B | TEST | 1669 | 1907 | 780 | 831 | -0.293827 | 0.021193 | 3.0235 | -0.024628 |
| B_interfaces vs R94B | TEST | 1701 | 1899 | 744 | 791 | -0.244444 | 0.022036 | 3.0185 | -0.025221 |
| A_collapsed vs R89B | TEST | 1403 | 1513 | 654 | 688 | -0.135802 | 0.010662 | 2.5383 | -0.012217 |
| B_interfaces vs R89B | TEST | 1412 | 1482 | 607 | 643 | -0.086420 | 0.011505 | 2.4914 | -0.012565 |

100*mean(current_cost < baseline_cost), cost win rate.
100*mean((baseline_cost-current_cost)/baseline_cost), mean relative cost improvement.
Cost/regret deltas are current minus baseline; G is positive for improvement.
Both-wrong cost/regret contributions use all problem rows as denominator; gross gain/loss are separate CSV columns.

## Historical References
| Model | Status | Reason |
| --- | --- | --- |
| R94B | complete |  |
| R89B | complete |  |

## Scientific Boundary
Complete A/B fresh parameter hashes and shapes match; logical batch=512.
The original full-reference hash is provenance, not the hash of the retained original-parameter subset.
same seeded task/shuffle protocol; actual per-update sample IDs not independently archived.
single seed2; no multi-seed significance or automatic promotion.
whole saved historical models; not fresh parameter/budget matched controls.
TEST participated in earlier R&D; not a pristine blind test; no finish-stage reselection.

Per-problem and native LIB metrics: metrics.csv. Pair decompositions: comparison.csv.
Instance-level events: events.csv. Input/output hashes: finish_complete.json.
Per-arm method-distribution, NSS and learning-curve PDFs remain in each full_model_statistics directory.
