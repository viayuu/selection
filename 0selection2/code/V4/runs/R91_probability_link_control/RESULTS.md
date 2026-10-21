# R91 Locked Results

Report delivery recovered on admin01 CPU on 2026-10-22 (Asia/Shanghai). A report, B report and compare all exited 0. Both arms completed 57,600 successful updates on 2026-10-20. This is a reporting repair, not a new experiment.

## Macro Outcomes

All18 problems have equal weight. Costs are descriptive equal-problem means, not pooled raw-cost comparisons. Native int64 winner labels were verified against original FP64 argmin, never FP32 argmin. SBS was fitted on the frozen 81,000 TRAIN instances per problem; Oracle is the per-instance minimum within the legal candidate pool.

| Model | Split | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R84A historical | VAL | 71.085185 | 11.416611477 | -0.194190 | 0.393443 | 0.403494 |
| R84A historical | TEST | 71.235802 | 11.373843044 | -0.197053 | 0.386140 | 0.394847 |
| R88B historical | VAL | 71.058025 | 11.416634105 | -0.192477 | 0.395188 | 0.405128 |
| R88B historical | TEST | 71.220988 | 11.373838370 | -0.195592 | 0.387648 | 0.396339 |
| R91A utility CE | VAL | 71.072840 | 11.416850456 | -0.191608 | 0.396065 | 0.405976 |
| R91A utility CE | TEST | 71.293827 | 11.373669575 | -0.196579 | 0.386647 | 0.395084 |
| R91B utility CE/risk/KL | VAL | 71.091358 | 11.416996733 | -0.190903 | 0.396778 | 0.408374 |
| R91B utility CE/risk/KL | TEST | 71.408642 | 11.374208708 | -0.194204 | 0.389022 | 0.398966 |

Common VAL SBS/Oracle costs: 11.427978473 / 11.371691902. Common TEST SBS/Oracle costs: 11.385299246 / 11.329932369. The full CSV/PDF reports contain ALL and every problem's Top1, mean_cost, SBS cost, Oracle cost, vs_SBS, vs_Oracle and actual regret. Ratio-of-means metrics and mean per-instance actual regret are distinct; ALL averages each metric across problems.

## Fixed Gates And Recovery

The prelocked joint VAL screen requires strictly higher Top1 and nonincreasing macro mean_cost and actual regret against BOTH R84A and R88B. Both arms FAIL. All four arm/reference comparisons increase VAL cost and regret. R84A and same-role R88B are historical references, not newly matched original-loss controls.

| Arm / VAL reference | Top1 delta (pp) | Cost delta | Actual regret delta (pp) | Pass |
| --- | ---: | ---: | ---: | --- |
| A / R84A | -0.012346 | +0.000238979 | +0.002482 | No |
| A / R88B | +0.014815 | +0.000216352 | +0.000849 | No |
| B / R84A | +0.006173 | +0.000385257 | +0.004880 | No |
| B / R88B | +0.033333 | +0.000362629 | +0.003247 | No |

The target list remains the same36 R84A VAL-defined problem/method groups. There are 4,625 VAL and 4,598 TEST true target winners. These are true-winner counts, not model selections.

| Arm / reference / split | Correct / selected | Recall (%) | Rescued | Lost | New false picks | Target regret delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / R84A / VAL | 10 / 24 | 0.216216 | 3 | 4 | 10 | +0.0002639 |
| A / R84A / TEST | 11 / 26 | 0.239234 | 2 | 4 | 6 | +0.0007054 |
| A / R88B / VAL | 10 / 24 | 0.216216 | 4 | 3 | 8 | -0.0002737 |
| A / R88B / TEST | 11 / 26 | 0.239234 | 2 | 3 | 7 | +0.0007082 |
| B / R84A / VAL | 18 / 39 | 0.389189 | 9 | 2 | 19 | -0.0005853 |
| B / R84A / TEST | 26 / 53 | 0.565463 | 16 | 3 | 19 | -0.0002167 |
| B / R88B / VAL | 18 / 39 | 0.389189 | 11 | 2 | 16 | -0.0011229 |
| B / R88B / TEST | 26 / 53 | 0.565463 | 17 | 3 | 21 | -0.0002140 |

B-minus-A: VAL Top1 +0.018519 pp, cost +0.000146277, actual regret +0.002398 pp; TEST Top1 +0.114815 pp, cost +0.000539133, actual regret +0.003881 pp. B recovers more target winners, but recall remains below 0.6%; target-local regret gains do not establish global cost/regret improvement. No multi-seed significance claim is made.

Recommendation: retain R84A as the reference. Neither R91 arm demonstrates the required joint VAL improvement or a convincing overall recovery/cost/regret gain; do not promote B based on TEST Top1 or fewer zero groups alone.

## Delivery And Provenance

Each arm uses its own unique original VAL lock on every dataset, with original maximin inference and no per-task assembly:

| Arm | Selected update | Checkpoint SHA256 |
| --- | ---: | --- |
| A_utility_ce | 37440 | fc889238acd827fa141c9e24fad07d5bc1f64f58c21a47da73b0ea9b5e543e8b |
| B_utility_probabilities | 40320 | cdcb5f1da1e3c30c59bb751ad0f8a019416e8b56d41ab36928dd2ee9de86296d |

Data: `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`, frozen per-problem TRAIN/VAL/TEST = 81,000/4,500/4,500, all18. Existing candidate exclusions and the single `reld_mt` alias remain unchanged. Old holdouts and prior TEST feedback participated in development; this is not an untouched blind evaluation.

Primary round summary: `RESULTS.pdf` (2 pages). Supporting summary CSVs: `RESULTS_macro.csv`, `RESULTS_target36.csv`.

For BOTH `A_utility_ce/full_model_statistics` and `B_utility_probabilities/full_model_statistics`:

- `MODEL_STATISTICS.pdf`: full22-page report, including same-lock NSS4 table, ALL/per-problem and LIB metrics.
- `selection_distributions.pdf`:18 problem pages; TEST then LIB for TSP/CVRP, paired model-selection and true-winner bars with native Top1 titles.
- `target_recovery_and_train.pdf`: same-lock cached TRAIN/VAL/TEST diagnostics; accompanying CSVs include recovery versus R84A/R88B and new-false-pick regret contributions.
- `nss_comparison.csv`: TSP TEST, CVRP TEST, TSPLIB and CVRPLIB, both Top1 and mean_cost. Reuses historical R79 NSS weights and already-saved exact4500 TEST/native-LIB predictions, with no new NSS inference or training.

Matched comparison: `comparison_artifacts/COMPARISON.pdf` and its CSVs. Recovery completion: `report_only_recovery_complete.json`. Command/timestamp/exits and protected before/after hashes: `report_only_recovery/20261021T163350838344Z/recovery_receipt.json`.

Original PDF failure timestamps (UTC): A `2026-10-20T12:44:31.564511+00:00`, B `2026-10-20T12:44:31.560255+00:00`. Original failed worker/controller receipts and `postprocess_status.json` remain byte-identical. Recovered reporter status is stored separately under the dated recovery directory. The frozen `r91_reporting.py` remains unchanged at SHA256 `f439abea1436ebcb7ebb4d35e31746fdfb25f11e0c3ac56b80af03d8df34f399`.

Runtime guards disabled checkpoint loading, model calls and train/test/LIB inference. Environment: empty `CUDA_VISIBLE_DEVICES`, `MPLBACKEND=Agg`, OMP/MKL/OpenBLAS/NumExpr threads1. No training, inference, TEST/LIB rerun, selection changes, retries, goal/index modifications or Git actions were performed by this repair task. Main owns state/goal reconciliation and the final commit.
