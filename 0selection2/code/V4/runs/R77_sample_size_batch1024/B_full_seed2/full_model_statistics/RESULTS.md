# R77B locked full-model statistics

[Main report, 22 pages](MODEL_STATISTICS.pdf) | [Selection distributions, 18 pages](selection_distributions.pdf)

Status: complete. Retain R77B; R78 candidates did not pass gate.
One checkpoint for every task/dataset: epoch 28, update 34776; SHA256 `31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b`.

| Split | ALL native Top1 (%) | ALL actual regret (%) |
|---|---:|---:|
| VAL | 70.8943089 | 0.4013267 |
| TEST | 71.2252175 | 0.3940143 |

ALL is equal-weight across 18 problems. SBS identities were fixed on FULL TRAIN; LIB posthoc SBS is supplementary only.
Primary Top1 uses saved native/FP64 winners. NSS original FP32 Top1 is a separate compatibility column.
New VAL/TEST data version: `selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`; LIB uses original native TSPLIB/CVRPLIB, not old TEST.
TEST has partly informed subsequent research; this is not a claim of a completely untouched blind set.
This report only reads existing predictions/metrics. No training, inference, checkpoint reselection, new trials, or commits.

CSV: `metrics_by_problem.csv` (including Top2/3), `nss_comparison.csv`, `selection_distribution.csv`, `single_method_comparison.csv`.
Exact formulas, source paths, locks, candidate order and three-row-per-dataset alignment checks: `protocol.json`.
