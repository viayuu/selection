# R108 Report-Only Recovery

Original canonical model reports, runtime sources, protocols, weights and predictions are unchanged.
Derived NSS CSVs only remove BOM and add provenance metadata; all original fields are byte-for-value equal after CSV parsing.
TRAIN-fitted SBS; native FP64-ind winner; ALL is equal-problem macro. Raw macro costs are descriptive only.
One VAL-selected checkpoint per arm (update43200); no training, inference, calibration or reselection.

| arm | split | Top1_pct | mean_cost | vs_SBS_pct | vs_Oracle_pct | actual_regret_pct |
|---|---|---|---|---|---|---|
| A_prior_ce | VAL | 71.080247 | 11.417570 | -0.191155 | 0.396431 | 0.408898 |
| A_prior_ce | TEST | 71.338272 | 11.374471 | -0.193688 | 0.389484 | 0.400252 |
| B_density_ratio | VAL | 71.061728 | 11.417242 | -0.193730 | 0.393849 | 0.406392 |
| B_density_ratio | TEST | 71.217284 | 11.374505 | -0.193817 | 0.389359 | 0.399144 |

All18 VAL/TEST per-problem metrics: metrics.csv.
Fixed36 paired recovery: comparison_artifacts/target_recovery_summary_B_vs_A.csv.
NSS4: normalized/<arm>/full_model_statistics/nss_comparison.csv.
Existing per-arm selection_distributions.pdf and MODEL_STATISTICS.pdf remain valid primary per-model deliveries.
Original canonical watcher remains blocked on its original malformed CSV; recovery gate does not claim it succeeded.
TEST/LIB have development exposure; this report establishes delivery, not selection improvement.
