# R120 interim snapshot: validation update 48960

This is an interim validation report, not the final locked TEST result.

| Problem | Model Top1 | TRAIN-fitted SBS Top1 | Selected deployments |
| --- | ---: | ---: | ---: |
| MIS | 87.900% | 73.700% | 2 |
| JSSP | 58.800% | 58.800% | 1 |
| FJSP | 42.200% | 14.800% | 7 |
| BPP | 98.100% | 98.100% | 1 |
| MKP | 93.556% | 93.556% | 1 |
| SMTWTP | 95.253% | 91.067% | 3 |

All new-task Top1 values are native-objective any-best accuracy.
Canonical winner frequency is only a fixed-order display of exact ties.
The report reads saved validation predictions only; no TEST inference or GPU work.

ETA is an observed-rate full-budget projection; early stopping is unknown.
Observed status: {"observed_utc": "2026-10-29T17:58:36.087896+00:00", "successful_updates": 49728, "max_updates": 57600, "completed_validation_count": 34, "last_completed_validation_update": 48960, "training_start_reference": "/public/home/shiys/0selection2/code/V4/runs/R120_multidomain_selector/initialization.json", "elapsed_seconds": 19889.81532406807, "measured_training_seconds_per_successful_update": 0.39628585438557595, "remaining_training_and_validation_hours_if_full_budget": 0.8755309681183294, "projected_full_budget_finish_cst": "2026-10-30T02:51:07.999381+08:00", "final_test_lib_report_allowance_minutes": [10, 20], "estimate_boundary": "Observed-rate projection, not a deadline; early stopping may finish sooner. Final TEST/LIB/report overhead is an allowance, not a measured R120 result.", "best_observed_validation_update": 44640, "best_observed_is_final_lock": false}
