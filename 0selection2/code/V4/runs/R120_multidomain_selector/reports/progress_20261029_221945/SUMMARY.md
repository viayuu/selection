# R120 interim snapshot: validation update 15840

This is an interim validation report, not the final locked TEST result.

| Problem | Model Top1 | TRAIN-fitted SBS Top1 | Selected deployments |
| --- | ---: | ---: | ---: |
| MIS | 73.700% | 73.700% | 1 |
| JSSP | 58.800% | 58.800% | 1 |
| FJSP | 41.900% | 14.800% | 3 |
| BPP | 98.100% | 98.100% | 1 |
| MKP | 93.556% | 93.556% | 1 |
| SMTWTP | 94.187% | 91.067% | 3 |

All new-task Top1 values are native-objective any-best accuracy.
Canonical winner frequency is only a fixed-order display of exact ties.
The report reads saved validation predictions only; no TEST inference or GPU work.

ETA is an observed-rate full-budget projection; early stopping is unknown.
Observed status: {"observed_utc": "2026-10-29T14:19:40.325461+00:00", "successful_updates": 17112, "max_updates": 57600, "completed_validation_count": 11, "last_completed_validation_update": 15840, "training_start_reference": "/public/home/shiys/0selection2/code/V4/runs/R120_multidomain_selector/initialization.json", "elapsed_seconds": 6754.052888870239, "measured_training_seconds_per_successful_update": 0.39122031809257163, "remaining_training_and_validation_hours_if_full_budget": 4.443491009916335, "projected_full_budget_finish_cst": "2026-10-30T02:46:16.893097+08:00", "final_test_lib_report_allowance_minutes": [10, 20], "estimate_boundary": "Observed-rate projection, not a deadline; early stopping may finish sooner. Final TEST/LIB/report overhead is an allowance, not a measured R120 result.", "best_observed_validation_update": 14400, "best_observed_is_final_lock": false}
