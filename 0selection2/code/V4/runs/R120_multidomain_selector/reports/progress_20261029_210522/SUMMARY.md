# R120 interim snapshot: validation update 5760

This is an interim validation report, not the final locked TEST result.

| Problem | Model Top1 | TRAIN-fitted SBS Top1 | Selected deployments |
| --- | ---: | ---: | ---: |
| MIS | 73.700% | 73.700% | 1 |
| JSSP | 58.800% | 58.800% | 1 |
| FJSP | 29.900% | 14.800% | 1 |
| BPP | 98.100% | 98.100% | 1 |
| MKP | 93.556% | 93.556% | 1 |
| SMTWTP | 91.067% | 91.067% | 1 |

All new-task Top1 values are native-objective any-best accuracy.
Canonical winner frequency is only a fixed-order display of exact ties.
The report reads saved validation predictions only; no TEST inference or GPU work.

ETA is an observed-rate full-budget projection; early stopping is unknown.
Observed status: {"observed_utc": "2026-10-29T13:05:19.540692+00:00", "successful_updates": 5784, "max_updates": 57600, "completed_validation_count": 4, "last_completed_validation_update": 5760, "training_start_reference": "/public/home/shiys/0selection2/code/V4/runs/R120_multidomain_selector/initialization.json", "elapsed_seconds": 2293.2681200504303, "measured_training_seconds_per_successful_update": 0.3925819792014284, "remaining_training_and_validation_hours_if_full_budget": 5.706998168179314, "projected_full_budget_finish_cst": "2026-10-30T02:47:44.734097+08:00", "final_test_lib_report_allowance_minutes": [10, 20], "estimate_boundary": "Observed-rate projection, not a deadline; early stopping may finish sooner. Final TEST/LIB/report overhead is an allowance, not a measured R120 result.", "best_observed_validation_update": 5760, "best_observed_is_final_lock": false}
