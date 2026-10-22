# R98 Frozen Expected-Cost Context

Primary delivery: [RESULTS.pdf](RESULTS.pdf). CSV and provenance: full_model_statistics/.

Bank SHA256: `8b96a234b1b0a97f36128317c1c00b62daeffec8b2153f6a8033763e4ca95df4`

## Method And Evidence Boundaries

- One predeclared full18 R98 expected-cost context bank; no SGD, fine-tuning, VAL grid or task-wise assembly.
- Official frozen TabICLRegressor: each legal method predicts the mean of raw FP64 relative gap (c-cmin)/cmin.
- Every nonconstant method fits all 81,000 historical TRAIN rows in natural order/frequency, sequentially; exact constants alone skip fitting.
- Decisions and Top-k use argmin RAW predicted mean gap, with ascending global-ID exact ties. Negative predictions are NOT clipped.
- Normalized exp(-predicted gap/0.01) is formal descriptive output only: NOT calibrated probabilities and NEVER the ranking rule.
- Fixed n_estimators=1, batch_size=1, norm_methods=[none], kv_cache=False, GPU-only/no CPU offload, Flash attention only.
- Input-only canonical R92 357 statistics; immutable R97 task-TRAIN-constant drop. No query labels/costs/winners enter forward.
- TRAIN self-context accuracy, epochs, SGD curves and training loss: N/A. Historical TRAIN outcomes are authorized context only.
- SBS is fixed on R84 full 81,000 TRAIN rows. Oracle is the per-instance minimum over the unchanged legal candidate pool.
- ALL is equal-problem macro across full18. Mixed raw mean_cost is descriptive, never a pooled optimization objective.
- vs_SBS / vs_Oracle = 100*(ratio of mean costs - 1); actual regret = mean_i[100*(selected_i-oracle_i)/oracle_i].
- NSS reuses saved R79 historical70,114 TRAIN weights; not a new81,000 retrain. Missing stays missing; incompatible evidence fails.
- R89B is the main historical control; R84A is secondary. External public pretraining is NOT matched compute.
- Old VAL/TEST members informed development. These results are NOT a wholly independent blind TEST.
- Fixed36 recovery must include rescued/lost true winners, new false picks, net recovery and cost/regret, not just fewer zero groups.
- Timing includes K sequential full-context fits/mean calls. Nested per-method times are breakdowns, NOT added twice to the enclosing call.
- Context/target loading and cached-query fingerprint verification are separate. Fresh raw-graph-to-statistics/cache-build latency is excluded for VAL/TEST.
- LIB native_input_to_statistics includes input loading/geometry/357 statistics, without opening labels before prediction. Recorded sums are not end-to-end latency.

## Full Metrics

| problem | split | n | top1_pct | mean_cost | sbs_cost | oracle_cost | vs_sbs_pct | vs_oracle_pct | actual_regret_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | VAL | 81000 | 71.055556 | 11.416108 | 11.427978 | 11.371692 | -0.201463 | 0.386023 | 0.393104 |
| TSP | VAL | 4500 | 45.488889 | 8.023554 | 8.045354 | 7.990533 | -0.270965 | 0.413248 | 0.401609 |
| CVRP | VAL | 4500 | 47.844444 | 18.146278 | 18.269377 | 18.064087 | -0.673802 | 0.454994 | 0.586960 |
| ATSP | VAL | 4500 | 77.955556 | 1.637715 | 1.676905 | 1.632471 | -2.337028 | 0.321253 | 0.318342 |
| OVRP | VAL | 4500 | 41.666667 | 7.459614 | 7.469579 | 7.415777 | -0.133409 | 0.591137 | 0.601848 |
| VRPB | VAL | 4500 | 80.888889 | 9.354878 | 9.354876 | 9.318750 | 0.000023 | 0.387699 | 0.396120 |
| VRPL | VAL | 4500 | 36.155556 | 12.431675 | 12.437959 | 12.383444 | -0.050529 | 0.389477 | 0.396544 |
| VRPTW | VAL | 4500 | 80.111111 | 18.819021 | 18.819431 | 18.766139 | -0.002177 | 0.281798 | 0.298417 |
| OVRPTW | VAL | 4500 | 83.755556 | 10.143753 | 10.143730 | 10.116859 | 0.000225 | 0.265827 | 0.262634 |
| OVRPB | VAL | 4500 | 88.511111 | 6.357254 | 6.357254 | 6.341078 | 0.000000 | 0.255104 | 0.265013 |
| OVRPL | VAL | 4500 | 42.022222 | 7.436053 | 7.445709 | 7.390416 | -0.129688 | 0.617522 | 0.629409 |
| VRPBL | VAL | 4500 | 78.644444 | 9.370116 | 9.371523 | 9.328366 | -0.015012 | 0.447555 | 0.451822 |
| VRPBTW | VAL | 4500 | 78.355556 | 19.950619 | 19.950430 | 19.851084 | 0.000949 | 0.501413 | 0.476837 |
| VRPLTW | VAL | 4500 | 78.733333 | 18.765557 | 18.765871 | 18.707843 | -0.001673 | 0.308505 | 0.323737 |
| OVRPBL | VAL | 4500 | 89.200000 | 6.369902 | 6.370093 | 6.352824 | -0.003005 | 0.268823 | 0.279289 |
| OVRPBTW | VAL | 4500 | 83.466667 | 10.660808 | 10.661098 | 10.623245 | -0.002726 | 0.353590 | 0.328107 |
| OVRPLTW | VAL | 4500 | 83.444444 | 10.157488 | 10.157423 | 10.129818 | 0.000644 | 0.273156 | 0.276084 |
| VRPBLTW | VAL | 4500 | 79.444444 | 19.796011 | 19.797044 | 19.706420 | -0.005216 | 0.454629 | 0.442948 |
| OVRPBLTW | VAL | 4500 | 83.311111 | 10.609643 | 10.609955 | 10.571302 | -0.002948 | 0.362686 | 0.340155 |
| ALL | TEST | 81000 | 71.216049 | 11.373265 | 11.385299 | 11.329932 | -0.202978 | 0.380106 | 0.386229 |
| TSP | TEST | 4500 | 46.111111 | 7.969327 | 7.989621 | 7.937593 | -0.254011 | 0.399783 | 0.392775 |
| CVRP | TEST | 4500 | 48.022222 | 17.999160 | 18.124983 | 17.918509 | -0.694197 | 0.450102 | 0.581560 |
| ATSP | TEST | 4500 | 77.688889 | 1.634959 | 1.674207 | 1.629441 | -2.344286 | 0.338620 | 0.336843 |
| OVRP | TEST | 4500 | 40.266667 | 7.445255 | 7.455750 | 7.400487 | -0.140761 | 0.604927 | 0.613092 |
| VRPB | TEST | 4500 | 80.400000 | 9.351294 | 9.351810 | 9.315298 | -0.005514 | 0.386420 | 0.389834 |
| VRPL | TEST | 4500 | 37.422222 | 12.347781 | 12.356138 | 12.300958 | -0.067634 | 0.380643 | 0.387112 |
| VRPTW | TEST | 4500 | 80.755556 | 18.691252 | 18.690545 | 18.637408 | 0.003780 | 0.288900 | 0.299563 |
| OVRPTW | TEST | 4500 | 83.888889 | 10.093168 | 10.093063 | 10.068135 | 0.001045 | 0.248637 | 0.239591 |
| OVRPB | TEST | 4500 | 88.133333 | 6.371035 | 6.370826 | 6.352225 | 0.003288 | 0.296112 | 0.303819 |
| OVRPL | TEST | 4500 | 41.288889 | 7.462453 | 7.472542 | 7.418033 | -0.135011 | 0.598807 | 0.604881 |
| VRPBL | TEST | 4500 | 79.400000 | 9.396103 | 9.396379 | 9.355967 | -0.002936 | 0.428990 | 0.432601 |
| VRPBTW | TEST | 4500 | 78.866667 | 19.665472 | 19.665725 | 19.577666 | -0.001285 | 0.448504 | 0.444593 |
| VRPLTW | TEST | 4500 | 79.288889 | 18.802478 | 18.803136 | 18.745143 | -0.003497 | 0.305870 | 0.319739 |
| OVRPBL | TEST | 4500 | 89.333333 | 6.375431 | 6.375741 | 6.359310 | -0.004860 | 0.253503 | 0.263993 |
| OVRPBTW | TEST | 4500 | 84.844444 | 10.659524 | 10.659591 | 10.624748 | -0.000635 | 0.327305 | 0.299216 |
| OVRPLTW | TEST | 4500 | 82.844444 | 10.085940 | 10.086112 | 10.059001 | -0.001703 | 0.267812 | 0.268407 |
| VRPBLTW | TEST | 4500 | 79.733333 | 19.705960 | 19.707068 | 19.613284 | -0.005623 | 0.472516 | 0.452579 |
| OVRPBLTW | TEST | 4500 | 83.600000 | 10.662174 | 10.662149 | 10.625574 | 0.000241 | 0.344459 | 0.321924 |
| TSP | TSPLIB | 49 | 24.489796 | 8.215202 | 8.240435 | 8.142167 | -0.306213 | 0.896996 | 0.986103 |
| CVRP | CVRPLIB | 100 | 56.000000 | 67.403215 | 67.245364 | 66.945093 | 0.234738 | 0.684324 | 0.598571 |

## NSS Four Datasets

| Dataset | Status | NSS Top1 % | R98 Top1 % | NSS cost | R98 cost |
| --- | --- | --- | --- | --- | --- |
| TSP TEST | complete | 45.888889 | 46.111111 | 7.970401 | 7.969327 |
| CVRP TEST | complete | 45.977778 | 48.022222 | 18.004802 | 17.999160 |
| TSPLIB | complete | 20.408163 | 24.489796 | 8.217907 | 8.215202 |
| CVRPLIB | complete | 70.000000 | 56.000000 | 67.101112 | 67.403215 |

## Historical Full18 TEST

| Reference | Status | Top1 delta pp | Cost delta | Regret delta pp |
| --- | --- | --- | --- | --- |
| R89B | complete | -0.203704 | -0.000206 | -0.003417 |
| R84A | complete | -0.019753 | -0.000578 | -0.008618 |

Fixed36 summaries: [{"all_test_regret_delta_pp": -0.0034168856066205, "baseline": "R89B", "correct_count": 32, "false_pick_count": 38, "lost": 41, "net_recovery": -13, "new_false_pick_count": 35, "new_false_pick_regret_contribution_pp": 0.001242500973217591, "recall_pct": 0.6959547629404089, "rescued": 28, "selected_count": 70, "split": "TEST", "status": "complete", "target_groups": 36, "target_regret_delta_pp": 0.002322908675512585, "winner_count": 4598}, {"all_test_regret_delta_pp": -0.008617718017882732, "baseline": "R84A", "correct_count": 32, "false_pick_count": 38, "lost": 10, "net_recovery": 19, "new_false_pick_count": 36, "new_false_pick_regret_contribution_pp": 0.0012914037793557336, "recall_pct": 0.6959547629404089, "rescued": 29, "selected_count": 70, "split": "TEST", "status": "complete", "target_groups": 36, "target_regret_delta_pp": -0.000715152753567838, "winner_count": 4598}]
