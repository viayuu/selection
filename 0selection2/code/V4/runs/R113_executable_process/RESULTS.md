# R113 Joint Closeout

## What Completed?
All three independent rc0 train/finalize pipelines; full18 VAL/TEST, native LIB, PDFs and post-lock process diagnostics.
Saved predictions only. One unique whole-model VAL-selected checkpoint per arm; no model execution or TEST tuning.
| Arm | Selected update | Checkpoint SHA256 |
| --- | --- | --- |
| A_scratch | 40320 | 7bbe2d341261c17d03966ac75af91764d0dceacc1aac6d794f8024aa8619596b |
| B_snapshot | 51840 | f3e09c85074311ccf0e3b376d14a2bac587aa4766bff94600853c51dd7114129 |
| C_transition | 40320 | de0523bd1f7beadf51456ff50767a03d76cc96756278f3578c523310b2abd286 |
Data: selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018

## Results
| Model | Split | Problem | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R113/A_scratch | VAL | ALL | 70.927160 | 11.417807 | 11.427978 | 11.371692 | -0.185309 | 0.402390 | 0.413414 |
| R113/B_snapshot | VAL | ALL | 70.835802 | 11.417715 | 11.427978 | 11.371692 | -0.185514 | 0.402182 | 0.412755 |
| R113/C_transition | VAL | ALL | 70.809877 | 11.417839 | 11.427978 | 11.371692 | -0.184394 | 0.403311 | 0.414188 |
| R113/A_scratch | TEST | ALL | 71.164198 | 11.374716 | 11.385299 | 11.329932 | -0.190165 | 0.393068 | 0.402034 |
| R113/B_snapshot | TEST | ALL | 71.188889 | 11.374399 | 11.385299 | 11.329932 | -0.193219 | 0.389976 | 0.398368 |
| R113/C_transition | TEST | ALL | 71.059259 | 11.374651 | 11.385299 | 11.329932 | -0.190058 | 0.393183 | 0.401596 |
| R94B | VAL | ALL | 71.428395 | 11.416279 | 11.427978 | 11.371692 | -0.198659 | 0.388932 | 0.399079 |
| R94B | TEST | ALL | 71.577778 | 11.372234 | 11.385299 | 11.329932 | -0.211169 | 0.371939 | 0.379115 |
| R89B | VAL | ALL | 71.248148 | 11.416294 | 11.427978 | 11.371692 | -0.197193 | 0.390414 | 0.400676 |
| R89B | TEST | ALL | 71.419753 | 11.373470 | 11.385299 | 11.329932 | -0.201446 | 0.381702 | 0.389646 |
Native/FP64 ind (not FP32 argmin). SBS fitted on frozen full TRAIN; Oracle is the legal-pool per-instance minimum.
ALL equal-problem; raw cost descriptive; ratios of means are not per-instance actual regret.

## Paired Changes
| Comparison | Split | Problem | Rescued | Harmed | Wrong cheaper | Wrong dearer | dCost | Wrong dCost | dRegret pp | W % | G % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_snapshot vs A_scratch | VAL | ALL | 1007 | 1081 | 534 | 491 | -0.000093 | -0.000208 | -0.000659 | 1.902469 | -0.000307 |
| C_transition vs A_scratch | VAL | ALL | 920 | 1015 | 471 | 457 | 0.000031 | -0.000120 | 0.000774 | 1.717284 | -0.001568 |
| C_transition vs B_snapshot | VAL | ALL | 788 | 809 | 413 | 444 | 0.000124 | 0.000091 | 0.001433 | 1.482716 | -0.002021 |
| B_snapshot vs A_scratch | TEST | ALL | 1007 | 987 | 549 | 484 | -0.000317 | -0.000222 | -0.003666 | 1.919753 | 0.002923 |
| C_transition vs A_scratch | TEST | ALL | 871 | 956 | 485 | 454 | -0.000065 | -0.000171 | -0.000438 | 1.674074 | -0.000202 |
| C_transition vs B_snapshot | TEST | ALL | 769 | 874 | 410 | 432 | 0.000252 | 0.000032 | 0.003228 | 1.455556 | -0.003789 |
| A_scratch vs R94B | VAL | ALL | 1733 | 2139 | 780 | 899 | 0.001528 | 0.000031 | 0.014335 | 3.101235 | -0.020137 |
| B_snapshot vs R94B | VAL | ALL | 1548 | 2028 | 744 | 825 | 0.001436 | -0.000136 | 0.013676 | 2.828395 | -0.019526 |
| C_transition vs R94B | VAL | ALL | 1710 | 2211 | 793 | 890 | 0.001560 | -0.000125 | 0.015109 | 3.088889 | -0.021003 |
| A_scratch vs R94B | TEST | ALL | 1690 | 2025 | 809 | 908 | 0.002482 | 0.000298 | 0.022919 | 3.085185 | -0.026324 |
| B_snapshot vs R94B | TEST | ALL | 1592 | 1907 | 774 | 802 | 0.002165 | 0.000079 | 0.019253 | 2.919753 | -0.022589 |
| C_transition vs R94B | TEST | ALL | 1695 | 2115 | 802 | 857 | 0.002417 | 0.000104 | 0.022481 | 3.082716 | -0.025959 |
| A_scratch vs R89B | VAL | ALL | 1474 | 1734 | 634 | 835 | 0.001513 | 0.000287 | 0.012738 | 2.601235 | -0.014729 |
| B_snapshot vs R89B | VAL | ALL | 1557 | 1891 | 681 | 844 | 0.001420 | 0.000150 | 0.012079 | 2.761728 | -0.014248 |
| C_transition vs R89B | VAL | ALL | 1587 | 1942 | 683 | 882 | 0.001545 | 0.000186 | 0.013512 | 2.801235 | -0.015707 |
| A_scratch vs R89B | TEST | ALL | 1465 | 1672 | 729 | 794 | 0.001245 | 0.000308 | 0.012389 | 2.707407 | -0.013824 |
| B_snapshot vs R89B | TEST | ALL | 1607 | 1794 | 740 | 755 | 0.000928 | 0.000069 | 0.008722 | 2.895062 | -0.010197 |
| C_transition vs R89B | TEST | ALL | 1589 | 1881 | 738 | 787 | 0.001180 | 0.000157 | 0.011950 | 2.871605 | -0.013485 |
Rescued=baseline wrong/current correct; harmed=the reverse. dCost/dRegret=current-baseline.
W=100*mean(current_cost < baseline_cost); G=100*mean((baseline_cost-current_cost)/baseline_cost). Positive G is improvement.
Both-wrong net cost/regret and gross gain/harm are separate comparison.csv columns, normalized by all problem rows.

## Historical References
| Model | Status | Reason |
| --- | --- | --- |
| R94B | complete |  |
| R89B | complete |  |
Only complete exact-data whole historical models; no available-task subsets or best-of-version stitching.

## Fixed Historical Target36
{"count": 36, "identity": {"baseline_checkpoint_sha256": "23fb9a5cf621045138fcdcfe5e6613a1bfd061c8d4dfeca108db3548dbf8031f", "baseline_state_hash": "ad0a6d9c68c6f117b477807f6f8be52662032a8baea348dda6144c46091303e2", "baseline_val_prediction_sha256": "8a1aea9caf608d12fbc83d89704a50eb37d461665c7609ad3e25778cdab932a7", "data_root": "/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018", "rule": "A locked VAL: chosen percentage <= 0.1 and native winner count > 0", "threshold_chosen_pct": 0.1}, "path": "/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/comparison_artifacts/target_list.json", "sha256": "406ded5a21d8b73c2a41b87dc2bc3cc14330663ae1aa4f59ca6fbfde83770fe4", "status": "complete"}
Target membership is frozen from historical VAL, not newly chosen using TEST or R113 predictions.
target_recovery.csv: per-method rescues/harms/new false picks; winner-partition FIXED36 + OUTSIDE36 exactly cover every row.
New false means a changed prediction to a non-winner; winner partitions classify these by true-winner membership, not chosen-method membership.

## Scientific Boundary
same fresh full model; only own TRAIN-pretrained processor differs; final mode always predicted feedback; resource-only migration: B/C processor8640 completed on 4090, selector resumed at persisted20160 on 3090; unsaved B2286/C1872 updates replayed.
single seed2; descriptive evidence, no multi-seed significance or automatic promotion.
whole saved exact-data references; not fresh parameter/budget matched controls.
development-exposed TEST; no TEST/LIB checkpoint selection, tuning or per-task assembly.
same seeded task/shuffle protocol; no claim of independently archived selector per-update row IDs.
post-lock fixed input-only VAL subset; teacher evaluation only, never query/model selection.

## Full18 And LIB Tables

### A_scratch
| Model | Split | Problem | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R113/A_scratch | VAL | TSP | 46.133333 | 8.025839 | 8.045354 | 7.990533 | -0.242561 | 0.441847 | 0.431160 |
| R113/A_scratch | VAL | CVRP | 46.066667 | 18.168150 | 18.269377 | 18.064087 | -0.554080 | 0.576078 | 0.766788 |
| R113/A_scratch | VAL | ATSP | 77.066667 | 1.638757 | 1.676905 | 1.632471 | -2.274892 | 0.385080 | 0.382087 |
| R113/A_scratch | VAL | OVRP | 40.533333 | 7.462796 | 7.469579 | 7.415777 | -0.090817 | 0.634039 | 0.649252 |
| R113/A_scratch | VAL | VRPB | 80.911111 | 9.355259 | 9.354876 | 9.318750 | 0.004093 | 0.391784 | 0.400225 |
| R113/A_scratch | VAL | VRPL | 37.666667 | 12.432522 | 12.437959 | 12.383444 | -0.043716 | 0.396320 | 0.404664 |
| R113/A_scratch | VAL | VRPTW | 80.022222 | 18.818706 | 18.819431 | 18.766139 | -0.003852 | 0.280118 | 0.296341 |
| R113/A_scratch | VAL | OVRPTW | 83.777778 | 10.143920 | 10.143730 | 10.116859 | 0.001877 | 0.267484 | 0.264822 |
| R113/A_scratch | VAL | OVRPB | 88.511111 | 6.357254 | 6.357254 | 6.341078 | 0.000000 | 0.255104 | 0.265013 |
| R113/A_scratch | VAL | OVRPL | 41.044444 | 7.439167 | 7.445709 | 7.390416 | -0.087865 | 0.659658 | 0.674834 |
| R113/A_scratch | VAL | VRPBL | 78.866667 | 9.368634 | 9.371523 | 9.328366 | -0.030827 | 0.431666 | 0.434729 |
| R113/A_scratch | VAL | VRPBTW | 78.466667 | 19.949860 | 19.950430 | 19.851084 | -0.002860 | 0.497585 | 0.475711 |
| R113/A_scratch | VAL | VRPLTW | 78.800000 | 18.765198 | 18.765871 | 18.707843 | -0.003586 | 0.306586 | 0.321785 |
| R113/A_scratch | VAL | OVRPBL | 89.200000 | 6.370093 | 6.370093 | 6.352824 | 0.000000 | 0.271836 | 0.283581 |
| R113/A_scratch | VAL | OVRPBTW | 83.488889 | 10.660941 | 10.661098 | 10.623245 | -0.001474 | 0.354847 | 0.329078 |
| R113/A_scratch | VAL | OVRPLTW | 83.444444 | 10.157488 | 10.157423 | 10.129818 | 0.000641 | 0.273153 | 0.276076 |
| R113/A_scratch | VAL | VRPBLTW | 79.422222 | 19.796072 | 19.797044 | 19.706420 | -0.004911 | 0.454936 | 0.442751 |
| R113/A_scratch | VAL | OVRPBLTW | 83.266667 | 10.609877 | 10.609955 | 10.571302 | -0.000737 | 0.364904 | 0.342559 |
| R113/A_scratch | VAL | ALL | 70.927160 | 11.417807 | 11.427978 | 11.371692 | -0.185309 | 0.402390 | 0.413414 |
| R113/A_scratch | TEST | TSP | 46.822222 | 7.972356 | 7.989621 | 7.937593 | -0.216094 | 0.437948 | 0.429889 |
| R113/A_scratch | TEST | CVRP | 46.777778 | 18.018578 | 18.124983 | 17.918509 | -0.587066 | 0.558468 | 0.728056 |
| R113/A_scratch | TEST | ATSP | 76.800000 | 1.635377 | 1.674207 | 1.629441 | -2.319312 | 0.364279 | 0.363963 |
| R113/A_scratch | TEST | OVRP | 40.266667 | 7.447659 | 7.455750 | 7.400487 | -0.108524 | 0.637405 | 0.650098 |
| R113/A_scratch | TEST | VRPB | 80.600000 | 9.350486 | 9.351810 | 9.315298 | -0.014159 | 0.377741 | 0.380171 |
| R113/A_scratch | TEST | VRPL | 37.911111 | 12.349939 | 12.356138 | 12.300958 | -0.050168 | 0.398187 | 0.408057 |
| R113/A_scratch | TEST | VRPTW | 80.933333 | 18.689677 | 18.690545 | 18.637408 | -0.004645 | 0.280451 | 0.292392 |
| R113/A_scratch | TEST | OVRPTW | 83.933333 | 10.093063 | 10.093063 | 10.068135 | 0.000000 | 0.247590 | 0.238522 |
| R113/A_scratch | TEST | OVRPB | 88.133333 | 6.370826 | 6.370826 | 6.352225 | 0.000000 | 0.292814 | 0.301388 |
| R113/A_scratch | TEST | OVRPL | 40.511111 | 7.465732 | 7.472542 | 7.418033 | -0.091124 | 0.643016 | 0.651711 |
| R113/A_scratch | TEST | VRPBL | 79.555556 | 9.395548 | 9.396379 | 9.355967 | -0.008839 | 0.423061 | 0.426270 |
| R113/A_scratch | TEST | VRPBTW | 78.911111 | 19.664458 | 19.665725 | 19.577666 | -0.006445 | 0.443321 | 0.439425 |
| R113/A_scratch | TEST | VRPLTW | 79.288889 | 18.802732 | 18.803136 | 18.745143 | -0.002149 | 0.307222 | 0.321761 |
| R113/A_scratch | TEST | OVRPBL | 89.311111 | 6.375636 | 6.375741 | 6.359310 | -0.001647 | 0.256724 | 0.267898 |
| R113/A_scratch | TEST | OVRPBTW | 84.800000 | 10.659602 | 10.659591 | 10.624748 | 0.000104 | 0.328046 | 0.300056 |
| R113/A_scratch | TEST | OVRPLTW | 82.933333 | 10.085761 | 10.086112 | 10.059001 | -0.003479 | 0.266031 | 0.266588 |
| R113/A_scratch | TEST | VRPBLTW | 79.844444 | 19.705408 | 19.707068 | 19.613284 | -0.008423 | 0.469702 | 0.449915 |
| R113/A_scratch | TEST | OVRPBLTW | 83.622222 | 10.662042 | 10.662149 | 10.625574 | -0.001004 | 0.343209 | 0.320459 |
| R113/A_scratch | TEST | ALL | 71.164198 | 11.374716 | 11.385299 | 11.329932 | -0.190165 | 0.393068 | 0.402034 |
| R113/A_scratch | TSPLIB | TSP | 24.489796 | 8.209452 | 8.240435 | 8.142167 | -0.375986 | 0.826381 | 0.877394 |
| R113/A_scratch | CVRPLIB | CVRP | 48.000000 | 67.599797 | 67.245364 | 66.945093 | 0.527074 | 0.977972 | 0.835197 |

### B_snapshot
| Model | Split | Problem | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R113/B_snapshot | VAL | TSP | 46.222222 | 8.025671 | 8.045354 | 7.990533 | -0.244649 | 0.439744 | 0.428883 |
| R113/B_snapshot | VAL | CVRP | 45.622222 | 18.165689 | 18.269377 | 18.064087 | -0.567554 | 0.562451 | 0.743303 |
| R113/B_snapshot | VAL | ATSP | 76.755556 | 1.638772 | 1.676905 | 1.632471 | -2.273973 | 0.386024 | 0.381334 |
| R113/B_snapshot | VAL | OVRP | 40.200000 | 7.463029 | 7.469579 | 7.415777 | -0.087693 | 0.637186 | 0.655131 |
| R113/B_snapshot | VAL | VRPB | 80.888889 | 9.356540 | 9.354876 | 9.318750 | 0.017791 | 0.405536 | 0.409909 |
| R113/B_snapshot | VAL | VRPL | 37.755556 | 12.431705 | 12.437959 | 12.383444 | -0.050288 | 0.389719 | 0.397254 |
| R113/B_snapshot | VAL | VRPTW | 79.955556 | 18.819210 | 18.819431 | 18.766139 | -0.001172 | 0.282805 | 0.300130 |
| R113/B_snapshot | VAL | OVRPTW | 83.800000 | 10.142991 | 10.143730 | 10.116859 | -0.007285 | 0.258298 | 0.255231 |
| R113/B_snapshot | VAL | OVRPB | 88.488889 | 6.357287 | 6.357254 | 6.341078 | 0.000517 | 0.255623 | 0.265553 |
| R113/B_snapshot | VAL | OVRPL | 40.622222 | 7.439188 | 7.445709 | 7.390416 | -0.087581 | 0.659945 | 0.682102 |
| R113/B_snapshot | VAL | VRPBL | 78.711111 | 9.369141 | 9.371523 | 9.328366 | -0.025413 | 0.437105 | 0.438621 |
| R113/B_snapshot | VAL | VRPBTW | 78.400000 | 19.950564 | 19.950430 | 19.851084 | 0.000670 | 0.501132 | 0.478652 |
| R113/B_snapshot | VAL | VRPLTW | 78.844444 | 18.764366 | 18.765871 | 18.707843 | -0.008018 | 0.302140 | 0.317335 |
| R113/B_snapshot | VAL | OVRPBL | 89.200000 | 6.369956 | 6.370093 | 6.352824 | -0.002162 | 0.269668 | 0.281371 |
| R113/B_snapshot | VAL | OVRPBTW | 83.577778 | 10.660387 | 10.661098 | 10.623245 | -0.006672 | 0.349630 | 0.323286 |
| R113/B_snapshot | VAL | OVRPLTW | 83.444444 | 10.157488 | 10.157423 | 10.129818 | 0.000641 | 0.273153 | 0.276076 |
| R113/B_snapshot | VAL | VRPBLTW | 79.422222 | 19.795963 | 19.797044 | 19.706420 | -0.005461 | 0.454383 | 0.441800 |
| R113/B_snapshot | VAL | OVRPBLTW | 83.133333 | 10.610916 | 10.609955 | 10.571302 | 0.009051 | 0.374729 | 0.353618 |
| R113/B_snapshot | VAL | ALL | 70.835802 | 11.417715 | 11.427978 | 11.371692 | -0.185514 | 0.402182 | 0.412755 |
| R113/B_snapshot | TEST | TSP | 46.600000 | 7.971642 | 7.989621 | 7.937593 | -0.225031 | 0.428952 | 0.420407 |
| R113/B_snapshot | TEST | CVRP | 47.488889 | 18.013341 | 18.124983 | 17.918509 | -0.615957 | 0.529243 | 0.682940 |
| R113/B_snapshot | TEST | ATSP | 76.888889 | 1.635298 | 1.674207 | 1.629441 | -2.324029 | 0.359433 | 0.359101 |
| R113/B_snapshot | TEST | OVRP | 40.622222 | 7.446669 | 7.455750 | 7.400487 | -0.121796 | 0.624034 | 0.638238 |
| R113/B_snapshot | TEST | VRPB | 80.533333 | 9.350770 | 9.351810 | 9.315298 | -0.011120 | 0.380792 | 0.382870 |
| R113/B_snapshot | TEST | VRPL | 37.266667 | 12.350350 | 12.356138 | 12.300958 | -0.046848 | 0.401522 | 0.410964 |
| R113/B_snapshot | TEST | VRPTW | 80.911111 | 18.689981 | 18.690545 | 18.637408 | -0.003020 | 0.282082 | 0.294236 |
| R113/B_snapshot | TEST | OVRPTW | 83.911111 | 10.093165 | 10.093063 | 10.068135 | 0.001018 | 0.248610 | 0.239784 |
| R113/B_snapshot | TEST | OVRPB | 88.111111 | 6.370831 | 6.370826 | 6.352225 | 0.000085 | 0.292899 | 0.301464 |
| R113/B_snapshot | TEST | OVRPL | 40.955556 | 7.464290 | 7.472542 | 7.418033 | -0.110422 | 0.623577 | 0.635793 |
| R113/B_snapshot | TEST | VRPBL | 79.577778 | 9.396131 | 9.396379 | 9.355967 | -0.002637 | 0.429291 | 0.431657 |
| R113/B_snapshot | TEST | VRPBTW | 78.777778 | 19.665714 | 19.665725 | 19.577666 | -0.000059 | 0.449736 | 0.447167 |
| R113/B_snapshot | TEST | VRPLTW | 79.288889 | 18.802721 | 18.803136 | 18.745143 | -0.002208 | 0.307163 | 0.321437 |
| R113/B_snapshot | TEST | OVRPBL | 89.288889 | 6.375594 | 6.375741 | 6.359310 | -0.002305 | 0.256065 | 0.267333 |
| R113/B_snapshot | TEST | OVRPBTW | 84.844444 | 10.659403 | 10.659591 | 10.624748 | -0.001768 | 0.326168 | 0.298295 |
| R113/B_snapshot | TEST | OVRPLTW | 82.911111 | 10.085766 | 10.086112 | 10.059001 | -0.003422 | 0.266088 | 0.266661 |
| R113/B_snapshot | TEST | VRPBLTW | 79.844444 | 19.705309 | 19.707068 | 19.613284 | -0.008928 | 0.469194 | 0.450144 |
| R113/B_snapshot | TEST | OVRPBLTW | 83.577778 | 10.662202 | 10.662149 | 10.625574 | 0.000499 | 0.344718 | 0.322138 |
| R113/B_snapshot | TEST | ALL | 71.188889 | 11.374399 | 11.385299 | 11.329932 | -0.193219 | 0.389976 | 0.398368 |
| R113/B_snapshot | TSPLIB | TSP | 30.612245 | 8.191982 | 8.240435 | 8.142167 | -0.587995 | 0.611814 | 0.681863 |
| R113/B_snapshot | CVRPLIB | CVRP | 49.000000 | 67.547212 | 67.245364 | 66.945093 | 0.448875 | 0.899422 | 0.817800 |

### C_transition
| Model | Split | Problem | Top1 % | mean cost | SBS | Oracle | vs SBS % | vs Oracle % | Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R113/C_transition | VAL | TSP | 46.377778 | 8.025277 | 8.045354 | 7.990533 | -0.249542 | 0.434817 | 0.424897 |
| R113/C_transition | VAL | CVRP | 45.377778 | 18.166357 | 18.269377 | 18.064087 | -0.563897 | 0.566149 | 0.754053 |
| R113/C_transition | VAL | ATSP | 76.533333 | 1.638824 | 1.676905 | 1.632471 | -2.270906 | 0.389174 | 0.383269 |
| R113/C_transition | VAL | OVRP | 40.488889 | 7.462934 | 7.469579 | 7.415777 | -0.088961 | 0.635908 | 0.652383 |
| R113/C_transition | VAL | VRPB | 80.755556 | 9.357355 | 9.354876 | 9.318750 | 0.026496 | 0.414274 | 0.420284 |
| R113/C_transition | VAL | VRPL | 37.488889 | 12.432677 | 12.437959 | 12.383444 | -0.042469 | 0.397573 | 0.404102 |
| R113/C_transition | VAL | VRPTW | 79.933333 | 18.819044 | 18.819431 | 18.766139 | -0.002055 | 0.281920 | 0.299278 |
| R113/C_transition | VAL | OVRPTW | 83.777778 | 10.143681 | 10.143730 | 10.116859 | -0.000479 | 0.265121 | 0.262001 |
| R113/C_transition | VAL | OVRPB | 88.511111 | 6.357254 | 6.357254 | 6.341078 | 0.000000 | 0.255104 | 0.265013 |
| R113/C_transition | VAL | OVRPL | 40.511111 | 7.439488 | 7.445709 | 7.390416 | -0.083559 | 0.663997 | 0.683722 |
| R113/C_transition | VAL | VRPBL | 78.644444 | 9.368975 | 9.371523 | 9.328366 | -0.027182 | 0.435328 | 0.438040 |
| R113/C_transition | VAL | VRPBTW | 78.533333 | 19.950080 | 19.950430 | 19.851084 | -0.001757 | 0.498693 | 0.476813 |
| R113/C_transition | VAL | VRPLTW | 78.800000 | 18.765198 | 18.765871 | 18.707843 | -0.003586 | 0.306586 | 0.321785 |
| R113/C_transition | VAL | OVRPBL | 89.222222 | 6.369919 | 6.370093 | 6.352824 | -0.002736 | 0.269093 | 0.280813 |
| R113/C_transition | VAL | OVRPBTW | 83.488889 | 10.660758 | 10.661098 | 10.623245 | -0.003188 | 0.353127 | 0.327110 |
| R113/C_transition | VAL | OVRPLTW | 83.466667 | 10.157456 | 10.157423 | 10.129818 | 0.000326 | 0.272837 | 0.275733 |
| R113/C_transition | VAL | VRPBLTW | 79.422222 | 19.795785 | 19.797044 | 19.706420 | -0.006362 | 0.453478 | 0.441899 |
| R113/C_transition | VAL | OVRPBLTW | 83.244444 | 10.610037 | 10.609955 | 10.571302 | 0.000768 | 0.366415 | 0.344190 |
| R113/C_transition | VAL | ALL | 70.809877 | 11.417839 | 11.427978 | 11.371692 | -0.184394 | 0.403311 | 0.414188 |
| R113/C_transition | TEST | TSP | 46.444444 | 7.972612 | 7.989621 | 7.937593 | -0.212895 | 0.441169 | 0.431930 |
| R113/C_transition | TEST | CVRP | 46.933333 | 18.014831 | 18.124983 | 17.918509 | -0.607736 | 0.537559 | 0.696996 |
| R113/C_transition | TEST | ATSP | 76.244444 | 1.635618 | 1.674207 | 1.629441 | -2.304909 | 0.379078 | 0.377662 |
| R113/C_transition | TEST | OVRP | 40.022222 | 7.446964 | 7.455750 | 7.400487 | -0.117834 | 0.628025 | 0.643717 |
| R113/C_transition | TEST | VRPB | 80.577778 | 9.350723 | 9.351810 | 9.315298 | -0.011618 | 0.380293 | 0.383297 |
| R113/C_transition | TEST | VRPL | 37.000000 | 12.351206 | 12.356138 | 12.300958 | -0.039915 | 0.408486 | 0.417404 |
| R113/C_transition | TEST | VRPTW | 80.911111 | 18.689960 | 18.690545 | 18.637408 | -0.003131 | 0.281970 | 0.293635 |
| R113/C_transition | TEST | OVRPTW | 83.933333 | 10.093063 | 10.093063 | 10.068135 | 0.000000 | 0.247590 | 0.238522 |
| R113/C_transition | TEST | OVRPB | 88.111111 | 6.370831 | 6.370826 | 6.352225 | 0.000085 | 0.292899 | 0.301464 |
| R113/C_transition | TEST | OVRPL | 40.800000 | 7.465215 | 7.472542 | 7.418033 | -0.098050 | 0.636040 | 0.646632 |
| R113/C_transition | TEST | VRPBL | 79.577778 | 9.395129 | 9.396379 | 9.355967 | -0.013307 | 0.418574 | 0.421350 |
| R113/C_transition | TEST | VRPBTW | 78.777778 | 19.665390 | 19.665725 | 19.577666 | -0.001705 | 0.448082 | 0.443552 |
| R113/C_transition | TEST | VRPLTW | 79.266667 | 18.802741 | 18.803136 | 18.745143 | -0.002098 | 0.307273 | 0.321935 |
| R113/C_transition | TEST | OVRPBL | 89.311111 | 6.375696 | 6.375741 | 6.359310 | -0.000712 | 0.257662 | 0.269048 |
| R113/C_transition | TEST | OVRPBTW | 84.844444 | 10.659346 | 10.659591 | 10.624748 | -0.002298 | 0.325635 | 0.297742 |
| R113/C_transition | TEST | OVRPLTW | 82.866667 | 10.086112 | 10.086112 | 10.059001 | 0.000000 | 0.269520 | 0.270132 |
| R113/C_transition | TEST | VRPBLTW | 79.844444 | 19.706160 | 19.707068 | 19.613284 | -0.004608 | 0.473536 | 0.452441 |
| R113/C_transition | TEST | OVRPBLTW | 83.600000 | 10.662116 | 10.662149 | 10.625574 | -0.000306 | 0.343910 | 0.321275 |
| R113/C_transition | TEST | ALL | 71.059259 | 11.374651 | 11.385299 | 11.329932 | -0.190058 | 0.393183 | 0.401596 |
| R113/C_transition | TSPLIB | TSP | 28.571429 | 8.195919 | 8.240435 | 8.142167 | -0.540216 | 0.660169 | 0.759088 |
| R113/C_transition | CVRPLIB | CVRP | 60.000000 | 67.295075 | 67.245364 | 66.945093 | 0.073924 | 0.522789 | 0.652926 |

Full precision: metrics.csv / comparison.csv / target_recovery*.csv / process_comparison.csv.
Per-arm method-distribution and NSS four-dataset tables remain in full_model_statistics/.
