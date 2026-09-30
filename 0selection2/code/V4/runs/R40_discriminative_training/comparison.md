# R40: Discriminative Training

Seed 2, full 18-task train/validation. No test evaluation.
A: dual-stream classification. B: direct global classifier. Both use the identical winner-cost loss.
Both are trained from scratch; the shared encoder starts with identical weights.
The sample sequence matches R39A; 64-instance real updates increase the update count tenfold without gradient accumulation.
Learning rate is scheduled per successful update, not per historical R39 step.
ALL percentages average per-problem percentages, not ratios of aggregate costs.

## Validation Comparison

| Model | Point | Top1 | Top2 | Top3 | Mean cost | vs_SBS (%) | Actual regret (%) | CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R39A reference | best | 0.4843 | 0.8198 | 0.9274 | 11.170767 | -1.0554 | 0.9451 | 1.10238 |
| R39A reference | final | 0.4843 | 0.8202 | 0.9279 | 11.171171 | -1.0496 | 0.9510 | 1.10238 |
| R39A reference | last5 | 0.4833 | 0.8198 | 0.9279 | 11.171554 | -1.0479 | 0.9540 | 1.10213 |
| A | best | 0.4793 | 0.8139 | 0.9222 | 11.173208 | -1.0392 | 0.9621 | 1.11253 |
| A | final | 0.4501 | 0.7956 | 0.9074 | 11.196538 | -0.8527 | 1.1742 | 2.22593 |
| A | last5 | 0.4507 | 0.7953 | 0.9080 | 11.196843 | -0.8501 | 1.1756 | 2.17957 |
| B | best | 0.4811 | 0.8168 | 0.9235 | 11.173831 | -1.0351 | 0.9626 | 1.13026 |
| B | final | 0.4588 | 0.7980 | 0.9102 | 11.191902 | -0.8937 | 1.1212 | 1.76024 |
| B | last5 | 0.4592 | 0.7979 | 0.9099 | 11.191993 | -0.8957 | 1.1178 | 1.75856 |

## Full Training-Set Fit

All training metrics below use eval mode, natural distribution and complete tail batches.
The +5 percentage-point fit flag is a screening criterion, not a release criterion.

| Model | Top1 | Delta vs R39A (pp) | CE | Mean cost | Actual regret (%) | Fit flag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| R39A reference | 0.5005 | +0.0000 | 1.05451 | 11.170912 | 0.8983 | False |
| A | 0.7412 | +24.0767 | 0.56708 | 11.110848 | 0.3780 | True |
| B | 0.7065 | +20.6017 | 0.65360 | 11.118722 | 0.4405 | True |

## Paired B Minus A

Best checkpoints use the original validation-cost rule. Terminal and last-five rows below use each run's own endpoint.
| Point | Top1 (pp) | Mean cost | vs_SBS (pp) | CE |
| --- | ---: | ---: | ---: | ---: |
| best | +0.1833 | +0.000623 | +0.0041 | +0.01772 |
| final | +0.8722 | -0.004637 | -0.0410 | -0.46569 |
| last5 | +0.8567 | -0.004849 | -0.0456 | -0.42101 |

## Explicit Early Stop and Matched Budget

A was stopped by the user after checkpoint epoch 56 (151200 saved successful updates). B completed 60 epochs.
Unsaved updates after the A checkpoint are excluded. This is not a completed 60/60 pair.
Architecture comparisons at equal budget use epoch 56 and its preceding five validation points.

| Group | Point | Epoch | Top1 | Mean cost | vs_SBS (%) | Actual regret (%) | CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | matched_final | 56 | 0.4501 | 11.196538 | -0.8527 | 1.1742 | 2.22593 |
| A | matched_last5 | 56 | 0.4507 | 11.196843 | -0.8501 | 1.1756 | 2.17957 |
| B | matched_final | 56 | 0.4595 | 11.191455 | -0.8970 | 1.1161 | 1.74921 |
| B | matched_last5 | 56 | 0.4597 | 11.191918 | -0.8948 | 1.1198 | 1.73456 |

### Full Train at Matched Epoch 55

| Group | Top1 | CE | Mean cost | Actual regret (%) |
| --- | ---: | ---: | ---: | ---: |
| A | 0.7410 | 0.56964 | 11.110969 | 0.3778 |
| B | 0.7036 | 0.65897 | 11.119475 | 0.4434 |

## dual_stream_seed2: Best Validation Epoch 13

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4793 | 0.8139 | 0.9222 | 11.173208 | 11.341290 (-1.0392%) | 11.058917 (+0.9251%) |
| TSP | 0.3410 | 0.5840 | 0.7470 | 7.920992 | 7.957803 (-0.4626%) | 7.869479 (+0.6546%) |
| CVRP | 0.4670 | 0.6680 | 0.8090 | 18.212615 | 18.294333 (-0.4467%) | 18.109962 (+0.5668%) |
| ATSP | 0.7700 | 0.9650 | 0.9930 | 1.635230 | 1.673524 (-2.2882%) | 1.629583 (+0.3465%) |
| OVRP | 0.3520 | 0.6580 | 0.8430 | 7.455382 | 7.461660 (-0.0841%) | 7.403457 (+0.7014%) |
| VRPB | 0.5390 | 0.9170 | 0.9910 | 9.333966 | 9.340269 (-0.0675%) | 9.268251 (+0.7090%) |
| VRPL | 0.2900 | 0.5450 | 0.7160 | 12.411251 | 12.407213 (+0.0325%) | 12.344029 (+0.5446%) |
| VRPTW | 0.4270 | 0.7380 | 0.9110 | 17.910530 | 18.657081 (-4.0014%) | 17.660252 (+1.4172%) |
| OVRPTW | 0.5240 | 0.9330 | 0.9990 | 10.036884 | 10.048435 (-0.1149%) | 9.958517 (+0.7869%) |
| OVRPB | 0.5910 | 0.9680 | 0.9850 | 6.316222 | 6.321981 (-0.0911%) | 6.260687 (+0.8870%) |
| OVRPL | 0.3870 | 0.6800 | 0.8600 | 7.431471 | 7.437562 (-0.0819%) | 7.379566 (+0.7034%) |
| VRPBL | 0.5410 | 0.9150 | 0.9900 | 9.356229 | 9.359268 (-0.0325%) | 9.284586 (+0.7716%) |
| VRPBTW | 0.4560 | 0.7960 | 0.9260 | 19.040633 | 19.669740 (-3.1984%) | 18.744386 (+1.5805%) |
| VRPLTW | 0.4500 | 0.7910 | 0.9260 | 17.642229 | 18.509149 (-4.6837%) | 17.419424 (+1.2791%) |
| OVRPBL | 0.5720 | 0.9550 | 0.9850 | 6.303877 | 6.310260 (-0.1012%) | 6.245926 (+0.9278%) |
| OVRPBTW | 0.4900 | 0.9130 | 0.9990 | 10.502212 | 10.506777 (-0.0435%) | 10.379805 (+1.1793%) |
| OVRPLTW | 0.5020 | 0.9270 | 1.0000 | 10.125646 | 10.127205 (-0.0154%) | 10.039835 (+0.8547%) |
| VRPBLTW | 0.4230 | 0.7770 | 0.9200 | 18.955168 | 19.519778 (-2.8925%) | 18.651532 (+1.6279%) |
| OVRPBLTW | 0.5050 | 0.9200 | 0.9990 | 10.527204 | 10.541174 (-0.1325%) | 10.411234 (+1.1139%) |

| Problem | Selected solver distribution |
| --- | --- |
| TSP | BQ: 55.0%; DIFUSCO: 0.0%; DIFUSCO500: 5.4%; ELG: 23.7%; LEHD: 0.1%; OMNI: 0.0%; T2T: 0.5%; T2T500: 15.3% |
| CVRP | BQ: 7.4%; ELG: 0.0%; ICAM: 15.3%; LEHD: 4.5%; MVMOE: 0.0%; MoSES_CaDA: 2.5%; MoSES_RF: 10.2%; OMNI: 40.9%; RELD_CVRP: 19.2%; RouteFinder: 0.0% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 29.1%; MATNET: 11.6%; MATPOENET: 0.0%; UNICO_MatPOENet: 59.3% |
| OVRP | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 42.4%; MoSES_RF: 55.2%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 2.4% |
| VRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 51.1%; RELD_MTL: 48.9%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 34.2%; MoSES_RF: 40.2%; RELD_MOEL: 0.0%; RELD_MTL: 23.9%; RouteFinder: 1.7% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 19.7%; MoSES_RF: 18.1%; RELD_MOEL: 5.1%; RELD_MTL: 50.5%; RouteFinder: 6.6% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 40.0%; RELD_MTL: 60.0%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 46.1%; RELD_MTL: 53.9%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 55.8%; MoSES_RF: 43.3%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 0.9% |
| VRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 49.7%; RELD_MTL: 50.3%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 2.8%; MoSES_RF: 9.4%; RELD_MOEL: 4.8%; RELD_MTL: 72.9%; RouteFinder: 10.1% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 18.6%; MoSES_RF: 11.7%; RELD_MOEL: 7.3%; RELD_MTL: 51.3%; RouteFinder: 11.1% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 36.7%; RELD_MTL: 63.3%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 29.2%; RELD_MTL: 70.8%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 37.4%; RELD_MTL: 62.6%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 3.6%; MoSES_RF: 0.8%; RELD_MOEL: 4.7%; RELD_MTL: 74.1%; RouteFinder: 16.8% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 30.6%; RELD_MTL: 69.4%; RouteFinder: 0.0% |

## direct_seed2: Best Validation Epoch 24

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4811 | 0.8168 | 0.9235 | 11.173831 | 11.341290 (-1.0351%) | 11.058917 (+0.9294%) |
| TSP | 0.3720 | 0.6060 | 0.7610 | 7.917078 | 7.957803 (-0.5118%) | 7.869479 (+0.6049%) |
| CVRP | 0.4830 | 0.6860 | 0.8200 | 18.211300 | 18.294333 (-0.4539%) | 18.109962 (+0.5596%) |
| ATSP | 0.7820 | 0.9680 | 0.9940 | 1.634879 | 1.673524 (-2.3092%) | 1.629583 (+0.3250%) |
| OVRP | 0.3670 | 0.6570 | 0.8460 | 7.455398 | 7.461660 (-0.0839%) | 7.403457 (+0.7016%) |
| VRPB | 0.5390 | 0.9180 | 0.9940 | 9.327996 | 9.340269 (-0.1314%) | 9.268251 (+0.6446%) |
| VRPL | 0.3140 | 0.5570 | 0.7290 | 12.404990 | 12.407213 (-0.0179%) | 12.344029 (+0.4939%) |
| VRPTW | 0.4340 | 0.7440 | 0.9140 | 17.922192 | 18.657081 (-3.9389%) | 17.660252 (+1.4832%) |
| OVRPTW | 0.4770 | 0.9310 | 1.0000 | 10.046159 | 10.048435 (-0.0227%) | 9.958517 (+0.8801%) |
| OVRPB | 0.5800 | 0.9670 | 0.9860 | 6.317548 | 6.321981 (-0.0701%) | 6.260687 (+0.9082%) |
| OVRPL | 0.3910 | 0.6810 | 0.8520 | 7.430885 | 7.437562 (-0.0898%) | 7.379566 (+0.6954%) |
| VRPBL | 0.5160 | 0.9090 | 0.9910 | 9.360230 | 9.359268 (+0.0103%) | 9.284586 (+0.8147%) |
| VRPBTW | 0.4510 | 0.7980 | 0.9200 | 19.053043 | 19.669740 (-3.1353%) | 18.744386 (+1.6467%) |
| VRPLTW | 0.4470 | 0.7900 | 0.9190 | 17.651101 | 18.509149 (-4.6358%) | 17.419424 (+1.3300%) |
| OVRPBL | 0.5570 | 0.9520 | 0.9840 | 6.306907 | 6.310260 (-0.0531%) | 6.245926 (+0.9763%) |
| OVRPBTW | 0.4980 | 0.9110 | 0.9990 | 10.505681 | 10.506777 (-0.0104%) | 10.379805 (+1.2127%) |
| OVRPLTW | 0.5100 | 0.9270 | 0.9990 | 10.123455 | 10.127205 (-0.0370%) | 10.039835 (+0.8329%) |
| VRPBLTW | 0.4310 | 0.7800 | 0.9160 | 18.933484 | 19.519778 (-3.0036%) | 18.651532 (+1.5117%) |
| OVRPBLTW | 0.5110 | 0.9200 | 0.9990 | 10.526629 | 10.541174 (-0.1380%) | 10.411234 (+1.1084%) |

| Problem | Selected solver distribution |
| --- | --- |
| TSP | BQ: 56.0%; DIFUSCO: 0.0%; DIFUSCO500: 5.0%; ELG: 14.2%; LEHD: 0.2%; OMNI: 0.0%; T2T: 1.6%; T2T500: 23.0% |
| CVRP | BQ: 9.2%; ELG: 1.1%; ICAM: 8.7%; LEHD: 11.9%; MVMOE: 0.3%; MoSES_CaDA: 3.0%; MoSES_RF: 6.3%; OMNI: 39.7%; RELD_CVRP: 19.8%; RouteFinder: 0.0% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 27.3%; MATNET: 17.1%; MATPOENET: 0.0%; UNICO_MatPOENet: 55.6% |
| OVRP | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 46.1%; MoSES_RF: 41.6%; RELD_MOEL: 0.3%; RELD_MTL: 0.4%; RouteFinder: 11.6% |
| VRPB | MTPOMO: 0.0%; MVMOE: 0.1%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 44.4%; RELD_MTL: 55.5%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 42.3%; MoSES_RF: 41.4%; RELD_MOEL: 2.6%; RELD_MTL: 7.0%; RouteFinder: 6.7% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 22.2%; MoSES_RF: 15.0%; RELD_MOEL: 11.6%; RELD_MTL: 44.2%; RouteFinder: 7.0% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.1%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 28.6%; RELD_MTL: 71.3%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 49.0%; RELD_MTL: 51.0%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 50.9%; MoSES_RF: 36.9%; RELD_MOEL: 0.3%; RELD_MTL: 0.5%; RouteFinder: 11.4% |
| VRPBL | MTPOMO: 0.1%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 40.5%; RELD_MTL: 59.4%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 0.2%; MoSES_CaDA: 3.2%; MoSES_RF: 6.6%; RELD_MOEL: 8.3%; RELD_MTL: 71.5%; RouteFinder: 10.2% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 17.7%; MoSES_RF: 7.3%; RELD_MOEL: 11.3%; RELD_MTL: 51.7%; RouteFinder: 12.0% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 45.2%; RELD_MTL: 54.8%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 34.0%; RELD_MTL: 66.0%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 29.0%; RELD_MTL: 71.0%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 4.0%; MoSES_RF: 1.3%; RELD_MOEL: 8.8%; RELD_MTL: 73.0%; RouteFinder: 12.9% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.1%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 23.6%; RELD_MTL: 76.3%; RouteFinder: 0.0% |
