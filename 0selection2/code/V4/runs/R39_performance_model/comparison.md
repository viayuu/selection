# R39 Performance Prediction

Completed runs: 2/2. All numbers below are validation, not test.
A: winner-cost classification decision. B: CE + centered-cost MSE, performance decision.
Classification Top1 is never combined with the performance policy's cost.
The ALL percentage is a mean of per-problem percentages, not a ratio of aggregate costs.

## Main Comparison

| Group | Point | Seeds | Top1 | Mean cost | vs_SBS (%) | Actual regret (%) | Classification Top1 | Classification CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | best | 1 | 0.4843 | 11.170767 | -1.0554 | 0.9451 | 0.4843 | 1.10238 |
| A | final | 1 | 0.4843 | 11.171171 | -1.0496 | 0.9510 | 0.4843 | 1.10238 |
| A | last5 | 1 | 0.4833 | 11.171554 | -1.0479 | 0.9540 | 0.4833 | 1.10213 |
| B | best | 1 | 0.4671 | 11.172470 | -1.0320 | 0.9695 | 0.4854 | 1.09185 |
| B | final | 1 | 0.4661 | 11.172786 | -1.0279 | 0.9751 | 0.4857 | 1.08969 |
| B | last5 | 1 | 0.4661 | 11.173017 | -1.0269 | 0.9770 | 0.4861 | 1.08966 |

## Paired B Minus A

| Point | Top1 (pp) | Mean cost | vs_SBS (pp) | Actual regret (pp) |
| --- | ---: | ---: | ---: | ---: |
| best | -1.7167 | +0.001702 | +0.0233 | +0.0244 |
| final | -1.8167 | +0.001615 | +0.0217 | +0.0241 |
| last5 | -1.7278 | +0.001463 | +0.0210 | +0.0230 |

1 paired seed(s), from-scratch training. Paired differences are in paired_differences.csv.
This is a single-seed method-screening comparison, not evidence of robustness across training seeds.

## Family Summary: Best Validation Checkpoints

| Group | Family | Seeds | Main Top1 | Mean cost | vs_SBS (%) | Classification Top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | TSP | 1 | 0.3750 | 7.916323 | -0.5212 | 0.3750 |
| A | CVRP | 1 | 0.4770 | 18.208151 | -0.4711 | 0.4770 |
| A | ATSP | 1 | 0.7780 | 1.635017 | -2.3010 | 0.7780 |
| A | MVRP | 1 | 0.4725 | 11.554288 | -1.0469 | 0.4725 |
| B | TSP | 1 | 0.3450 | 7.913790 | -0.5531 | 0.3770 |
| B | CVRP | 1 | 0.4460 | 18.212296 | -0.4484 | 0.4870 |
| B | ATSP | 1 | 0.7690 | 1.635556 | -2.2687 | 0.7820 |
| B | MVRP | 1 | 0.4565 | 11.556187 | -1.0204 | 0.4727 |

## performance_seed2: Best Validation Epoch 47

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4671 | 0.8040 | 0.9148 | 11.172470 | 11.341290 (-1.0320%) | 11.058917 (+0.9318%) |
| TSP | 0.3450 | 0.5550 | 0.6920 | 7.913790 | 7.957803 (-0.5531%) | 7.869479 (+0.5631%) |
| CVRP | 0.4460 | 0.6120 | 0.7320 | 18.212296 | 18.294333 (-0.4484%) | 18.109962 (+0.5651%) |
| ATSP | 0.7690 | 0.9660 | 0.9900 | 1.635556 | 1.673524 (-2.2687%) | 1.629583 (+0.3665%) |
| OVRP | 0.3110 | 0.6220 | 0.8380 | 7.461931 | 7.461660 (+0.0036%) | 7.403457 (+0.7898%) |
| VRPB | 0.5100 | 0.9170 | 0.9940 | 9.339432 | 9.340269 (-0.0090%) | 9.268251 (+0.7680%) |
| VRPL | 0.2830 | 0.5320 | 0.7210 | 12.405768 | 12.407213 (-0.0116%) | 12.344029 (+0.5002%) |
| VRPTW | 0.4510 | 0.7530 | 0.9160 | 17.885252 | 18.657081 (-4.1369%) | 17.660252 (+1.2741%) |
| OVRPTW | 0.4880 | 0.9240 | 0.9980 | 10.043930 | 10.048435 (-0.0448%) | 9.958517 (+0.8577%) |
| OVRPB | 0.5840 | 0.9690 | 0.9850 | 6.316309 | 6.321981 (-0.0897%) | 6.260687 (+0.8884%) |
| OVRPL | 0.3380 | 0.6340 | 0.8500 | 7.437821 | 7.437562 (+0.0035%) | 7.379566 (+0.7894%) |
| VRPBL | 0.5230 | 0.9060 | 0.9900 | 9.359349 | 9.359268 (+0.0009%) | 9.284586 (+0.8052%) |
| VRPBTW | 0.4330 | 0.7930 | 0.9280 | 19.038195 | 19.669740 (-3.2107%) | 18.744386 (+1.5675%) |
| VRPLTW | 0.4450 | 0.7980 | 0.9360 | 17.638137 | 18.509149 (-4.7058%) | 17.419424 (+1.2556%) |
| OVRPBL | 0.5570 | 0.9550 | 0.9860 | 6.306617 | 6.310260 (-0.0577%) | 6.245926 (+0.9717%) |
| OVRPBTW | 0.4830 | 0.9100 | 0.9970 | 10.501095 | 10.506777 (-0.0541%) | 10.379805 (+1.1685%) |
| OVRPLTW | 0.4970 | 0.9250 | 0.9980 | 10.128852 | 10.127205 (+0.0163%) | 10.039835 (+0.8866%) |
| VRPBLTW | 0.4410 | 0.7800 | 0.9180 | 18.946917 | 19.519778 (-2.9348%) | 18.651532 (+1.5837%) |
| OVRPBLTW | 0.5040 | 0.9210 | 0.9980 | 10.533205 | 10.541174 (-0.0756%) | 10.411234 (+1.1715%) |

| Problem | Selected arm distribution |
| --- | --- |
| TSP | BQ: 40.4%; DIFUSCO: 0.0%; DIFUSCO500: 9.2%; ELG: 26.1%; LEHD: 0.0%; OMNI: 0.0%; T2T: 3.5%; T2T500: 20.8% |
| CVRP | BQ: 0.7%; ELG: 0.0%; ICAM: 9.6%; LEHD: 2.8%; MVMOE: 2.7%; MoSES_CaDA: 2.6%; MoSES_RF: 6.9%; OMNI: 43.0%; RELD_CVRP: 25.7%; RouteFinder: 6.0% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 35.0%; MATNET: 9.9%; MATPOENET: 0.0%; UNICO_MatPOENet: 55.1% |
| OVRP | MTPOMO: 0.2%; MVMOE: 0.0%; MoSES_CaDA: 33.5%; MoSES_RF: 40.9%; RELD_MOEL: 0.2%; RELD_MTL: 1.2%; RouteFinder: 24.0% |
| VRPB | MTPOMO: 0.2%; MVMOE: 0.4%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 32.7%; RELD_MTL: 66.7%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 29.0%; MoSES_RF: 42.0%; RELD_MOEL: 1.5%; RELD_MTL: 4.5%; RouteFinder: 23.0% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 1.0%; MoSES_CaDA: 21.3%; MoSES_RF: 18.4%; RELD_MOEL: 16.5%; RELD_MTL: 36.4%; RouteFinder: 6.4% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.3%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 43.2%; RELD_MTL: 56.5%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 34.7%; RELD_MTL: 65.3%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.2%; MVMOE: 0.0%; MoSES_CaDA: 35.6%; MoSES_RF: 41.2%; RELD_MOEL: 0.4%; RELD_MTL: 0.6%; RouteFinder: 22.0% |
| VRPBL | MTPOMO: 0.0%; MVMOE: 0.3%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 28.5%; RELD_MTL: 71.2%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 0.8%; MoSES_CaDA: 2.4%; MoSES_RF: 6.5%; RELD_MOEL: 27.2%; RELD_MTL: 51.0%; RouteFinder: 12.1% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 3.0%; MoSES_CaDA: 20.0%; MoSES_RF: 12.8%; RELD_MOEL: 24.4%; RELD_MTL: 29.1%; RouteFinder: 10.7% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 37.3%; RELD_MTL: 62.7%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.1%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 32.0%; RELD_MTL: 67.9%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 32.9%; RELD_MTL: 67.1%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 0.3%; MoSES_CaDA: 2.9%; MoSES_RF: 1.9%; RELD_MOEL: 25.2%; RELD_MTL: 55.7%; RouteFinder: 14.0% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 28.4%; RELD_MTL: 71.6%; RouteFinder: 0.0% |

## winner_cost_seed2: Best Validation Epoch 53

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4843 | 0.8198 | 0.9274 | 11.170767 | 11.341290 (-1.0554%) | 11.058917 (+0.9084%) |
| TSP | 0.3750 | 0.6030 | 0.7600 | 7.916323 | 7.957803 (-0.5212%) | 7.869479 (+0.5953%) |
| CVRP | 0.4770 | 0.6890 | 0.8320 | 18.208151 | 18.294333 (-0.4711%) | 18.109962 (+0.5422%) |
| ATSP | 0.7780 | 0.9700 | 0.9930 | 1.635017 | 1.673524 (-2.3010%) | 1.629583 (+0.3334%) |
| OVRP | 0.3540 | 0.6420 | 0.8430 | 7.453958 | 7.461660 (-0.1032%) | 7.403457 (+0.6821%) |
| VRPB | 0.5520 | 0.9190 | 0.9960 | 9.331662 | 9.340269 (-0.0921%) | 9.268251 (+0.6842%) |
| VRPL | 0.3190 | 0.5730 | 0.7190 | 12.403579 | 12.407213 (-0.0293%) | 12.344029 (+0.4824%) |
| VRPTW | 0.4540 | 0.7530 | 0.9240 | 17.908853 | 18.657081 (-4.0104%) | 17.660252 (+1.4077%) |
| OVRPTW | 0.5050 | 0.9330 | 1.0000 | 10.041513 | 10.048435 (-0.0689%) | 9.958517 (+0.8334%) |
| OVRPB | 0.5760 | 0.9680 | 0.9860 | 6.316737 | 6.321981 (-0.0829%) | 6.260687 (+0.8953%) |
| OVRPL | 0.3930 | 0.6840 | 0.8610 | 7.431928 | 7.437562 (-0.0757%) | 7.379566 (+0.7096%) |
| VRPBL | 0.5220 | 0.9130 | 0.9890 | 9.358460 | 9.359268 (-0.0086%) | 9.284586 (+0.7957%) |
| VRPBTW | 0.4300 | 0.8030 | 0.9300 | 19.037025 | 19.669740 (-3.2167%) | 18.744386 (+1.5612%) |
| VRPLTW | 0.4530 | 0.8090 | 0.9380 | 17.625535 | 18.509149 (-4.7739%) | 17.419424 (+1.1832%) |
| OVRPBL | 0.5890 | 0.9540 | 0.9830 | 6.303944 | 6.310260 (-0.1001%) | 6.245926 (+0.9289%) |
| OVRPBTW | 0.5090 | 0.9130 | 0.9990 | 10.497551 | 10.506777 (-0.0878%) | 10.379805 (+1.1344%) |
| OVRPLTW | 0.5070 | 0.9270 | 1.0000 | 10.123481 | 10.127205 (-0.0368%) | 10.039835 (+0.8331%) |
| VRPBLTW | 0.4240 | 0.7840 | 0.9410 | 18.948331 | 19.519778 (-2.9275%) | 18.651532 (+1.5913%) |
| OVRPBLTW | 0.5000 | 0.9200 | 1.0000 | 10.531762 | 10.541174 (-0.0893%) | 10.411234 (+1.1577%) |

| Problem | Selected arm distribution |
| --- | --- |
| TSP | BQ: 57.5%; DIFUSCO: 0.0%; DIFUSCO500: 3.5%; ELG: 14.8%; LEHD: 0.2%; OMNI: 0.0%; T2T: 2.1%; T2T500: 21.9% |
| CVRP | BQ: 3.8%; ELG: 0.0%; ICAM: 11.0%; LEHD: 6.0%; MVMOE: 0.0%; MoSES_CaDA: 0.3%; MoSES_RF: 10.9%; OMNI: 39.4%; RELD_CVRP: 28.6%; RouteFinder: 0.0% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 30.1%; MATNET: 14.4%; MATPOENET: 0.0%; UNICO_MatPOENet: 55.5% |
| OVRP | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 40.0%; MoSES_RF: 45.2%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 14.8% |
| VRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 56.7%; RELD_MTL: 43.3%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 45.0%; MoSES_RF: 37.4%; RELD_MOEL: 6.4%; RELD_MTL: 8.6%; RouteFinder: 2.6% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 24.3%; MoSES_RF: 14.1%; RELD_MOEL: 22.8%; RELD_MTL: 30.7%; RouteFinder: 8.1% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 60.6%; RELD_MTL: 39.4%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 36.3%; RELD_MTL: 63.7%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 49.2%; MoSES_RF: 36.0%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 14.8% |
| VRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 55.9%; RELD_MTL: 44.1%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 3.6%; MoSES_RF: 4.8%; RELD_MOEL: 22.1%; RELD_MTL: 55.2%; RouteFinder: 14.3% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 23.2%; MoSES_RF: 9.0%; RELD_MOEL: 26.2%; RELD_MTL: 28.2%; RouteFinder: 13.4% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 30.2%; RELD_MTL: 69.8%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 52.1%; RELD_MTL: 47.9%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 62.1%; RELD_MTL: 37.9%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 3.3%; MoSES_RF: 0.7%; RELD_MOEL: 27.1%; RELD_MTL: 52.1%; RouteFinder: 16.8% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 51.1%; RELD_MTL: 48.9%; RouteFinder: 0.0% |
