# R34_winner_cost: val

Validation-selected checkpoint: epoch 50.
Native winner labels; ALL percentages are means of per-problem percentages.
SBS is the best fixed solver on this evaluated split (a hindsight comparator). Oracle is pool VBS, not the proven routing optimum.

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4847** | **0.8173** | **0.9256** | **11.1708** | **—** | **11.3413 (-1.054%)** | **11.0589 (+0.909%)** |
| TSP | 0.363 | 0.595 | 0.746 | 7.9180 | — | 7.9578 (-0.501%) | 7.8695 (+0.616%) |
| CVRP | 0.475 | 0.671 | 0.822 | 18.2045 | — | 18.2943 (-0.491%) | 18.1100 (+0.522%) |
| ATSP | 0.775 | 0.970 | 0.994 | 1.6354 | — | 1.6735 (-2.275%) | 1.6296 (+0.360%) |
| OVRP | 0.356 | 0.639 | 0.839 | 7.4558 | — | 7.4617 (-0.078%) | 7.4035 (+0.707%) |
| VRPB | 0.564 | 0.918 | 0.994 | 9.3297 | — | 9.3403 (-0.113%) | 9.2683 (+0.663%) |
| VRPL | 0.331 | 0.563 | 0.727 | 12.4007 | — | 12.4072 (-0.053%) | 12.3440 (+0.459%) |
| VRPTW | 0.468 | 0.767 | 0.924 | 17.8891 | — | 18.6571 (-4.116%) | 17.6603 (+1.296%) |
| OVRPTW | 0.496 | 0.933 | 1.000 | 10.0427 | — | 10.0484 (-0.057%) | 9.9585 (+0.845%) |
| OVRPB | 0.580 | 0.968 | 0.986 | 6.3152 | — | 6.3220 (-0.107%) | 6.2607 (+0.871%) |
| OVRPL | 0.386 | 0.690 | 0.860 | 7.4313 | — | 7.4376 (-0.083%) | 7.3796 (+0.702%) |
| VRPBL | 0.532 | 0.914 | 0.992 | 9.3578 | — | 9.3593 (-0.015%) | 9.2846 (+0.789%) |
| VRPBTW | 0.437 | 0.790 | 0.925 | 19.0527 | — | 19.6697 (-3.137%) | 18.7444 (+1.645%) |
| VRPLTW | 0.453 | 0.803 | 0.933 | 17.6347 | — | 18.5091 (-4.725%) | 17.4194 (+1.236%) |
| OVRPBL | 0.584 | 0.954 | 0.989 | 6.3032 | — | 6.3103 (-0.111%) | 6.2459 (+0.918%) |
| OVRPBTW | 0.497 | 0.913 | 0.998 | 10.4990 | — | 10.5068 (-0.074%) | 10.3798 (+1.148%) |
| OVRPLTW | 0.495 | 0.927 | 0.999 | 10.1251 | — | 10.1272 (-0.021%) | 10.0398 (+0.849%) |
| VRPBLTW | 0.429 | 0.777 | 0.933 | 18.9480 | — | 19.5198 (-2.929%) | 18.6515 (+1.590%) |
| OVRPBLTW | 0.503 | 0.920 | 0.999 | 10.5315 | — | 10.5412 (-0.092%) | 10.4112 (+1.155%) |

## Arm Distribution

| Problem | Selection frequencies |
| --- | --- |
| TSP | BQ: 54.80%; DIFUSCO: 0.00%; DIFUSCO500: 5.30%; ELG: 15.80%; LEHD: 0.10%; OMNI: 0.00%; T2T: 1.20%; T2T500: 22.80% |
| CVRP | BQ: 3.10%; ELG: 0.00%; ICAM: 13.70%; LEHD: 6.30%; MVMOE: 1.10%; MoSES_CaDA: 5.30%; MoSES_RF: 7.90%; OMNI: 37.80%; RELD_CVRP: 24.80%; RouteFinder: 0.00% |
| ATSP | GLOP: 0.00%; ICAM_ATSP: 27.60%; MATNET: 16.30%; MATPOENET: 0.00%; UNICO_MatPOENet: 56.10% |
| OVRP | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 53.00%; MoSES_RF: 41.70%; RELD_MOEL: 0.00%; RELD_MTL: 0.00%; RouteFinder: 5.30% |
| VRPB | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 58.20%; RELD_MTL: 41.80%; RouteFinder: 0.00% |
| VRPL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 55.90%; MoSES_RF: 31.20%; RELD_MOEL: 1.80%; RELD_MTL: 9.80%; RouteFinder: 1.30% |
| VRPTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 24.90%; MoSES_RF: 16.30%; RELD_MOEL: 24.90%; RELD_MTL: 27.30%; RouteFinder: 6.60% |
| OVRPTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 58.40%; RELD_MTL: 41.60%; RouteFinder: 0.00% |
| OVRPB | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 40.00%; RELD_MTL: 60.00%; RouteFinder: 0.00% |
| OVRPL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 58.60%; MoSES_RF: 33.60%; RELD_MOEL: 0.00%; RELD_MTL: 0.00%; RouteFinder: 7.80% |
| VRPBL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 57.50%; RELD_MTL: 42.50%; RouteFinder: 0.00% |
| VRPBTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 2.10%; MoSES_RF: 7.20%; RELD_MOEL: 20.70%; RELD_MTL: 58.80%; RouteFinder: 11.20% |
| VRPLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 25.40%; MoSES_RF: 10.70%; RELD_MOEL: 22.70%; RELD_MTL: 32.00%; RouteFinder: 9.20% |
| OVRPBL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 29.70%; RELD_MTL: 70.30%; RouteFinder: 0.00% |
| OVRPBTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 47.70%; RELD_MTL: 52.30%; RouteFinder: 0.00% |
| OVRPLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 52.00%; RELD_MTL: 48.00%; RouteFinder: 0.00% |
| VRPBLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 3.70%; MoSES_RF: 2.10%; RELD_MOEL: 15.30%; RELD_MTL: 64.40%; RouteFinder: 14.50% |
| OVRPBLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 41.60%; RELD_MTL: 58.40%; RouteFinder: 0.00% |
