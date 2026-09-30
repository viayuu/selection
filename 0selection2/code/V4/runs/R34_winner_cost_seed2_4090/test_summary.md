# R34_winner_cost: test

Validation-selected checkpoint: epoch 50.
Native winner labels; ALL percentages are means of per-problem percentages.
SBS is the best fixed solver on this evaluated split (a hindsight comparator). Oracle is pool VBS, not the proven routing optimum.

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4734** | **0.8128** | **0.9220** | **11.1729** | **—** | **11.3339 (-0.998%)** | **11.0567 (+0.951%)** |
| TSP | 0.355 | 0.594 | 0.733 | 8.0467 | — | 8.0829 (-0.448%) | 7.9946 (+0.650%) |
| CVRP | 0.449 | 0.689 | 0.808 | 18.2537 | — | 18.3323 (-0.429%) | 18.1586 (+0.524%) |
| ATSP | 0.757 | 0.956 | 0.993 | 1.6328 | — | 1.6697 (-2.208%) | 1.6246 (+0.508%) |
| OVRP | 0.346 | 0.651 | 0.832 | 7.4669 | — | 7.4695 (-0.035%) | 7.4087 (+0.785%) |
| VRPB | 0.517 | 0.917 | 0.987 | 9.2848 | — | 9.2951 (-0.110%) | 9.2178 (+0.727%) |
| VRPL | 0.292 | 0.530 | 0.699 | 12.3575 | — | 12.3607 (-0.025%) | 12.2950 (+0.508%) |
| VRPTW | 0.424 | 0.755 | 0.922 | 17.6677 | — | 18.4357 (-4.166%) | 17.4240 (+1.399%) |
| OVRPTW | 0.487 | 0.908 | 0.999 | 10.1166 | — | 10.1233 (-0.066%) | 10.0336 (+0.827%) |
| OVRPB | 0.552 | 0.964 | 0.993 | 6.2927 | — | 6.2963 (-0.058%) | 6.2329 (+0.959%) |
| OVRPL | 0.369 | 0.666 | 0.860 | 7.4452 | — | 7.4537 (-0.114%) | 7.3937 (+0.697%) |
| VRPBL | 0.550 | 0.939 | 0.991 | 9.2471 | — | 9.2492 (-0.022%) | 9.1773 (+0.761%) |
| VRPBTW | 0.439 | 0.783 | 0.937 | 18.9996 | — | 19.5263 (-2.698%) | 18.6905 (+1.653%) |
| VRPLTW | 0.457 | 0.760 | 0.924 | 17.6243 | — | 18.4269 (-4.356%) | 17.3725 (+1.449%) |
| OVRPBL | 0.583 | 0.971 | 0.990 | 6.2991 | — | 6.3068 (-0.123%) | 6.2418 (+0.918%) |
| OVRPBTW | 0.508 | 0.904 | 0.998 | 10.6074 | — | 10.6090 (-0.015%) | 10.4782 (+1.233%) |
| OVRPLTW | 0.490 | 0.921 | 1.000 | 10.1027 | — | 10.1036 (-0.009%) | 10.0172 (+0.853%) |
| VRPBLTW | 0.441 | 0.797 | 0.931 | 19.0800 | — | 19.6770 (-3.034%) | 18.7886 (+1.551%) |
| OVRPBLTW | 0.506 | 0.925 | 0.999 | 10.5871 | — | 10.5928 (-0.054%) | 10.4710 (+1.108%) |

## Arm Distribution

| Problem | Selection frequencies |
| --- | --- |
| TSP | BQ: 59.60%; DIFUSCO: 0.10%; DIFUSCO500: 5.50%; ELG: 13.20%; LEHD: 0.00%; OMNI: 0.00%; T2T: 0.80%; T2T500: 20.80% |
| CVRP | BQ: 3.60%; ELG: 0.00%; ICAM: 12.90%; LEHD: 6.90%; MVMOE: 0.90%; MoSES_CaDA: 5.20%; MoSES_RF: 8.30%; OMNI: 35.50%; RELD_CVRP: 26.70%; RouteFinder: 0.00% |
| ATSP | GLOP: 0.00%; ICAM_ATSP: 26.50%; MATNET: 16.80%; MATPOENET: 0.00%; UNICO_MatPOENet: 56.70% |
| OVRP | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 54.00%; MoSES_RF: 40.40%; RELD_MOEL: 0.00%; RELD_MTL: 0.00%; RouteFinder: 5.60% |
| VRPB | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 59.00%; RELD_MTL: 41.00%; RouteFinder: 0.00% |
| VRPL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 55.40%; MoSES_RF: 32.60%; RELD_MOEL: 1.60%; RELD_MTL: 9.10%; RouteFinder: 1.30% |
| VRPTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 25.50%; MoSES_RF: 16.20%; RELD_MOEL: 25.80%; RELD_MTL: 27.30%; RouteFinder: 5.20% |
| OVRPTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 60.00%; RELD_MTL: 40.00%; RouteFinder: 0.00% |
| OVRPB | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 41.30%; RELD_MTL: 58.70%; RouteFinder: 0.00% |
| OVRPL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 59.40%; MoSES_RF: 33.30%; RELD_MOEL: 0.00%; RELD_MTL: 0.00%; RouteFinder: 7.30% |
| VRPBL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 58.10%; RELD_MTL: 41.90%; RouteFinder: 0.00% |
| VRPBTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 2.30%; MoSES_RF: 8.60%; RELD_MOEL: 19.30%; RELD_MTL: 59.10%; RouteFinder: 10.70% |
| VRPLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 25.60%; MoSES_RF: 9.90%; RELD_MOEL: 22.40%; RELD_MTL: 32.90%; RouteFinder: 9.20% |
| OVRPBL | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 34.40%; RELD_MTL: 65.60%; RouteFinder: 0.00% |
| OVRPBTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 47.00%; RELD_MTL: 53.00%; RouteFinder: 0.00% |
| OVRPLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 53.40%; RELD_MTL: 46.60%; RouteFinder: 0.00% |
| VRPBLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 4.00%; MoSES_RF: 1.80%; RELD_MOEL: 15.60%; RELD_MTL: 60.80%; RouteFinder: 17.80% |
| OVRPBLTW | MTPOMO: 0.00%; MVMOE: 0.00%; MoSES_CaDA: 0.00%; MoSES_RF: 0.00%; RELD_MOEL: 39.20%; RELD_MTL: 60.80%; RouteFinder: 0.00% |
