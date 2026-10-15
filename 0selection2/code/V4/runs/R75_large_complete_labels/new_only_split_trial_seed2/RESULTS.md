# New-Only Split Trial

Seed: 2. 77904 instances per task, 1402272 in total.

Each task has TRAIN=70114, VALIDATION=3895, TEST=3895. The first 10000 historical instances are excluded.

Selection uses exact input scale, generation distribution and capacity type. Winners are only opened after the indices are locked; they do not influence sampling. There was no split-seed search, solver execution, model inference or training.

TV (%) is half the sum of absolute winner-frequency differences. These are descriptive marginal similarities, not proof of identical joint distributions.

## Summary

Across all 18 problems, the winner marginals are close under this single fixed input-stratified split.

| Comparison | Mean TV % | Maximum TV % |
|---|---:|---:|
| Train / Validation | 1.07 | 2.14 |
| Train / Test | 1.20 | 2.11 |
| Validation / Test | 1.53 | 2.77 |

The largest individual solver-frequency difference across any two splits is 2.54 percentage points.

Aggregate sizes: TRAIN=1262052, VALIDATION=70110, TEST=70110.

## Per-Problem Similarity

| Problem | Train/Val TV % | Train/Test TV % | Val/Test TV % | Max Method Gap pp |
|---|---:|---:|---:|---:|
| TSP | 0.87 | 2.07 | 1.72 | 1.92 |
| CVRP | 1.48 | 1.76 | 2.31 | 0.87 |
| ATSP | 0.93 | 0.47 | 0.46 | 0.91 |
| OVRP | 1.64 | 2.11 | 2.59 | 2.03 |
| VRPB | 1.47 | 0.98 | 2.18 | 1.59 |
| VRPL | 1.87 | 1.75 | 2.21 | 1.67 |
| VRPTW | 0.89 | 1.38 | 0.56 | 1.22 |
| OVRPTW | 0.33 | 0.88 | 1.10 | 0.92 |
| OVRPB | 0.66 | 1.49 | 1.87 | 1.69 |
| OVRPL | 2.14 | 0.89 | 2.13 | 2.00 |
| VRPBL | 0.53 | 1.08 | 1.57 | 1.31 |
| VRPBTW | 1.71 | 1.09 | 2.77 | 2.54 |
| VRPLTW | 0.91 | 1.36 | 1.69 | 1.67 |
| OVRPBL | 0.48 | 1.11 | 1.51 | 1.10 |
| OVRPBTW | 1.55 | 1.34 | 0.44 | 1.55 |
| OVRPLTW | 0.95 | 0.57 | 0.95 | 0.93 |
| VRPBLTW | 0.60 | 0.60 | 0.69 | 0.69 |
| OVRPBLTW | 0.31 | 0.74 | 0.69 | 0.71 |

## TSP

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| BQ | 29.48 | 29.76 | 29.45 |
| DIFUSCO | 8.68 | 8.47 | 7.78 |
| DIFUSCO500 | 1.23 | 1.13 | 1.08 |
| ELG | 7.39 | 7.60 | 7.24 |
| LEHD | 16.95 | 16.46 | 16.35 |
| OMNI | 0.58 | 0.54 | 0.72 |
| T2T | 32.14 | 32.53 | 34.07 |
| T2T500 | 3.55 | 3.52 | 3.31 |

## CVRP

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| BQ | 12.83 | 12.48 | 13.02 |
| ELG | 2.04 | 1.82 | 2.16 |
| ICAM | 18.77 | 18.25 | 18.05 |
| LEHD | 14.43 | 14.33 | 13.56 |
| MVMOE | 2.15 | 2.57 | 2.44 |
| MoSES_CaDA | 2.61 | 2.44 | 3.18 |
| MoSES_RF | 3.08 | 3.08 | 3.16 |
| OMNI | 24.35 | 24.24 | 24.85 |
| RELD_CVRP | 18.08 | 18.64 | 18.10 |
| RouteFinder | 1.66 | 2.16 | 1.49 |

## ATSP

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| GLOP | 0.30 | 0.28 | 0.28 |
| ICAM_ATSP | 30.44 | 29.53 | 29.99 |
| MATNET | 14.29 | 14.69 | 14.35 |
| MATPOENET | 7.96 | 8.16 | 8.09 |
| UNICO_MatPOENet | 47.01 | 47.34 | 47.29 |

## OVRP

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.84 | 0.67 | 0.98 |
| MVMOE | 0.44 | 0.33 | 0.39 |
| MoSES_CaDA | 32.33 | 32.32 | 33.79 |
| MoSES_RF | 31.84 | 33.30 | 31.27 |
| RELD_MOEL | 5.75 | 5.49 | 6.26 |
| RELD_MTL | 7.65 | 7.83 | 7.32 |
| RouteFinder | 21.16 | 20.05 | 20.00 |

## VRPB

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 1.81 | 1.93 | 2.36 |
| MVMOE | 5.62 | 5.65 | 5.80 |
| MoSES_CaDA | 0.36 | 0.31 | 0.28 |
| MoSES_RF | 0.38 | 0.31 | 0.23 |
| RELD_MOEL | 40.53 | 41.51 | 39.97 |
| RELD_MTL | 50.97 | 49.63 | 51.22 |
| RouteFinder | 0.32 | 0.67 | 0.13 |

## VRPL

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 2.72 | 3.03 | 2.93 |
| MVMOE | 2.46 | 2.44 | 2.64 |
| MoSES_CaDA | 26.47 | 26.11 | 25.67 |
| MoSES_RF | 27.40 | 28.96 | 27.86 |
| RELD_MOEL | 11.66 | 11.22 | 11.55 |
| RELD_MTL | 14.56 | 13.79 | 15.46 |
| RouteFinder | 14.74 | 14.45 | 13.89 |

## VRPTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 9.39 | 9.24 | 8.81 |
| MoSES_CaDA | 0.30 | 0.36 | 0.46 |
| MoSES_RF | 0.04 | 0.00 | 0.05 |
| RELD_MOEL | 40.57 | 39.92 | 39.79 |
| RELD_MTL | 49.59 | 50.42 | 50.81 |
| RouteFinder | 0.10 | 0.05 | 0.08 |

## OVRPTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 7.89 | 7.96 | 7.03 |
| MoSES_CaDA | 0.10 | 0.21 | 0.08 |
| MoSES_RF | 0.02 | 0.08 | 0.03 |
| RELD_MOEL | 44.82 | 44.49 | 45.06 |
| RELD_MTL | 47.14 | 47.24 | 47.75 |
| RouteFinder | 0.04 | 0.03 | 0.05 |

## OVRPB

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 1.93 | 2.21 | 2.11 |
| MVMOE | 1.85 | 1.62 | 1.80 |
| MoSES_CaDA | 0.25 | 0.26 | 0.21 |
| MoSES_RF | 0.30 | 0.26 | 0.21 |
| RELD_MOEL | 41.60 | 41.98 | 40.36 |
| RELD_MTL | 53.81 | 53.43 | 55.12 |
| RouteFinder | 0.26 | 0.26 | 0.21 |

## OVRPL

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.78 | 1.00 | 1.18 |
| MVMOE | 0.48 | 0.62 | 0.51 |
| MoSES_CaDA | 33.27 | 31.27 | 32.94 |
| MoSES_RF | 32.18 | 33.58 | 32.20 |
| RELD_MOEL | 5.67 | 5.93 | 6.11 |
| RELD_MTL | 7.39 | 7.24 | 7.34 |
| RouteFinder | 20.23 | 20.36 | 19.72 |

## VRPBL

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 1.80 | 1.77 | 1.57 |
| MVMOE | 6.15 | 6.37 | 6.11 |
| MoSES_CaDA | 0.40 | 0.39 | 0.49 |
| MoSES_RF | 0.51 | 0.46 | 0.54 |
| RELD_MOEL | 40.00 | 39.56 | 40.87 |
| RELD_MTL | 50.78 | 51.07 | 49.96 |
| RouteFinder | 0.37 | 0.39 | 0.46 |

## VRPBTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.02 | 0.03 | 0.00 |
| MVMOE | 10.55 | 10.96 | 10.47 |
| MoSES_CaDA | 0.75 | 0.62 | 0.82 |
| MoSES_RF | 0.07 | 0.08 | 0.10 |
| RELD_MOEL | 39.04 | 40.33 | 38.07 |
| RELD_MTL | 49.33 | 47.78 | 50.32 |
| RouteFinder | 0.23 | 0.21 | 0.21 |

## VRPLTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 9.22 | 10.12 | 9.65 |
| MoSES_CaDA | 0.38 | 0.39 | 0.36 |
| MoSES_RF | 0.06 | 0.03 | 0.05 |
| RELD_MOEL | 40.62 | 40.49 | 39.36 |
| RELD_MTL | 49.60 | 48.86 | 50.53 |
| RouteFinder | 0.13 | 0.13 | 0.05 |

## OVRPBL

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 1.84 | 1.57 | 2.26 |
| MVMOE | 1.86 | 2.16 | 1.75 |
| MoSES_CaDA | 0.27 | 0.15 | 0.33 |
| MoSES_RF | 0.30 | 0.28 | 0.31 |
| RELD_MOEL | 41.49 | 41.54 | 42.11 |
| RELD_MTL | 53.99 | 54.12 | 53.02 |
| RouteFinder | 0.25 | 0.18 | 0.23 |

## OVRPBTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 8.61 | 8.16 | 8.01 |
| MoSES_CaDA | 0.32 | 0.23 | 0.28 |
| MoSES_RF | 0.06 | 0.05 | 0.08 |
| RELD_MOEL | 42.87 | 44.42 | 44.18 |
| RELD_MTL | 48.10 | 47.09 | 47.45 |
| RouteFinder | 0.05 | 0.05 | 0.00 |

## OVRPLTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 7.93 | 8.06 | 8.50 |
| MoSES_CaDA | 0.13 | 0.15 | 0.10 |
| MoSES_RF | 0.03 | 0.03 | 0.00 |
| RELD_MOEL | 44.62 | 43.70 | 44.18 |
| RELD_MTL | 47.24 | 48.04 | 47.16 |
| RouteFinder | 0.05 | 0.03 | 0.05 |

## VRPBLTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.02 | 0.00 | 0.00 |
| MVMOE | 10.60 | 11.19 | 10.78 |
| MoSES_CaDA | 0.76 | 0.77 | 0.74 |
| MoSES_RF | 0.13 | 0.13 | 0.03 |
| RELD_MOEL | 38.86 | 38.59 | 39.28 |
| RELD_MTL | 49.38 | 49.09 | 48.93 |
| RouteFinder | 0.25 | 0.23 | 0.23 |

## OVRPBLTW

| Solver | Train % | Validation % | Test % |
|---|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 |
| MVMOE | 8.64 | 8.34 | 8.34 |
| MoSES_CaDA | 0.36 | 0.49 | 0.39 |
| MoSES_RF | 0.04 | 0.03 | 0.05 |
| RELD_MOEL | 42.99 | 43.03 | 43.70 |
| RELD_MTL | 47.92 | 48.06 | 47.50 |
| RouteFinder | 0.05 | 0.05 | 0.03 |

## Boundaries

This trial preserves the available new-label scenario; it does not repair solver constraint semantics. It cannot reconstruct missing large-scale TSP/CVRP tail instances. Any future model using these heldout splits must be trained from scratch, not resumed from a model that saw these instances.
