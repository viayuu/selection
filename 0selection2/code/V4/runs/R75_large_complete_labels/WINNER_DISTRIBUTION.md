# R75 Native Winner Distribution

All values are the percentage of instances whose original native winner is the named solver. No new TEST inference or solver execution was performed. TEST columns reuse the saved previously completed original full TEST report. Added TRAIN excludes the first10000 rows.

TV is half the sum of absolute solver-frequency differences; it describes marginal distribution shift, not an accuracy ceiling or evidence of its cause.

| Problem | Big TRAIN vs VAL TV % | Big TRAIN vs TEST TV % | Old TRAIN vs VAL TV % | VAL vs TEST TV % |
|---|---:|---:|---:|---:|
| TSP | 23.14 | 23.37 | 3.35 | 5.00 |
| CVRP | 8.87 | 8.29 | 4.13 | 5.50 |
| ATSP | 1.06 | 1.50 | 1.15 | 2.30 |
| OVRP | 1.80 | 2.39 | 2.32 | 2.90 |
| VRPB | 1.69 | 3.57 | 1.19 | 2.80 |
| VRPL | 4.08 | 3.92 | 4.10 | 6.10 |
| VRPTW | 41.20 | 40.50 | 3.31 | 4.20 |
| OVRPTW | 1.34 | 2.15 | 0.95 | 2.90 |
| OVRPB | 1.26 | 2.21 | 0.78 | 2.20 |
| OVRPL | 1.55 | 2.50 | 1.94 | 3.90 |
| VRPBL | 2.73 | 3.10 | 2.74 | 2.50 |
| VRPBTW | 21.63 | 21.33 | 1.82 | 3.90 |
| VRPLTW | 39.07 | 38.67 | 2.42 | 3.60 |
| OVRPBL | 2.07 | 1.55 | 2.00 | 3.10 |
| OVRPBTW | 2.10 | 2.19 | 1.59 | 0.90 |
| OVRPLTW | 2.09 | 1.89 | 1.76 | 0.60 |
| VRPBLTW | 21.09 | 22.57 | 2.35 | 3.20 |
| OVRPBLTW | 1.96 | 1.70 | 1.96 | 0.70 |

## TSP

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| BQ | 30.77 | 29.49 | 29.64 | 29.70 | 28.00 |
| DIFUSCO | 5.21 | 8.62 | 8.23 | 4.80 | 5.20 |
| DIFUSCO500 | 12.83 | 1.22 | 2.54 | 11.60 | 13.20 |
| ELG | 7.24 | 7.40 | 7.38 | 6.60 | 7.80 |
| LEHD | 16.23 | 16.89 | 16.82 | 17.20 | 18.50 |
| OMNI | 0.45 | 0.58 | 0.57 | 2.50 | 0.30 |
| T2T | 10.90 | 32.26 | 29.83 | 10.90 | 11.40 |
| T2T500 | 16.37 | 3.54 | 5.00 | 16.70 | 15.60 |

## CVRP

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| BQ | 12.97 | 12.87 | 12.88 | 12.40 | 12.20 |
| ELG | 1.87 | 2.02 | 2.00 | 1.90 | 1.50 |
| ICAM | 11.60 | 18.72 | 18.00 | 10.20 | 11.30 |
| LEHD | 16.08 | 14.36 | 14.53 | 15.00 | 14.70 |
| MVMOE | 1.50 | 2.18 | 2.11 | 1.80 | 2.60 |
| MoSES_CaDA | 3.14 | 2.62 | 2.67 | 2.50 | 2.60 |
| MoSES_RF | 3.49 | 3.09 | 3.13 | 4.00 | 3.90 |
| OMNI | 25.33 | 24.34 | 24.44 | 28.10 | 24.40 |
| RELD_CVRP | 22.34 | 18.11 | 18.54 | 21.90 | 25.40 |
| RouteFinder | 1.68 | 1.70 | 1.70 | 2.20 | 1.40 |

## ATSP

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| GLOP | 0.21 | 0.28 | 0.28 | 0.30 | 0.50 |
| ICAM_ATSP | 30.59 | 30.42 | 30.44 | 30.20 | 30.10 |
| MATNET | 14.07 | 14.31 | 14.29 | 13.90 | 14.80 |
| MATPOENET | 8.19 | 8.02 | 8.04 | 7.60 | 8.80 |
| UNICO_MatPOENet | 46.94 | 46.97 | 46.96 | 48.00 | 45.80 |

## OVRP

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.83 | 0.82 | 0.82 | 0.80 | 1.40 |
| MVMOE | 0.45 | 0.43 | 0.43 | 0.50 | 0.40 |
| MoSES_CaDA | 32.11 | 32.38 | 32.36 | 32.60 | 31.90 |
| MoSES_RF | 32.40 | 31.93 | 31.98 | 30.20 | 31.70 |
| RELD_MOEL | 5.37 | 5.79 | 5.75 | 6.10 | 6.90 |
| RELD_MTL | 7.45 | 7.65 | 7.63 | 8.50 | 8.30 |
| RouteFinder | 21.39 | 20.99 | 21.03 | 21.30 | 19.40 |

## VRPB

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 1.87 | 1.82 | 1.82 | 1.70 | 2.20 |
| MVMOE | 5.71 | 5.67 | 5.67 | 6.60 | 5.90 |
| MoSES_CaDA | 0.05 | 0.35 | 0.32 | 0.00 | 0.00 |
| MoSES_RF | 0.05 | 0.38 | 0.35 | 0.00 | 0.10 |
| RELD_MOEL | 41.00 | 40.48 | 40.54 | 41.30 | 43.50 |
| RELD_MTL | 51.31 | 50.97 | 51.00 | 50.40 | 48.30 |
| RouteFinder | 0.01 | 0.33 | 0.29 | 0.00 | 0.00 |

## VRPL

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 3.09 | 2.78 | 2.81 | 2.80 | 3.30 |
| MVMOE | 2.44 | 2.46 | 2.46 | 2.20 | 2.60 |
| MoSES_CaDA | 26.58 | 26.50 | 26.51 | 28.30 | 25.20 |
| MoSES_RF | 27.47 | 27.55 | 27.54 | 27.60 | 25.60 |
| RELD_MOEL | 11.18 | 11.58 | 11.54 | 10.50 | 13.60 |
| RELD_MTL | 14.59 | 14.46 | 14.47 | 11.70 | 13.80 |
| RouteFinder | 14.65 | 14.67 | 14.67 | 16.90 | 15.90 |

## VRPTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 6.29 | 9.34 | 9.04 | 7.10 | 6.60 |
| MoSES_CaDA | 15.72 | 0.31 | 1.85 | 14.30 | 15.40 |
| MoSES_RF | 16.81 | 0.04 | 1.72 | 16.90 | 18.80 |
| RELD_MOEL | 22.44 | 40.43 | 38.63 | 23.10 | 24.10 |
| RELD_MTL | 25.49 | 49.76 | 47.34 | 23.60 | 23.80 |
| RouteFinder | 13.25 | 0.11 | 1.43 | 15.00 | 11.30 |

## OVRPTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 7.50 | 7.92 | 7.88 | 6.60 | 9.10 |
| MoSES_CaDA | 0.04 | 0.10 | 0.09 | 0.10 | 0.00 |
| MoSES_RF | 0.02 | 0.03 | 0.03 | 0.00 | 0.00 |
| RELD_MOEL | 45.50 | 44.86 | 44.93 | 45.70 | 42.90 |
| RELD_MTL | 46.91 | 47.06 | 47.04 | 47.60 | 47.90 |
| RouteFinder | 0.03 | 0.04 | 0.04 | 0.00 | 0.10 |

## OVRPB

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 1.64 | 1.95 | 1.92 | 1.40 | 2.00 |
| MVMOE | 1.69 | 1.82 | 1.80 | 1.50 | 1.40 |
| MoSES_CaDA | 0.12 | 0.24 | 0.23 | 0.10 | 0.00 |
| MoSES_RF | 0.07 | 0.28 | 0.26 | 0.20 | 0.10 |
| RELD_MOEL | 42.52 | 41.57 | 41.67 | 42.30 | 40.40 |
| RELD_MTL | 53.85 | 53.87 | 53.87 | 54.50 | 56.00 |
| RouteFinder | 0.11 | 0.26 | 0.25 | 0.00 | 0.10 |

## OVRPL

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.96 | 0.82 | 0.83 | 0.80 | 0.40 |
| MVMOE | 0.56 | 0.48 | 0.49 | 0.70 | 0.60 |
| MoSES_CaDA | 33.16 | 33.09 | 33.10 | 33.30 | 33.10 |
| MoSES_RF | 32.29 | 32.25 | 32.26 | 33.40 | 30.70 |
| RELD_MOEL | 5.05 | 5.67 | 5.61 | 5.60 | 5.10 |
| RELD_MTL | 7.42 | 7.44 | 7.44 | 6.80 | 8.10 |
| RouteFinder | 20.56 | 20.24 | 20.27 | 19.40 | 22.00 |

## VRPBL

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 2.06 | 1.79 | 1.82 | 2.10 | 1.30 |
| MVMOE | 6.57 | 6.16 | 6.20 | 6.50 | 4.80 |
| MoSES_CaDA | 0.03 | 0.40 | 0.37 | 0.00 | 0.00 |
| MoSES_RF | 0.05 | 0.52 | 0.48 | 0.00 | 0.00 |
| RELD_MOEL | 41.16 | 40.05 | 40.16 | 38.60 | 40.20 |
| RELD_MTL | 50.10 | 50.71 | 50.64 | 52.80 | 53.70 |
| RouteFinder | 0.03 | 0.37 | 0.33 | 0.00 | 0.00 |

## VRPBTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.01 | 0.02 | 0.02 | 0.00 | 0.00 |
| MVMOE | 8.96 | 10.56 | 10.40 | 7.70 | 9.90 |
| MoSES_CaDA | 4.20 | 0.72 | 1.07 | 4.70 | 4.40 |
| MoSES_RF | 8.35 | 0.07 | 0.90 | 8.00 | 9.70 |
| RELD_MOEL | 30.30 | 39.02 | 38.15 | 30.10 | 29.70 |
| RELD_MTL | 36.25 | 49.38 | 48.06 | 37.20 | 35.70 |
| RouteFinder | 11.93 | 0.23 | 1.40 | 12.30 | 10.60 |

## VRPLTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 6.92 | 9.28 | 9.05 | 6.30 | 7.40 |
| MoSES_CaDA | 16.09 | 0.37 | 1.95 | 17.30 | 15.80 |
| MoSES_RF | 13.33 | 0.05 | 1.38 | 11.80 | 12.30 |
| RELD_MOEL | 22.79 | 40.58 | 38.80 | 24.00 | 21.90 |
| RELD_MTL | 25.82 | 49.59 | 47.22 | 25.70 | 27.10 |
| RouteFinder | 15.04 | 0.12 | 1.61 | 14.90 | 15.50 |

## OVRPBL

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 2.05 | 1.83 | 1.85 | 2.40 | 0.90 |
| MVMOE | 1.71 | 1.87 | 1.85 | 1.90 | 1.60 |
| MoSES_CaDA | 0.09 | 0.27 | 0.25 | 0.20 | 0.00 |
| MoSES_RF | 0.12 | 0.29 | 0.27 | 0.00 | 0.20 |
| RELD_MOEL | 41.79 | 41.50 | 41.53 | 40.00 | 42.70 |
| RELD_MTL | 54.15 | 54.01 | 54.03 | 55.50 | 54.40 |
| RouteFinder | 0.09 | 0.23 | 0.22 | 0.00 | 0.20 |

## OVRPBTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 8.82 | 8.59 | 8.61 | 8.60 | 9.50 |
| MoSES_CaDA | 0.03 | 0.31 | 0.28 | 0.00 | 0.00 |
| MoSES_RF | 0.05 | 0.06 | 0.06 | 0.00 | 0.00 |
| RELD_MOEL | 43.61 | 43.09 | 43.15 | 45.20 | 44.40 |
| RELD_MTL | 47.36 | 47.90 | 47.85 | 46.10 | 46.00 |
| RouteFinder | 0.13 | 0.04 | 0.05 | 0.10 | 0.10 |

## OVRPLTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 7.66 | 8.01 | 7.97 | 7.20 | 7.80 |
| MoSES_CaDA | 0.12 | 0.12 | 0.12 | 0.10 | 0.10 |
| MoSES_RF | 0.00 | 0.03 | 0.03 | 0.00 | 0.00 |
| RELD_MOEL | 44.58 | 44.53 | 44.53 | 43.30 | 42.90 |
| RELD_MTL | 47.64 | 47.27 | 47.31 | 49.40 | 49.20 |
| RouteFinder | 0.00 | 0.05 | 0.04 | 0.00 | 0.00 |

## VRPBLTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.03 | 0.01 | 0.01 | 0.00 | 0.10 |
| MVMOE | 9.02 | 10.60 | 10.44 | 9.20 | 9.20 |
| MoSES_CaDA | 4.73 | 0.75 | 1.15 | 5.60 | 4.70 |
| MoSES_RF | 6.25 | 0.12 | 0.73 | 5.40 | 8.00 |
| RELD_MOEL | 30.20 | 38.94 | 38.07 | 31.50 | 29.50 |
| RELD_MTL | 35.74 | 49.32 | 47.97 | 34.70 | 35.20 |
| RouteFinder | 14.03 | 0.25 | 1.63 | 13.60 | 13.30 |

## OVRPBLTW

| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |
|---|---:|---:|---:|---:|---:|
| MTPOMO | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MVMOE | 8.38 | 8.62 | 8.60 | 7.90 | 7.50 |
| MoSES_CaDA | 0.04 | 0.37 | 0.34 | 0.00 | 0.00 |
| MoSES_RF | 0.03 | 0.04 | 0.04 | 0.10 | 0.00 |
| RELD_MOEL | 43.01 | 43.00 | 43.00 | 44.90 | 44.70 |
| RELD_MTL | 48.48 | 47.93 | 47.98 | 47.10 | 47.80 |
| RouteFinder | 0.06 | 0.04 | 0.05 | 0.00 | 0.00 |

Retained-first10000 label frequency counts match the old TRAIN report for 18/18 tasks. This is a count comparison, not a per-instance provenance audit.
