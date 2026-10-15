# R77 Validation-Only Overview

Generated: 2026-10-14T20:11:48.804228+00:00

Saved VAL metrics/scores only. No TEST input, model inference, fitting, checkpoint or protocol change.

| Arm | Successful update | Top1 % | Actual regret % | vs common FULL TRAIN SBS % | CE | Zero-pick supported classes |
|---|---:|---:|---:|---:|---:|---:|
| A | 49680 | 69.4808 | 0.539679 | -0.067623 | 0.853683 | 16 |
| B | 49680 | 70.8415 | 0.410186 | -0.184136 | 0.708262 | 26 |

Latest shared update: 49680. Use the matched JSON section for A/B comparisons.

Zero-pick means positive VAL winner support but no predictions. Near-zero <=0.1% selection and rare VAL support <1% are descriptive only. Nonzero selection is not necessarily correct or beneficial.

## Class Changes From First VAL

### A

At update49680: 16/85 supported classes have zero predictions; 13/19 rare-support classes have zero predictions.

Recovered selection (not a correctness claim): ATSP/MATNET, CVRP/ELG, CVRP/ICAM, CVRP/LEHD, CVRP/MVMOE, CVRP/MoSES_CaDA, CVRP/MoSES_RF, CVRP/OMNI, CVRP/RouteFinder, OVRP/MoSES_RF, OVRP/RouteFinder, OVRP/reld_mt, OVRPB/MTPOMO, OVRPB/MVMOE, OVRPBL/MTPOMO, OVRPBL/MVMOE, OVRPBLTW/MVMOE, OVRPBTW/MVMOE, OVRPL/RouteFinder, OVRPL/reld_mt, OVRPLTW/MVMOE, OVRPLTW/MoSES_CaDA, OVRPTW/MVMOE, OVRPTW/MoSES_CaDA, TSP/BQ, TSP/DIFUSCO, TSP/DIFUSCO500, TSP/ELG, TSP/LEHD, TSP/OMNI, TSP/T2T500, VRPB/MTPOMO, VRPB/MVMOE, VRPBL/MTPOMO, VRPBL/MVMOE, VRPBLTW/MVMOE, VRPBTW/MVMOE, VRPL/MVMOE, VRPL/RouteFinder, VRPL/reld_mt, VRPLTW/MVMOE, VRPLTW/MoSES_CaDA, VRPLTW/MoSES_RF, VRPLTW/RouteFinder, VRPTW/MVMOE, VRPTW/MoSES_CaDA, VRPTW/RouteFinder

Newly zero: 

### B

At update49680: 26/85 supported classes have zero predictions; 18/19 rare-support classes have zero predictions.

Recovered selection (not a correctness claim): ATSP/MATNET, ATSP/MATPOENET, CVRP/BQ, CVRP/ELG, CVRP/ICAM, CVRP/LEHD, CVRP/MVMOE, CVRP/MoSES_CaDA, CVRP/MoSES_RF, CVRP/OMNI, OVRP/RouteFinder, OVRP/reld_mt, OVRPB/MTPOMO, OVRPB/MVMOE, OVRPBL/MTPOMO, OVRPBL/MVMOE, OVRPBLTW/MVMOE, OVRPBTW/MVMOE, OVRPL/RouteFinder, OVRPL/reld_mt, OVRPLTW/MVMOE, OVRPLTW/MoSES_CaDA, OVRPTW/MVMOE, TSP/BQ, TSP/ELG, TSP/T2T500, VRPB/MTPOMO, VRPB/MVMOE, VRPBL/MTPOMO, VRPBL/MVMOE, VRPBLTW/MVMOE, VRPBTW/MVMOE, VRPL/RouteFinder, VRPL/reld_mt, VRPLTW/MVMOE, VRPTW/MVMOE, VRPTW/MoSES_CaDA

Newly zero: 

## Continuity

A continued this R77 run's own saved update2484 checkpoint, not historical weights or a Git commit; model/optimizer/scaler/RNG/shuffle restored; no architecture/optimization formula change

Architecture/objective/optimizer/LR/batch/clipping/R-Drop/init/task/budget/common-SBS configs are equal. The GRES correction did not edit the shared R76 objective, R43 loss or model factory. Geometry mean/std and TRAIN populations are the predeclared arm-specific differences.

Cold-class diagnostics never feed checkpoint choice, LR, budget or TEST. No Git commits were made.
