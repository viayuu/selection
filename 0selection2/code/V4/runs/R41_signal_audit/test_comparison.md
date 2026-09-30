# R41 frozen test comparison

All four checkpoints were selected before R41 using validation cost. No training or test-based reselection.
Native `ind` is the Top1 target; original raw costs are aggregated in FP64. No LIB datasets in this round.
ALL percentages are the mean of per-problem percentages, not ratios of ALL average costs.

| Model | Locked epoch | Top1 | Top2 | Top3 | Mean cost | vs SBS | vs Oracle | Actual regret | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R34 | 50 | 0.4734 | 0.8128 | 0.9220 | 11.172881 | -0.9982% | +0.9508% | 0.9819% | 1.1128 |
| R39A | 53 | 0.4712 | 0.8141 | 0.9245 | 11.173855 | -0.9936% | +0.9555% | 0.9866% | 1.1113 |
| R40A | 13 | 0.4684 | 0.8082 | 0.9208 | 11.172058 | -0.9967% | +0.9517% | 0.9808% | 1.1206 |
| R40B | 24 | 0.4729 | 0.8080 | 0.9197 | 11.175296 | -0.9813% | +0.9677% | 0.9970% | 1.1394 |

## R34

Checkpoint: `code/V4/runs/R34_winner_cost_seed2_4090/best.pt`; epoch 50.

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4734 | 0.8128 | 0.9220 | 11.172881 | 11.333921 (-0.9982%) | 11.056695 (+0.9508%) | 0.9819% |
| TSP | 0.3550 | 0.5940 | 0.7330 | 8.046653 | 8.082872 (-0.4481%) | 7.994650 (+0.6505%) | 0.6831% |
| CVRP | 0.4490 | 0.6890 | 0.8080 | 18.253697 | 18.332332 (-0.4289%) | 18.158602 (+0.5237%) | 0.6347% |
| ATSP | 0.7570 | 0.9560 | 0.9930 | 1.632819 | 1.669680 (-2.2076%) | 1.624559 (+0.5084%) | 0.5266% |
| OVRP | 0.3460 | 0.6510 | 0.8320 | 7.466884 | 7.469509 (-0.0351%) | 7.408692 (+0.7855%) | 0.8002% |
| VRPB | 0.5170 | 0.9170 | 0.9870 | 9.284798 | 9.295063 (-0.1104%) | 9.217786 (+0.7270%) | 0.7316% |
| VRPL | 0.2920 | 0.5300 | 0.6990 | 12.357540 | 12.360694 (-0.0255%) | 12.295043 (+0.5083%) | 0.5146% |
| VRPTW | 0.4240 | 0.7550 | 0.9220 | 17.667681 | 18.435676 (-4.1658%) | 17.423958 (+1.3988%) | 1.4612% |
| OVRPTW | 0.4870 | 0.9080 | 0.9990 | 10.116629 | 10.123299 (-0.0659%) | 10.033608 (+0.8274%) | 0.8401% |
| OVRPB | 0.5520 | 0.9640 | 0.9930 | 6.292654 | 6.296312 (-0.0581%) | 6.232885 (+0.9589%) | 0.9732% |
| OVRPL | 0.3690 | 0.6660 | 0.8600 | 7.445191 | 7.453650 (-0.1135%) | 7.393675 (+0.6968%) | 0.7032% |
| VRPBL | 0.5500 | 0.9390 | 0.9910 | 9.247149 | 9.249180 (-0.0220%) | 9.177293 (+0.7612%) | 0.7766% |
| VRPBTW | 0.4390 | 0.7830 | 0.9370 | 18.999568 | 19.526309 (-2.6976%) | 18.690529 (+1.6535%) | 1.6371% |
| VRPLTW | 0.4570 | 0.7600 | 0.9240 | 17.624261 | 18.426862 (-4.3556%) | 17.372466 (+1.4494%) | 1.5095% |
| OVRPBL | 0.5830 | 0.9710 | 0.9900 | 6.299073 | 6.306825 (-0.1229%) | 6.241775 (+0.9180%) | 0.9265% |
| OVRPBTW | 0.5080 | 0.9040 | 0.9980 | 10.607421 | 10.608986 (-0.0148%) | 10.478190 (+1.2333%) | 1.2853% |
| OVRPLTW | 0.4900 | 0.9210 | 1.0000 | 10.102700 | 10.103602 (-0.0089%) | 10.017218 (+0.8534%) | 0.8909% |
| VRPBLTW | 0.4410 | 0.7970 | 0.9310 | 19.080042 | 19.676957 (-3.0336%) | 18.788554 (+1.5514%) | 1.6290% |
| OVRPBLTW | 0.5060 | 0.9250 | 0.9990 | 10.587091 | 10.592773 (-0.0536%) | 10.471024 (+1.1085%) | 1.1511% |

### Solver pick distribution

| Problem | Solver | Pick fraction | Native winner fraction |
| --- | --- | ---: | ---: |
| TSP | BQ | 59.60% | 28.00% |
| TSP | DIFUSCO | 0.10% | 5.20% |
| TSP | DIFUSCO500 | 5.50% | 13.20% |
| TSP | ELG | 13.20% | 7.80% |
| TSP | LEHD | 0.00% | 18.50% |
| TSP | OMNI | 0.00% | 0.30% |
| TSP | T2T | 0.80% | 11.40% |
| TSP | T2T500 | 20.80% | 15.60% |
| CVRP | BQ | 3.60% | 12.20% |
| CVRP | ELG | 0.00% | 1.50% |
| CVRP | ICAM | 12.90% | 11.30% |
| CVRP | LEHD | 6.90% | 14.70% |
| CVRP | MVMOE | 0.90% | 2.60% |
| CVRP | MoSES_CaDA | 5.20% | 2.60% |
| CVRP | MoSES_RF | 8.30% | 3.90% |
| CVRP | OMNI | 35.50% | 24.40% |
| CVRP | RELD_CVRP | 26.70% | 25.40% |
| CVRP | RouteFinder | 0.00% | 1.40% |
| ATSP | GLOP | 0.00% | 0.50% |
| ATSP | ICAM_ATSP | 26.50% | 30.10% |
| ATSP | MATNET | 16.80% | 14.80% |
| ATSP | MATPOENET | 0.00% | 8.80% |
| ATSP | UNICO_MatPOENet | 56.70% | 45.80% |
| OVRP | MTPOMO | 0.00% | 1.40% |
| OVRP | MVMOE | 0.00% | 0.40% |
| OVRP | MoSES_CaDA | 54.00% | 31.90% |
| OVRP | MoSES_RF | 40.40% | 31.70% |
| OVRP | RELD_MOEL | 0.00% | 6.90% |
| OVRP | RELD_MTL | 0.00% | 8.30% |
| OVRP | RouteFinder | 5.60% | 19.40% |
| VRPB | MTPOMO | 0.00% | 2.20% |
| VRPB | MVMOE | 0.00% | 5.90% |
| VRPB | MoSES_CaDA | 0.00% | 0.00% |
| VRPB | MoSES_RF | 0.00% | 0.10% |
| VRPB | RELD_MOEL | 59.00% | 43.50% |
| VRPB | RELD_MTL | 41.00% | 48.30% |
| VRPB | RouteFinder | 0.00% | 0.00% |
| VRPL | MTPOMO | 0.00% | 3.30% |
| VRPL | MVMOE | 0.00% | 2.60% |
| VRPL | MoSES_CaDA | 55.40% | 25.20% |
| VRPL | MoSES_RF | 32.60% | 25.60% |
| VRPL | RELD_MOEL | 1.60% | 13.60% |
| VRPL | RELD_MTL | 9.10% | 13.80% |
| VRPL | RouteFinder | 1.30% | 15.90% |
| VRPTW | MTPOMO | 0.00% | 0.00% |
| VRPTW | MVMOE | 0.00% | 6.60% |
| VRPTW | MoSES_CaDA | 25.50% | 15.40% |
| VRPTW | MoSES_RF | 16.20% | 18.80% |
| VRPTW | RELD_MOEL | 25.80% | 24.10% |
| VRPTW | RELD_MTL | 27.30% | 23.80% |
| VRPTW | RouteFinder | 5.20% | 11.30% |
| OVRPTW | MTPOMO | 0.00% | 0.00% |
| OVRPTW | MVMOE | 0.00% | 9.10% |
| OVRPTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPTW | MoSES_RF | 0.00% | 0.00% |
| OVRPTW | RELD_MOEL | 60.00% | 42.90% |
| OVRPTW | RELD_MTL | 40.00% | 47.90% |
| OVRPTW | RouteFinder | 0.00% | 0.10% |
| OVRPB | MTPOMO | 0.00% | 2.00% |
| OVRPB | MVMOE | 0.00% | 1.40% |
| OVRPB | MoSES_CaDA | 0.00% | 0.00% |
| OVRPB | MoSES_RF | 0.00% | 0.10% |
| OVRPB | RELD_MOEL | 41.30% | 40.40% |
| OVRPB | RELD_MTL | 58.70% | 56.00% |
| OVRPB | RouteFinder | 0.00% | 0.10% |
| OVRPL | MTPOMO | 0.00% | 0.40% |
| OVRPL | MVMOE | 0.00% | 0.60% |
| OVRPL | MoSES_CaDA | 59.40% | 33.10% |
| OVRPL | MoSES_RF | 33.30% | 30.70% |
| OVRPL | RELD_MOEL | 0.00% | 5.10% |
| OVRPL | RELD_MTL | 0.00% | 8.10% |
| OVRPL | RouteFinder | 7.30% | 22.00% |
| VRPBL | MTPOMO | 0.00% | 1.30% |
| VRPBL | MVMOE | 0.00% | 4.80% |
| VRPBL | MoSES_CaDA | 0.00% | 0.00% |
| VRPBL | MoSES_RF | 0.00% | 0.00% |
| VRPBL | RELD_MOEL | 58.10% | 40.20% |
| VRPBL | RELD_MTL | 41.90% | 53.70% |
| VRPBL | RouteFinder | 0.00% | 0.00% |
| VRPBTW | MTPOMO | 0.00% | 0.00% |
| VRPBTW | MVMOE | 0.00% | 9.90% |
| VRPBTW | MoSES_CaDA | 2.30% | 4.40% |
| VRPBTW | MoSES_RF | 8.60% | 9.70% |
| VRPBTW | RELD_MOEL | 19.30% | 29.70% |
| VRPBTW | RELD_MTL | 59.10% | 35.70% |
| VRPBTW | RouteFinder | 10.70% | 10.60% |
| VRPLTW | MTPOMO | 0.00% | 0.00% |
| VRPLTW | MVMOE | 0.00% | 7.40% |
| VRPLTW | MoSES_CaDA | 25.60% | 15.80% |
| VRPLTW | MoSES_RF | 9.90% | 12.30% |
| VRPLTW | RELD_MOEL | 22.40% | 21.90% |
| VRPLTW | RELD_MTL | 32.90% | 27.10% |
| VRPLTW | RouteFinder | 9.20% | 15.50% |
| OVRPBL | MTPOMO | 0.00% | 0.90% |
| OVRPBL | MVMOE | 0.00% | 1.60% |
| OVRPBL | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBL | MoSES_RF | 0.00% | 0.20% |
| OVRPBL | RELD_MOEL | 34.40% | 42.70% |
| OVRPBL | RELD_MTL | 65.60% | 54.40% |
| OVRPBL | RouteFinder | 0.00% | 0.20% |
| OVRPBTW | MTPOMO | 0.00% | 0.00% |
| OVRPBTW | MVMOE | 0.00% | 9.50% |
| OVRPBTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBTW | RELD_MOEL | 47.00% | 44.40% |
| OVRPBTW | RELD_MTL | 53.00% | 46.00% |
| OVRPBTW | RouteFinder | 0.00% | 0.10% |
| OVRPLTW | MTPOMO | 0.00% | 0.00% |
| OVRPLTW | MVMOE | 0.00% | 7.80% |
| OVRPLTW | MoSES_CaDA | 0.00% | 0.10% |
| OVRPLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPLTW | RELD_MOEL | 53.40% | 42.90% |
| OVRPLTW | RELD_MTL | 46.60% | 49.20% |
| OVRPLTW | RouteFinder | 0.00% | 0.00% |
| VRPBLTW | MTPOMO | 0.00% | 0.10% |
| VRPBLTW | MVMOE | 0.00% | 9.20% |
| VRPBLTW | MoSES_CaDA | 4.00% | 4.70% |
| VRPBLTW | MoSES_RF | 1.80% | 8.00% |
| VRPBLTW | RELD_MOEL | 15.60% | 29.50% |
| VRPBLTW | RELD_MTL | 60.80% | 35.20% |
| VRPBLTW | RouteFinder | 17.80% | 13.30% |
| OVRPBLTW | MTPOMO | 0.00% | 0.00% |
| OVRPBLTW | MVMOE | 0.00% | 7.50% |
| OVRPBLTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBLTW | RELD_MOEL | 39.20% | 44.70% |
| OVRPBLTW | RELD_MTL | 60.80% | 47.80% |
| OVRPBLTW | RouteFinder | 0.00% | 0.00% |

## R39A

Checkpoint: `code/V4/runs/R39_performance_model/winner_cost_seed2/best.pt`; epoch 53.

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4712 | 0.8141 | 0.9245 | 11.173855 | 11.333921 (-0.9936%) | 11.056695 (+0.9555%) | 0.9866% |
| TSP | 0.3720 | 0.6060 | 0.7520 | 8.044285 | 8.082872 (-0.4774%) | 7.994650 (+0.6209%) | 0.6470% |
| CVRP | 0.4620 | 0.6930 | 0.8210 | 18.252950 | 18.332332 (-0.4330%) | 18.158602 (+0.5196%) | 0.6289% |
| ATSP | 0.7630 | 0.9620 | 0.9940 | 1.631219 | 1.669680 (-2.3035%) | 1.624559 (+0.4099%) | 0.4125% |
| OVRP | 0.3420 | 0.6350 | 0.8300 | 7.465689 | 7.469509 (-0.0511%) | 7.408692 (+0.7693%) | 0.7777% |
| VRPB | 0.5200 | 0.9170 | 0.9880 | 9.287785 | 9.295063 (-0.0783%) | 9.217786 (+0.7594%) | 0.7736% |
| VRPL | 0.2860 | 0.5280 | 0.7150 | 12.356667 | 12.360694 (-0.0326%) | 12.295043 (+0.5012%) | 0.5087% |
| VRPTW | 0.4320 | 0.7560 | 0.9250 | 17.669732 | 18.435676 (-4.1547%) | 17.423958 (+1.4106%) | 1.4640% |
| OVRPTW | 0.4720 | 0.9080 | 0.9990 | 10.117832 | 10.123299 (-0.0540%) | 10.033608 (+0.8394%) | 0.8540% |
| OVRPB | 0.5460 | 0.9640 | 0.9890 | 6.296385 | 6.296312 (+0.0012%) | 6.232885 (+1.0188%) | 1.0331% |
| OVRPL | 0.3640 | 0.6620 | 0.8600 | 7.445866 | 7.453650 (-0.1044%) | 7.393675 (+0.7059%) | 0.7127% |
| VRPBL | 0.5540 | 0.9380 | 0.9900 | 9.245328 | 9.249180 (-0.0416%) | 9.177293 (+0.7413%) | 0.7513% |
| VRPBTW | 0.4160 | 0.7810 | 0.9320 | 18.997982 | 19.526309 (-2.7057%) | 18.690529 (+1.6450%) | 1.6647% |
| VRPLTW | 0.4370 | 0.7850 | 0.9210 | 17.635882 | 18.426862 (-4.2925%) | 17.372466 (+1.5163%) | 1.5730% |
| OVRPBL | 0.5640 | 0.9710 | 0.9900 | 6.303353 | 6.306825 (-0.0550%) | 6.241775 (+0.9866%) | 1.0053% |
| OVRPBTW | 0.5140 | 0.9030 | 0.9980 | 10.607479 | 10.608986 (-0.0142%) | 10.478190 (+1.2339%) | 1.2710% |
| OVRPLTW | 0.4940 | 0.9210 | 0.9990 | 10.102181 | 10.103602 (-0.0141%) | 10.017218 (+0.8482%) | 0.8625% |
| VRPBLTW | 0.4470 | 0.7990 | 0.9390 | 19.080610 | 19.676957 (-3.0307%) | 18.788554 (+1.5544%) | 1.6557% |
| OVRPBLTW | 0.4970 | 0.9240 | 0.9990 | 10.588166 | 10.592773 (-0.0435%) | 10.471024 (+1.1187%) | 1.1629% |

### Solver pick distribution

| Problem | Solver | Pick fraction | Native winner fraction |
| --- | --- | ---: | ---: |
| TSP | BQ | 59.90% | 28.00% |
| TSP | DIFUSCO | 0.10% | 5.20% |
| TSP | DIFUSCO500 | 4.40% | 13.20% |
| TSP | ELG | 13.20% | 7.80% |
| TSP | LEHD | 0.30% | 18.50% |
| TSP | OMNI | 0.00% | 0.30% |
| TSP | T2T | 2.20% | 11.40% |
| TSP | T2T500 | 19.90% | 15.60% |
| CVRP | BQ | 3.10% | 12.20% |
| CVRP | ELG | 0.00% | 1.50% |
| CVRP | ICAM | 11.90% | 11.30% |
| CVRP | LEHD | 5.70% | 14.70% |
| CVRP | MVMOE | 0.00% | 2.60% |
| CVRP | MoSES_CaDA | 0.20% | 2.60% |
| CVRP | MoSES_RF | 10.30% | 3.90% |
| CVRP | OMNI | 36.30% | 24.40% |
| CVRP | RELD_CVRP | 32.50% | 25.40% |
| CVRP | RouteFinder | 0.00% | 1.40% |
| ATSP | GLOP | 0.00% | 0.50% |
| ATSP | ICAM_ATSP | 29.70% | 30.10% |
| ATSP | MATNET | 13.90% | 14.80% |
| ATSP | MATPOENET | 0.00% | 8.80% |
| ATSP | UNICO_MatPOENet | 56.40% | 45.80% |
| OVRP | MTPOMO | 0.00% | 1.40% |
| OVRP | MVMOE | 0.00% | 0.40% |
| OVRP | MoSES_CaDA | 39.90% | 31.90% |
| OVRP | MoSES_RF | 44.80% | 31.70% |
| OVRP | RELD_MOEL | 0.00% | 6.90% |
| OVRP | RELD_MTL | 0.20% | 8.30% |
| OVRP | RouteFinder | 15.10% | 19.40% |
| VRPB | MTPOMO | 0.00% | 2.20% |
| VRPB | MVMOE | 0.00% | 5.90% |
| VRPB | MoSES_CaDA | 0.00% | 0.00% |
| VRPB | MoSES_RF | 0.00% | 0.10% |
| VRPB | RELD_MOEL | 57.80% | 43.50% |
| VRPB | RELD_MTL | 42.20% | 48.30% |
| VRPB | RouteFinder | 0.00% | 0.00% |
| VRPL | MTPOMO | 0.00% | 3.30% |
| VRPL | MVMOE | 0.00% | 2.60% |
| VRPL | MoSES_CaDA | 44.60% | 25.20% |
| VRPL | MoSES_RF | 39.50% | 25.60% |
| VRPL | RELD_MOEL | 5.30% | 13.60% |
| VRPL | RELD_MTL | 8.30% | 13.80% |
| VRPL | RouteFinder | 2.30% | 15.90% |
| VRPTW | MTPOMO | 0.00% | 0.00% |
| VRPTW | MVMOE | 0.00% | 6.60% |
| VRPTW | MoSES_CaDA | 24.40% | 15.40% |
| VRPTW | MoSES_RF | 14.70% | 18.80% |
| VRPTW | RELD_MOEL | 22.20% | 24.10% |
| VRPTW | RELD_MTL | 31.10% | 23.80% |
| VRPTW | RouteFinder | 7.60% | 11.30% |
| OVRPTW | MTPOMO | 0.00% | 0.00% |
| OVRPTW | MVMOE | 0.00% | 9.10% |
| OVRPTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPTW | MoSES_RF | 0.00% | 0.00% |
| OVRPTW | RELD_MOEL | 62.30% | 42.90% |
| OVRPTW | RELD_MTL | 37.70% | 47.90% |
| OVRPTW | RouteFinder | 0.00% | 0.10% |
| OVRPB | MTPOMO | 0.00% | 2.00% |
| OVRPB | MVMOE | 0.00% | 1.40% |
| OVRPB | MoSES_CaDA | 0.00% | 0.00% |
| OVRPB | MoSES_RF | 0.00% | 0.10% |
| OVRPB | RELD_MOEL | 34.80% | 40.40% |
| OVRPB | RELD_MTL | 65.20% | 56.00% |
| OVRPB | RouteFinder | 0.00% | 0.10% |
| OVRPL | MTPOMO | 0.00% | 0.40% |
| OVRPL | MVMOE | 0.00% | 0.60% |
| OVRPL | MoSES_CaDA | 50.60% | 33.10% |
| OVRPL | MoSES_RF | 35.40% | 30.70% |
| OVRPL | RELD_MOEL | 0.20% | 5.10% |
| OVRPL | RELD_MTL | 0.20% | 8.10% |
| OVRPL | RouteFinder | 13.60% | 22.00% |
| VRPBL | MTPOMO | 0.00% | 1.30% |
| VRPBL | MVMOE | 0.00% | 4.80% |
| VRPBL | MoSES_CaDA | 0.00% | 0.00% |
| VRPBL | MoSES_RF | 0.00% | 0.00% |
| VRPBL | RELD_MOEL | 57.90% | 40.20% |
| VRPBL | RELD_MTL | 42.10% | 53.70% |
| VRPBL | RouteFinder | 0.00% | 0.00% |
| VRPBTW | MTPOMO | 0.00% | 0.00% |
| VRPBTW | MVMOE | 0.00% | 9.90% |
| VRPBTW | MoSES_CaDA | 3.30% | 4.40% |
| VRPBTW | MoSES_RF | 5.60% | 9.70% |
| VRPBTW | RELD_MOEL | 22.00% | 29.70% |
| VRPBTW | RELD_MTL | 56.10% | 35.70% |
| VRPBTW | RouteFinder | 13.00% | 10.60% |
| VRPLTW | MTPOMO | 0.00% | 0.00% |
| VRPLTW | MVMOE | 0.00% | 7.40% |
| VRPLTW | MoSES_CaDA | 23.10% | 15.80% |
| VRPLTW | MoSES_RF | 9.60% | 12.30% |
| VRPLTW | RELD_MOEL | 24.70% | 21.90% |
| VRPLTW | RELD_MTL | 30.60% | 27.10% |
| VRPLTW | RouteFinder | 12.00% | 15.50% |
| OVRPBL | MTPOMO | 0.00% | 0.90% |
| OVRPBL | MVMOE | 0.00% | 1.60% |
| OVRPBL | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBL | MoSES_RF | 0.00% | 0.20% |
| OVRPBL | RELD_MOEL | 31.90% | 42.70% |
| OVRPBL | RELD_MTL | 68.10% | 54.40% |
| OVRPBL | RouteFinder | 0.00% | 0.20% |
| OVRPBTW | MTPOMO | 0.00% | 0.00% |
| OVRPBTW | MVMOE | 0.00% | 9.50% |
| OVRPBTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBTW | RELD_MOEL | 51.80% | 44.40% |
| OVRPBTW | RELD_MTL | 48.20% | 46.00% |
| OVRPBTW | RouteFinder | 0.00% | 0.10% |
| OVRPLTW | MTPOMO | 0.00% | 0.00% |
| OVRPLTW | MVMOE | 0.00% | 7.80% |
| OVRPLTW | MoSES_CaDA | 0.00% | 0.10% |
| OVRPLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPLTW | RELD_MOEL | 62.90% | 42.90% |
| OVRPLTW | RELD_MTL | 37.10% | 49.20% |
| OVRPLTW | RouteFinder | 0.00% | 0.00% |
| VRPBLTW | MTPOMO | 0.00% | 0.10% |
| VRPBLTW | MVMOE | 0.00% | 9.20% |
| VRPBLTW | MoSES_CaDA | 4.40% | 4.70% |
| VRPBLTW | MoSES_RF | 0.70% | 8.00% |
| VRPBLTW | RELD_MOEL | 26.90% | 29.50% |
| VRPBLTW | RELD_MTL | 49.30% | 35.20% |
| VRPBLTW | RouteFinder | 18.70% | 13.30% |
| OVRPBLTW | MTPOMO | 0.00% | 0.00% |
| OVRPBLTW | MVMOE | 0.00% | 7.50% |
| OVRPBLTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBLTW | RELD_MOEL | 51.60% | 44.70% |
| OVRPBLTW | RELD_MTL | 48.40% | 47.80% |
| OVRPBLTW | RouteFinder | 0.00% | 0.00% |

## R40A

Checkpoint: `code/V4/runs/R40_discriminative_training/dual_stream_seed2/best.pt`; epoch 13.

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4684 | 0.8082 | 0.9208 | 11.172058 | 11.333921 (-0.9967%) | 11.056695 (+0.9517%) | 0.9808% |
| TSP | 0.3470 | 0.5900 | 0.7310 | 8.047198 | 8.082872 (-0.4413%) | 7.994650 (+0.6573%) | 0.6764% |
| CVRP | 0.4480 | 0.6630 | 0.8060 | 18.260297 | 18.332332 (-0.3929%) | 18.158602 (+0.5600%) | 0.6827% |
| ATSP | 0.7380 | 0.9540 | 0.9920 | 1.631986 | 1.669680 (-2.2576%) | 1.624559 (+0.4572%) | 0.4606% |
| OVRP | 0.3480 | 0.6450 | 0.8300 | 7.466311 | 7.469509 (-0.0428%) | 7.408692 (+0.7777%) | 0.7867% |
| VRPB | 0.5050 | 0.9180 | 0.9890 | 9.287810 | 9.295063 (-0.0780%) | 9.217786 (+0.7597%) | 0.7687% |
| VRPL | 0.2820 | 0.5160 | 0.7090 | 12.361822 | 12.360694 (+0.0091%) | 12.295043 (+0.5431%) | 0.5510% |
| VRPTW | 0.4310 | 0.7480 | 0.9070 | 17.651965 | 18.435676 (-4.2511%) | 17.423958 (+1.3086%) | 1.3697% |
| OVRPTW | 0.4940 | 0.9080 | 0.9990 | 10.113738 | 10.123299 (-0.0944%) | 10.033608 (+0.7986%) | 0.8032% |
| OVRPB | 0.5490 | 0.9640 | 0.9890 | 6.297996 | 6.296312 (+0.0268%) | 6.232885 (+1.0446%) | 1.0605% |
| OVRPL | 0.3620 | 0.6550 | 0.8600 | 7.446497 | 7.453650 (-0.0960%) | 7.393675 (+0.7144%) | 0.7204% |
| VRPBL | 0.5540 | 0.9390 | 0.9910 | 9.249223 | 9.249180 (+0.0005%) | 9.177293 (+0.7838%) | 0.7913% |
| VRPBTW | 0.4360 | 0.7750 | 0.9330 | 18.983420 | 19.526309 (-2.7803%) | 18.690529 (+1.5671%) | 1.5659% |
| VRPLTW | 0.4450 | 0.7580 | 0.9120 | 17.612777 | 18.426862 (-4.4179%) | 17.372466 (+1.3833%) | 1.4443% |
| OVRPBL | 0.5550 | 0.9710 | 0.9910 | 6.302504 | 6.306825 (-0.0685%) | 6.241775 (+0.9729%) | 0.9781% |
| OVRPBTW | 0.4720 | 0.9040 | 0.9990 | 10.611580 | 10.608986 (+0.0245%) | 10.478190 (+1.2730%) | 1.3149% |
| OVRPLTW | 0.5260 | 0.9210 | 0.9990 | 10.093550 | 10.103602 (-0.0995%) | 10.017218 (+0.7620%) | 0.7845% |
| VRPBLTW | 0.4580 | 0.7930 | 0.9370 | 19.079970 | 19.676957 (-3.0339%) | 18.788554 (+1.5510%) | 1.6366% |
| OVRPBLTW | 0.4820 | 0.9250 | 1.0000 | 10.598399 | 10.592773 (+0.0531%) | 10.471024 (+1.2165%) | 1.2582% |

### Solver pick distribution

| Problem | Solver | Pick fraction | Native winner fraction |
| --- | --- | ---: | ---: |
| TSP | BQ | 57.70% | 28.00% |
| TSP | DIFUSCO | 0.00% | 5.20% |
| TSP | DIFUSCO500 | 6.80% | 13.20% |
| TSP | ELG | 20.60% | 7.80% |
| TSP | LEHD | 0.00% | 18.50% |
| TSP | OMNI | 0.00% | 0.30% |
| TSP | T2T | 0.40% | 11.40% |
| TSP | T2T500 | 14.50% | 15.60% |
| CVRP | BQ | 7.50% | 12.20% |
| CVRP | ELG | 0.00% | 1.50% |
| CVRP | ICAM | 15.70% | 11.30% |
| CVRP | LEHD | 5.00% | 14.70% |
| CVRP | MVMOE | 0.00% | 2.60% |
| CVRP | MoSES_CaDA | 1.80% | 2.60% |
| CVRP | MoSES_RF | 9.70% | 3.90% |
| CVRP | OMNI | 38.60% | 24.40% |
| CVRP | RELD_CVRP | 21.70% | 25.40% |
| CVRP | RouteFinder | 0.00% | 1.40% |
| ATSP | GLOP | 0.00% | 0.50% |
| ATSP | ICAM_ATSP | 27.80% | 30.10% |
| ATSP | MATNET | 11.90% | 14.80% |
| ATSP | MATPOENET | 0.00% | 8.80% |
| ATSP | UNICO_MatPOENet | 60.30% | 45.80% |
| OVRP | MTPOMO | 0.00% | 1.40% |
| OVRP | MVMOE | 0.00% | 0.40% |
| OVRP | MoSES_CaDA | 42.00% | 31.90% |
| OVRP | MoSES_RF | 55.70% | 31.70% |
| OVRP | RELD_MOEL | 0.00% | 6.90% |
| OVRP | RELD_MTL | 0.00% | 8.30% |
| OVRP | RouteFinder | 2.30% | 19.40% |
| VRPB | MTPOMO | 0.00% | 2.20% |
| VRPB | MVMOE | 0.00% | 5.90% |
| VRPB | MoSES_CaDA | 0.00% | 0.00% |
| VRPB | MoSES_RF | 0.00% | 0.10% |
| VRPB | RELD_MOEL | 50.80% | 43.50% |
| VRPB | RELD_MTL | 49.20% | 48.30% |
| VRPB | RouteFinder | 0.00% | 0.00% |
| VRPL | MTPOMO | 0.00% | 3.30% |
| VRPL | MVMOE | 0.00% | 2.60% |
| VRPL | MoSES_CaDA | 34.80% | 25.20% |
| VRPL | MoSES_RF | 41.80% | 25.60% |
| VRPL | RELD_MOEL | 0.00% | 13.60% |
| VRPL | RELD_MTL | 21.90% | 13.80% |
| VRPL | RouteFinder | 1.50% | 15.90% |
| VRPTW | MTPOMO | 0.00% | 0.00% |
| VRPTW | MVMOE | 0.00% | 6.60% |
| VRPTW | MoSES_CaDA | 19.40% | 15.40% |
| VRPTW | MoSES_RF | 18.10% | 18.80% |
| VRPTW | RELD_MOEL | 4.10% | 24.10% |
| VRPTW | RELD_MTL | 52.40% | 23.80% |
| VRPTW | RouteFinder | 6.00% | 11.30% |
| OVRPTW | MTPOMO | 0.00% | 0.00% |
| OVRPTW | MVMOE | 0.00% | 9.10% |
| OVRPTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPTW | MoSES_RF | 0.00% | 0.00% |
| OVRPTW | RELD_MOEL | 39.30% | 42.90% |
| OVRPTW | RELD_MTL | 60.70% | 47.90% |
| OVRPTW | RouteFinder | 0.00% | 0.10% |
| OVRPB | MTPOMO | 0.00% | 2.00% |
| OVRPB | MVMOE | 0.00% | 1.40% |
| OVRPB | MoSES_CaDA | 0.00% | 0.00% |
| OVRPB | MoSES_RF | 0.00% | 0.10% |
| OVRPB | RELD_MOEL | 47.00% | 40.40% |
| OVRPB | RELD_MTL | 53.00% | 56.00% |
| OVRPB | RouteFinder | 0.00% | 0.10% |
| OVRPL | MTPOMO | 0.00% | 0.40% |
| OVRPL | MVMOE | 0.00% | 0.60% |
| OVRPL | MoSES_CaDA | 56.10% | 33.10% |
| OVRPL | MoSES_RF | 43.10% | 30.70% |
| OVRPL | RELD_MOEL | 0.00% | 5.10% |
| OVRPL | RELD_MTL | 0.00% | 8.10% |
| OVRPL | RouteFinder | 0.80% | 22.00% |
| VRPBL | MTPOMO | 0.00% | 1.30% |
| VRPBL | MVMOE | 0.00% | 4.80% |
| VRPBL | MoSES_CaDA | 0.00% | 0.00% |
| VRPBL | MoSES_RF | 0.00% | 0.00% |
| VRPBL | RELD_MOEL | 48.90% | 40.20% |
| VRPBL | RELD_MTL | 51.10% | 53.70% |
| VRPBL | RouteFinder | 0.00% | 0.00% |
| VRPBTW | MTPOMO | 0.00% | 0.00% |
| VRPBTW | MVMOE | 0.00% | 9.90% |
| VRPBTW | MoSES_CaDA | 2.60% | 4.40% |
| VRPBTW | MoSES_RF | 8.90% | 9.70% |
| VRPBTW | RELD_MOEL | 5.10% | 29.70% |
| VRPBTW | RELD_MTL | 73.00% | 35.70% |
| VRPBTW | RouteFinder | 10.40% | 10.60% |
| VRPLTW | MTPOMO | 0.00% | 0.00% |
| VRPLTW | MVMOE | 0.00% | 7.40% |
| VRPLTW | MoSES_CaDA | 18.80% | 15.80% |
| VRPLTW | MoSES_RF | 11.00% | 12.30% |
| VRPLTW | RELD_MOEL | 8.00% | 21.90% |
| VRPLTW | RELD_MTL | 50.60% | 27.10% |
| VRPLTW | RouteFinder | 11.60% | 15.50% |
| OVRPBL | MTPOMO | 0.00% | 0.90% |
| OVRPBL | MVMOE | 0.00% | 1.60% |
| OVRPBL | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBL | MoSES_RF | 0.00% | 0.20% |
| OVRPBL | RELD_MOEL | 39.10% | 42.70% |
| OVRPBL | RELD_MTL | 60.90% | 54.40% |
| OVRPBL | RouteFinder | 0.00% | 0.20% |
| OVRPBTW | MTPOMO | 0.00% | 0.00% |
| OVRPBTW | MVMOE | 0.00% | 9.50% |
| OVRPBTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBTW | RELD_MOEL | 26.50% | 44.40% |
| OVRPBTW | RELD_MTL | 73.50% | 46.00% |
| OVRPBTW | RouteFinder | 0.00% | 0.10% |
| OVRPLTW | MTPOMO | 0.00% | 0.00% |
| OVRPLTW | MVMOE | 0.00% | 7.80% |
| OVRPLTW | MoSES_CaDA | 0.00% | 0.10% |
| OVRPLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPLTW | RELD_MOEL | 37.40% | 42.90% |
| OVRPLTW | RELD_MTL | 62.60% | 49.20% |
| OVRPLTW | RouteFinder | 0.00% | 0.00% |
| VRPBLTW | MTPOMO | 0.00% | 0.10% |
| VRPBLTW | MVMOE | 0.00% | 9.20% |
| VRPBLTW | MoSES_CaDA | 3.70% | 4.70% |
| VRPBLTW | MoSES_RF | 1.00% | 8.00% |
| VRPBLTW | RELD_MOEL | 6.30% | 29.50% |
| VRPBLTW | RELD_MTL | 69.20% | 35.20% |
| VRPBLTW | RouteFinder | 19.80% | 13.30% |
| OVRPBLTW | MTPOMO | 0.00% | 0.00% |
| OVRPBLTW | MVMOE | 0.00% | 7.50% |
| OVRPBLTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBLTW | RELD_MOEL | 28.40% | 44.70% |
| OVRPBLTW | RELD_MTL | 71.60% | 47.80% |
| OVRPBLTW | RouteFinder | 0.00% | 0.00% |

## R40B

Checkpoint: `code/V4/runs/R40_discriminative_training/direct_seed2/best.pt`; epoch 24.

| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.4729 | 0.8080 | 0.9197 | 11.175296 | 11.333921 (-0.9813%) | 11.056695 (+0.9677%) | 0.9970% |
| TSP | 0.3520 | 0.5890 | 0.7550 | 8.047019 | 8.082872 (-0.4436%) | 7.994650 (+0.6550%) | 0.6775% |
| CVRP | 0.4560 | 0.6730 | 0.8180 | 18.263372 | 18.332332 (-0.3762%) | 18.158602 (+0.5770%) | 0.7096% |
| ATSP | 0.7690 | 0.9610 | 0.9920 | 1.631245 | 1.669680 (-2.3019%) | 1.624559 (+0.4115%) | 0.4174% |
| OVRP | 0.3690 | 0.6440 | 0.8310 | 7.462689 | 7.469509 (-0.0913%) | 7.408692 (+0.7288%) | 0.7371% |
| VRPB | 0.5220 | 0.9170 | 0.9870 | 9.293158 | 9.295063 (-0.0205%) | 9.217786 (+0.8177%) | 0.8537% |
| VRPL | 0.2920 | 0.5190 | 0.6930 | 12.359675 | 12.360694 (-0.0082%) | 12.295043 (+0.5257%) | 0.5365% |
| VRPTW | 0.4200 | 0.7450 | 0.9140 | 17.674062 | 18.435676 (-4.1312%) | 17.423958 (+1.4354%) | 1.5133% |
| OVRPTW | 0.4970 | 0.9090 | 0.9990 | 10.118851 | 10.123299 (-0.0439%) | 10.033608 (+0.8496%) | 0.8645% |
| OVRPB | 0.5530 | 0.9630 | 0.9910 | 6.292767 | 6.296312 (-0.0563%) | 6.232885 (+0.9607%) | 0.9497% |
| OVRPL | 0.3720 | 0.6670 | 0.8490 | 7.448861 | 7.453650 (-0.0643%) | 7.393675 (+0.7464%) | 0.7476% |
| VRPBL | 0.5300 | 0.9290 | 0.9860 | 9.249460 | 9.249180 (+0.0030%) | 9.177293 (+0.7864%) | 0.7956% |
| VRPBTW | 0.4270 | 0.7840 | 0.9260 | 19.009920 | 19.526309 (-2.6446%) | 18.690529 (+1.7088%) | 1.7072% |
| VRPLTW | 0.4440 | 0.7490 | 0.8960 | 17.613923 | 18.426862 (-4.4117%) | 17.372466 (+1.3899%) | 1.4747% |
| OVRPBL | 0.5830 | 0.9700 | 0.9960 | 6.301993 | 6.306825 (-0.0766%) | 6.241775 (+0.9648%) | 0.9690% |
| OVRPBTW | 0.4940 | 0.9000 | 0.9990 | 10.610889 | 10.608986 (+0.0179%) | 10.478190 (+1.2664%) | 1.3019% |
| OVRPLTW | 0.5030 | 0.9210 | 1.0000 | 10.099943 | 10.103602 (-0.0362%) | 10.017218 (+0.8258%) | 0.8507% |
| VRPBLTW | 0.4550 | 0.7810 | 0.9230 | 19.077512 | 19.676957 (-3.0464%) | 18.788554 (+1.5379%) | 1.5768% |
| OVRPBLTW | 0.4740 | 0.9230 | 0.9990 | 10.599983 | 10.592773 (+0.0681%) | 10.471024 (+1.2316%) | 1.2639% |

### Solver pick distribution

| Problem | Solver | Pick fraction | Native winner fraction |
| --- | --- | ---: | ---: |
| TSP | BQ | 57.90% | 28.00% |
| TSP | DIFUSCO | 0.10% | 5.20% |
| TSP | DIFUSCO500 | 4.50% | 13.20% |
| TSP | ELG | 12.00% | 7.80% |
| TSP | LEHD | 0.10% | 18.50% |
| TSP | OMNI | 0.00% | 0.30% |
| TSP | T2T | 1.80% | 11.40% |
| TSP | T2T500 | 23.60% | 15.60% |
| CVRP | BQ | 8.50% | 12.20% |
| CVRP | ELG | 0.60% | 1.50% |
| CVRP | ICAM | 9.50% | 11.30% |
| CVRP | LEHD | 12.00% | 14.70% |
| CVRP | MVMOE | 0.40% | 2.60% |
| CVRP | MoSES_CaDA | 3.80% | 2.60% |
| CVRP | MoSES_RF | 5.00% | 3.90% |
| CVRP | OMNI | 37.30% | 24.40% |
| CVRP | RELD_CVRP | 22.70% | 25.40% |
| CVRP | RouteFinder | 0.20% | 1.40% |
| ATSP | GLOP | 0.00% | 0.50% |
| ATSP | ICAM_ATSP | 27.00% | 30.10% |
| ATSP | MATNET | 16.90% | 14.80% |
| ATSP | MATPOENET | 0.00% | 8.80% |
| ATSP | UNICO_MatPOENet | 56.10% | 45.80% |
| OVRP | MTPOMO | 0.00% | 1.40% |
| OVRP | MVMOE | 0.00% | 0.40% |
| OVRP | MoSES_CaDA | 48.10% | 31.90% |
| OVRP | MoSES_RF | 40.50% | 31.70% |
| OVRP | RELD_MOEL | 0.10% | 6.90% |
| OVRP | RELD_MTL | 0.50% | 8.30% |
| OVRP | RouteFinder | 10.80% | 19.40% |
| VRPB | MTPOMO | 0.20% | 2.20% |
| VRPB | MVMOE | 0.00% | 5.90% |
| VRPB | MoSES_CaDA | 0.00% | 0.00% |
| VRPB | MoSES_RF | 0.00% | 0.10% |
| VRPB | RELD_MOEL | 46.20% | 43.50% |
| VRPB | RELD_MTL | 53.60% | 48.30% |
| VRPB | RouteFinder | 0.00% | 0.00% |
| VRPL | MTPOMO | 0.00% | 3.30% |
| VRPL | MVMOE | 0.00% | 2.60% |
| VRPL | MoSES_CaDA | 43.70% | 25.20% |
| VRPL | MoSES_RF | 41.10% | 25.60% |
| VRPL | RELD_MOEL | 2.20% | 13.60% |
| VRPL | RELD_MTL | 7.30% | 13.80% |
| VRPL | RouteFinder | 5.70% | 15.90% |
| VRPTW | MTPOMO | 0.00% | 0.00% |
| VRPTW | MVMOE | 0.00% | 6.60% |
| VRPTW | MoSES_CaDA | 21.50% | 15.40% |
| VRPTW | MoSES_RF | 15.60% | 18.80% |
| VRPTW | RELD_MOEL | 12.00% | 24.10% |
| VRPTW | RELD_MTL | 45.40% | 23.80% |
| VRPTW | RouteFinder | 5.50% | 11.30% |
| OVRPTW | MTPOMO | 0.00% | 0.00% |
| OVRPTW | MVMOE | 0.00% | 9.10% |
| OVRPTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPTW | MoSES_RF | 0.00% | 0.00% |
| OVRPTW | RELD_MOEL | 26.00% | 42.90% |
| OVRPTW | RELD_MTL | 74.00% | 47.90% |
| OVRPTW | RouteFinder | 0.00% | 0.10% |
| OVRPB | MTPOMO | 0.00% | 2.00% |
| OVRPB | MVMOE | 0.00% | 1.40% |
| OVRPB | MoSES_CaDA | 0.00% | 0.00% |
| OVRPB | MoSES_RF | 0.00% | 0.10% |
| OVRPB | RELD_MOEL | 47.70% | 40.40% |
| OVRPB | RELD_MTL | 52.30% | 56.00% |
| OVRPB | RouteFinder | 0.00% | 0.10% |
| OVRPL | MTPOMO | 0.00% | 0.40% |
| OVRPL | MVMOE | 0.00% | 0.60% |
| OVRPL | MoSES_CaDA | 52.30% | 33.10% |
| OVRPL | MoSES_RF | 37.00% | 30.70% |
| OVRPL | RELD_MOEL | 0.20% | 5.10% |
| OVRPL | RELD_MTL | 0.60% | 8.10% |
| OVRPL | RouteFinder | 9.90% | 22.00% |
| VRPBL | MTPOMO | 0.10% | 1.30% |
| VRPBL | MVMOE | 0.10% | 4.80% |
| VRPBL | MoSES_CaDA | 0.00% | 0.00% |
| VRPBL | MoSES_RF | 0.00% | 0.00% |
| VRPBL | RELD_MOEL | 41.60% | 40.20% |
| VRPBL | RELD_MTL | 58.20% | 53.70% |
| VRPBL | RouteFinder | 0.00% | 0.00% |
| VRPBTW | MTPOMO | 0.00% | 0.00% |
| VRPBTW | MVMOE | 0.40% | 9.90% |
| VRPBTW | MoSES_CaDA | 4.00% | 4.40% |
| VRPBTW | MoSES_RF | 6.10% | 9.70% |
| VRPBTW | RELD_MOEL | 7.40% | 29.70% |
| VRPBTW | RELD_MTL | 72.60% | 35.70% |
| VRPBTW | RouteFinder | 9.50% | 10.60% |
| VRPLTW | MTPOMO | 0.00% | 0.00% |
| VRPLTW | MVMOE | 0.00% | 7.40% |
| VRPLTW | MoSES_CaDA | 17.10% | 15.80% |
| VRPLTW | MoSES_RF | 6.10% | 12.30% |
| VRPLTW | RELD_MOEL | 12.10% | 21.90% |
| VRPLTW | RELD_MTL | 53.20% | 27.10% |
| VRPLTW | RouteFinder | 11.50% | 15.50% |
| OVRPBL | MTPOMO | 0.00% | 0.90% |
| OVRPBL | MVMOE | 0.00% | 1.60% |
| OVRPBL | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBL | MoSES_RF | 0.00% | 0.20% |
| OVRPBL | RELD_MOEL | 48.80% | 42.70% |
| OVRPBL | RELD_MTL | 51.20% | 54.40% |
| OVRPBL | RouteFinder | 0.00% | 0.20% |
| OVRPBTW | MTPOMO | 0.00% | 0.00% |
| OVRPBTW | MVMOE | 0.10% | 9.50% |
| OVRPBTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBTW | RELD_MOEL | 31.80% | 44.40% |
| OVRPBTW | RELD_MTL | 68.10% | 46.00% |
| OVRPBTW | RouteFinder | 0.00% | 0.10% |
| OVRPLTW | MTPOMO | 0.00% | 0.00% |
| OVRPLTW | MVMOE | 0.00% | 7.80% |
| OVRPLTW | MoSES_CaDA | 0.00% | 0.10% |
| OVRPLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPLTW | RELD_MOEL | 30.20% | 42.90% |
| OVRPLTW | RELD_MTL | 69.80% | 49.20% |
| OVRPLTW | RouteFinder | 0.00% | 0.00% |
| VRPBLTW | MTPOMO | 0.00% | 0.10% |
| VRPBLTW | MVMOE | 0.00% | 9.20% |
| VRPBLTW | MoSES_CaDA | 3.90% | 4.70% |
| VRPBLTW | MoSES_RF | 2.70% | 8.00% |
| VRPBLTW | RELD_MOEL | 8.20% | 29.50% |
| VRPBLTW | RELD_MTL | 71.80% | 35.20% |
| VRPBLTW | RouteFinder | 13.40% | 13.30% |
| OVRPBLTW | MTPOMO | 0.00% | 0.00% |
| OVRPBLTW | MVMOE | 0.00% | 7.50% |
| OVRPBLTW | MoSES_CaDA | 0.00% | 0.00% |
| OVRPBLTW | MoSES_RF | 0.00% | 0.00% |
| OVRPBLTW | RELD_MOEL | 22.20% | 44.70% |
| OVRPBLTW | RELD_MTL | 77.80% | 47.80% |
| OVRPBLTW | RouteFinder | 0.00% | 0.00% |

## Interpretation

1. These test results do not show a substantial R40 improvement. R34 remains the strongest strict Top1 of the four frozen policies.
2. Raw mean cost and macro relative cost are different aggregates. R40A has a slightly smaller raw mean cost, but not a better macro vs SBS than R34.
3. Do not tune models, checkpoints, or the label audit sample using this test comparison. Original sampling remains locked.
4. Validation replay uses historical evaluation batch sizes: R34/R39A 640, R40A/R40B 256. An initial batch256 R34 replay changed a few near-equal FP32 choices; that failed pre-test attempt is preserved separately.
5. These are four single-seed frozen models; small differences are not evidence of a reliable architectural advantage.
