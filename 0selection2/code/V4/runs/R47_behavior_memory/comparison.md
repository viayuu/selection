# R47B: instance-conditioned performance memory

Only B, seed2, was trained from scratch. R47A was not trained.
R45A is a frozen historical reference, not a freshly matched R47A control.
Train-only references; original FP64 costs; native winners; ALL is a per-problem macro average.

## Test

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R45A_historical | ALL | 0.4791 | 0.8174 | 0.9243 | 11.168366 | -1.0320% | 0.9432% |
| R47B | ALL | 0.4736 | 0.8141 | 0.9228 | 11.173784 | -0.9971% | 0.9850% |
| size_prior | ALL | 0.4144 | 0.7421 | 0.8438 | 11.298356 | -0.3060% | 1.6205% |

R47B minus historical R45A: Top1 -0.5500 percentage points; mean_cost +0.005418; actual regret relative change +4.433%.
Engineering screening target reached: False. For the regret target, a Top1 drop greater than1pp is considered material.
This is a single-seed direction screen, not a statistical or causal claim about retrieval.

## Training

Stop epoch: 39; validation-best epoch: 31; successful updates: 55,458.
| Point | Top1 | CE | mean_cost | vs_SBS | actual regret |
|---|---:|---:|---:|---:|---:|
| Best val | 0.4873 | 1.1002 | 11.173151 | -1.0453% | 0.9532% |
| Final train | 0.5151 | 1.0391 | 11.166102 | -1.1519% | 0.8592% |
| Final val | 0.4832 | 1.1025 | 11.174242 | -1.0313% | 0.9717% |
| Last5 val | 0.4833 | 1.1027 | 11.174279 | -1.0323% | 0.9705% |

## Validation mechanism check

The fixed best model is reevaluated without retraining. Whole behavior rows (gap vector and native winner) are permuted together within exact problem and node count. Keys stay unchanged.

| Memory | Top1 | mean_cost | actual regret |
|---|---:|---:|---:|
| Real | 0.4873 | 11.173151 | 0.9532% |
| Shuffled | 0.4571 | 11.248300 | 1.3134% |
Changed validation choices: 4,161; retrieved reference indices and weights remain unchanged. Per-problem results: mechanism_check_per_problem.csv.
Sensitivity shows reliance on the stored association, not by itself generalizable usefulness or a complete causal mechanism.

## Families

| Model | Family | Top1 | mean_cost | actual regret |
|---|---|---:|---:|---:|
| R45A_historical | TSP | 0.3590 | 8.044507 | 0.6504% |
| R45A_historical | CVRP | 0.4810 | 18.247945 | 0.6138% |
| R45A_historical | ATSP | 0.7750 | 1.629876 | 0.3274% |
| R45A_historical | MVRP | 0.4673 | 11.540551 | 1.0258% |
| R47B | TSP | 0.3750 | 8.044306 | 0.6410% |
| R47B | CVRP | 0.4770 | 18.247443 | 0.6107% |
| R47B | ATSP | 0.7800 | 1.630181 | 0.3459% |
| R47B | MVRP | 0.4595 | 11.547078 | 1.0755% |
| size_prior | TSP | 0.2100 | 8.071628 | 0.9022% |
| size_prior | CVRP | 0.2540 | 18.332332 | 0.9733% |
| size_prior | ATSP | 0.6630 | 1.635086 | 0.6361% |
| size_prior | MVRP | 0.4221 | 11.688757 | 1.7772% |

## Paired decisions versus historical R45A

| Change | Instances | Contribution to mean_cost change | Contribution to regret change |
|---|---:|---:|---:|
| corrected | 1415 | -0.019513 | -0.16152pp |
| harmed | 1514 | +0.023999 | +0.19322pp |
| both_wrong_cheaper | 439 | -0.005225 | -0.03394pp |
| both_wrong_dearer | 508 | +0.006157 | +0.04406pp |
| both_wrong_equal | 0 | +0.000000 | +0.00000pp |
| all | 18000 | +0.005418 | +0.04182pp |

## Implementation boundary

The reference graph summaries are detached epoch snapshots, refreshed in eval mode at epoch start and before every formal evaluation. Query Encoder, shared key map on both sides, behavior value/correction, and the main selector are jointly trainable.
Every train/train_eval query excludes its base-instance group; val/test only read train memory and also exclude input duplicates. Identity grouping covers coordinate permutations/D4 copies to1e-6 and identical ordered matrices; no external lineage is inferred.
No source embeddings, label regeneration, extra seed, new loss, candidate cropping, or test-driven checkpoint selection.
The average-gap reference is a train-only problem/size-quartile prior, not the R47 decision rule.

## Per problem

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R45A_historical | TSP | 0.3590 | 0.6170 | 0.7520 | 8.044507 | -0.4746% | 0.6504% |
| R45A_historical | CVRP | 0.4810 | 0.6920 | 0.8220 | 18.247945 | -0.4603% | 0.6138% |
| R45A_historical | ATSP | 0.7750 | 0.9620 | 0.9950 | 1.629876 | -2.3839% | 0.3274% |
| R45A_historical | OVRP | 0.3740 | 0.6490 | 0.8310 | 7.461105 | -0.1125% | 0.7130% |
| R45A_historical | VRPB | 0.5170 | 0.9180 | 0.9880 | 9.287440 | -0.0820% | 0.7681% |
| R45A_historical | VRPL | 0.3020 | 0.5280 | 0.6960 | 12.354608 | -0.0492% | 0.4926% |
| R45A_historical | VRPTW | 0.4350 | 0.7710 | 0.9240 | 17.657704 | -4.2199% | 1.4119% |
| R45A_historical | OVRPTW | 0.4910 | 0.9080 | 0.9990 | 10.114022 | -0.0916% | 0.8190% |
| R45A_historical | OVRPB | 0.5510 | 0.9640 | 0.9920 | 6.295778 | -0.0085% | 1.0279% |
| R45A_historical | OVRPL | 0.3700 | 0.6860 | 0.8570 | 7.444869 | -0.1178% | 0.6943% |
| R45A_historical | VRPBL | 0.5470 | 0.9400 | 0.9910 | 9.245516 | -0.0396% | 0.7462% |
| R45A_historical | VRPBTW | 0.4430 | 0.7850 | 0.9350 | 18.972974 | -2.8338% | 1.5118% |
| R45A_historical | VRPLTW | 0.4450 | 0.7850 | 0.9280 | 17.603499 | -4.4683% | 1.3870% |
| R45A_historical | OVRPBL | 0.5740 | 0.9710 | 0.9910 | 6.301951 | -0.0773% | 0.9841% |
| R45A_historical | OVRPBTW | 0.5140 | 0.9030 | 0.9970 | 10.608161 | -0.0078% | 1.2650% |
| R45A_historical | OVRPLTW | 0.5020 | 0.9210 | 1.0000 | 10.097594 | -0.0595% | 0.8210% |
| R45A_historical | VRPBLTW | 0.4490 | 0.7890 | 0.9410 | 19.071665 | -3.0761% | 1.5746% |
| R45A_historical | OVRPBLTW | 0.4950 | 0.9250 | 0.9990 | 10.591372 | -0.0132% | 1.1698% |
| R47B | TSP | 0.3750 | 0.6090 | 0.7470 | 8.044306 | -0.4771% | 0.6410% |
| R47B | CVRP | 0.4770 | 0.6990 | 0.8290 | 18.247443 | -0.4631% | 0.6107% |
| R47B | ATSP | 0.7800 | 0.9630 | 0.9960 | 1.630181 | -2.3656% | 0.3459% |
| R47B | OVRP | 0.3560 | 0.6460 | 0.8310 | 7.464274 | -0.0701% | 0.7638% |
| R47B | VRPB | 0.5220 | 0.9180 | 0.9880 | 9.286858 | -0.0883% | 0.7610% |
| R47B | VRPL | 0.3090 | 0.5320 | 0.7040 | 12.355486 | -0.0421% | 0.4993% |
| R47B | VRPTW | 0.4320 | 0.7550 | 0.9170 | 17.654885 | -4.2352% | 1.3976% |
| R47B | OVRPTW | 0.4580 | 0.9080 | 0.9980 | 10.124810 | +0.0149% | 0.9226% |
| R47B | OVRPB | 0.5630 | 0.9640 | 0.9900 | 6.294150 | -0.0343% | 0.9895% |
| R47B | OVRPL | 0.3730 | 0.6670 | 0.8620 | 7.445822 | -0.1050% | 0.7152% |
| R47B | VRPBL | 0.5500 | 0.9380 | 0.9890 | 9.246673 | -0.0271% | 0.7573% |
| R47B | VRPBTW | 0.4080 | 0.7770 | 0.9260 | 19.015483 | -2.6161% | 1.7405% |
| R47B | VRPLTW | 0.4300 | 0.7670 | 0.9190 | 17.616357 | -4.3985% | 1.4886% |
| R47B | OVRPBL | 0.5680 | 0.9710 | 0.9930 | 6.302577 | -0.0674% | 0.9738% |
| R47B | OVRPBTW | 0.5180 | 0.9030 | 0.9990 | 10.605439 | -0.0334% | 1.2541% |
| R47B | OVRPLTW | 0.4810 | 0.9210 | 0.9990 | 10.105198 | +0.0158% | 0.8951% |
| R47B | VRPBLTW | 0.4380 | 0.7900 | 0.9240 | 19.095136 | -2.9569% | 1.7532% |
| R47B | OVRPBLTW | 0.4870 | 0.9250 | 0.9990 | 10.593028 | +0.0024% | 1.2217% |
| size_prior | TSP | 0.2100 | 0.3310 | 0.4500 | 8.071628 | -0.1391% | 0.9022% |
| size_prior | CVRP | 0.2540 | 0.5000 | 0.6500 | 18.332332 | +0.0000% | 0.9733% |
| size_prior | ATSP | 0.6630 | 0.9520 | 0.9900 | 1.635086 | -2.0719% | 0.6361% |
| size_prior | OVRP | 0.3190 | 0.6390 | 0.8300 | 7.470514 | +0.0135% | 0.8457% |
| size_prior | VRPB | 0.4830 | 0.9180 | 0.9400 | 9.295063 | +0.0000% | 0.8484% |
| size_prior | VRPL | 0.2970 | 0.5080 | 0.6670 | 12.354726 | -0.0483% | 0.4905% |
| size_prior | VRPTW | 0.2880 | 0.5030 | 0.7520 | 18.145477 | -1.5741% | 4.3848% |
| size_prior | OVRPTW | 0.4790 | 0.9080 | 0.9800 | 10.123299 | +0.0000% | 0.9229% |
| size_prior | OVRPB | 0.5600 | 0.9640 | 0.9770 | 6.296312 | +0.0000% | 1.0382% |
| size_prior | OVRPL | 0.3870 | 0.6640 | 0.8580 | 7.445682 | -0.1069% | 0.7128% |
| size_prior | VRPBL | 0.5370 | 0.9390 | 0.9520 | 9.249180 | +0.0000% | 0.7814% |
| size_prior | VRPBTW | 0.3570 | 0.6540 | 0.7600 | 19.526309 | +0.0000% | 3.8798% |
| size_prior | VRPLTW | 0.2990 | 0.5100 | 0.6730 | 18.135649 | -1.5804% | 4.1786% |
| size_prior | OVRPBL | 0.5440 | 0.9710 | 0.9760 | 6.306825 | +0.0000% | 1.0493% |
| size_prior | OVRPBTW | 0.4600 | 0.9040 | 0.9910 | 10.608986 | +0.0000% | 1.2996% |
| size_prior | OVRPLTW | 0.4920 | 0.9210 | 0.9650 | 10.103602 | +0.0000% | 0.8997% |
| size_prior | VRPBLTW | 0.3520 | 0.6470 | 0.7860 | 19.676957 | +0.0000% | 4.1316% |
| size_prior | OVRPBLTW | 0.4780 | 0.9250 | 0.9920 | 10.592773 | +0.0000% | 1.1946% |
