# R43: full-pool pairwise selection

一个 seed=2；A/B 从头训练。全部模型使用原生 ind；成本从原始标签以 FP64 汇总。
R42A 只复用已冻结的 best；R43 按验证 macro_vs_sbs_pct 严格最小锁定，test 不参与选型。
ALL 的归一化百分比是逐问题宏平均，不是两个 ALL 平均成本直接相除。
沿用历史口径：SBS 是对应 split 成本表中均值最低的固定 solver；Oracle 是逐实例最低成本。
A 对 R42A 同时改变监督与 maximin 分数；A/B 的差异才对应显式比较 Decoder。

## Test 总览

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | ALL | 0.4735 | 0.8137 | 0.9224 | 11.172455 | -0.9999% | 0.9715% |
| R43A | ALL | 0.4794 | 0.8157 | 0.9247 | 11.168108 | -1.0324% | 0.9433% |
| R43B | ALL | 0.4756 | 0.8173 | 0.9260 | 11.169289 | -1.0234% | 0.9523% |

## 训练与验证选型

| Model | Stop epoch | Best epoch | Best val Top1 | Best val vs_SBS | Final train Top1 | Final val Top1 | Last5 val Top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R43A | 40 | 35 | 0.4884 | -1.0668% | 0.4961 | 0.4874 | 0.4883 |
| R43B | 40 | 33 | 0.4901 | -1.0663% | 0.5084 | 0.4877 | 0.4887 |

### 最终与末五轮验证

| Model | Point | Top1 | CE | Mean cost | vs_SBS | Actual regret |
|---|---|---:|---:|---:|---:|---:|
| R43A | Final | 0.4874 | 1.0885 | 11.170022 | -1.0591% | 0.9363% |
| R43A | Last5 mean | 0.4883 | 1.0886 | 11.169995 | -1.0609% | 0.9344% |
| R43B | Final | 0.4877 | 1.0901 | 11.169239 | -1.0639% | 0.9333% |
| R43B | Last5 mean | 0.4887 | 1.0905 | 11.169436 | -1.0620% | 0.9356% |

## 分问题族 Test

| Model | Family | Top1 | Mean cost | vs_SBS | Actual regret |
|---|---|---:|---:|---:|---:|
| R42A | TSP | 0.3720 | 8.042250 | -0.5026% | 0.6142% |
| R42A | CVRP | 0.4360 | 18.265892 | -0.3624% | 0.6570% |
| R42A | ATSP | 0.7670 | 1.630234 | -2.3625% | 0.3485% |
| R42A | MVRP | 0.4632 | 11.544387 | -0.9848% | 1.0578% |
| R43A | TSP | 0.3700 | 8.044075 | -0.4800% | 0.6424% |
| R43A | CVRP | 0.4660 | 18.250420 | -0.4468% | 0.6381% |
| R43A | ATSP | 0.7720 | 1.629884 | -2.3834% | 0.3284% |
| R43A | MVRP | 0.4681 | 11.540104 | -1.0182% | 1.0247% |
| R43B | TSP | 0.3500 | 8.046002 | -0.4561% | 0.6713% |
| R43B | CVRP | 0.4810 | 18.252043 | -0.4380% | 0.6441% |
| R43B | ATSP | 0.7780 | 1.630031 | -2.3746% | 0.3379% |
| R43B | MVRP | 0.4634 | 11.541275 | -1.0102% | 1.0325% |

## 强候选比较

真实 Top3 内部按实例平均，只统计原始成本不相等的方法对；零 margin 不算正确。
第二个诊断固定比较 R42A 提名的两个方法，不是新模型自己选的容易比较。

| Model | true Top3 accuracy | R42A nominated Top2 accuracy |
|---|---:|---:|
| R42A | 0.6978 | 0.5763 |
| R43A | 0.7005 | 0.5842 |
| R43B | 0.7000 | 0.5791 |

## 相对 R42A 的纠正与伤害

成本与 regret 差值：负值表示改善；分类正确性使用 native winner。

| Model | Change | Instances | Delta mean cost | Delta actual regret |
|---|---|---:|---:|---:|
| R43A | corrected | 946 | -0.013749 | -0.11357% |
| R43A | harmed | 839 | +0.011178 | +0.09251% |
| R43A | both_wrong_changed | 635 | -0.001776 | -0.00711% |
| R43A | all | 18000 | -0.004347 | -0.02817% |
| R43B | corrected | 1339 | -0.020335 | -0.16386% |
| R43B | harmed | 1302 | +0.018162 | +0.14730% |
| R43B | both_wrong_changed | 846 | -0.000993 | -0.00264% |
| R43B | all | 18000 | -0.003166 | -0.01919% |

## 结果判断

- R43A 相对 R42A：Top1 +0.594 个百分点；actual regret -0.02817 个百分点；mean_cost -0.004347。
- R43B 相对 R42A：Top1 +0.206 个百分点；actual regret -0.01919 个百分点；mean_cost -0.003166。
- 显式比较 B 相对匹配对照 A：Top1 -0.389 个百分点；actual regret +0.00898 个百分点。
- R43A 纠正 946 例、新增错误 839 例，净增加正确选择 107 例；不能只看纠正数量。
- R43B 纠正 1339 例、新增错误 1302 例，净增加正确选择 37 例；不能只看纠正数量。
- 本轮优先保留更简单的 A 作为 R43 候选：显式比较 B 的整体准确率与成本都未超过 A，不支持为了复杂度继续扩大 Decoder。
- 本轮只有一个从头训练 seed，不把微小差异解释为稳定优势；最终选择指标优先于 pair loss 或比较正确率。

## 逐问题 Test

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | TSP | 0.3720 | 0.6140 | 0.7410 | 8.042250 | -0.5026% | 0.6142% |
| R43A | TSP | 0.3700 | 0.6200 | 0.7480 | 8.044075 | -0.4800% | 0.6424% |
| R43B | TSP | 0.3500 | 0.6100 | 0.7490 | 8.046002 | -0.4561% | 0.6713% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | CVRP | 0.4360 | 0.6770 | 0.8100 | 18.265892 | -0.3624% | 0.6570% |
| R43A | CVRP | 0.4660 | 0.6930 | 0.8210 | 18.250420 | -0.4468% | 0.6381% |
| R43B | CVRP | 0.4810 | 0.6980 | 0.8290 | 18.252043 | -0.4380% | 0.6441% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | ATSP | 0.7670 | 0.9560 | 0.9930 | 1.630234 | -2.3625% | 0.3485% |
| R43A | ATSP | 0.7720 | 0.9620 | 0.9960 | 1.629884 | -2.3834% | 0.3284% |
| R43B | ATSP | 0.7780 | 0.9620 | 0.9960 | 1.630031 | -2.3746% | 0.3379% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRP | 0.3460 | 0.6430 | 0.8310 | 7.465951 | -0.0476% | 0.7856% |
| R43A | OVRP | 0.3580 | 0.6440 | 0.8360 | 7.463935 | -0.0746% | 0.7570% |
| R43B | OVRP | 0.3680 | 0.6420 | 0.8330 | 7.463432 | -0.0814% | 0.7486% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPB | 0.4940 | 0.9180 | 0.9880 | 9.290834 | -0.0455% | 0.8003% |
| R43A | VRPB | 0.5050 | 0.9180 | 0.9890 | 9.291716 | -0.0360% | 0.8089% |
| R43B | VRPB | 0.5080 | 0.9180 | 0.9890 | 9.289792 | -0.0567% | 0.7896% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPL | 0.3030 | 0.5360 | 0.7040 | 12.355418 | -0.0427% | 0.4963% |
| R43A | VRPL | 0.3090 | 0.5270 | 0.6980 | 12.353379 | -0.0592% | 0.4829% |
| R43B | VRPL | 0.3090 | 0.5230 | 0.6950 | 12.355442 | -0.0425% | 0.4936% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPTW | 0.4360 | 0.7520 | 0.9170 | 17.656235 | -4.2279% | 1.3925% |
| R43A | VRPTW | 0.4430 | 0.7550 | 0.9250 | 17.645494 | -4.2862% | 1.3410% |
| R43B | VRPTW | 0.4330 | 0.7800 | 0.9260 | 17.641668 | -4.3069% | 1.3125% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPTW | 0.4750 | 0.9080 | 0.9990 | 10.117615 | -0.0561% | 0.8431% |
| R43A | OVRPTW | 0.4840 | 0.9080 | 0.9990 | 10.116015 | -0.0720% | 0.8304% |
| R43B | OVRPTW | 0.4690 | 0.9080 | 0.9990 | 10.120668 | -0.0260% | 0.8709% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPB | 0.5590 | 0.9640 | 0.9910 | 6.295345 | -0.0154% | 1.0205% |
| R43A | OVRPB | 0.5580 | 0.9640 | 0.9910 | 6.295010 | -0.0207% | 1.0079% |
| R43B | OVRPB | 0.5700 | 0.9640 | 0.9920 | 6.292480 | -0.0608% | 0.9749% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPL | 0.3660 | 0.6670 | 0.8610 | 7.446749 | -0.0926% | 0.7223% |
| R43A | OVRPL | 0.3700 | 0.6800 | 0.8590 | 7.445066 | -0.1152% | 0.6975% |
| R43B | OVRPL | 0.3640 | 0.6710 | 0.8610 | 7.446607 | -0.0945% | 0.7236% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPBL | 0.5690 | 0.9390 | 0.9900 | 9.245790 | -0.0366% | 0.7467% |
| R43A | VRPBL | 0.5560 | 0.9380 | 0.9910 | 9.246575 | -0.0282% | 0.7584% |
| R43B | VRPBL | 0.5470 | 0.9400 | 0.9900 | 9.244701 | -0.0484% | 0.7395% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPBTW | 0.4560 | 0.7800 | 0.9370 | 18.977081 | -2.8128% | 1.5216% |
| R43A | VRPBTW | 0.4360 | 0.7760 | 0.9380 | 18.984880 | -2.7728% | 1.5542% |
| R43B | VRPBTW | 0.4260 | 0.7860 | 0.9410 | 18.988186 | -2.7559% | 1.5765% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPLTW | 0.4570 | 0.7750 | 0.9260 | 17.624781 | -4.3528% | 1.4805% |
| R43A | VRPLTW | 0.4460 | 0.7820 | 0.9300 | 17.587025 | -4.5577% | 1.3034% |
| R43B | VRPLTW | 0.4420 | 0.7860 | 0.9270 | 17.599454 | -4.4902% | 1.3817% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPBL | 0.5610 | 0.9710 | 0.9920 | 6.304361 | -0.0391% | 1.0106% |
| R43A | OVRPBL | 0.5860 | 0.9710 | 0.9920 | 6.299670 | -0.1134% | 0.9440% |
| R43B | OVRPBL | 0.5650 | 0.9710 | 0.9950 | 6.302030 | -0.0760% | 0.9789% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPBTW | 0.5100 | 0.9040 | 0.9990 | 10.608291 | -0.0065% | 1.2802% |
| R43A | OVRPBTW | 0.5150 | 0.9040 | 0.9980 | 10.608407 | -0.0055% | 1.2823% |
| R43B | OVRPBTW | 0.5230 | 0.9040 | 1.0000 | 10.607103 | -0.0177% | 1.2667% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPLTW | 0.4820 | 0.9210 | 1.0000 | 10.104459 | +0.0085% | 0.9023% |
| R43A | OVRPLTW | 0.5050 | 0.9210 | 1.0000 | 10.097362 | -0.0618% | 0.8300% |
| R43B | OVRPLTW | 0.4860 | 0.9210 | 1.0000 | 10.102824 | -0.0077% | 0.8647% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | VRPBLTW | 0.4650 | 0.7960 | 0.9250 | 19.073493 | -3.0669% | 1.6051% |
| R43A | VRPBLTW | 0.4500 | 0.7940 | 0.9350 | 19.075731 | -3.0555% | 1.6052% |
| R43B | VRPBLTW | 0.4480 | 0.8020 | 0.9460 | 19.075079 | -3.0588% | 1.5977% |

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R42A | OVRPBLTW | 0.4690 | 0.9250 | 1.0000 | 10.599406 | +0.0626% | 1.2594% |
| R43A | OVRPBLTW | 0.5010 | 0.9250 | 0.9990 | 10.591296 | -0.0139% | 1.1676% |
| R43B | OVRPBLTW | 0.4930 | 0.9250 | 1.0000 | 10.589655 | -0.0294% | 1.1683% |

## 臂分布

| Model | Problem | Solver picks |
|---|---|---|
| R42A | TSP | BQ: 54.9% / DIFUSCO: 0.0% / DIFUSCO500: 3.8% / ELG: 15.6% / LEHD: 0.0% / OMNI: 0.0% / T2T: 0.2% / T2T500: 25.5% |
| R42A | CVRP | BQ: 2.6% / ELG: 0.0% / ICAM: 7.9% / LEHD: 5.7% / MVMOE: 0.0% / MoSES_CaDA: 2.6% / MoSES_RF: 7.5% / OMNI: 47.9% / RELD_CVRP: 25.8% / RouteFinder: 0.0% |
| R42A | ATSP | GLOP: 0.0% / ICAM_ATSP: 28.1% / MATNET: 13.3% / MATPOENET: 0.0% / UNICO_MatPOENet: 58.6% |
| R42A | OVRP | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 47.9% / MoSES_RF: 40.7% / RELD_MOEL: 0.1% / RELD_MTL: 0.2% / RouteFinder: 11.1% |
| R42A | VRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 48.4% / RELD_MTL: 51.6% / RouteFinder: 0.0% |
| R42A | VRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 42.6% / MoSES_RF: 41.1% / RELD_MOEL: 0.5% / RELD_MTL: 15.3% / RouteFinder: 0.5% |
| R42A | VRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 21.1% / MoSES_RF: 14.3% / RELD_MOEL: 14.4% / RELD_MTL: 44.1% / RouteFinder: 6.1% |
| R42A | OVRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 65.0% / RELD_MTL: 35.0% / RouteFinder: 0.0% |
| R42A | OVRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 36.2% / RELD_MTL: 63.8% / RouteFinder: 0.0% |
| R42A | OVRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 55.5% / MoSES_RF: 30.8% / RELD_MOEL: 0.0% / RELD_MTL: 0.1% / RouteFinder: 13.6% |
| R42A | VRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 48.0% / RELD_MTL: 52.0% / RouteFinder: 0.0% |
| R42A | VRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.9% / MoSES_RF: 4.9% / RELD_MOEL: 20.8% / RELD_MTL: 59.8% / RouteFinder: 13.6% |
| R42A | VRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 22.2% / MoSES_RF: 12.0% / RELD_MOEL: 17.2% / RELD_MTL: 41.3% / RouteFinder: 7.3% |
| R42A | OVRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 28.4% / RELD_MTL: 71.6% / RouteFinder: 0.0% |
| R42A | OVRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 57.0% / RELD_MTL: 43.0% / RouteFinder: 0.0% |
| R42A | OVRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 60.1% / RELD_MTL: 39.9% / RouteFinder: 0.0% |
| R42A | VRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 1.9% / MoSES_RF: 0.8% / RELD_MOEL: 23.3% / RELD_MTL: 57.0% / RouteFinder: 17.0% |
| R42A | OVRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 52.6% / RELD_MTL: 47.4% / RouteFinder: 0.0% |
| R43A | TSP | BQ: 59.6% / DIFUSCO: 0.0% / DIFUSCO500: 4.8% / ELG: 13.8% / LEHD: 0.0% / OMNI: 0.0% / T2T: 0.7% / T2T500: 21.1% |
| R43A | CVRP | BQ: 5.8% / ELG: 0.0% / ICAM: 11.9% / LEHD: 6.1% / MVMOE: 0.0% / MoSES_CaDA: 0.6% / MoSES_RF: 10.0% / OMNI: 37.5% / RELD_CVRP: 28.1% / RouteFinder: 0.0% |
| R43A | ATSP | GLOP: 0.0% / ICAM_ATSP: 30.2% / MATNET: 14.6% / MATPOENET: 0.0% / UNICO_MatPOENet: 55.2% |
| R43A | OVRP | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 45.1% / MoSES_RF: 42.4% / RELD_MOEL: 0.2% / RELD_MTL: 0.2% / RouteFinder: 12.1% |
| R43A | VRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 55.4% / RELD_MTL: 44.6% / RouteFinder: 0.0% |
| R43A | VRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 42.9% / MoSES_RF: 45.0% / RELD_MOEL: 2.4% / RELD_MTL: 8.4% / RouteFinder: 1.3% |
| R43A | VRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 22.5% / MoSES_RF: 13.6% / RELD_MOEL: 17.5% / RELD_MTL: 39.6% / RouteFinder: 6.8% |
| R43A | OVRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 61.0% / RELD_MTL: 39.0% / RouteFinder: 0.0% |
| R43A | OVRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 38.2% / RELD_MTL: 61.8% / RouteFinder: 0.0% |
| R43A | OVRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 52.1% / MoSES_RF: 36.0% / RELD_MOEL: 0.0% / RELD_MTL: 0.3% / RouteFinder: 11.6% |
| R43A | VRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 53.3% / RELD_MTL: 46.7% / RouteFinder: 0.0% |
| R43A | VRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 2.0% / MoSES_RF: 5.2% / RELD_MOEL: 19.4% / RELD_MTL: 59.4% / RouteFinder: 14.0% |
| R43A | VRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 21.2% / MoSES_RF: 9.4% / RELD_MOEL: 21.2% / RELD_MTL: 37.9% / RouteFinder: 10.3% |
| R43A | OVRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 30.8% / RELD_MTL: 69.2% / RouteFinder: 0.0% |
| R43A | OVRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 51.4% / RELD_MTL: 48.6% / RouteFinder: 0.0% |
| R43A | OVRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 53.4% / RELD_MTL: 46.6% / RouteFinder: 0.0% |
| R43A | VRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 3.0% / MoSES_RF: 0.8% / RELD_MOEL: 20.1% / RELD_MTL: 58.1% / RouteFinder: 18.0% |
| R43A | OVRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 44.0% / RELD_MTL: 56.0% / RouteFinder: 0.0% |
| R43B | TSP | BQ: 56.2% / DIFUSCO: 0.1% / DIFUSCO500: 7.8% / ELG: 12.4% / LEHD: 0.0% / OMNI: 0.0% / T2T: 1.9% / T2T500: 21.6% |
| R43B | CVRP | BQ: 8.1% / ELG: 0.0% / ICAM: 9.6% / LEHD: 5.5% / MVMOE: 0.3% / MoSES_CaDA: 1.3% / MoSES_RF: 8.4% / OMNI: 39.8% / RELD_CVRP: 27.0% / RouteFinder: 0.0% |
| R43B | ATSP | GLOP: 0.0% / ICAM_ATSP: 28.4% / MATNET: 16.4% / MATPOENET: 0.0% / UNICO_MatPOENet: 55.2% |
| R43B | OVRP | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 43.8% / MoSES_RF: 44.9% / RELD_MOEL: 0.3% / RELD_MTL: 0.1% / RouteFinder: 10.9% |
| R43B | VRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 57.0% / RELD_MTL: 43.0% / RouteFinder: 0.0% |
| R43B | VRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 40.2% / MoSES_RF: 41.4% / RELD_MOEL: 2.4% / RELD_MTL: 13.5% / RouteFinder: 2.5% |
| R43B | VRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 21.5% / MoSES_RF: 16.0% / RELD_MOEL: 13.3% / RELD_MTL: 41.0% / RouteFinder: 8.2% |
| R43B | OVRPTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 73.3% / RELD_MTL: 26.7% / RouteFinder: 0.0% |
| R43B | OVRPB | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 40.1% / RELD_MTL: 59.9% / RouteFinder: 0.0% |
| R43B | OVRPL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 49.4% / MoSES_RF: 39.4% / RELD_MOEL: 0.1% / RELD_MTL: 0.1% / RouteFinder: 11.0% |
| R43B | VRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 57.5% / RELD_MTL: 42.5% / RouteFinder: 0.0% |
| R43B | VRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 3.1% / MoSES_RF: 5.7% / RELD_MOEL: 22.6% / RELD_MTL: 54.4% / RouteFinder: 14.2% |
| R43B | VRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 20.5% / MoSES_RF: 9.3% / RELD_MOEL: 18.6% / RELD_MTL: 39.1% / RouteFinder: 12.5% |
| R43B | OVRPBL | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 39.0% / RELD_MTL: 61.0% / RouteFinder: 0.0% |
| R43B | OVRPBTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 57.2% / RELD_MTL: 42.8% / RouteFinder: 0.0% |
| R43B | OVRPLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 65.3% / RELD_MTL: 34.7% / RouteFinder: 0.0% |
| R43B | VRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 3.4% / MoSES_RF: 1.3% / RELD_MOEL: 21.1% / RELD_MTL: 55.2% / RouteFinder: 19.0% |
| R43B | OVRPBLTW | MTPOMO: 0.0% / MVMOE: 0.0% / MoSES_CaDA: 0.0% / MoSES_RF: 0.0% / RELD_MOEL: 55.1% / RELD_MTL: 44.9% / RouteFinder: 0.0% |
