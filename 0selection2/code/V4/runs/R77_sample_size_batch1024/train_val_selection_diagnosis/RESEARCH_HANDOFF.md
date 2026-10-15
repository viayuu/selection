# R78 研究输入：R77 / R76 TRAIN与VAL选择诊断

接收方：Astra，仅用于 papers/ideas 与 goal4 方案讨论。本脚本不创建 goal、不做代码重审、不决定或启动新训练。
**严格隔离：没有读取任何 R77 TEST 预测、TEST metrics、TEST结果或含TEST数字的status文件。** `final_test_lock.json` 仅作为VAL选权重的元数据；本文不含TEST数字。
## 来源状态

| 模型 | 状态 | 实际TRAIN/题 | VAL/题 | selected epoch / update |
| --- | --- | ---: | ---: | --- |
| R76_full_bs9856 | ready | 70,114 | 3,895 | 18 / n/a |
| R77_A_10k_bs1024 | ready | 10,000 | 3,895 | 13 / 16146 |
| R77_B_full_bs1024 | ready | 70,114 | 3,895 | 28 / 34776 |

## 核心数字

| 模型/split | 实例数 | Top1 | 宏CE | 宏actual regret | zero方法-问题对 | zero获胜比例 | near-zero对 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R76_full_bs9856/train | 1,262,052 | 69.3696% | 0.752703 | 0.460650% | 55 | 20.2345% | 1 |
| R76_full_bs9856/val | 70,110 | 69.2369% | 0.755865 | 0.464503% | 54 | 20.3281% | 1 |
| R77_A_10k_bs1024/train | 180,000 | 70.9189% | 0.693358 | 0.405340% | 24 | 1.7772% | 13 |
| R77_A_10k_bs1024/val | 70,110 | 70.1997% | 0.724285 | 0.418613% | 32 | 4.9337% | 5 |
| R77_B_full_bs1024/train | 1,262,052 | 71.1767% | 0.704273 | 0.393012% | 25 | 1.7795% | 13 |
| R77_B_full_bs1024/val | 70,110 | 70.8943% | 0.713642 | 0.401327% | 30 | 3.7769% | 6 |

## 口径与限制

A的TRAIN分母只有实际训练的10,000例/题，不重新推理其余60,114例；B和R76的TRAIN是70,114例/题。跨模型优先比较比率，而不是把winner样本数差当改进。VAL是相同的3,895例/题。
zero为selected_count=0且native truewinner>0；near-zero为0<selected_pct≤0.1%且truewinner>0，固定描述性阈值，不拟合规则。模型选择频数与真实winner频数分别保留。
复用R76 reader：最终maximin logits的masked FP32 softmax，温度1；排名/选法用原logits，并列按global solver ID。native标签不由FP32成本argmin替代。成本保留FP64。
条件概率/排名/CE在本方法是真实winner时计算；另有漏选条件统计及5/25/50/75/95分位数。Δcost=所选成本−native winner成本，mean_costloss_contribution=该方法漏选Δcost总和/本题N，costloss share分母为本题全部损失；不同题原始成本不相加。actual regret是所选相对FP64 Oracle的逐样本百分比均值。
ECE为固定20等宽bins，Brier为多类平方误差和；不拟合温度。R77与R76同时改变batch、update轴LR/预算；A还改变TRAIN样本和几何stats，不能从此表声称单一batch变量的因果改进。
实现角色：Solmax；same-family provisional，不声称独立外部审查。只有锁定/预算一致性与真实row映射两项输入guard，未运行额外tests、prep、GPU、training或profile。
## 完整文件

`zero_methods.csv`：完整零选择但真实获胜的方法-问题对表；`near_zero_methods.csv`与`zero_or_near_zero_classes.csv`：稀少选择类及class calibration。
`method_summary.csv`：全部方法的真实/选择频数、winner条件概率/排名/CE/成本贡献；`missed_candidate_summary.csv`与`missed_selection_frequencies.csv`：漏选候选概率及选法流向。
`summary.csv`、`problem_summary.csv`、`calibration_summary.csv`：整体与逐题CE/Top1/成本regret；`comparison_vs_r76.csv`、`method_comparison_vs_r76.csv`、`problem_comparison_vs_r76.csv`：与oldbs9856的同口径并列表及差值。
`source_readiness.json`与`protocol.json`：准确源路径、checkpoint与输入hash。各模型独立目录保留TRAIN/VAL原表；R76历史输出只读，未覆盖。

## B最终native监督交接

B唯一锁定为epoch28/update34,776；A为epoch13/update16,146。R76、A、B的TRAIN/VAL全部18题已经ready，不含任何R77 TEST数字。Astra请先使用这个完整基础作papers/ideas分析，再讨论goal4；这里不创建或启动目标4训练。

| B split | zero方法-问题对 | 覆盖native winner样本数 | 占全部样本 | near-zero对 | Top1 | CE | actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 25 | 22,458 / 1,262,052 | 1.7795% | 13 | 71.1767% | 0.704273 | 0.393012% |
| val | 30 | 2,648 / 70,110 | 3.7769% | 6 | 70.8943% | 0.713642 | 0.401327% |

TRAIN/VAL共同zero且真实获胜的方法对为23个，任一split为zero的名称对并集为32个。
观察1：B的TSP/LEHD在VAL有641个native winner、模型选0次，条件平均概率20.6230%、平均排名2.8066，贡献本题36.2325%成本损失。零选择不等于没有概率质量。
观察2：ATSP/MATPOENET在B VAL有318个native winner、模型选0次，条件平均概率11.1728%、平均排名2.2516，贡献本题33.5961%成本损失。这只是标签条件观测，不指定修正规则。
`B_train_val_native_supervision.csv`按问题/native方法名与global ID对齐全部候选，含TRAIN/VAL真实winner与模型选择的分别计数、概率分位数、排名、native CE和成本贡献；两种频数不能互换。
`B_train_val_zero_native_names.csv`给出任一split剩余zero的方法名及两侧完整条件统计；`B_zero_methods.csv`保留两个split各自的完整zero行。native标签沿用保存winner，不使用FP32 argmin重标。

### B TRAIN：全部剩余zero真实winner方法

| 问题 | native方法名 | truewinner数/比例 | 模型选择数 | 条件平均P | 条件平均rank | 本题成本损失贡献 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSP | DIFUSCO | 6,084 / 8.6773% | 0 | 10.7542% | 3.6032 | 11.6081% |
| TSP | DIFUSCO500 | 865 / 1.2337% | 0 | 4.2755% | 6.0150 | 1.6504% |
| TSP | OMNI | 404 / 0.5762% | 0 | 2.9493% | 6.6535 | 0.7959% |
| CVRP | RouteFinder | 1,163 / 1.6587% | 0 | 6.7713% | 6.5125 | 2.2021% |
| ATSP | GLOP | 210 / 0.2995% | 0 | 1.5598% | 3.5286 | 1.8588% |
| ATSP | MATPOENET | 5,579 / 7.9570% | 0 | 10.9258% | 2.2276 | 31.7650% |
| OVRP | MTPOMO | 660 / 0.9413% | 0 | 2.3285% | 5.1530 | 1.7680% |
| OVRP | MVMOE | 404 / 0.5762% | 0 | 1.9856% | 5.5446 | 0.9913% |
| VRPL | MTPOMO | 2,314 / 3.3003% | 0 | 6.9066% | 5.3470 | 5.9346% |
| VRPL | MVMOE | 2,283 / 3.2561% | 0 | 7.4986% | 5.3167 | 6.0695% |
| VRPTW | MTPOMO | 4 / 0.0057% | 0 | 0.1211% | 6.0000 | 0.0299% |
| VRPTW | MoSES_RF | 98 / 0.1398% | 0 | 1.5084% | 3.8571 | 1.3062% |
| VRPTW | RouteFinder | 304 / 0.4336% | 0 | 2.5039% | 3.1118 | 3.8042% |
| OVRPTW | MoSES_RF | 58 / 0.0827% | 0 | 1.0137% | 4.1897 | 1.0050% |
| OVRPTW | RouteFinder | 158 / 0.2253% | 0 | 1.4291% | 2.9937 | 2.2460% |
| OVRPL | MTPOMO | 616 / 0.8786% | 0 | 2.0356% | 5.2175 | 1.7644% |
| OVRPL | MVMOE | 430 / 0.6133% | 0 | 1.9370% | 5.5256 | 1.0424% |
| VRPBTW | MTPOMO | 75 / 0.1070% | 0 | 0.3734% | 2.6400 | 0.5336% |
| VRPLTW | MTPOMO | 4 / 0.0057% | 0 | 0.0519% | 6.0000 | 0.0306% |
| VRPLTW | MoSES_RF | 140 / 0.1997% | 0 | 1.0449% | 4.0357 | 1.9126% |
| VRPLTW | RouteFinder | 294 / 0.4193% | 0 | 2.3522% | 3.0850 | 3.6325% |
| OVRPLTW | MoSES_RF | 71 / 0.1013% | 0 | 0.9184% | 4.1972 | 0.9319% |
| OVRPLTW | RouteFinder | 165 / 0.2353% | 0 | 1.9882% | 2.9758 | 2.1685% |
| VRPBLTW | MTPOMO | 72 / 0.1027% | 0 | 0.3271% | 2.5972 | 0.4918% |
| OVRPBLTW | MTPOMO | 3 / 0.0043% | 0 | 0.2522% | 2.6667 | 0.0156% |

### B VAL：全部剩余zero真实winner方法

| 问题 | native方法名 | truewinner数/比例 | 模型选择数 | 条件平均P | 条件平均rank | 本题成本损失贡献 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSP | DIFUSCO | 330 / 8.4724% | 0 | 10.3274% | 3.5727 | 13.8159% |
| TSP | DIFUSCO500 | 44 / 1.1297% | 0 | 4.0891% | 6.2955 | 1.2387% |
| TSP | LEHD | 641 / 16.4570% | 0 | 20.6230% | 2.8066 | 36.2325% |
| TSP | OMNI | 21 / 0.5392% | 0 | 2.5873% | 6.9048 | 0.7266% |
| CVRP | RouteFinder | 84 / 2.1566% | 0 | 7.2027% | 6.4524 | 3.4004% |
| ATSP | GLOP | 11 / 0.2824% | 0 | 1.3049% | 3.5455 | 0.6604% |
| ATSP | MATPOENET | 318 / 8.1643% | 0 | 11.1728% | 2.2516 | 33.5961% |
| OVRP | MTPOMO | 30 / 0.7702% | 0 | 2.3663% | 5.1667 | 1.1686% |
| OVRP | MVMOE | 17 / 0.4365% | 0 | 1.7881% | 5.5882 | 0.6148% |
| VRPL | MTPOMO | 137 / 3.5173% | 0 | 6.9724% | 5.3504 | 6.7177% |
| VRPL | MVMOE | 131 / 3.3633% | 0 | 7.2552% | 5.2672 | 5.6624% |
| VRPTW | MoSES_RF | 2 / 0.0513% | 0 | 2.2823% | 3.0000 | 0.2038% |
| VRPTW | RouteFinder | 16 / 0.4108% | 0 | 1.3600% | 3.2500 | 2.7198% |
| OVRPTW | MoSES_CaDA | 29 / 0.7445% | 0 | 3.1986% | 2.2414 | 6.0265% |
| OVRPTW | MoSES_RF | 5 / 0.1284% | 0 | 0.7681% | 4.0000 | 0.9062% |
| OVRPTW | RouteFinder | 9 / 0.2311% | 0 | 2.2348% | 3.0000 | 3.0860% |
| OVRPB | MTPOMO | 221 / 5.6739% | 0 | 4.5967% | 2.2851 | 48.9586% |
| OVRPB | MVMOE | 234 / 6.0077% | 0 | 7.5911% | 2.2479 | 51.0414% |
| OVRPL | MTPOMO | 41 / 1.0526% | 0 | 2.1282% | 5.2195 | 1.8901% |
| OVRPL | MVMOE | 27 / 0.6932% | 0 | 1.4998% | 5.6296 | 1.1535% |
| VRPBL | MTPOMO | 205 / 5.2632% | 0 | 8.7402% | 2.2146 | 31.9235% |
| VRPBTW | MTPOMO | 4 / 0.1027% | 0 | 0.2072% | 2.5000 | 0.3685% |
| VRPLTW | MTPOMO | 1 / 0.0257% | 0 | 0.0956% | 6.0000 | 0.0138% |
| VRPLTW | MoSES_CaDA | 50 / 1.2837% | 0 | 8.3195% | 2.2000 | 7.4429% |
| VRPLTW | MoSES_RF | 6 / 0.1540% | 0 | 1.1688% | 4.3333 | 0.7140% |
| VRPLTW | RouteFinder | 21 / 0.5392% | 0 | 2.0186% | 3.3333 | 2.5348% |
| OVRPBTW | MTPOMO | 1 / 0.0257% | 0 | 0.1358% | 2.0000 | 0.1034% |
| OVRPLTW | MoSES_RF | 4 / 0.1027% | 0 | 1.8926% | 4.2500 | 1.3683% |
| OVRPLTW | RouteFinder | 5 / 0.1284% | 0 | 1.1591% | 3.0000 | 1.1193% |
| VRPBLTW | MTPOMO | 3 / 0.0770% | 0 | 0.1754% | 2.0000 | 0.3933% |

## 最终manifest

源manifest：`/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/selected_train_val_predictions/manifest.json`，complete=true、input_splits=train/val。`final_train_val_manifest_reference.json`保留两臂确切TRAIN/VAL路径、每题行数及锁定身份。本次交接刷新仅读取已有CSV与manifest，没有重新读NPZ、核验旧R76标签/CE或执行推理/tests/modelaudit。
