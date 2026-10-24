# full18 平台与零选择：锁定结果的定量分解

整理日期：2026-10-24；沿用任务目录日期 20261023。这是既有结果的 CPU 定量证据，不是新训练或论文综合。

## 最关键的八条实测结论

1. **平台不等于所有题都在 71%。** 统一主模型 R94B 的 TEST 为 **71.5778%**，错 **23,022/81,000**；TSP、CVRP、OVRP、OVRPL、VRPL 五题贡献 **55.6251%** 的错误，却只贡献 **39.8157%** 的 actual regret。应分别解释分类难题和成本损失。[E1](evidence_per_problem_metrics.csv)
2. **固定漏选组重要，但不是平台的大部分。** 严格零 29 组占 R94B 错误 **11.6758%**、regret **11.3265%**；近零 7 组另占 **7.7882% / 9.8850%**。即使固定 36 组漏掉的 **4,481** 例全部无代价救回，Top1 也只到 **77.1099%**。这是答案已知的诊断修复预算，不是可部署收益。[E2](evidence_zero_error_regret_budget.csv)
3. **先验很强，额外实例收益确实存在，但较小且高度不均。** 只用 TRAIN 拟合的 problem×精确N×scale×distribution×capacity_type 众数规则，TEST Top1 为 **69.7198%**；R94B 额外 **+1.8580 pp**，即纠正 **3,789**、伤害 **2,284**、净多对 **1,505**。TSP/CVRP 增益分别 **+9.3778/+15.0889 pp**，VRPL 仅 **+0.0889 pp**。[E3](evidence_train_condition_priors.csv)
4. **“总正例很多”与“局部冠军占优”不是一回事。** TSP 的 **450** 个真实条件组，每组 TRAIN 中位数 **178** 例；LEHD/DIFUSCO 在全部组都有正例，组内 winner 中位数分别 **32/14**，但 TRAIN 众数规则没有一个组选择它们。CVRP 则分成 **900** 组、中位数 **89** 例，RouteFinder 的 TEST winner 中 **58/67** 所在组 TRAIN 正例不足 10 个。稀疏性解释不能在这些目标之间一概而论。[E4](evidence_condition_support.csv)、[E5](evidence_condition_method_support.csv)
5. **大目标既非只在 heldout 漏，也非都差一点点。** R94B 的 TSP/LEHD 在 TRAIN 仅 **15/14,279** 命中，TEST **2/762**；DIFUSCO 分别 **0/6,806、0/334**。TEST winner 的分数 margin 中位数为 **-0.7883/-1.4886**；LEHD 多在 rank2/3，DIFUSCO 的 **226/334** 在 rank4。不能把两者都称为临界 argmax 错误。[E6](evidence_focus_rank_margin_cost.csv)、[E7](evidence_focus_rank_histogram.csv)
6. **平台不能主要归咎于几乎无成本的标签分歧。** R94B 全部错例只有 **116/23,022=0.5039%** 为零成本损失；所选方法高于 Oracle **>1%** 的 **9,820** 例，占错误 **42.6549%**，却贡献 regret 的 **81.5822%**。若用真正最优与次优成本间隔衡量，间隔 >1% 的错例仍有 **7,599** 例。[E8](evidence_error_gap_bins.csv)
7. **有互补，但大量共同错，且纠正与伤害近乎抵消。** 三个近期代表模型共同错 **20,842** 例，占 R94B 错误 **90.5308%**、regret **87.7300%**；R102A 相对 R94B 纠正 **1,114**、伤害 **1,104**，净增仅 **10**。这不是 ensemble 成绩或不可约错误估计。[E9](evidence_paired_test_changes.csv)、[E10](evidence_three_model_error_union.csv)
8. **现有选型冲突的解释预算很小。** 成本锁定点与同轨迹最高 VAL Top1 的差，R84A/R89B/R94B/R102A 分别为 **0 / 0 / 0.02963 / 0.06543 pp**。近期三个模型 selected TRAIN-VAL Top1 差为 **0.4430 / 0.5195 / 0.4767 pp**；不能照搬旧小样本严重过拟合叙事，也不能据此断言没有任务局部过拟合。[E11](evidence_val_checkpoint_objective_gap.csv)、[E1](evidence_per_problem_metrics.csv)

## 口径与边界

- 主表始终用 **同一个 R94B 锁定模型**，不是逐题挑最优。R89B 为历史参照；R102A 为补充强参照，保留 **cached-feature 效能不等于 online 一致性认证** 的边界。它比 R94B TEST 多对 10 例，不据此改选主模型。
- NEW90k 每题 TRAIN/VAL/TEST = **81,000/4,500/4,500**，18 题合计 **1,458,000/81,000/81,000**。18 题等权宏平均；因为各题等样本数，本文错误计数的池化比例恰好等于宏平均贡献。跨题 raw cost 不作为好坏判据。[原汇总](../../../code/V4/runs/R106_native_encoder_adaptation/preparation/completed_context_experiments/RESULTS.md)
- winner 直接读取保存的 **native/FP64 ind**，不重算或替换为 FP32 argmin。Oracle 为合法候选成本的逐实例最小值；合法池沿用 NEW90k 的 B 题删三方法、删 RELD_MTL、单一 reld_mt 规则。SBS 在完整冻结 TRAIN 拟合。[原锁定](../../../code/V4/runs/R94_pretrained_node_fusion/B/final_test_lock.json)
- `actual_regret_pct = 100 * mean_i((C_selected-C_oracle)/C_oracle)`，先题内后等题平均；`vs_SBS = 100*(mean_cost/SBS_cost-1)`、`vs_Oracle = 100*(mean_cost/Oracle_cost-1)` 是均值比，不能混称。本文成本均来自保存 NPZ/CSV，不求解补数。[指标实现](../../../code/V4/r103_conditional_signal.py)
- 冻结 36 组来自 **R84A 锁定 VAL**：29 组当时选择 0 次，另 7 组选择比例大于 0 且不超过 0.1%。这是历史类别，不代表当前模型/TEST 仍选择 0 次。每个实例按它的真实 winner 归组，严格零、近零、其它三类互斥。[原清单](../../../code/V4/runs/R84_new90k_attention/comparison_artifacts/target_list.json)
- 不训练、不调用模型/solver/GPU，不换 checkpoint、不重写标签、不新建划分、不读取在跑版本 partial 结果、不接管任务、不提交。仅补算必要已存预测；没有全量哈希/数组审计。旧 TEST 已参与研发，不能称盲测；无多 seed 显著性结论。

## 1. 哪些题承担剩余错误，哪些题真正超过先验？

全部为 R94B TEST，每行分母 **4,500**；错误份额分母 **23,022**。regret 单位 %。条件先验严格来自 TRAIN：按真实 metadata 完整条件统计 native winner 众数，计数并列按稳定 global solver ID；未调平滑、阈值或分箱，未用 VAL/TEST 标签拟合。所有 heldout 条件组均在 TRAIN 出现，无 fallback 实际触发。

| 问题 | Top1 % | actual regret % | 错误数 | 全18错误份额 % | TRAIN条件先验 Top1 % | 模型净增益 pp |
|---|---:|---:|---:|---:|---:|---:|
| TSP | 47.0222 | 0.407885 | 2,384 | 10.3553 | 37.6444 | +9.3778 |
| CVRP | 48.4667 | 0.668543 | 2,319 | 10.0730 | 33.3778 | +15.0889 |
| ATSP | 77.1778 | 0.354722 | 1,027 | 4.4610 | 76.1556 | +1.0222 |
| OVRP | 40.4667 | 0.620311 | 2,679 | 11.6367 | 38.5111 | +1.9556 |
| VRPB | 81.0667 | 0.354383 | 852 | 3.7008 | 80.4000 | +0.6667 |
| VRPL | 37.6667 | 0.400175 | 2,805 | 12.1840 | 37.5778 | +0.0889 |
| VRPTW | 81.3556 | 0.257739 | 839 | 3.6443 | 80.8444 | +0.5111 |
| OVRPTW | 84.0889 | 0.248191 | 716 | 3.1101 | 83.9333 | +0.1556 |
| OVRPB | 88.3333 | 0.257984 | 525 | 2.2804 | 88.1333 | +0.2000 |
| OVRPL | 41.8000 | 0.620141 | 2,619 | 11.3761 | 40.6667 | +1.1333 |
| VRPBL | 80.4222 | 0.368104 | 881 | 3.8268 | 79.4222 | +1.0000 |
| VRPBTW | 79.1111 | 0.425437 | 940 | 4.0831 | 78.7556 | +0.3556 |
| VRPLTW | 79.4889 | 0.312011 | 923 | 4.0092 | 79.2444 | +0.2444 |
| OVRPBL | 89.5111 | 0.242431 | 472 | 2.0502 | 89.2889 | +0.2222 |
| OVRPBTW | 85.0889 | 0.284473 | 671 | 2.9146 | 84.8222 | +0.2667 |
| OVRPLTW | 83.2667 | 0.276826 | 753 | 3.2708 | 82.8667 | +0.4000 |
| VRPBLTW | 80.2667 | 0.419068 | 888 | 3.8572 | 79.7333 | +0.5333 |
| OVRPBLTW | 83.8000 | 0.305655 | 729 | 3.1665 | 83.5778 | +0.2222 |
| **ALL，等题权重** | **71.5778** | **0.379115** | **23,022** | **100** | **69.7198** | **+1.8580** |

来源：[E1](evidence_per_problem_metrics.csv)，筛选 `model_id=R94B,split=TEST`；[E3](evidence_train_condition_priors.csv)，`prior_rule=problem_N_scale_distribution_capacity_TRAIN_mode`。E1 另存 TRAIN、VAL、TEST 的 mean_cost、SBS、Oracle、vs_SBS、vs_Oracle、regret；不靠 ALL raw cost 掩盖各题。

**先验分解，而非把高 Top1 全称实例识别。** R94B TEST 对 problem-only、problem×精确N、完整条件 TRAIN 众数的净增益分别为 **4.8914 / 2.3988 / 1.8580 pp**；对应先验 Top1 为 **66.6864 / 69.1790 / 69.7198%**。完整条件先验 actual regret **0.462566%**，模型为 **0.379115%**，减少 **0.083451 pp**。OVRPTW、OVRPLTW 虽有 Top1 增益，相对该先验 regret 反而增加 **0.009669 / 0.006694 pp**。[E3](evidence_train_condition_priors.csv)

**不要混同两种诊断。** 既有 R103 的 R94B VAL 同题同精确N保频次置换，使 Top1 **71.4284→67.7134%**，差 **3.7150 pp**；TSP/CVRP/VRPL 分别差 **9.3744/24.1311/4.5167 pp**。它保持模型选择频次，但没有控制 distribution/capacity；这里的 TRAIN 众数基线不同，完整条件 VAL 净增益仅 **1.8840 pp**。两者共同反对“完全没有实例信息”，不能相减成结构因果效应，也不能把经验先验差额全归因于细粒度结构，模型还可能更好地平滑估计条件频率。[R103原表](evidence_r103_permutation.csv)

## 2. 零与近零最多解释多少平台？

下表只含 TEST，18 题共 81,000 例。组内漏选数除以模型全部错误得到错误份额；该组 winner 实例的归一化成本损失除以模型全部归一化损失得到 regret 份额。**不把目标误选的成本再次叠加到 winner 分组预算，避免重复计数。**

| 模型 | 历史类别 | winner数 | 正确 / 漏选 | 误选目标次数 | 全部错误份额 % | regret份额 % | 完美修复最多增益 pp |
|---|---|---:|---:|---:|---:|---:|---:|
| R89B | 严格零29 | 2,731 | 15 / 2,716 | 16 | 11.7322 | 11.7380 | 3.3531 |
| R89B | 近零7 | 1,867 | 30 / 1,837 | 34 | 7.9352 | 10.6961 | 2.2679 |
| R94B | 严格零29 | 2,731 | 43 / 2,688 | 41 | 11.6758 | 11.3265 | 3.3185 |
| R94B | 近零7 | 1,867 | 74 / 1,793 | 72 | 7.7882 | 9.8850 | 2.2136 |
| R102A | 严格零29 | 2,731 | 69 / 2,662 | 59 | 11.5679 | 10.6809 | 3.2864 |
| R102A | 近零7 | 1,867 | 66 / 1,801 | 58 | 7.8264 | 9.9729 | 2.2235 |

来源：[E2](evidence_zero_error_regret_budget.csv)，`problem=ALL,split=TEST`。R94B 两类合计 regret 贡献 **0.080416 pp**，其余候选 winner 实例贡献 **0.298699 pp**，即 **78.7885%**；其余错误 **18,541** 例，即 **80.5360%**。固定 36 组同时完美修复后的 Top1/actual regret 为 **77.1099% / 0.298699%**；R102A 对应 **77.1000% / 0.300280%**。这是“只修复该组且不伤害其它实例”的诊断预算，既非可部署上限，也非总体学习上限。

## 3. 大目标到底离 argmax 多远，损失有多大？

`margin = 保存的该方法decision logit - 其它合法方法最高decision logit`。这里只在同模型、同题解释它；它不是校准概率、真实后验间隔或成本 gap。排序按保存分数和稳定 global-ID tie 规则。下表均为 R94B TEST，分母为该方法 native winner 数；成本差均以同一行 FP64 Oracle 为分母。

| 目标 | 命中/winner | winner平均rank | winner margin P05 / P50 / P95 | winner所选成本超额均值 / 中位数 % | 所选成本减winner成本均值 |
|---|---:|---:|---:|---:|---:|
| TSP / LEHD | 2/762 | 2.7717 | -1.9783 / -0.7883 / -0.3244 | 0.8613 / 0.5792 | 0.075833 |
| TSP / DIFUSCO | 0/334 | 3.6707 | -2.6058 / -1.4886 / -0.8066 | 0.5487 / 0.2864 | 0.042217 |
| CVRP / RouteFinder | 0/67 | 6.3134 | -3.8136 / -1.1464 / -0.5493 | 1.2237 / 0.9250 | 0.132821 |
| CVRP / LEHD | 106/611 | 2.6285 | -2.8299 / -0.5971 / +0.5733 | 1.3779 / 0.9769 | 0.175853 |
| ATSP / MATPOENET | 0/370 | 2.2541 | -3.1858 / -2.3656 / -0.6872 | 1.4330 / 0.9341 | 0.023224 |
| VRPL / MVMOE | 0/155 | 5.2581 | -3.2646 / -1.2973 / -0.6312 | 0.7634 / 0.6471 | 0.099211 |
| VRPL / MTPOMO | 0/149 | 5.2349 | -3.4403 / -1.4893 / -0.6866 | 0.6685 / 0.5239 | 0.080558 |

来源：[E6](evidence_focus_rank_margin_cost.csv)。最后一列是题内 raw cost 差，不跨题相加比较。这里 selected-winner 差等于 selected-Oracle 差；native 并列赢家仍按固定标签判 Top1。

**有同方法区分，不等于能赢过其它方法。** LEHD/DIFUSCO 的 winner-vs-nonwinner margin AUC 为 **0.6709/0.6123**；winner margin 四分位范围分别 **[-1.0742,-0.5632] / [-1.8441,-1.2739]**，nonwinner 为 **[-1.7621,-0.7178] / [-2.2688,-1.3985]**，明显重叠。CVRP/RouteFinder AUC 可达 **0.8714**，但仍无一命中、平均 rank **6.3134**。AUC 只是已存分数的关联区分量，不是冠军可辨认性或可修复比例证明。[E6](evidence_focus_rank_margin_cost.csv)

**名次结构也不一样。** TSP/LEHD 的 rank1/2/3/≥4 为 **2/342/325/93**；DIFUSCO 为 **0/39/51/244**，其中 rank4 **226**。不能仅用“把第二名推上去”解释全部漏选，也不能把 Top2/Top3 覆盖直接当部署收益。[E7](evidence_focus_rank_histogram.csv)

### 小 gap 与大 gap 分开看

“所选超额”是 `(selected-Oracle)/Oracle`；“最优次优间隔”是 `(second-lowest cost-lowest cost)/lowest cost`。后者才描述成本层面的 winner 竞争，前者描述已经发生的决策损失。下表按 **错例** 分母，精确零与正小 gap 分开。

| 错误集合 | 错误数 | 所选零超额 | 所选(0,0.1%] | 所选(0.1%,0.5%] | 所选(0.5%,1%] | 所选>1% |
|---|---:|---:|---:|---:|---:|---:|
| 全18 | 23,022 | 116 | 1,843 | 6,016 | 5,227 | 9,820 |
| TSP/LEHD winner | 760 | 0 | 91 | 253 | 190 | 226 |
| TSP/DIFUSCO winner | 334 | 25 | 68 | 119 | 65 | 57 |

来源：[E8](evidence_error_gap_bins.csv)，`model=R94B,split=TEST,gap_kind=selected_excess_pct`。全18所选 >1% 的错误贡献 actual regret **0.309291 pp / 0.379115%=81.5822%**；≤0.1%（含精确零）的错误合计 **1,959/23,022=8.5093%**，只贡献 regret **0.2897%**。不支持“平台主要是没有实际代价的近并列”。

按最优次优 gap 重新分类，全18 **>0.5%** 的错例 **12,453/23,022=54.0917%**，其中 **>1%** 为 **7,599/23,022=33.0076%**；LEHD 错例的 gap >0.5% 有 **274/760=36.0526%**，DIFUSCO 为 **70/334=20.9581%**。DIFUSCO 有 **58** 个错例的最优次优成本精确并列，但只有 **25** 个错例所选成本也是 Oracle；不能用前者替代“零 regret 错误”。这些都不等于认定标签随机或噪声。[E8](evidence_error_gap_bins.csv)，`gap_kind=oracle_runnerup_gap_pct`

## 4. 条件空间是否使 81k 变稀疏？

只用现成 `input_metadata.npz` 的 scale/distribution/capacity_type 与已保存真实 nodes，并核对 `release_source_indices == metadata.source_index`；**未从索引猜分布**。坐标分布名称仅照录 metadata，未重新认证生成器。CVRP/MVRP 的真实节点数含 depot，scale 与 nodes 没有静默混用。完整逐条件 TRAIN/VAL/TEST 样本数与所有候选 winner 计数见 [E12](evidence_condition_cells.csv)，未生成新 split。

| 问题范围 | 每题TRAIN条件组数 | 每组样本 min / median / mean / max | metadata分布与容量类型 | VAL/TEST 未覆盖实例 |
|---|---:|---:|---|---:|
| TSP | 450 | 89 / 178 / 180 / 292 | Gaussian；none | 0/4,500；0/4,500 |
| CVRP | 900 | 16 / 89 / 90 / 196 | Gaussian；random_triangular、scale | 0/4,500；0/4,500 |
| ATSP | 81 | 996 / 997 / 1,000 / 1,005 | matrix_closure；none | 0/4,500；0/4,500 |
| 其它15题，每题 | 204 | 396 / 397 / 397.0588 / 404 | clustered、gaussian_mixture、ring、uniform；mvrp_default | 各0/4,500；各0/4,500 |

来源：[E4](evidence_condition_support.csv)。全18共 **4,491** 个 TRAIN 条件单元，VAL/TEST 在这些粗条件上全部有训练覆盖，不代表在更细几何/需求结构上覆盖充分。既有 R103 只按精确N：TSP/CVRP 各 **450** 组，ATSP **81** 组，其它每题 **51** 组；不要把 R103 的组数当完整条件组数。[R103分组](evidence_r103_group_coverage.csv)

| 方法 | TRAIN总winner | 有正例条件组/全部组 | 每组winner中位数（含零） | TRAIN先验选该方法的组数 | TEST winner位于TRAIN正例<10的组 |
|---|---:|---:|---:|---:|---:|
| TSP/LEHD | 14,279 | 450/450 | 32 | 0 | 6/762，0.7874% |
| TSP/DIFUSCO | 6,806 | 450/450 | 14 | 0 | 47/334，14.0719% |
| CVRP/LEHD | 11,655 | 863/900 | 12 | 81 | 82/611，13.4206% |
| CVRP/RouteFinder | 1,354 | 323/900 | 0 | 0 | 58/67，86.5672% |
| CVRP/OMNI | 19,683 | 890/900 | 14 | 321 | 101/1,114，9.0664% |

来源：[E5](evidence_condition_method_support.csv)。先验使用固定global-ID并列规则；它选谁不等于真实后验谁必占优。CVRP/RouteFinder 的正例条件组内 winner 中位数为 **3**，TEST winner 有 **9/67** 位于 TRAIN 对该方法零正例组；TSP 两个目标均 **0** 个 TEST winner 位于 TRAIN 零正例组。因而 **粗条件缺正例可能限制 RouteFinder，但不能统一解释 LEHD/DIFUSCO**；更细条件的有效样本量仍未知。

## 5. TRAIN 也漏吗？成本锁点错过多少 VAL Top1？

全部 TRAIN 均使用该版本最终 selected checkpoint 的已保存统计，非末步、非优化过程 mini-batch accuracy。每题 TRAIN 81k、VAL/TEST 各4.5k，ALL 等题权重。

| 模型 | TRAIN Top1 % | VAL Top1 % | TEST Top1 % | TRAIN-VAL pp | VAL actual regret % | TEST actual regret % |
|---|---:|---:|---:|---:|---:|---:|
| R89B | 71.6912 | 71.2481 | 71.4198 | 0.4430 | 0.400676 | 0.389646 |
| R94B | 71.9479 | 71.4284 | 71.5778 | 0.5195 | 0.399079 | 0.379115 |
| R102A | 71.7780 | 71.3012 | 71.5901 | 0.4767 | 0.396175 | 0.378443 |

来源：[E1](evidence_per_problem_metrics.csv)。R94B 的 TSP/LEHD **TRAIN 15/14,279、VAL 2/792、TEST 2/762**；DIFUSCO **0/6,806、0/373、0/334**。LEHD TRAIN/VAL/TEST winner margin 中位数 **-0.7735/-0.7567/-0.7883**；DIFUSCO **-1.5029/-1.5673/-1.4886**，不是只在 heldout 才整体跨不过决策边界。[E6](evidence_focus_rank_margin_cost.csv)

仅描述各轨迹已有 **40** 个 VAL 点，不保存/替换最高 Top1 checkpoint，不读它的新 TEST 结果：

| 模型 | 既定成本锁定步 | 轨迹最高Top1步 | 锁点VAL Top1 % | 轨迹最高VAL Top1 % | 差 pp | 最高Top1点相对锁点actual regret增加 pp |
|---|---:|---:|---:|---:|---:|---:|
| R84A | 37,440 | 37,440 | 71.085185 | 71.085185 | 0 | 0 |
| R89B | 36,000 | 36,000 | 71.248148 | 71.248148 | 0 | 0 |
| R94B | 54,720 | 53,280 | 71.428395 | 71.458025 | 0.029630 | 0.003402 |
| R102A | 33,120 | 44,640 | 71.301235 | 71.366667 | 0.065432 | 0.009644 |

来源：[E11](evidence_val_checkpoint_objective_gap.csv)，原 `metrics.csv` 与 `final_test_lock.json`。既定锁定指标是 **macro_vs_sbs_pct 最小**，不是 actual regret 最小，更不是 Top1 最大；原始各点 vs_SBS 和 actual regret 分列保留。这里最多只发现 **0.065432 pp** 的已观测轨迹选型损失，不能把它外推成“更换训练目标也只可能提高这么多”。最高 VAL Top1 本身是轨迹事后描述，不是重新授权的部署指标。

**旧 R77 单列。** OLD77904/NEW70114 每题 TRAIN/VAL/TEST 为 **70,114/3,895/3,895**，不进入上述排名与合并预算。其 TSP/LEHD TRAIN 为 **1/11,881**、TEST **0/637**；DIFUSCO TRAIN **0/6,084**、TEST **0/303**。这只说明类似现象在旧发布版也出现，不能跨数据归因当前模型优劣。[旧数据单列](evidence_old_r77_separate.csv)

## 6. 共同错还是互补？纠正是否伴随伤害？

以下 TEST 配对只用相同实例与候选顺序的既存锁定预测，分母都是 81,000；不是重训对照。

| 候选相对参考 | 改变选择 | 纠正 | 伤害 | 净多对 | actual regret差 pp |
|---|---:|---:|---:|---:|---:|
| R94B vs R89B | 4,869 | 1,756 | 1,628 | +128 | -0.010531 |
| R102A vs R94B | 3,198 | 1,114 | 1,104 | +10 | -0.000673 |
| R102A vs R89B | 4,771 | 1,723 | 1,585 | +138 | -0.011203 |

来源：[E9](evidence_paired_test_changes.csv)。R94B 相对 R89B：纠正贡献 regret **-0.045469 pp**，伤害贡献 **+0.034684 pp**，改变选择但双方都错贡献 **+0.000255 pp**，净收益来自抵消后的余量。R102A 相对 R94B：纠正 **-0.023949 pp**、伤害 **+0.023745 pp**、双方错但改变选择 **-0.000469 pp**。逐题纠正/伤害的 score gap 分位数保存在 [配对margin](evidence_paired_test_margin.csv)，不将不同题的 logit 尺度混成校准置信度。

**近期三模型 TEST 集合。** 任一模型正确 **60,158/81,000=74.2691%**，共同错误 **20,842**，占 R94B 错误 **90.5308%**、regret **87.7300%**；固定36的 winner 中仅 **179/4,598** 被任一模型命中，**4,419** 被全部漏掉。知道答案后才能从各模型挑中正确者，这个集合覆盖率不是可部署 ensemble，更不是准确率上限。[E10](evidence_three_model_error_union.csv)

**旧21模型 VAL 集合单列。** 复用 R96，不再读取其21套数组：**17,332/23,289=74.4214%** 的 R89B 错误为共同错，贡献其 VAL regret **75.9043%**；固定36任一命中 **146/4,625**，全漏 **4,479**。R96 建表时未纳入后来的 R94/R95 等，且这是 VAL；不能与上面的三模型 TEST 数字作时间进步比较。[R96原表](evidence_r96_common_errors.csv)

**正对照仍成立但不能泛化。** 已有 R94 匹配 B-vs-A TEST 纠正 **1,699**、伤害 **1,466**、净 **233**；它和本表 R94B-vs-R89B 的参考不同。不能把本表历史比较当预训练单因素效应，也不能把有限净收益总结成“预训练完全无用”。[原匹配结果](evidence_recent_matched_ab.csv)

## 7. 观察、排除与仍未知

| 已观察 | 可排除或限制的简单解释 | 仍然未知 |
|---|---|---|
| 固定36仅约五分之一错误和regret；其它winner贡献大头 | “只要救零选择，平台就基本消失” | 不增加误选的可恢复比例；其它错误的机制是否相同 |
| LEHD/DIFUSCO粗条件都有TRAIN正例，但先验不选；TRAIN也持续漏 | “仅仅总正例不够”或“仅未见样本过拟合”作为统一主因 | 更细优势区域是否稀疏；现有表示、优化、读出各自责任 |
| 实例关联和TRAIN条件先验外净收益非零 | “完全只是题型/规模记忆、没有实例对应信息” | 超出完整条件的可泛化结构信息还有多少；先验估计噪声的份额 |
| 大成本间隔错例很多，零成本错例很少 | “主要只是native tie或无实际代价的小gap” | 隐藏执行状态、标签生成稳定性、任务语义是否有未审计因素 |
| 同方法margin有AUC但常远低于其它候选；纠正和伤害抵消 | “同方法排序可用，就已能安全跨方法argmax” | 可靠地识别净纠正区域需要什么信息；margin与成本差的因果关系 |
| 共错集中、选型差小 | “简单选另一个历史模型/另一个现存VAL点即可大幅突破” | 更换信息/表示/目标后的可达精度；Bayes风险与不可约噪声均未测定 |

**最大信息价值的科学问题：** 在固定 problem×精确N×distribution×capacity_type 的条件组内部，允许输入中是否存在一种能跨冻结划分稳定区分“LEHD/DIFUSCO 等目标真正胜出、且成本优势非微小”与“外观相近但目标会输”的信息，使新增正确纠正显著超过新增伤害？如果有，现有保存分数没有保留它，还是保留后跨方法比较失效？现有证据确认这个问题值得区分，**尚未给出答案**；具体新方向由主线程完成。

## 证据与复核

- [CPU分析脚本](evidence_analysis.py)：只导入 numpy/pandas 等分析库，无模型模块；所有输出限定本目录。读取三个代表版本锁定 VAL/TEST，TRAIN 只取条件统计必要字段与少数目标题的已存预测，不读取其它 checkpoint 数组。
- [来源清单](evidence_sources.csv)、[机器可读口径与锁定身份](evidence_manifest.json)：原 CSV/metadata/NPZ 路径、大小、mtime，较小文本的 SHA256；大型 NPZ 未作全量哈希审计。保存现有 checkpoint 哈希只用于引用，未打开权重。
- 必要核对已通过：使用到的预测 Top1/actual regret 与原完整报告一致；metadata 与预测的 release-source 索引对齐；36/29组分母及三模型正确/误选计数复现原报告；分析前后来源 stat 未改变。无成本的材料不补求解，字段缺失仍按 missing，不能把缺失当 0。
- 本文所有推论限于现有发布版、已锁定单seed模型、已参与研发的 heldout；不声称全面工程审计、online 计时认证、独立盲测、不可约噪声估计或理论精度上限。
