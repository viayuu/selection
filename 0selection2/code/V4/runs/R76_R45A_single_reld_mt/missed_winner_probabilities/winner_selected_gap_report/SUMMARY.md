# R76 真实winner与实际selected：成对差距

仅描述已有TEST错例，不调参、不训练、不推理。默认原52个zero-with-winner组；概率未经校准，不是实际胜率。每个selected概率取该例实际选中的方法，同一子集严格配对，不借用其它样本的BQ/T2T等全组均值。

## 先看数字

| 范围 | 错例数 | Δp均值/中位/P90 (pp) | literal成本差均值/中位/P90 (%) | 全18actual regret贡献 (pp) |
| --- | ---: | --- | --- | ---: |
| 全部错例 | 21,426 | 34.131/26.838/82.546 | 1.5188/0.8993/3.5144 | 0.464143599 |
| 原zero组 | 14,077 | 45.362/41.351/86.912 | 1.6032/0.9756/3.6166 | 0.321906120 |

Δp=100×(p_selected−p_nativewinner)，排名距离=winner_rank−1；literal成本差=100×(c_selected−c_nativewinner)/c_nativewinner；actual regret=100×(c_selected−c_oracle)/c_oracle。两种成本口径分别保留，不改native身份。

分母来自problem_summary：18题，各题3,895例，共70,110例。每例贡献=actual_regret_pct/(18×本题全体N)，再加总；**不是**错例actual regret均值，**不是**对各题raw成本求和。各桶错例占比以该scope错例数为分母，贡献及贡献占比仍用全18口径。

## TSP / LEHD：同一批实际漏选

637个LEHD真实获胜错例：native平均p **18.8060%**，实际selected平均p **42.0062%**；Δp均值/中位/P90 **23.2002/22.1640/37.9123pp**。literal成本差均值/中位/P90 **0.877485/0.556792/1.957622%**。

| 实际selected | 次数 | native均p (%) | selected均p (%) | Δp均值 (pp) | literal成本差均值 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BQ | 322 | 22.0007 | 46.9904 | 24.9897 | 0.923853 |
| T2T | 293 | 16.0625 | 37.5327 | 21.4702 | 0.859843 |
| ELG | 22 | 8.5871 | 28.6360 | 20.0489 | 0.433788 |

## 最主要原zero成对错误

| 问题 | native → 实际selected | 次数 | 均p native/selected (%) | Δp均值 (pp) | 成本差均值 (%) | 全18贡献 (pp) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| VRPBLTW | MVMOE → reld_mt | 791 | 24.05/75.85 | 51.80 | 2.2509 | 0.025396 |
| VRPBTW | MVMOE → reld_mt | 819 | 23.17/76.72 | 53.55 | 2.1094 | 0.024641 |
| OVRPBLTW | MVMOE → reld_mt | 644 | 21.87/78.08 | 56.21 | 1.9285 | 0.017714 |
| OVRPBTW | MVMOE → reld_mt | 591 | 21.96/77.99 | 56.03 | 1.9341 | 0.016304 |
| VRPBL | MVMOE → reld_mt | 599 | 18.22/79.61 | 61.39 | 1.6900 | 0.014439 |
| VRPTW | MVMOE → reld_mt | 693 | 22.76/76.77 | 54.01 | 1.4236 | 0.014071 |
| VRPLTW | MVMOE → reld_mt | 726 | 23.53/76.02 | 52.48 | 1.3186 | 0.013654 |
| VRPB | MVMOE → reld_mt | 552 | 16.43/81.13 | 64.71 | 1.6841 | 0.013259 |

## 分桶与排名

概率差固定[0,1,2,5,10,20,40,100]pp，首桶含0，后续右闭；成本差负值、exact=0、(0,.1]、(.1,.5]、(.5,1]、(1,2]、(2,5]、>5%分开。完整桶计数、错例占比及全18贡献见两个bins CSV和离线HTML。

| 范围 | winner排第2 | 第3 | >3 | selected/native成本并列 | native成本非最小 | literal负差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 63.74% | 20.32% | 15.93% | 104 | 0 | 0 |
| zero | 56.11% | 24.95% | 18.95% | 79 | 0 | 0 |

并列仅指该错例selected成本与native成本精确相等；未读取全池成本，不能据此枚举其它候选的并列。异常行保留在cost_exceptions.csv，不改native标签。

## 最少核对与文件

条数21426、zero条数14077及52组与原CSV一致；两个scope的分桶次数/贡献合计一致；仅检查给定TSP index1算例：Δp=16.157674789429pp，排名距离=3，literal成本差=0.200793619605%，actual regret=0.200793619605%。未重复概率/校准分析或扩展来源审计。

- sample_gaps.csv：逐例成对差值、两种成本口径和全体分母贡献。
- by_winner.csv / by_pair.csv：紧凑聚合，zero_group明确区分原zero范围与其余错例。
- scope_summary.csv / probability_gap_bins.csv / cost_gap_bins.csv / cost_exceptions.csv：全局/zero分位数、分桶和成本例外。
- figures/global_top18_zero_pairs.png、figures/TSP_zero_winners.png：优先两图；共18题小图及index.html由finish阶段从聚合CSV生成。
- 输入仅missed_samples.csv、method_summary_test.csv、problem_summary_test.csv；无NPZ、训练、GPU、新infer、Git或goal操作。
