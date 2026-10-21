# R88 锁定结果短报告

**结论：预定 VAL gate 未通过；未观察到 fixed36 的实质恢复，研究 goal 仍未解决，由主线程控制。** 本报告仅汇总已完成的锁定结果，不作论文论述或下一步研究建议。

## 1. 口径与唯一模型

自动 compare 已于 2026-10-20 13:34:45 CST 完成，回执 complete、exit_status=0。NEW90k 冻结数据版本为 selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018；每题 TRAIN/VAL/TEST 为 81,000/4,500/4,500，沿用全部 18 题合法候选池。winner 为 native/FP64 ind，非 FP32 argmin；SBS 在完整 TRAIN 拟合。ALL 各题等权，mean_cost 仅为跨题描述量，不能作为混合 raw cost 的整体胜负依据。

A 为 TSP 几何 canonical anchor，B 为锁定 BQ/LEHD 部署的物理 node0 execution anchor；仅 TSP 原零值 slot3 改变，其他 17 题输入路径不变，不新增参数或损失。两臂均按完整 VAL macro_vs_sbs_pct 严格最小值选择唯一 checkpoint，精确平局保留先到者，随后固定规则评估 TEST/LIB；未按 TEST 重选或逐题拼接。

锁定点：A update 37,440 / epoch 26，SHA256 前16位 38dd57441afcfe90；B update 38,880 / epoch 27，1e23f0dce7677334；R84A 历史同数据参考为 23fb9a5cf62104513，并非本轮匹配对照。完整 SHA 与推理规则见锁文件。

## 2. ALL 锁定指标

| 划分 | 模型 | Top1 (%) | mean_cost | actual regret (%) | vs_SBS (%) | vs_Oracle (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VAL | R84A | 71.085185 | 11.416611477 | 0.403494295 | -0.194190 | 0.393443 |
| VAL | R88A | 71.058025 | 11.416770631 | 0.405442096 | -0.192104 | 0.395538 |
| VAL | R88B | 71.058025 | 11.416634105 | 0.405127692 | -0.192477 | 0.395188 |
| TEST | R84A | 71.235802 | 11.373843044 | 0.394846788 | -0.197053 | 0.386140 |
| TEST | R88A | 71.251852 | 11.373727552 | 0.392723937 | -0.198439 | 0.384739 |
| TEST | R88B | 71.220988 | 11.373838370 | 0.396338679 | -0.195592 | 0.387648 |

三模型共用：VAL SBS cost=11.427978473、Oracle cost=11.371691902；TEST SBS cost=11.385299246、Oracle cost=11.329932369。Oracle 为逐实例合法池最低成本；vs_SBS、vs_Oracle 先按题计算均值比再等权，actual regret 则先平均逐实例相对 Oracle 的损失再按题等权，二者不可混用。

## 3. 严格预定 VAL Gate

protocol_lock.json 要求 B 对匹配 A 和历史 R84A **同时**满足：Top1 严格增加，mean_cost 不增，actual regret 不增。以下判定使用 CSV 原值及 native 整数计数，不设舍入容差；差值仅作显示舍入。

| VAL 比较 | ΔTop1 (pp) | Δmean_cost | Δregret (pp) | 三条件：Top1 / cost / regret |
| --- | ---: | ---: | ---: | --- |
| B - A | 0（精确） | -0.000136525979 | -0.000314404394 | 否 / 是 / 是：FAIL |
| B - R84A | -0.027160494 | +0.000022627696 | +0.001633397070 | 否 / 否 / 否：FAIL |
| A - R84A | -0.027160494 | +0.000159153675 | +0.001947801465 | 否 / 否 / 否 |

A/B 的锁定 macro_top1 原值均为 0.7105802469135802，且均正确 57,557/81,000；R84A 为 57,579/81,000。B 对 A 在 VAL 纠正/伤害 767/767，净 0；TEST 为 768/793，净 -25。TEST 的 B-A 为 Top1 -0.030864198 pp、mean_cost +0.000110819、regret +0.003614742 pp，不能改变 VAL gate。

<!-- pagebreak -->

# R88 诊断与证据边界

## 4. 固定 36 组 Target 的真实恢复

目标沿用 R84A 锁定 VAL 的 36 组 zero/near-zero 列表（选择比例不超过 0.1%，且 native winner 数大于 0），不重定义。VAL/TEST 真 winner 总数为 4,625/4,598；历史 R84A 分别正确 11/13。rescued/lost/new false 均为相对指定基线的逐实例变化，不是选择次数变化；precision=正确/选择，recall=正确/真 winner。

| 划分 | 模型 / 基线 | 正确 / 选择 / 真winner | rescued | lost | new false | precision (%) | recall (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | A / R84A | 15 / 27 / 4625 | 5 | 1 | 7 | 55.5556 | 0.3243 |
| VAL | B / R84A | 9 / 18 / 4625 | 1 | 3 | 5 | 50.0000 | 0.1946 |
| VAL | B / A | 9 / 18 / 4625 | 0 | 6 | 2 | 50.0000 | 0.1946 |
| TEST | A / R84A | 15 / 33 / 4598 | 3 | 1 | 7 | 45.4545 | 0.3262 |
| TEST | B / R84A | 12 / 21 / 4598 | 1 | 2 | 1 | 57.1429 | 0.2610 |
| TEST | B / A | 12 / 21 / 4598 | 0 | 3 | 1 | 57.1429 | 0.2610 |

B 对 A 在两个留出集都没有新增 rescue；TEST precision 虽上升，选择数与正确数却同时下降，recall 未改善。B 对历史 R84A 各仅 rescue 1 个，lost 分别为 3/2，不构成目标恢复。

## 5. 锁定 Checkpoint 的 TRAIN 支持

以下每题均为 81,000 例；选择数、正确数与真 winner 数分开列示，不把真 winner 频数当模型选择频数。

| TRAIN 题目 / 方法 | 真 winner 数 | A 选择 / 正确 | B 选择 / 正确 |
| --- | ---: | ---: | ---: |
| TSP / LEHD | 14279 | 26 / 9 | 20 / 6 |
| TSP / DIFUSCO | 6806 | 0 / 0 | 0 / 0 |
| TSP / DIFUSCO500 | 873 | 0 / 0 | 0 / 0 |
| ATSP / MATPOENET | 6476 | 22 / 12 | 0 / 0 |
| CVRP / LEHD | 11655 | 2079 / 1004 | 2772 / 1310 |

TSP LEHD recall：A 0.0630%、B 0.0420%；DIFUSCO 两臂均 0%；MATPOENET 为 A 0.1853%、B 0%。零选择的 precision 未定义，不补成 0。TRAIN fixed36：A 正确/选择/真 winner 为 331/622/83,279，B 为 290/496/83,279（B 对 A rescue/lost/new false=46/87/38）。TRAIN macro Top1 A/B 为 71.251578%/71.297325%；有充分 TRAIN 支持的这些方法仍受压制，不能只用未见实例过拟合解释，也不能据此确定唯一根因。

## 6. TSP 局部还是跨任务变化

下表为已有 perproblem CSV 的 B-A 描述性差值；Top1/regret 单位为 pp，cost 为原成本差。仅列 TSP 和主要非 TSP 变化作定位，各题完整指标保留在原 CSV/PDF。

| 题目 | VAL ΔTop1 | TEST ΔTop1 | VAL Δcost | TEST Δcost | VAL Δregret | TEST Δregret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | -0.111111 | +0.022222 | +0.000362197 | -0.000313339 | +0.003613 | -0.003029 |
| CVRP | +0.488889 | +0.244444 | -0.000481319 | +0.000587774 | -0.002779 | +0.011379 |
| ATSP | +0.155556 | -0.400000 | +0.000320593 | +0.000605488 | +0.022281 | +0.039900 |
| OVRP | -0.533333 | -0.644444 | +0.000455456 | +0.000830332 | +0.006565 | +0.010915 |

TSP 的 VAL/TEST 净正确为 -5/+1；其余 17 题为 +5/-26，分别有 17/16 题出现预测变化（2,013/2,066 例）。因此观察到的变化并不局限于 TSP，且方向混合；两臂共享全题训练、锁定点不同，不能将非 TSP 变化或 ALL 微小变化直接归因为角色标记，更不能宣称找到整体平台根因。

**边界与交付：**旧 TEST 已参与此前研发，本轮不宣称全新盲测；LIB 的历史实际 execution role 尚未全面认证，不能据此作机制确认。本次仅读小 CSV 与锁定记录，未新增推理、训练、TEST/LIB 执行、权重加载或 NPZ 重放；未改运行文件/protocol，未启动 review、实验、通知或 Git 操作。

来源：[锁定协议](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/protocol_lock.json)；[A 锁](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/A_canonical_anchor/final_test_lock.json)；[B 锁](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/B_execution_anchor/final_test_lock.json)；[R84A 指标](/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/A_baseline/full_model_statistics/metrics_by_problem.csv)；[A 各题完整指标](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/A_canonical_anchor/full_model_statistics/metrics_by_problem.csv)；[B 各题完整指标](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/B_execution_anchor/full_model_statistics/metrics_by_problem.csv)；[B/A 配对差值](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/comparison_artifacts/paired_changes_B_vs_A.csv)；[B/A fixed36](/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/comparison_artifacts/target_recovery_summary_B_vs_A.csv)。A/B 相对 R84A 的 target_recovery_summary.csv、TRAIN 两表及 method_selection_train_val_test.csv 均取自各臂同一 full_model_statistics 目录；未补缺失值，也未重复生成已有完整分布报告。
