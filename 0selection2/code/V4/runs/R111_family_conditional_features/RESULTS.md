# R111 结果收束与发布准备

状态：**实验全链 completed rc0；Git 仅准备，尚未 stage/commit/push。**
完成时间：2026-10-25 14:27:59（Asia/Shanghai，06:27:59 UTC）。本次仅读取既有结果与源码，没有训练、TEST、LIB、计时、模型加载或预测重放。

## 1. 完成了什么，效果如何？

单组 `A_family_conditional`，完整18题，seed2、逻辑 batch1024、57,600次成功更新。唯一 checkpoint 为 **update41760 / epoch29**，按严格最低完整 VAL `macro_vs_sbs_pct` 选择，精确并列保留先到者；不是按 TEST、Top1 或 actual regret 选型。TRAIN/VAL/TEST/LIB 使用同一 checkpoint，没有逐题拼接。

`PLAN.md` 保留启动前冻结原文，其中“尚未启动”为历史状态；当前状态以本报告和完成回执为准，不改写已冻结 PLAN/训练代码。

| R111 ALL | 实例数 | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | 81,000 | 71.113580 | 11.417042545 | 11.427978473 | 11.371691902 | -0.190972 | 0.396696 | 0.407969 |
| TEST | 81,000 | 71.330864 | 11.374659020 | 11.385299246 | 11.329932369 | -0.189515 | 0.393731 | 0.403318 |

口径：冻结 `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`，每题81,000 TRAIN / 4,500 VAL / 4,500 TEST；保留八个带B问题的候选删除及单一 `reld_mt`，不恢复候选、不合并两种 ReLD。Top1 使用 native/FP64 `ind`，不是 FP32 成本 argmin。SBS 在完整新81,000 TRAIN逐题拟合，再在相应划分上计算成本；Oracle 为逐实例合法候选池最低成本。

每题 `vs_SBS = 100*(mean_cost/SBS cost-1)`、`vs_Oracle = 100*(mean_cost/Oracle cost-1)`；actual regret 为逐实例 `100*(selected_cost/oracle_cost-1)` 的均值。ALL 对18题等权，先算逐题比值再平均；不能拿 ALL 三个 raw cost 再相除冒充这些指标。ALL raw cost 仅作描述，不是跨题统一量纲性能。

完整 TRAIN/VAL/TEST 的逐题七项指标以及两 LIB 已在 `A_family_conditional/full_model_statistics/metrics_by_problem.csv`（59行）和既有 `MODEL_STATISTICS.pdf` 中，本次不重复评估。

## 2. 与同数据历史版本相比？

下表均为各版本原先按 VAL 锁定的单一完整模型；R110 A/B 分开列出，不按数据集择优拼接。已小范围核对三份历史 protocol：数据路径、划分行数、候选池、别名、winner 与 TRAIN-SBS 均与 R111 一致。不是新跑的参数量/FLOP/执行路径匹配对照，不能作因果或显著性结论。旧 TEST 已参与研发，不称全新盲测。

| 模型 | checkpoint update | 划分 | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| R94B | 54720 | VAL | 71.428395 | 11.416278963 | -0.198659 | 0.388932 | 0.399079 |
| R94B | 54720 | TEST | 71.577778 | 11.372233605 | -0.211169 | 0.371939 | 0.379115 |
| R110 A_tied_ce | 43200 | VAL | 71.103704 | 11.416779039 | -0.192984 | 0.394679 | 0.405093 |
| R110 A_tied_ce | 43200 | TEST | 71.307407 | 11.374706635 | -0.186717 | 0.396557 | 0.406464 |
| R110 B_private_ce | 21600 | VAL | 70.896296 | 11.417566031 | -0.184524 | 0.403215 | 0.412389 |
| R110 B_private_ce | 21600 | TEST | 71.233333 | 11.374359128 | -0.192514 | 0.390729 | 0.399449 |

结论：**R111 全链完成，但没有一致改善。** 相对 R94B，VAL/TEST Top1 分别下降0.314815/0.246914个百分点，actual regret 上升0.008890/0.024202个百分点。相对 R110 A，TEST Top1 增0.023457个百分点、regret降0.003147个百分点，但 VAL 成本指标略差；相对 R110 B，TEST Top1 增0.097531个百分点，regret反而升0.003869个百分点。单seed、小幅混合变化不构成优越性证据。

历史数字源：`../R94_pretrained_node_fusion/B/final_metrics.json`、`../R110_method_conditioned_ce/{A_tied_ce,B_private_ce}/final_metrics.json`；R111 已有 `historical_R110_summary.csv` 与这些记录一致。各划分 SBS/Oracle cost 与上表 R111 相同。

## 3. 零选择是真的救回，还是带来伤害？

固定目标来自原 R84A 锁定 VAL：选择率<=0.1%且真实 winner 非零，共36个“问题/方法”组，其中原严格零选择29组、近零7组；不根据 R111 TEST 重选目标。“零”是原 VAL 分类，不表示在所有模型/划分上都零选择。以下直接汇总既有 CSV，不读取预测数组。

| 目标范围 | 相对基线 | 划分 | 真救回 | 失去原有正确选择 | 净救回 | 新增误选 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 固定36组 | R84A | VAL | 23 | 3 | +20 | 21 |
| 固定36组 | R84A | TEST | 14 | 4 | +10 | 29 |
| 固定36组 | R94B | VAL | 11 | 106 | -95 | 11 |
| 固定36组 | R94B | TEST | 5 | 99 | -94 | 23 |
| 其中原严格零29组 | R84A | VAL | 7 | 0 | +7 | 7 |
| 其中原严格零29组 | R84A | TEST | 7 | 0 | +7 | 13 |
| 其中原严格零29组 | R94B | VAL | 4 | 41 | -37 | 4 |
| 其中原严格零29组 | R94B | TEST | 3 | 38 | -35 | 8 |

真救回指目标真实 winner 原来未选对、现在选对；“失去”指相反转变；新增误选是新选了目标方法但它不是真实 winner，不等同于从正确变错误，三列不能直接相加当净准确率。

R111 TEST 固定36组实际选61次，正确23次、错误38次，目标 winner 共4,598例，recall仅0.500217%；不能把61次选择说成61次救回。当前 VAL/TEST 仍有25/21个“有真实 winner 但零选择”的组；这些是当期全候选统计，不是固定36组的成功数量。

全人群对 R94B：VAL 真纠正1,596、伤害1,851，净-255；TEST 真纠正1,582、伤害1,782，净-200，TEST ALL raw mean_cost 增0.002425415。扣除固定36组 winner 人群，非目标人群 VAL 净-160、TEST 净-106；不存在仅靠目标净救回来解释整体提升。对 R84A TEST 虽固定目标净+10，整体 raw mean_cost 仍增0.000815976、actual regret增0.008471个百分点。**零选择组数下降不是效果改善证据。**

数据源：`target_recovery_summary_vs_R84A.csv`、`target_recovery_summary_vs_R94B.csv`、逐组 `target_recovery_vs_*.csv`、全人群 `paired_changes_vs_*.csv`、`zero_selection_summary.csv`，均位于 `A_family_conditional/full_model_statistics/`。R110 没有现成配对救回/伤害 CSV，本次不补算；不据汇总指标虚构配对人数。

## 4. 已有交付与完成证据

主要 PDF（均为既有文件，本次不重绘）：

- `A_family_conditional/full_model_statistics/MODEL_STATISTICS.pdf`：完整模型指标，22页。
- `A_family_conditional/full_model_statistics/selection_distributions.pdf`：18题同轴模型/真实winner分布；TSP、CVRP各含 TEST、LIB。
- `A_family_conditional/full_model_statistics/target_recovery_and_train.pdf`：固定目标、配对伤害、历史参照。
- `A_family_conditional/full_model_statistics/training_and_method_diagnostics.pdf`、`learning_curves.pdf`：既有诊断与学习曲线。

`nss_comparison.csv` 已有同一 R111 checkpoint 对历史 R79 NSS 的四数据集对照（不是 NEW90k 重训 NSS）：

| 数据集 | n | R111 Top1 (%) | NSS Top1 (%) | R111 mean_cost | NSS mean_cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP TEST | 4500 | 46.977778 | 45.888889 | 7.972805443 | 7.970400719 |
| CVRP TEST | 4500 | 47.911111 | 45.977778 | 18.016056388 | 18.004802368 |
| TSPLIB | 49 | 30.612245 | 20.408163 | 8.202366645 | 8.217907074 |
| CVRPLIB | 100 | 60.000000 | 70.000000 | 67.282793598 | 67.101111717 |

完成证据：`preparation/dispatch_recovery_01/queue.json` 为 complete/exit_status0（allocation_pipeline、report 均rc0），`worker.json` 的 train/finalize/timing/lib 均rc0，`entry.json` 同为rc0；最终 `A_family_conditional/postprocess_status.json` 为 complete=true。训练、TEST、LIB 分别另有 `training_complete.json`、`test_complete.json`、`lib_stage_complete.json`。本管线使用 postprocess 状态，不要求另造 `report_complete.json`。

checkpoint SHA256：`2c6011ea4cc80373810f1435c07d38d703665dcd1ca262ef99b72b89f6245d9f`；本次引用既有锁与各阶段绑定，未重新读取权重或数组来冒充新审计。首次启动在缓存行政元数据比较时失败、未进入训练；原失败回执和 `preparation/recovery_01/RECOVERY.md` 保留，与后续 completed 链分开。

## 5. Git 交给 main 统一锁执行

`publication_manifest.json` 包含 Git根目录 `/public/home/shiys` 下的精确 `stage_paths`、完整保守源码闭包、SHA256、是否已在 HEAD、依赖角色、小报告与完成证据。准备基线 HEAD 为 `370e0b49a6783036ba0899fdbc995f3d2230a27f`（R110）；不是 R111 已发布的声明。源码闭包包含原有已提交依赖，实际 allowlist 只纳入尚未提交/有差异的必要源码及本版小报告。

本次清单：99个源码文件（88个已在HEAD）；待提交66个精确路径，即11个必要源码、54个本版小报告/证据及manifest，文件正文约1.5MB。11个源码为 `r109_experiment.py`、`r109_reporting.py`、R111的 experiment/launch/model/reporting/publication 五个模块、`run_v4_r111.sh` 及三个 `test_r111_*.py`；完整依赖不依赖这段简称，以manifest逐文件路径及哈希为准。

R109 两组尚未结束（按 main 交接，本次不额外巡检、不改动）。其中 `r109_reporting.py` 和保守工厂依赖 `r109_experiment.py` 等若列入清单，仅作为源码依赖提交；**不纳入 R109 runtime/结果，不声称 R109 completed**。R112 刚运行的冻结源码、回执及其它作者修改完全不触碰。本次没有更新 goal、停止训练、stage、commit 或 push，也没有新增模型身份/独立审查背书。

main 持有 `/public/home/shiys/.git/codex-publication.lock` 后，可先在 easynco 中执行 `python -B -m code.V4.r111_publication verify`，再使用 `allowlist` 的 NUL 分隔输出交给 `git -C /public/home/shiys add --pathspec-from-file=- --pathspec-file-nul`。准备模块本身没有任何 Git写操作；HEAD或清单文件改变会拒绝继续，需 main 在锁内重新对齐。不要执行目录级 `git add runs/` 或 `git add .`。

明确排除：权重 `.pt/.pth`、预测及缓存 `.npz/.npy`、cache、wandb（含回执）、大 history/updates/训练日志、PNG、逐题图片目录和其它旧图。源码、协议及已有PDF/CSV可发布，但完整数值复现仍需本地冻结数据、R84 TRAIN-only统计与原始随机初始化、ID+34资产、相应缓存/权重/预测及历史参照；不是一个克隆后无外部资产即可推理的权重包。多seed不确定性、新匹配对照和 R110逐实例配对结论缺失，均不自动补跑。
