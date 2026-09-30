# R41 原始标签来源核对

本报告来自独立、只读的同模型族 provenance 核验，`review_independence=same-family`、`acceptance_status=provisional`。标签文件的来源一致，不等于原求解配置已经可复现。本轮没有下载替代权重，也没有用当前默认参数替换缺失的原配置。

## TSP

当前池为 `BQ, DIFUSCO, DIFUSCO500, ELG, LEHD, OMNI, T2T, T2T500`。

除 OMNI train 外，当前结果由 `0selection/neural-solver-selection/datasets/TSP{train,val}/results/` 的旧 NSS 文件导入。完整旧文件的 `id,cost,time` 投影为 `id,cost` 后可重现当前结果文件。实例坐标也与 NSS 一致，位于 `[0,1]`。导入入口为 `/public/home/shiys/easynco_v3_bridge/scripts/merge_nss_and_paper_source_into_data.py::merge_nss_one_dataset`。

| 方法 | 已找到的权重候选，不代表原标签权重绑定 | 缺失证据 |
| --- | --- | --- |
| BQ | `methods论文_mineru_output/source_code/bq-nco/pretrained_models/tsp.best` | 原命令、权重绑定、归一化、beam/解码预算、增强和 RNG |
| DIFUSCO | `EasyNCO/pretrained/DIFUSCO_pretrain/difusco_tsp100.ckpt` | 原命令/绑定、扩散与采样预算、归一化、增强和 RNG |
| DIFUSCO500 | `EasyNCO/pretrained/DIFUSCO_pretrain/difusco_tsp500.ckpt` | 同上；文件名不能证明原解码配置 |
| ELG | `methods论文_mineru_output/source_code/ELG/TSP/weights/ELG.pt` | 原命令/绑定、解码预算、归一化、增强和 RNG |
| LEHD | `EasyNCO/pretrained/lehd/lehd_tsp100.ckpt` | 原命令/绑定、重构预算、归一化、增强和 RNG |
| OMNI | `EasyNCO/pretrained/omni/omni_tsp_maml_fomaml.ckpt` | train 有 EasyNCO 记录，但 val 来自 NSS，不能把 train 配方直接当作 val 原配方 |
| T2T | `EasyNCO/pretrained/T2T_pretrain/t2t_tsp100.ckpt` | 原命令/绑定、扩散/搜索预算、归一化、增强和 RNG |
| T2T500 | 未找到已绑定的原权重；`EasyNCO/pretrained/T2T_pretrain/t2t_tsp500.ckpt` 不存在 | 权重及原求解配置 |

OMNI train 的记录在 `EasyNCO/results/nss_eval/omni_tsp/TSPtrain/eval.log`：顶层 greedy、batch1、seed1234、N个起点、8-fold、无迭代/微调；当前标签取 `no_aug_score`，不是最佳增强结果。该记录仍不能补齐其他方法或 OMNI val 的来源。

**排除替代配置：** 当前 ELG/T2T 标签不等于 EasyNCO 后来的 paperlike/cosine 输出，不能用那些配置声称复现了当前标签。

## OVRPTW

当前池为 `MTPOMO, MVMOE, MoSES_CaDA, MoSES_RF, RELD_MOEL, RELD_MTL, RouteFinder`。

| 方法 | 原入口/数据来源 | 已确认预算 | 本轮状态 |
| --- | --- | --- | --- |
| MTPOMO | `EasyNCO/results/label_v4/mvrp/{train,val}/mtpomo_ovrptw/` | N起点、aug1、batch64、seed1234 | 原运行的实际 decoder 没有被记录 |
| MVMOE | `EasyNCO/results/label_v4/mvrp/{train,val}/mvmoe_ovrptw/` | N起点、aug1、batch64、seed1234 | 同上 |
| MoSES_CaDA | `easynco_v3_bridge/scripts/paper_source_routefinder_runner.py` | 原源码调用8-fold x N起点，取最佳增强 | 原 rf_easnco_env/rl4co 版本及有效解码/RNG 绑定未恢复 |
| MoSES_RF | 同上 | 同上 | 同上 |
| RELD_MOEL | `easynco_v3_bridge/scripts/reld_bridge_common.py` | 显式argmax、N起点、aug1、sample1；旧batch128 | 已按此配置实际复跑 |
| RELD_MTL | 同上 | 同上 | 已按此配置实际复跑 |
| RouteFinder | `easynco_v3_bridge/scripts/paper_source_routefinder_runner.py` | 原源码调用8-fold x N起点，取最佳增强 | 原环境及有效解码/RNG 绑定未恢复 |

所有11000个 OVRPTW train/val 实例的 depot/customer 坐标、需求、service/TW 与 EasyNCO 原 tuple、paper-source NPZ 对齐。需求已归一化，capacity=1。paper-source 的 depot TW 为 `[0,inf]`，RELD 为 `[0,3]`。paper-source 权重分桶规则为 `N<=75 -> 50`，否则100，不能改为最近规模。

RELD 权重：

- `reld-nco-main/Multi-Task/pretrained/reld_mtl/epoch-5000.pt`，SHA256 `2a67f0c1ada0a05563280fcd56d8375b1b0007978bad55c625c862fc3cc3b7ab`。
- MOEL 的固定路径与当前完整哈希记录于 `solver_manifest.json`。历史任务记录来自 `easynco_v3_bridge/exports/reld_nss_like_no_aug_v1/_bridge_state/reld_bridge_events.jsonl`。
- 原完整 CLI、历史权重 digest、历史 RNG seed 未恢复。原函数、预算、输入和命名权重已确认；本轮固定当前权重，并以实际旧成本匹配补充证据。只迁移失效的旧文件系统根路径，不修改算法。

## 一个必须保留的解码歧义

MTPOMO/MVMOE 的204个旧 shard 配置都写了顶层 `decoder_strategy=greedy`，但没有 `settings.module.decoder_strategy`。

EasyNCO 的 git revision `7f50c94e9b4e312696d71014563bc7afa7ee0ff5`（2026-04-14）早于标签日志；所检查的7个 decoder-chain 文件与该 revision 一致：

- `eval.py:89` 没有将顶层 decoder_strategy 转交给 module。
- `phases/rl/ar_reinforce.py:256` 的 module 默认值为 sampling。
- `neural_solvers/pipeline/initialization.py:78` 只调用 eval，不会自动切到 greedy。
- `neural_solvers/methods/pomo/initialization.py:26` 传递实际策略；`neural_solvers/utils/post_search.py:46` 的 sampling 使用 multinomial。

因此，**sampling 有历史源码支持，greedy 只是顶层意图**。但运行目录没有源码快照、commit ID 或有效 decoder 日志；TensorBoard hparams 为空，无法将该 revision 绑定到当次运行。这里不冒充已经证明了原运行的 sampling，也不擅自改成 greedy 后跑标签。

## 依赖限制

当前 easynco 为 Python3.9，只有这一套环境；rl4co 未安装。RouteFinder 本地 `pyproject.toml` 要求 Python>=3.10，依赖未锁定的 rl4co main。仅安装某个当前版本无法恢复旧环境绑定，也不能补齐原 decoder 配置。本轮保留这些缺口，不改主实验环境或用别的运行预算替代。

这不证明旧标签错误或存在随机噪声。它证明的是：**旧结果保存完整，但部分标签生成的实验溯源信息不完整。**
