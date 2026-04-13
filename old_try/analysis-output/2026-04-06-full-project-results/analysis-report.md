# Strict Analysis Report

## Analysis Question
- 主问题 1：在可严格配对的 legacy 3000-instance selector 协议上，bandit 与 supervised ranking 哪一类更强，是否显著优于 `single_best_per_problem`？
- 主问题 2：在当前 NSS 2000-instance 协议上，`Neural-LinUCB` 是否稳定优于 `NeuralUCB-Diag`？
- 主问题 3：`2cmab2` 调参过程中，`top1` 的波动是否真正带来 `mean_cost` 改善？
- 主问题 4：更早期的 `1two_gate`、`1online`、`2cmab` 分支到底失败在什么地方？
- 主问题 5：当前 NSS 最优 checkpoint 迁移到 TSPLIB/CVRPLIB benchmark 后，`top1` 与 `mean_cost` 是否仍一致？

## Data Inventory
- 可做严格配对统计的家族：
  - `2cmab2` legacy 3000-instance traces（逐实例）
  - `2l2r` full 3000-instance traces（逐实例且含所有 arm 的 `true_cost`）
  - `2cmab2` NSS 2000-instance traces（逐实例，但只含选中 arm 与 oracle cost）
- 只能做描述性分析的家族：
  - `1two_gate`：单日志、无多 seed，仅可分析 collapse 与 plateau
  - `1online`：都是 smoke run，单实例、单步或极少步，不支持推断统计
  - `2cmab`：保存的是文本评测报告，不是逐实例原始表，只能做历史对照
- 统计边界：当前大多数对比没有多 seed 重复，因此所有显著性结论都只成立于“固定 test split 的逐实例配对比较”，不能外推成“跨 seed 稳健结论”。

## Executive Summary
- 在 legacy 3000-instance 协议上，没有任何可学习 selector 打败 `SingleBestPerProblem`；在可学习方法内部，最低 `mean_cost` 来自 `2cmab2 Neural-LinUCB (legacy)`，其 `mean_cost=11.8110`。
- 在同一协议上，`2l2r Best-vs-All alpha0` 没有带来 `mean_cost` 优势，`alpha2` 版本则出现显著退化，说明 ranking 分支对损失权重设置敏感。
- 在当前 NSS 2000-instance 协议上，最佳 run 是 `Neural-LinUCB NSS safe_bs64`，`mean_cost=13.2027`、`top1=0.4400`；它显著优于 `NeuralUCB-Diag NSS` 的 `mean_cost=13.2944`。
- `2cmab2` 的 stage2 调参显示：`top1` 可以从约 `0.213` 波动到 `0.361`，但 `mean_cost` 基本锁在 `11.850±0.004` 的极窄区间，证明 `top1` 不是可靠的唯一目标。
- 当前 NSS 最优模型在 TSPLIB/CVRPLIB benchmark 上出现明显的 `top1 / mean_cost` 失配：selector 的 `top1` 高于 single-best，但 `mean_cost` 反而更差，特别是 CVRP benchmark 上 gap 明显。
- 更早期 RL 原型的核心失败模式已经很清楚：`1two_gate` 在 100-200 step 内坍缩到 `gate1=elg, gate2=none`；`1online` smoke run 全部 `improvement=0.0`，甚至 reward 修正后也没有产生策略更新信号。

## Main Findings

### 1. Legacy 3000-instance selector family
- `SingleBestPerProblem`：`mean_cost=11.8096`，仍然是这个协议上的最强基线。
- `2l2r Best-vs-All alpha0`：`mean_cost=11.8117`，`top1=0.3900`。
- `2cmab2 Neural-LinUCB (legacy)`：`mean_cost=11.8110`，`top1=0.3033`。
- `2cmab2 NeuralUCB-Diag (legacy)`：`mean_cost=11.8117`，`top1=0.3900`。
- 观察：legacy 协议下，所有可学习方法都没能在 `mean_cost` 上超过 `SingleBestPerProblem`。可学习方法内部，legacy `Neural-LinUCB` 的 cost 最低；`alpha0` 与 legacy `NeuralUCB-Diag` 在 cost 上几乎重合，但 `alpha2` 明显更差。

### 2. Current NSS family
- `Neural-LinUCB NSS safe_bs64` 与 `server` 版表现接近，说明后续加复杂设置并没有带来决定性收益。
- 两个 `Neural-LinUCB` NSS run 都明显好于 `NeuralUCB-Diag NSS`，差异体现在 `mean_cost` 与 `top1` 两端，同时 `gap_to_oracle` 更小。
- 观察：当前最稳妥的 bandit 结论是“浅层线性头 + 周期性表示更新”优于当前的对角近似 NeuralUCB 实现。

### 3. Stage2 tuning shows top1 / mean_cost decoupling
- 最低 `mean_cost` 的 stage2 setting 不是最高 `top1` 的 setting。
- `s2_00_baseline_default` 与 `s2_04_full_linear_history` 的 `top1` 相差约 `0.082`，但 `mean_cost` 只差约 `0.00033`。
- 这意味着：如果论文只报 `top1`，会夸大某些调参带来的真实收益。

### 4. OOD benchmark mismatch
- 在 TSPLIB/CVRPLIB benchmark 上，selector 的 overall `top1=0.4027`，高于 single-best 的 `0.3557`。
- 但 overall `mean_cost=52.1834`，反而差于 single-best 的 `49.0652`。
- 解释边界：当前 benchmark 只拿到了 selector 的逐实例 trace，baseline 只有聚合指标，因此这里只能做描述性结论，不能做配对显著性检验。

### 5. Earlier failed branches remain informative
- `1two_gate`：`gate2=none` 很快升到 `1.00`，并在几百 step 内维持；eval `mean_len` 在 `5.7612` 处早早平台化。
- `1online`：现有 smoke run 全部只有单实例、单步或极少步；`improvement=0.0`，说明这些 run 只能验证代码通路，不能支持方法有效性。
- `2cmab`：历史文本报告显示 `Neural-LinUCB final mean_cost=12.0405`、`NeuralTS final mean_cost=11.9855`，整体弱于后来的 `2cmab2/2l2r` 3000-instance 结果。

## Caveats
- 没有多 seed 重复，因此不能做 seed-level significance claim。
- 当前 NSS family 的 per-instance trace 缺少所有 arm 的 `true_cost`，所以对 `single_best_per_problem` 的比较只能做 summary-level，不做逐实例显著性检验。
- benchmark family 也只有 selector trace 和 baseline 聚合统计，因此 OOD benchmark 结论只能是描述性的。
- `1two_gate`、`1online` 的大部分结果都是 smoke / prototype 级别，适合写失败案例，不适合作为主实验。

## Recommended Next Move
- 论文主结果应分成两个协议分别写：legacy 3000-instance 与 current NSS 2000-instance，不要混成一张表。
- 主文里应把 `mean_cost` 设为 primary metric，`top1` 作为辅助 metric。
- 若继续推进 bandit 主线，优先保留 `Neural-LinUCB`，谨慎投入 `NeuralUCB-Diag`。
- 若继续追求纯效果，`SingleBestPerProblem` 必须始终保留为强基线；`2l2r Best-vs-All alpha0` 适合作为与 bandit 主线平行的学习型对照。
- 若要讲失败经验，`1two_gate` collapse 与 benchmark 的 `top1 / mean_cost` 失配最有价值。

## Exact Numeric Tables

### Legacy 3000-instance family
| name                           |   count |   mean_cost |   ci_low |   ci_high |   top1_accuracy |   gain_vs_sbp |   gap_to_oracle |
|:-------------------------------|--------:|------------:|---------:|----------:|----------------:|--------------:|----------------:|
| Oracle                         |    3000 |     11.7622 |  11.6120 |   11.9125 |          1.0000 |        0.0474 |          0.0000 |
| SingleBestPerProblem           |    3000 |     11.8096 |  11.6581 |   11.9612 |          0.4273 |        0.0000 |          0.0474 |
| 2cmab2 Neural-LinUCB (legacy)  |    3000 |     11.8110 |  11.6593 |   11.9627 |          0.3033 |       -0.0014 |          0.0488 |
| 2cmab2 NeuralUCB-Diag (legacy) |    3000 |     11.8117 |  11.6598 |   11.9636 |          0.3900 |       -0.0021 |          0.0495 |
| 2l2r Best-vs-All alpha0        |    3000 |     11.8117 |  11.6598 |   11.9636 |          0.3900 |       -0.0021 |          0.0495 |
| 2l2r Listwise                  |    3000 |     11.8122 |  11.6603 |   11.9641 |          0.3557 |       -0.0026 |          0.0500 |
| 2l2r Pairwise                  |    3000 |     11.8130 |  11.6612 |   11.9648 |          0.2753 |       -0.0034 |          0.0508 |
| 2l2r Best-vs-All alpha2        |    3000 |     11.8163 |  11.6647 |   11.9678 |          0.1940 |       -0.0066 |          0.0540 |

### Current NSS 2000-instance family
| name                        |   count |   mean_cost |   ci_low |   ci_high |   top1_accuracy |   gap_to_oracle |
|:----------------------------|--------:|------------:|---------:|----------:|----------------:|----------------:|
| Oracle NSS                  |    2000 |     13.1191 | nan      |  nan      |          1.0000 |          0.0000 |
| Neural-LinUCB NSS safe_bs64 |    2000 |     13.2027 |  12.8508 |   13.5546 |          0.4400 |          0.0837 |
| Neural-LinUCB NSS server    |    2000 |     13.2076 |  12.8551 |   13.5602 |          0.4410 |          0.0886 |
| NeuralUCB-Diag NSS          |    2000 |     13.2944 |  12.9428 |   13.6461 |          0.2785 |          0.1754 |
| SingleBestPerProblem NSS    |    2000 |     13.3093 | nan      |  nan      |          0.3380 |          0.1903 |

### Stage2 tuning sweep
| tag                        |   mean_cost |   top1_accuracy |   alpha |   train_every |   rep_steps |   rep_buffer |   lin_buffer | freeze_encoder   |
|:---------------------------|------------:|----------------:|--------:|--------------:|------------:|-------------:|-------------:|:-----------------|
| s2_00_baseline_default     |     11.8502 |          0.2787 |  1.0000 |           100 |          10 |         5000 |        10000 | False            |
| s2_02_more_explore         |     11.8504 |          0.3447 |  2.0000 |           200 |           2 |         2000 |        10000 | False            |
| s2_04_full_linear_history  |     11.8505 |          0.3607 |  2.0000 |           200 |           2 |         2000 |            0 | False            |
| s2_03_less_explore         |     11.8508 |          0.2127 |  0.5000 |           200 |           2 |         2000 |        10000 | False            |
| s2_06_freeze_graph_encoder |     11.8511 |          0.3207 |  2.0000 |           200 |           2 |         2000 |        10000 | True             |
| s2_07_low_lr               |     11.8513 |          0.3167 |  2.0000 |           200 |           2 |         2000 |        10000 | False            |
| s2_05_autosaea_reward      |     11.8525 |          0.3387 |  2.0000 |           200 |           2 |         2000 |        10000 | False            |
| s2_01_less_rep_fit         |     11.8540 |          0.2760 |  1.0000 |           200 |           2 |         2000 |        10000 | False            |
