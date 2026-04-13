---
type: results-report
date: 2026-04-06
experiment_line: full-project-selection-lines
round: 0
purpose: retrospective-report
status: active
source_artifacts:
  - analysis-output/2026-04-06-full-project-results/analysis-report.md
  - analysis-output/2026-04-06-full-project-results/stats-appendix.md
  - analysis-output/2026-04-06-full-project-results/figure-catalog.md
linked_experiments: []
linked_results: []
---

# Full Project Selection Lines / Round 00 / retrospective-report / 2026-04-06

## Executive Summary

- 这份报告回顾的是整个“组合优化方法选择”项目，而不是单个 run。分析对象同时包含早期失败尝试和当前主线，包括 `1two_gate`、`1online`、`2cmab`、`2cmab2`、`2l2r` 以及 NSS 协议下的新 `2cmab2`。
- 当前最稳的结论有两条。第一，在 legacy 3000-instance 协议上，没有任何学习型 selector 在主指标 `mean_cost` 上击败 `SingleBestPerProblem`；如果只比较学习型方法，`2cmab2 Neural-LinUCB (legacy)` 的 `mean_cost` 最低。第二，在当前 NSS 2000-instance 协议上，`Neural-LinUCB` 明显优于当前实现的 `NeuralUCB-Diag`，因此 bandit 主线应继续围绕 `Neural-LinUCB` 展开。
- 当前最重要的负结果也已经很清楚。`top1` 与 `mean_cost` 并不等价。无论是 `2cmab2` 的 stage2 调参，还是 TSPLIB/CVRPLIB benchmark 的 OOD 评测，都显示 `top1` 的改善不能保证 `mean_cost` 改善。后续论文和实验决策必须把 `mean_cost` 作为 primary metric。
- 早期失败实验提供了有价值的反证。`1two_gate` 很快坍缩到 `gate1=elg, gate2=none`，说明双 gate RL 容易在离散动作上过早塌缩；`1online` 现有 smoke run 全部 `improvement=0.0`，说明当时更多是在打通 pipeline，而不是在形成有效的 online learning。
- 当前仓库未绑定 Obsidian project memory，因此这份报告仅写入本地 markdown；没有尝试做 Obsidian write-back。文件名中的 `r00` 是占位 round，后续若要制度化归档，建议再统一编号。

## Experiment Identity and Decision Context

- Experiment line: `full-project-selection-lines`
- Round: `r00`。这不是一个真实实验轮次，而是一次跨分支的阶段性复盘。
- Purpose: 给论文写作和下一轮实验决策提供统一判断，明确哪些分支值得继续、哪些分支应停止、哪些分支只保留为失败经验。
- 这次复盘面对的真实决策压力有三点：
- 第一，项目已经积累了多条 selector/RL 分支，如果不做协议分层，很容易把不可比结果混在一起。
- 第二，当前论文写作需要一个清晰主线，必须知道 bandit、ranking、强 baseline 和失败 RL 原型各自处在什么位置。
- 第三，当前结果已经暴露出 `top1` 与 `mean_cost` 失配的问题，如果不在现在纠正，会直接影响后面的模型选择和论文表述。

## Setup and Evaluation Protocol

- 本次报告严格区分四类证据家族，避免把不同协议的结果混成同一结论。

### A. Legacy 3000-instance family

- 对象：`2cmab2` legacy traces 与 `2l2r` full runs。
- 单位：逐测试实例配对，`n = 3000`。
- 主指标：`mean_cost`，越低越好。
- 辅助指标：`top1_accuracy`、`gap_to_oracle`。
- 可做的统计：逐实例配对检验，可以做 Wilcoxon 或 paired t-test，并做 Holm 校正。
- 这个家族里还可以从全 arm 真值反推出 `SingleBestPerProblem` 与 `Oracle`，因此适合作为“最严格主结果协议”。

### B. Current NSS 2000-instance family

- 对象：`2cmab2` NSS 版 `Neural-LinUCB` 与 `NeuralUCB-Diag`。
- 单位：逐测试实例配对，`n = 2000`。
- 主指标：`mean_cost`。
- 当前限制：selector trace 里没有所有 arm 的 `true_cost`，所以可以对 selector 方法之间做严格配对统计，但不能对 `SingleBestPerProblem NSS` 做逐实例显著性检验。

### C. Stage2 tuning family

- 对象：`2cmab2` stage2 的 8 组调参 run。
- 单位：逐测试实例，`n = 1500`。
- 目的：不是比较不同算法家族，而是检验“调参带来的 `top1` 提升是否真正改善 `mean_cost`”。

### D. Descriptive-only failure families

- 对象：`1two_gate`、`1online` smoke run、早期 `2cmab` 文本评测。
- 这些分支缺少足够的逐实例原始数据、重复 seed 或标准化协议，因此只能做描述性分析，不能当作强统计证据。

## Main Findings

### 1. Legacy 3000-instance 协议下，`SingleBestPerProblem` 仍然是最强基线

- `SingleBestPerProblem` 的 `mean_cost = 11.8096`，是这个协议上的最低非 oracle 成本。
- `2cmab2 Neural-LinUCB (legacy)` 的 `mean_cost = 11.8110`，在学习型方法中最低，但仍略差于 `SingleBestPerProblem`。
- `2cmab2 NeuralUCB-Diag (legacy)` 与 `2l2r Best-vs-All alpha0` 都是 `11.8117`，几乎重合。
- `2l2r Best-vs-All alpha2` 退化到 `11.8163`，说明 ranking 分支对损失权重敏感，而且调大后会明显伤害 `mean_cost`。
- 这组结果最重要的结论不是“哪一个学习方法第一”，而是“**当前学习型 selector 还没有越过 problem-wise strong baseline**”。这会直接影响论文中 baseline 的摆放方式，也意味着后续不能省略 `SingleBestPerProblem`。

### 2. 如果只比较学习型方法，legacy `Neural-LinUCB` 是更稳的 bandit 版本

- 在 legacy 协议里，`2cmab2 Neural-LinUCB (legacy)` 的 `mean_cost` 比 `2l2r alpha0` 更低，差值约 `0.0007`。
- 这个差值非常小，但逐实例 Wilcoxon 检验给出 `p = 0.0003`，Holm 校正后 `p = 0.0015`，effect size 约 `-0.0988`。这意味着差异方向稳定，但效应很小。
- 因此更谨慎的写法应该是：legacy 协议下，`Neural-LinUCB` 在学习型方法中占优，但这是**小效应优势**，不能写成数量级提升。

### 3. Current NSS 2000-instance 协议下，`Neural-LinUCB` 明显优于 `NeuralUCB-Diag`

- `Neural-LinUCB NSS safe_bs64`：`mean_cost = 13.2027`，`top1 = 0.4400`。
- `Neural-LinUCB NSS server`：`mean_cost = 13.2076`，`top1 = 0.4410`。
- `NeuralUCB-Diag NSS`：`mean_cost = 13.2944`，`top1 = 0.2785`。
- 两个 `Neural-LinUCB` 版本都显著优于 `NeuralUCB-Diag`。`safe_bs64 vs NeuralUCB-Diag` 的逐实例 Wilcoxon 检验 `p = 0.0000`，effect size 约 `-0.5354`；`server vs NeuralUCB-Diag` 的 effect size 约 `-0.5137`。
- `safe_bs64` 和 `server` 版本彼此几乎没有差别，`delta_mean_cost = -0.0049`，Wilcoxon `p = 0.9628`。这说明后来加入的更多工程复杂度并没有形成决定性收益。
- 这组证据已经足够支持一个清楚的工程决策：**当前 bandit 主线只保留 `Neural-LinUCB`，不再优先推进当前版本的 `NeuralUCB-Diag`**。

### 4. Stage2 tuning 证明 `top1` 不是可靠主指标

- 8 组 tuning 里，`top1_accuracy` 从 `0.2127` 到 `0.3607` 波动很大。
- 但 `mean_cost` 只在 `11.8502` 到 `11.8540` 的窄区间内波动，量级非常小。
- 例如 `s2_00_baseline_default` 与 `s2_04_full_linear_history` 的 `top1` 差了约 `0.082`，但 `mean_cost` 只差 `0.00033` 左右。
- 对应的逐实例检验也不支持 `mean_cost` 改善：`baseline_default vs full_linear_history` 的 Wilcoxon `p = 0.3312`。
- 这直接改变了后续实验选择准则。以后不能再把“top1 高一些”视为模型更好的充分证据，尤其不能在论文里把 `top1` 当唯一主表。

### 5. OOD benchmark 再次暴露 `top1 / mean_cost` 失配

- 在 TSPLIB/CVRPLIB benchmark 上，selector 的 overall `top1 = 0.4027`，高于 single-best 的 `0.3557`。
- 但 selector 的 overall `mean_cost = 52.1834`，反而差于 single-best 的 `49.0652`。
- 这个现象与 stage2 tuning 的发现方向一致，说明 `top1` 与真实成本目标之间存在结构性错位，而不是偶然波动。
- 目前 benchmark 只有 selector trace 和 baseline 聚合指标，没有 baseline 的逐实例原始表，因此这里不能做显著性检验。更保守但足够重要的结论是：**在 OOD 场景里，单纯提升“选到最佳 arm 的频率”并不自动等价于更低的平均成本**。

### 6. 早期失败实验为当前主线提供了反证

#### `1two_gate`

- `gate2=none` 很快升到接近 `1.00`，随后长期维持。
- eval `mean_len` 在 `5.7612` 附近很早就平台化。
- 这说明双 gate RL 在当时的设计下容易迅速找到一个“保守但无效”的固定动作组合，而不是持续探索更优方法。

#### `1online`

- 现有 smoke runs 全部只有单实例、单步或极少步。
- 已记录 run 的 `train_improvement = 0.0`，`original_improvement = 0.0`，`augmented_improvement = 0.0`。
- 即使 reward 修正后的 smoke run，`reward_mean_train` 也没有转化为有效 improvement。
- 因此 `1online` 目前只能证明 pipeline 一度被打通，不足以说明 online RL 路线已经有效。

#### 早期 `2cmab`

- 历史文本报告显示：`2cmab Neural-LinUCB final mean_cost = 12.0405`，`2cmab NeuralTS final mean_cost = 11.9855`。
- 这些结果整体弱于后来 `2cmab2/2l2r` 在更成熟协议上的结果。
- 它们的主要价值是提供项目演化背景，而不是作为当前论文主比较对象。

## Statistical Validation

- 这份报告里所有强结论都来自 `analysis-report.md` 和 `stats-appendix.md`，而不是写报告时重新发挥。
- Legacy family 的推断统计基于逐实例配对比较，`n = 3000`。由于差值不服从正态，主要使用 Wilcoxon signed-rank test，并在 planned contrasts 内使用 Holm correction。
- Current NSS family 的推断统计基于逐实例配对比较，`n = 2000`。同样主要使用 Wilcoxon signed-rank test。
- Stage2 tuning family 的对比也是逐实例配对，但这些检验的主要用途是反驳“top1 提升必然带来 cost 改善”，而不是证明某个调参方案显著更优。
- 当前可以被认为“统计上支持”的结论主要有三类：
- 第一，legacy 协议下，`2l2r alpha2` 相对 `SingleBestPerProblem` 出现显著退化，`p_holm = 0.0000`。
- 第二，current NSS 协议下，两个 `Neural-LinUCB` 版本都显著优于 `NeuralUCB-Diag`，且 effect size 达到中等偏大的量级。
- 第三，stage2 调参中，`top1` 的大幅变化并没有转化为显著的 `mean_cost` 改善。
- 当前**不能被升级成强结论**的内容也必须明确：
- 没有任何主家族有多 seed 重复，因此不能宣称“跨 seed 稳健性”。
- current NSS 协议下无法对 `SingleBestPerProblem` 做逐实例显著性检验，因为缺少 all-arm `true_cost`。
- benchmark 家族只有聚合 baseline 指标，不能做 paired significance。

## Figure-by-Figure Interpretation

### Figure 1: `figure-01-legacy-family-comparison.png`

- 这张图的任务是回答：在最严格、最可比的 legacy 3000-instance 协议上，到底谁最强。
- 图里最该注意的不是学习型方法之间的微小排序，而是 `SingleBestPerProblem` 仍然压住所有学习型方法。
- 这张图改变的项目认知是：当前 selector 研究还没有跨过最强 problem-wise baseline，因此后续论文不能把 baseline 设计得太弱。

### Figure 2: `figure-02-current-nss-comparison.png`

- 这张图用于压缩当前 NSS 协议下 bandit 主线的内部决策。
- 读图重点是：两个 `Neural-LinUCB` run 几乎并列，而且都明显好于 `NeuralUCB-Diag`。
- 这张图带来的决策变化是：当前 bandit 主线应该收缩，而不是继续并行维护多个复杂 bandit 版本。

### Figure 3: `figure-03-stage2-top1-vs-cost.png`

- 这张图专门用来回答一个容易误判的问题：调参得到更高 `top1`，是否就意味着 selector 更好。
- 图上的点会在 `top1` 方向展开，但在 `mean_cost` 方向几乎是扁平的。
- 这张图改变的不是某个模型排名，而是指标观。以后无论做调参、checkpoint 选择还是写论文，都不能再只看 `top1`。

### Figure 4: `figure-04-benchmark-ood-comparison.png`

- 这张图对应 OOD 检验，也就是“当前最优 selector 到真实 benchmark 上还是否成立”。
- 关键观察是：selector 的 `top1` 提高了，但 `mean_cost` 没有更好，甚至更差。
- 这张图强化了一个更广泛的信念：训练内或验证内的代理指标，如果和最终成本目标没有严格对齐，到了分布外环境就会暴露问题。

### Figure 5: `figure-05-two-gate-collapse.png`

- 这张图不是主结果图，而是失败证据图。
- 它存在的理由是让“为什么不继续投双 gate RL”这件事有具体证据，而不是只靠主观印象。
- 这张图支持的结论是：`1two_gate` 更适合作为失败经验和设计教训，不适合作为当前论文主线。

## Failure Cases / Negative Results / Limitations

- 当前最强的负结果之一，是所有学习型 selector 仍未在 legacy 主协议上超过 `SingleBestPerProblem`。这说明当前问题并不是“已经超过 baseline，只差一点修饰”，而是“学习信号与成本目标之间还存在系统偏差”。
- 当前 NSS 家族虽然支持 `Neural-LinUCB > NeuralUCB-Diag`，但不能据此声称 “Neural-LinUCB 已经全面优于 single-best”。因为这里缺少对 baseline 的逐实例真值表。
- Stage2 tuning 暴露出一个很实际的问题：如果训练目标更接近 rank/top1，而不是更接近最终 cost，那么调参很可能在优化一个和真实目标并不完全一致的 surrogate。
- Benchmark OOD 结果说明 current selector 还没有建立足够强的跨分布鲁棒性。这个问题对论文尤其重要，因为如果只展示 in-distribution split 上的 `top1` 改善，会掩盖真实部署风险。
- `1two_gate` 与 `1online` 的大部分结果都处于 prototype 或 smoke run 级别。它们非常适合写“我们为什么放弃这条路”，但不适合作为和主线并排的正式实验。
- 所有主家族都缺少 multi-seed repeated runs，因此这份报告只能在“固定 split 的逐实例比较”层面给出强结论，不能上升为 seed-robust general conclusion。

## What Changed Our Belief

- 早期项目可能隐含着一个较乐观假设：只要 selector 学会更常选中最优 arm，成本自然会跟着下降。现在这个假设已经被两套独立证据否定了。stage2 tuning 和 OOD benchmark 都显示 `top1` 可以提升，但 `mean_cost` 未必改善。
- 早期也可能认为 bandit、ranking 和 RL 分支会在同一条线上逐渐收敛成一个“大一统方法”。现在看更准确的判断是：这些分支实际上回答的是不同层次的问题，应该分协议、分功能来摆放，而不是混成一个 leaderboard。
- 对 bandit 主线的判断也发生了收缩。不是“所有 bandit 都值得继续做”，而是“当前数据协议下，值得继续保留的是 `Neural-LinUCB`，而不是当前版本的 `NeuralUCB-Diag`”。
- 对早期 RL 尝试的认知也更清楚了。它们不是“差一点就成功”，而是揭示了动作空间、reward 对齐和训练稳定性上的结构性问题。这个结论会帮助后面写 related negative results 或 design rationale。

## Next Actions

- 论文主结果建议按协议拆成两组表，而不是把 legacy 3000-instance 和 current NSS 2000-instance 混在一起。
- 主指标明确设为 `mean_cost`，`top1_accuracy` 只作为辅助指标；任何只在 `top1` 上更好的方案都不能直接称为更优模型。
- 当前 bandit 主线建议只保留 `Neural-LinUCB`。`NeuralUCB-Diag` 可以保留在补充材料或失败分析里，但不应继续作为主攻方向。
- `SingleBestPerProblem` 必须始终保留在正式实验里，因为它在 legacy 主协议上仍然是最强非 oracle 基线。
- 若继续推进 current NSS 方向，优先补的不是更多同类调参，而是能把 baseline 也还原到逐实例 all-arm 真值的评测协议，这样才能对 `SingleBestPerProblem` 做严格显著性比较。
- 若要为论文增加“负结果”价值，优先写入 `1two_gate` collapse 和 benchmark 中 `top1 / mean_cost` 失配这两类失败证据，因为它们最能解释为什么后续设计要转向。
- 如果后续还有时间补实验，最值得补的是 multi-seed repeated runs；这会显著提升目前结论的可信度层级。

## Artifact and Reproducibility Index

- 严格分析报告：`analysis-output/2026-04-06-full-project-results/analysis-report.md`
- 统计附录：`analysis-output/2026-04-06-full-project-results/stats-appendix.md`
- 图目录：`analysis-output/2026-04-06-full-project-results/figure-catalog.md`
- 主结果图 1：`analysis-output/2026-04-06-full-project-results/figures/figure-01-legacy-family-comparison.png`
- 主结果图 2：`analysis-output/2026-04-06-full-project-results/figures/figure-02-current-nss-comparison.png`
- 支撑图 1：`analysis-output/2026-04-06-full-project-results/figures/figure-03-stage2-top1-vs-cost.png`
- 支撑图 2：`analysis-output/2026-04-06-full-project-results/figures/figure-04-benchmark-ood-comparison.png`
- 失败案例图：`analysis-output/2026-04-06-full-project-results/figures/figure-05-two-gate-collapse.png`
- 关键数表：`analysis-output/2026-04-06-full-project-results/legacy_family_metrics.csv`
- 关键数表：`analysis-output/2026-04-06-full-project-results/current_nss_metrics.csv`
- 关键数表：`analysis-output/2026-04-06-full-project-results/stage2_tuning_metrics.csv`
- 配对检验表：`analysis-output/2026-04-06-full-project-results/legacy_family_paired_tests.csv`
- 配对检验表：`analysis-output/2026-04-06-full-project-results/current_nss_paired_tests.csv`
- 配对检验表：`analysis-output/2026-04-06-full-project-results/stage2_tuning_paired_tests.csv`
- 历史分支摘要：`analysis-output/2026-04-06-full-project-results/oneonline_smoke_summary.csv`
- 历史分支摘要：`analysis-output/2026-04-06-full-project-results/legacy_2cmab_history.csv`
