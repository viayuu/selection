# 研究简报

> **用于 `/idea-discovery` 或 `/research-pipeline` 的文档化输入模板。** 提供详细上下文，替代一行式提示。当前项目的已填写简报，供其他 AI 直接读取后继续工作。

## 问题陈述
当前研究问题是：能否训练一个**统一监督学习的 neural solver selector**，让**一个模型**同时支持多个 routing problem，并在实例级别从多个已有 solver 中选出最合适的方法。

这项工作的重点不是再设计一个新的 `TSP` 或 `CVRP` solver，而是把已有 solver 看成候选方法池，对每个实例做 solver selection。与 NSS 不同，当前目标不是“每个问题单独训练一个 selector”，而是训练一个**跨问题共享**的 selector，在统一框架下同时支持 `TSP / CVRP / ATSP / 15 种 MVRP 变体` 共 18 个问题。

这个问题的核心难点有三个。第一，不同问题支持的 solver 集合不同，因此输出空间不是固定多分类，当前的解决方法是建模为“全局 solver pool + per-instance feasible mask”。第二，不同问题的原始输入格式不同，因此必须建立统一实例表示。第三，一个共享模型既要复用跨问题的公共结构，又要显式区分不同问题的语义和约束差异。

## 背景

- **领域**: 组合优化 / neural solver selection / routing
- **子方向**: 多问题统一的监督学习 solver selector
- **已读关键论文**:
  - `NSS (Neural Solver Selection for Combinatorial Optimization)`
  - `URS`
  - `CoEKS / moe.md` 对应的跨问题专家化思路
  - 推荐系统多域/多任务建模文献：`MMoE / PLE / STAR`
- **当前代码路径**:
  - `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost`
- **当前数据路径**:
  - `/public/home/zhoucl/shiys/data`
- **当前结果整理路径**:
  - `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/observation_metrics`
- **已尝试的方法**:
  - 基于 NSS 监督学习骨架改造 unified selector
  - 构建全局 `solver pool`
  - 为每个实例构造 `feasible mask`
  - 用统一节点特征槽位编码不同问题实例
  - 显式加入 `problem_id / problem_features / scale / instance_stats`
  - 使用 `masked ranking loss`
  - 加入 `problem-balanced sampling` 与更复杂条件化变体做过对比实验
- **失败经验**:
  - 更复杂的条件化增强方案在 full-scale 上没有稳定优于简单主方法
  - benchmark / OOD 的 `CVRP/LIB` 仍然是最主要弱点

## 约束条件

- **算力**:
  - 当前机器：`2x RTX 3090 24GB`
  - 默认环境：`conda activate easynco_zhoucl`
  - 长任务默认放在 `tmux`
- **目标会议/期刊**:
  - 暂未在当前上下文中锁定具体 venue
  - 但整体标准按顶会级研究问题与实验严谨性来推进

## 期望方向

- [ ] 从零开始探索新研究方向
- [x] 改进现有方法：统一监督学习的多问题 neural solver selector
- [x] 诊断性研究 / 分析型论文

## 领域知识

当前主方法应该理解为：

- **simple unified selector**

不要把当前主方法理解成复杂结构。当前更可信的主线是：

- 统一实例表示
- 全局 solver pool
- per-instance feasible mask
- 共享 encoder
- ranking-based selection objective

当前实现上的关键认知：

- 输出空间不是固定多分类，而是 masked candidate scoring
- 统一表示是必须的，否则无法共享一个模型
- 显式问题信息是必须的，不能只靠原始几何让模型自己猜问题类型
- 当前训练只看 `cost`，不看 `time`
- 当前更适合看的指标不是单纯整体 accuracy，而是：
  - `val_macro_problem_oracle_ratio`
  - `test_macro_problem_oracle_ratio`
  - `benchmark_macro_problem_oracle_ratio`

当前 `val_top_1_ratio` 的定义是：

- `mean(predicted_top1_cost / oracle_cost)`

即：模型 top1 选择相对 oracle 的平均 cost 比例。`1.0` 最好，越小越好。

## 非目标

- 不再做按问题分别训练一个 selector 的路线
- 不把 `time` 作为当前阶段的监督目标
- 不关注 `old_try` 里的历史路线

## 已有结果（如有）

当前最重要的 full-scale run 是：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256`

当前最重要的结论是：

- **simple unified selector 是当前主方法**
- full-scale 上标准 `val/test` 结果已经较强
- full `test` 上显著优于 best single
- 当前主要弱点集中在 `CVRP/LIB`

关键数值：

- `val_macro_problem_oracle_ratio = 1.003591`
- `test_macro_problem_oracle_ratio = 1.003554`
- `benchmark_macro_problem_oracle_ratio = 1.017672`
- `test_macro_problem_gap_vs_best_single = -0.018162`

这说明：

- `test` 上相对 best single 有稳定收益
- `benchmark` 上仍然存在明显 gap

当前 review loop 最终状态：

- 文件：
  - `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/REVIEW_STATE.json`
- 状态：
  - `round = 5`
  - `status = completed`
  - `last_score = 7.0`
  - `last_verdict = almost`

说明：

- 主要实验包已经基本收敛
- 当前更多是结果整理、分析和后续写作 / 有针对性的改进

当前观测指标已经整理好，目录在：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/observation_metrics`

这个目录下已经有：

- `val_problem_metrics.csv`
- `test_problem_metrics.csv`
- `loss_curve.png`
- `val_top1_vs_single_methods.png`
- `test_top1_vs_single_methods.png`
- `val_mean_cost_vs_single_methods.png`
- `test_mean_cost_vs_single_methods.png`
- `val_arm_distribution.png`
- `test_arm_distribution.png`