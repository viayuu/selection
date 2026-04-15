# AGENTS.md

你不用看的目录：old_try文件夹。

本文件只记录 `/public/home/zhoucl/shiys` 当前正在做的研究问题、已经实现的内容、以及目前看到的效果。

如果后续实验结论变化，应直接更新本文件。

## 1. 现在正在做的问题

当前在做的是：

- **统一监督学习的 neural solver selector**
- 用 **一个模型** 同时支持多种 routing problem
- 不再按问题分别训练 selector

当前目标问题集合是：

- `TSP`
- `CVRP`
- `ATSP`
- `15` 种 `MVRP` 变体

合起来当前统一处理的是 **18 个问题**。

这个问题的核心难点是三件事：

1. 不同问题支持的 solver 数量不同
2. 不同问题的输入表征不同
3. 一个共享模型需要同时学会“区分问题”和“复用跨问题结构”

当前默认的问题形式应理解为：

- 统一实例表示
- 全局 solver pool
- per-instance feasible mask
- 学习 `score(instance, solver)`

## 2. 目前已经实现了什么

当前主代码目录是：

- [neural-solver-selection_unified_cost](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost)

当前已经实现完成的核心内容：

- 基于 NSS 监督学习骨架改出的统一 selector
- 一个共享的 `NSS-style hierarchical encoder`
- 统一实例输入接口
- 显式 `problem descriptor`
- 全局 `solver pool`
- `feasible mask`
- `masked ranking objective`
- 统一训练 / 验证 / 测试 / benchmark 分析流程


当前已经跑完并保留的关键 full runs：

- [round2_full_true_baseline-0415-040256](/public/home/zhoucl/shiys/nss代码/neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256)
- [round2_full_balanced_stats-0415-024024](/public/home/zhoucl/shiys/nss代码/neural-solver-selection_unified_cost/train_logs/round2_full_balanced_stats-0415-024024)

当前 review 记录与状态：

- [AUTO_REVIEW.md](/public/home/zhoucl/shiys/nss代码/neural-solver-selection_unified_cost/AUTO_REVIEW.md)
- [REVIEW_STATE.json](/public/home/zhoucl/shiys/nss代码/neural-solver-selection_unified_cost/REVIEW_STATE.json)

## 3. 当前数据与标签

当前最重要的数据目录是：

- [data](/public/home/zhoucl/shiys/data)

这个目录包含当前 unified selector 使用的主要导出数据：

- `TSPtrain / TSPval / TSPtest / TSPLIB`
- `CVRPtrain / CVRPval / CVRPtest / CVRPLIB`
- `ATSPtrain / ATSPval / ATSPtest`
- `15` 种 `MVRP` 变体的 `train / val / test`

当前标签形式是 NSS-like 格式：

- `dataset.pkl`
  - 合并后的实例列表
- `results/result_*.txt`
  - 每个方法对应一个文件
  - 每行格式：`instance_id,no_aug_score`
- `raw_label.pkl`
  - 每个实例的标签汇总，包含各方法的成本及最佳方法索引

## 4. 当前方法应该如何理解

当前最应该默认理解的主方法是：

- **simple unified selector**

而不是：

- 复杂的 `balanced+stats`
- 重型 `MoE / FiLM / heavy conditioning`

当前更准确的表述是：

- 一个共享的简单统一 selector
- 在统一输入和全局 solver pool 上做 masked ranking

`balanced+stats` 当前应被理解为：

- 一个已经实现并公平比较过的更复杂变体
- 但它不是当前最可信的主方法

## 5. 目前效果如何

当前最稳妥的实验结论是：

- **simple unified selector 是当前主方法**
- 它比 `balanced+stats` 更适合作为最终方法

full-scale 上目前看到的主要效果：

- simple unified selector 在 full `val/test` 上优于 `balanced+stats`
- simple unified selector 在 full `test` 上显著优于 best single
- simple unified selector 与 `balanced+stats` 在 full `benchmark` 上没有可靠差异
- 两者都在 `CVRP/LIB` benchmark 上明显弱于 best single

所以当前效果应理解为：

- 标准 `val/test` 上结果已经不错
- benchmark / OOD 的 `CVRP/LIB` 是当前最主要弱点

## 6. 当前观测指标整理情况

当前观测指标已经按单独目录整理好：

- [observation_metrics](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/observation_metrics)

对应指标定义文件：

- [观测指标.md](/public/home/zhoucl/shiys/观测指标/观测指标.md)

当前已经整理好的内容包括：

- 每个 run 的 `val_problem_metrics.csv`
- 每个 run 的 `test_problem_metrics.csv`
- `top1_vs_single_methods.png`
- `mean_cost_vs_single_methods.png`
- `arm_distribution.png`
- `loss_curve.png`

## 7. 当前最重要的指标

当前最该优先看的指标是：

- 每个问题的 `selector_top1_accuracy`
- 每个问题的 `best_top1_accuracy`
- 每个问题的 `selector_mean_cost`
- 每个问题的 `best_mean_cost`
- selector 在 `top1` 上超过了哪些单方法
- selector 在 `mean cost` 上超过了哪些单方法

训练和 run-level 汇总时，当前最重要的整体指标是：

- `val_macro_problem_oracle_ratio`
- `test_macro_problem_oracle_ratio`

## 9. 运行约定

- 默认环境：
  - `conda activate easynco_zhoucl`
- 长任务优先：
  - `tmux`
- 当前机器：
  - `2x RTX 3090 24GB`
