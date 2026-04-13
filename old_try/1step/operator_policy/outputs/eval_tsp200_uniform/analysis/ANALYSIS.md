# Eval 坍缩分析：`eval_tsp200_uniform`

## 结果概览

- 问题规模：`200`
- 数据集：`EasyNCO/data/datasets/test_dataset_tsp_uniform/test_tsp200_nums128_uniform.pt`
- 评测样本数：`128`
- rollout 步数：`100`
- 初始解均值：`10.9640`
- 最终解均值：`10.7190`
- 平均改进：`0.2450`
- 相对改进：`2.2343%`
- 单步平均改进：`0.002450`
- 总耗时：`382.94s`，吞吐：`0.3343` items/s

## 初始化坍缩

- 分布：`lehd=0.0000, elg=1.0000, difusco=0.0000`
- 最大占比：`1.0000`，判定：**完全坍缩**
- 主导初始化器：`elg`
- 解释：`init_distribution` 是在整个评测集上统计“每个实例最终选了哪个初始化器”的频率；这里若某一项为 1，表示所有实例都选了同一个初始化器。

## 迭代算子坍缩

- 分布：`two_opt=0.0000, lehd_rrc_step=1.0000, dact_2opt_step=0.0000`
- 最大占比：`1.0000`，判定：**完全坍缩**
- 主导算子：`lehd_rrc_step`
- 解释：`operator_distribution` 统计的是整个评测过程中“所有实例 × 所有 rollout step”的动作频率；若某一项为 1，表示每个实例在每一步都用了同一个 operator。

## Batch 过程稳定性

- 日志中解析到 `batch` 进度点：`1` 个。
- 首个进度点：items=`32`，avg_init=`10.9576`，avg_final=`10.6897`，improve=`0.2680`。
- 最后进度点：items=`32`，avg_init=`10.9576`，avg_final=`10.6897`，improve=`0.2680`。
- 若首尾数值变化很小，说明评测过程中统计量较稳定，没有明显因为后续 batch 而漂移。

## 注意事项

- [warn] checkpoint was trained at problem_size=100 but is now evaluated at problem_size=200
- 这说明当前 selector 是在 `tsp100` 上训练的，再外推到更大尺度；因此 200/500 的结果更应理解为“跨尺度泛化下的动作选择行为”。
- 这三个 eval 目录都没有逐实例 trace，因此这里的坍缩结论是“数据集整体层面”的，不是逐实例动作序列层面的。
