# Stage2 Neural-LinUCB 正式分析

## 1. 实验目的

本轮实验的目标是对 `Neural-LinUCB` 在 `TSP + CVRP` 初始化方法选择任务上的超参数进行更稳定的比较。

相较于上一轮 `1k / problem` 的小规模筛选实验，本轮实验主要做了两点增强：

1. 将数据规模扩大到每个问题 `5000` 条样本；
2. 将周期性验证频率设置为 `500 step`，使单个 run 在训练过程中产生约 `14` 个验证点。

这样做的目的，是降低小样本实验中：

- `val / test` 排名不稳定；
- 周期性验证点过少、难以观察训练轨迹；
- 单次结果被采样噪声放大的问题。

## 2. 实验设置

### 2.1 数据规模

- TSP: `5000`
- CVRP: `5000`
- 总样本数: `10000`

按当前默认比例切分后：

- train: `7000`
- val: `1500`
- test: `1500`

### 2.2 训练设置

- 方法：`Neural-LinUCB`
- epoch: `1`
- seed: `0`
- `eval_every_steps = 500`
- `checkpoint_every_steps = 1000`

### 2.3 参与比较的 8 组配置

1. `s2_00_baseline_default`
2. `s2_01_less_rep_fit`
3. `s2_02_more_explore`
4. `s2_03_less_explore`
5. `s2_04_full_linear_history`
6. `s2_05_autosaea_reward`
7. `s2_06_freeze_graph_encoder`
8. `s2_07_low_lr`

对应结果目录位于：

- [/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k)

## 3. 总体结果

以下以 `test_greedy.mean_cost` 作为主排名指标，数值越小越好。

| 排名 | 配置 | test mean_cost | 相对 single_best_per_problem | 相对 oracle | test top1 | val mean_cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `s2_00_baseline_default` | 11.850174 | -0.000327 | +0.049820 | 0.2787 | 11.766477 |
| 2 | `s2_02_more_explore` | 11.850449 | -0.000052 | +0.050095 | 0.3447 | 11.766350 |
| 3 | `s2_04_full_linear_history` | 11.850501 | +0.000000 | +0.050147 | 0.3607 | 11.766242 |
| 4 | `s2_03_less_explore` | 11.850756 | +0.000255 | +0.050402 | 0.2127 | 11.766056 |
| 5 | `s2_06_freeze_graph_encoder` | 11.851058 | +0.000557 | +0.050704 | 0.3207 | 11.766900 |
| 6 | `s2_07_low_lr` | 11.851349 | +0.000848 | +0.050995 | 0.3167 | 11.766187 |
| 7 | `s2_05_autosaea_reward` | 11.852463 | +0.001962 | +0.052109 | 0.3387 | 11.766307 |
| 8 | `s2_01_less_rep_fit` | 11.854026 | +0.003525 | +0.053672 | 0.2760 | 11.765718 |

统一 baseline 口径：

- `single_best_per_problem`: `11.850501`
- `single_best_global`: `11.851062`
- `oracle`: `11.800354`
- `random`: `15.163140`

## 4. 关键观察

### 4.1 最优配置是默认配置，但优势极小

本轮最优配置是：

- `s2_00_baseline_default`

其 `test_greedy.mean_cost = 11.850174`，只比 `single_best_per_problem = 11.850501` 好了：

- `0.000327`

这说明当前这批超参数改动并没有带来明显的实质性提升。更准确地说：

- 当前 selector 已基本追平 `single_best_per_problem`
- 但尚未明显突破这一基线

### 4.2 超参数之间的差距非常小

本轮 8 组实验之间的范围是：

- `test mean_cost range = 0.003852`
- `val mean_cost range = 0.001182`

这意味着：

1. 不同超参数设定之间的差距已经非常接近；
2. 仅凭这 8 组配置，很难指望再得到数量级更大的提升；
3. 继续围绕当前这些近邻超参数做密集搜索，收益大概率有限。

### 4.3 上一轮小规模实验筛出的“最优配置”在大规模下不再成立

上一轮 `1k / problem` 实验中，排名靠前的是：

- `low_lr`
- `less_rep_fit`
- `more_explore`

但本轮 `5k / problem` 实验中：

- `baseline_default` 变成了最好；
- `low_lr` 下降到第 6；
- `less_rep_fit` 下降到第 8。

这说明上一轮小规模实验的筛参结论不够稳定，用户此前对“小规模实验不够可靠”的判断是成立的。

### 4.4 val 排名与 test 排名仍然没有完全一致

以本轮最终 `val_greedy.mean_cost` 排名来看，最优的是：

1. `s2_01_less_rep_fit`
2. `s2_03_less_explore`
3. `s2_07_low_lr`

但最终 `test_greedy.mean_cost` 最优的是：

1. `s2_00_baseline_default`
2. `s2_02_more_explore`
3. `s2_04_full_linear_history`

这说明即使放大到 `5k / problem`，单 seed 下依然存在：

- `val` 不能完全稳定预测 `test` 排名

因此，从研究严谨性的角度，下一步如果要做最终结论，仍应加入：

- 多 seed 验证

### 4.5 top1 命中率和 mean_cost 不是同一件事

例如：

- `s2_04_full_linear_history` 的 `test top1 = 0.3607`，是本轮最高；
- 但它的 `test mean_cost = 11.850501`，并没有优于 `baseline_default`。

这说明：

- 更高的 top1 命中率，不一定带来更低的平均 cost；
- 某些配置可能“更常猜中最优 arm”，但一旦猜错，代价更高；
- 对当前任务来说，`mean_cost` 仍然应作为主指标，`top1` 更适合作为辅助解释指标。

## 5. 分问题结果

### 5.1 TSP

各配置在 `TSP test mean_cost` 上的差异非常小，基本都集中在：

- `7.8010 ~ 7.8024`

最优是：

- `s2_00_baseline_default`: `7.801060`

最差是：

- `s2_03_less_explore`: `7.802400`

范围只有：

- 约 `0.00134`

这说明在 TSP 上，不同超参数配置已经几乎收敛到同一水平。

### 5.2 CVRP

CVRP 的差异比 TSP 稍大，但仍然不大，集中在：

- `15.8993 ~ 15.9063`

最优是：

- `s2_00_baseline_default`: `15.899288`

最差是：

- `s2_01_less_rep_fit`: `15.906349`

范围约：

- `0.00706`

也就是说，本轮配置差异主要还是更体现在 CVRP 上，而不是 TSP。

## 6. 训练过程稳定性

每个 run 都包含 `14` 次周期性验证，`val_greedy.mean_cost` 的整体波动并不大。

例如：

- `s2_00_baseline_default`
  - first: `11.765608`
  - best: `11.764550`
  - last: `11.766477`
- `s2_07_low_lr`
  - first: `11.768515`
  - best: `11.764327`
  - last: `11.766187`
- `s2_05_autosaea_reward`
  - first: `11.766252`
  - best: `11.763684`
  - last: `11.766307`

可以看到：

1. 各配置在训练中都能出现局部更优点；
2. 但最终收敛点彼此很接近；
3. 说明问题已经不太像“单纯没把学习率/窗口调好”，而更像“当前方法框架本身的提升空间有限”。

## 7. 结果解释

基于本轮实验，更合理的解释是：

1. 当前 `Neural-LinUCB` 实现已经能稳定做到接近 `single_best_per_problem`；
2. 但围绕当前这些超参数的微调，无法继续显著拉开与 baseline 的差距；
3. 这意味着当前瓶颈更可能来自：
   - reward 口径与最终 cost 指标的错位；
   - 表示层训练目标本身的限制；
   - bandit 反馈过于稀疏，只基于被选 arm 做监督；
   - 当前 `phi(x,a)` 构造方式对方法差异的表达能力不足。

## 8. 当前最稳妥的结论

如果现在需要给出一个阶段性结论，那么更稳妥的表述应是：

> 在 `TSP + CVRP` 初始化方法选择任务上，当前 `Neural-LinUCB` 的多组超参数配置均能显著优于随机基线，并达到与 `single_best_per_problem` 非常接近的水平；然而，在 `5k / problem` 规模下，8 组超参数之间的性能差异极小，默认配置已经是最优或近似最优。由此可见，当前性能瓶颈不再主要来自这些近邻超参数，而更可能来自 reward 设计、表示层训练目标与 bandit 监督方式本身。

## 9. 下一步建议

### 建议 1：不要继续大规模扫当前这类近邻超参数

理由：

- 本轮已经表明收益很小；
- 差距量级过小，继续扫可能主要是在追噪声。

### 建议 2：如果要做更严谨结论，先做多 seed

建议保留 2 到 3 组代表性配置：

1. `s2_00_baseline_default`
2. `s2_02_more_explore`
3. `s2_04_full_linear_history`

然后跑：

- `seed = 0, 1, 2, 3, 4`

这样可以判断当前极小差距是否稳定存在。

### 建议 3：后续重点应转向结构性改动

例如：

1. 修改 reward 设计；
2. 修改表示层训练目标；
3. 重新考虑线性头与表示层的耦合方式；
4. 增强动作条件表示 `phi(x,a)` 的建模能力。

---

## 附：关键结果文件

最优配置结果：

- [s2_00_baseline_default/summary.json](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k/s2_00_baseline_default/summary.json)

次优配置结果：

- [s2_02_more_explore/summary.json](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k/s2_02_more_explore/summary.json)
- [s2_04_full_linear_history/summary.json](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k/s2_04_full_linear_history/summary.json)

表现较弱配置结果：

- [s2_01_less_rep_fit/summary.json](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k/s2_01_less_rep_fit/summary.json)

阶段 2 输出根目录：

- [/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k](/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k)
