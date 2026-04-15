# 交接文档

本文件只保留继续当前项目最需要的 4 类信息：

1. 我想做的事情 / idea
2. 当前是如何实现的
3. 当前数据集的构成与路径
4. 需要关注的指标

## 1. 我想做的事情 / idea

当前主线是：

- 做一个 **统一监督学习的 neural solver selector**
- 用 **一个模型** 同时支持多个 routing problem
- 不再像 NSS 那样按问题分别训练 selector

更具体地说，当前想做的不是“再设计一个单独 solver”，而是：

- 把已经存在的多个 solver 当作候选动作 / 候选专家
- 对每个具体实例，自动选择最合适的 solver
- 并且这个选择器不是只服务于单一问题，而是服务于一组不同 routing problem

所以这个项目的目标不是：

- 学一个新的 `TSP` solver
- 学一个新的 `CVRP` solver
- 或者为每个问题单独训练一个 selector

而是：

- 训练一个**跨问题共享**的 selector
- 让它在不同问题上复用底层表示能力
- 但又能识别不同问题的结构差异
- 最终在实例级别做 solver selection

当前统一处理的问题共有 **18 个**：

- `TSP`
- `CVRP`
- `ATSP`
- `15` 种 `MVRP` 变体：
  - `OVRP`
  - `VRPB`
  - `VRPL`
  - `VRPTW`
  - `OVRPTW`
  - `OVRPB`
  - `OVRPL`
  - `VRPBL`
  - `VRPBTW`
  - `VRPLTW`
  - `OVRPBL`
  - `OVRPBTW`
  - `OVRPLTW`
  - `VRPBLTW`
  - `OVRPBLTW`

这个问题的核心形式是：

- 统一实例表示
- 全局 solver pool
- per-instance feasible mask
- 学习 `score(instance, solver)`

这里最关键的建模思想有四个：

1. **输出空间不是固定多分类**
   - 不同问题支持的 solver 集合不同
   - 所以不能简单把问题写成“固定类别数的 softmax 分类”
   - 更自然的形式是：全局 solver pool + 每个实例自己的 feasible mask

2. **实例表示需要统一**
   - `TSP / CVRP / ATSP / 各种 MVRP` 的原始格式不一样
   - 必须把它们都转成统一的输入形式，才能让一个共享模型真正训练起来

3. **模型必须知道当前实例属于什么问题**
   - 不能只靠几何结构自己猜
   - 需要显式的问题信息，例如 problem id / problem feature

4. **优化目标本质上是 ranking，不只是分类**
   - 当前更关心的是选到 cost 更低的方法
   - 而不是只追求“类别标签是否完全命中”

从研究角度看，这个 idea 想解决的是一个比 NSS 更强的问题：

- NSS 证明了“实例级 solver selection”是有意义的
- 当前项目想进一步证明：
  - 一个共享 selector 也可以在多个问题上工作
  - 而且这种统一 selector 不只是概念上可行，标准 `val/test` 上也能真正跑出效果

当前这条线最重要的科学难点是：

- **动态候选集合**
  - 不同问题支持的 solver 不一样
- **跨问题共享与区分**
  - 既要共享，又不能把问题差异抹平
- **统一表示**
  - 输入格式必须统一
- **统一训练**
  - 一个训练流程同时兼容 18 个问题

当前主方法应理解为：

- **simple unified selector**

它的论文级主张应该理解为：

- 一个简单但统一的 supervised selector
- 基于共享 encoder、统一输入、全局 solver pool、feasible mask、ranking objective
- 已经可以在多问题设置下形成有竞争力的结果

不要把当前主方法理解成复杂结构。当前最重要的是一个简单、统一、共享的 selector。

## 2. 当前是如何实现的

当前真实项目根目录是：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost`

不要误用其它中间路径，当前活跃目录就是这个。

当前实现的核心内容：

- 基于 NSS 监督学习骨架改出的 unified selector
- 一个共享的 `NSS-style hierarchical encoder`
- 统一实例输入接口
- 显式 `problem descriptor`
- 全局 `solver pool`
- `feasible mask`
- `masked ranking objective`
- 统一训练 / 验证 / 测试 / benchmark 分析流程

更详细地说，当前实现可以按下面这条流水线理解：

### 2.1 数据读取与统一化

数据由 [dataset.py](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/dataset.py) 负责读取。

它做的事情包括：

- 读取每个数据集目录里的 `dataset.pkl`
- 读取 `results/result_*.txt`
- 统一整理出当前数据集支持的 solver 名称
- 把 solver 名称做 canonicalize，避免大小写或命名差异造成重复 solver
- 自动构建：
  - `solver_pool`
  - `problem_pool`

它会把不同问题的原始实例统一编码成节点特征矩阵：

- `TSP`：坐标型实例
- `ATSP`：矩阵型实例
- `CVRP`：dict 格式实例
- `MVRP`：tuple 格式实例

统一后的每个 sample 至少会包含：

- `nodes`
- `label`
- `costs`
- `times`
- `feasible_mask`
- `problem_id`
- `problem_features`
- `scale`
- `instance_stats`

其中最重要的是：

- `costs`
  - 对应全局 solver pool 上每个 solver 的 cost
- `feasible_mask`
  - 标识当前实例哪些 solver 可用
- `label`
  - 当前 cost 最小的 solver 索引

### 2.2 统一实例编码

原始实例会先被编码成统一节点特征。

当前节点特征维度是固定槽位形式，核心字段包括：

- 坐标相关
- demand / capacity 相关
- prize / penalty 相关
- service time / time window 相关
- depot / route 相关标记
- route limit 等约束相关字段

也就是说，当前实现不是给每类问题单独一套网络输入，而是：

- 所有问题共享固定维度节点特征
- 缺失的字段补零

### 2.3 问题语义与统计特征

除了节点特征本身，当前实现还会显式给模型提供：

- `problem_id`
- `problem_features`
- `scale`
- `instance_stats`

其中：

- `problem_id`
  - 用于 `problem embedding`
- `problem_features`
  - 是显式问题描述向量，表示这个问题有哪些属性
- `scale`
  - 表示实例规模
- `instance_stats`
  - 是从实例里提出来的连续统计特征

例如在当前实现中：

- 对 `ATSP` 会统计矩阵的均值、方差、非对称性等
- 对坐标类问题会统计坐标分布、demand、时间窗跨度、route limit 等

### 2.4 模型结构

模型定义在 [model.py](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/model.py)。

当前结构可以理解为 4 层：

1. **实例编码层**
   - 用 `NSS-style hierarchical encoder` 编码图实例
   - 得到图级表示 `graph_emb`

2. **问题信息融合层**
   - `problem_embedding`
   - `problem_feature_proj`
   - `scale_proj`
   - `stats_proj`
   - 共同形成实例侧的高层表示

3. **solver 表示层**
   - 每个 solver 有一个 `solver_embedding`
   - 如果提供 solver feature，也可以再额外投影融合

4. **pair scorer**
   - 对 `(instance, solver)` 配对表示打分
   - 输出每个 solver 的一个 score

当前 `scorer` 本质上是：

- 实例表示
- solver 表示
- 二者的交互项
- 拼接后过一个小 MLP

最终得到：

- 对全局 solver pool 中每个 solver 的一个 logit

然后再用：

- `feasible_mask`

把当前问题不支持的 solver 屏蔽掉。

### 2.5 损失函数

损失定义在 [loss.py](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/loss.py)。

当前主要实现了两种：

- `MaskedRankingLoss`
- `MaskedCrossEntropyLoss`

当前主线更重要的是：

- `MaskedRankingLoss`

它的逻辑是：

- 只在当前实例可行的 solver 上做排序学习
- 每一步取当前最优 cost 对应的 solver 作为监督目标
- 逐步把已经选过的 solver 从 mask 中移除

这使得训练目标更接近“方法排序 / 方法选择”本质，而不是普通分类。

### 2.6 训练流程

训练与评估流程在 [trainer.py](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/trainer.py)。

主要过程是：

- 构建 `train / val / test / benchmark` DataLoader
- 按 epoch 训练
- 每个 epoch 在 `val` 上评估
- 保存每个 epoch checkpoint
- 保存 `best_checkpoint_info.json`
- 按选定指标选择最佳 checkpoint

训练时支持：

- 普通采样
- `problem_balanced_sampling`

即：

- 如果开启，会按问题重新加权采样，避免大问题族压制小问题族

### 2.7 评估流程

当前评估时会自动计算：

- `acc`
- `oracle_cost`
- `single_best_problem_cost`
- `top_k_cost`
- `top_k_ratio`
- `macro_problem_top1_accuracy`
- `macro_problem_oracle_ratio`
- `macro_problem_gap_vs_best_single`

并且会把分析结果写到：

- `analysis_val`
- `analysis_test`
- `analysis_benchmark`

这些分析目录里会包含：

- `per_problem_summary.csv`
- 每个问题对应的 `*_single_method_compare.csv`
- `arm_distribution.csv`

### 2.8 观测指标整理脚本

当前为了适配用户要求，又额外写了：

- [organize_observation_metrics.py](/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/organize_observation_metrics.py)

它会把原始分析结果再整理成更适合直接阅读的产物：

- `val_problem_metrics.csv`
- `test_problem_metrics.csv`
- 汇总版 `top1_vs_single_methods.png`
- 汇总版 `mean_cost_vs_single_methods.png`
- 汇总版 `arm_distribution.png`
- `loss_curve.png`

也就是说，当前实现不只是“能训练”，而是：

- 已经形成了一整套从数据读取、训练、评估，到按用户指标整理结果的完整链路

最重要的代码文件：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/dataset.py`
- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/model.py`
- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/loss.py`
- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/trainer.py`
- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/run.py`
- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/organize_observation_metrics.py`

当前已经保留的关键 full run：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256`

当前结果结论应理解为：

- simple unified selector 是当前主方法
- 标准 `val/test` 上结果已经不错
- full `test` 上显著优于 best single
- 当前主要弱点是 `CVRP/LIB` benchmark / OOD 泛化

## 3. 当前数据集的构成、路径

当前活跃数据根目录是：

- `/public/home/zhoucl/shiys/data`

说明文件：

- `/public/home/zhoucl/shiys/data/README.md`

当前包含：

- `TSPtrain / TSPval / TSPtest / TSPLIB`
- `CVRPtrain / CVRPval / CVRPtest / CVRPLIB`
- `ATSPtrain / ATSPval / ATSPtest`
- `15` 种 MVRP 变体的 `train / val / test`

每个数据集目录结构一致：

- `dataset.pkl`
- `results/result_*.txt`
- `raw_label.pkl`

当前标签使用方式：

- 只用 `cost`
- 不考虑 `time`

这点很重要，因为当前实现是 cost-only selector。

## 4. 需要关注的指标

用户定义的指标文档在：

- `/public/home/zhoucl/shiys/观测指标/观测指标.md`

必须重点看：

对于 18 个问题中的每一个，在 `val/test` 上都要看：

1. `selector top1_accuracy`
2. `best top1`
3. `selector mean cost`
4. `best mean cost`
5. selector 在 `top1` 上超过了哪些方法
6. selector 在 `top1` 上没有超过哪些方法
7. selector 在 `mean cost` 上超过了哪些方法
8. selector 在 `mean cost` 上没有超过哪些方法

必须有这些图：

- `top1_vs_single_methods.png`
- `mean_cost_vs_single_methods.png`
- `arm_distribution.png`
- `loss_curve.png`

当前这些指标已经整理在：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/observation_metrics`

对应说明文件：

- `/public/home/zhoucl/shiys/neural-solver-selection_unified_cost/train_logs/observation_metrics/README.md`

当前每个保留 run 下都有：

- `val_problem_metrics.csv`
- `test_problem_metrics.csv`
- `val_problem_metrics.md`
- `test_problem_metrics.md`
- `loss_curve.png`
- `val_top1_vs_single_methods.png`
- `test_top1_vs_single_methods.png`
- `val_mean_cost_vs_single_methods.png`
- `test_mean_cost_vs_single_methods.png`
- `val_arm_distribution.png`
- `test_arm_distribution.png`

当前训练 / 汇总时最重要的整体指标是：

- `val_macro_problem_oracle_ratio`
- `test_macro_problem_oracle_ratio`
- `benchmark_macro_problem_oracle_ratio`

另外，`val_top_1_ratio` 的含义是：

- `mean(predicted_top1_cost / oracle_cost)`

也就是：

- top1 选择相对 oracle 的平均 cost 比例
- `1.0` 最好
- 越小越好
