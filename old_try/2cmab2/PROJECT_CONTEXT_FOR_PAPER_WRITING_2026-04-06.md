# 2cmab2 项目完整上下文交接文档

本文档的目的不是写论文正文，而是给“完全没有上下文的另一个 AI 模型”一份尽可能完整的项目背景说明，帮助它快速接手 `2cmab2` 这条研究线，继续写论文、整理实验、做结果解释或补充分析。

请把这份文档理解成：

- 当前项目的背景交接
- 关键代码与结果的路线图
- 设计选择与历史演化记录
- 主要实验与结论摘要
- 继续工作的注意事项

---

## 1. 一句话概括当前项目

`2cmab2` 当前做的是：

> 在 `TSP + CVRP` 两类组合优化问题上，做 **initialization-only 的方法选择**，将问题实例映射到一个初始化方法；核心方法是 **contextual bandit**，主实现是 **Neural-LinUCB**，当前主数据源是 **NSS 论文提供的 synthetic train/val/test 数据集**，并额外在 **TSPLIB / CVRPLIB** 上做 benchmark 测试。

这不是 step-level operator RL，也不是 NSS 原论文那种纯监督学习分类器。

---

## 2. 当前工作区里，`2cmab2` 处在什么位置

整个工作区 `/public/home/zhoucl/shiys` 不是一个单一项目，而是一个研究工作区。和当前论文最相关的目录大致如下：

- `2cmab2/`
  - 当前这条 contextual bandit / initialization selector 主线
- `EasyNCO/`
  - 组合优化平台源码
  - `2cmab2` 早期版本曾支持从 `EasyNCO/results/test` 构建数据
- `9nss论文/neural-solver-selection/`
  - NSS 原论文与代码
  - 当前主要用于复用 encoder 结构、手工特征思路、数据集与 benchmark
- `1two_gate/`
  - 其中的 `paper_encoder.py` 是当前 `2cmab2/encoders.py` 的直接复制来源之一
- `1online/`, `1step/`
  - 这些是旧的 RL / operator-selection 主线
  - **不是** 当前 `2cmab2` 论文主线

如果另一个 AI 模型要写当前论文，默认应该把注意力放在：

- `2cmab2/`
- `9nss论文/neural-solver-selection/`
- `2cmab2/references/`

而不是去深挖 `1online/` 或 `1step/`。

---

## 3. 当前研究问题是怎么演化过来的

### 3.1 最初的方向

最初希望做的是：

- 在组合优化问题上自动选择更合适的方法
- 并且一度考虑过更复杂的 RL / 迭代阶段 operator selection

但后来研究范围被收缩和明确化，变成了：

- 只关注 `initialization`
- 只做 one-shot selection
- 当前先只覆盖 `TSP + CVRP`

原因是：

- 这两个问题在现有平台和论文数据里可选方法足够多
- 更适合做“根据实例选择方法”的研究
- 比起迭代阶段 RL，更容易先形成一个清晰、可复现、可写论文的研究问题

### 3.2 中间经历过的两个数据方案

`2cmab2` 代码其实支持两条数据路线：

1. `easynco_results`
   - 从 `EasyNCO/results/test` 扫结果 JSON
   - 再回溯 `.pt` 数据集
   - 形成共享 bandit 数据集
   - 对应统一 `12` 臂版本

2. `nss`
   - 直接读取 `9nss论文/neural-solver-selection/datasets`
   - 使用 NSS 自带的 `train / val / test` split
   - 对应当前主线统一 `9` 臂版本

### 3.3 当前主线已经切换到 NSS 数据

当前默认配置见：

- `2cmab2/default_settings.py`

其中：

- `DATA_SOURCE = "nss"`

所以当前论文主线应该理解为：

> 基于 NSS 数据集训练一个 contextual bandit selector，然后在 NSS synthetic test 以及 TSPLIB/CVRPLIB benchmark 上评测。

### 3.4 不要混淆成 NSS 的监督学习

NSS 原论文做的是 supervised selector：

- 输入实例
- 直接预测最合适的 solver / rank
- 本质是监督学习

当前 `2cmab2` 做的不是这个范式，而是：

- 外层：contextual bandit 式逐步选臂
- 内层：用 bandit 收集到的 `(sample, arm, reward)` 数据训练表示层

更准确地说：

> `2cmab2` 是 bandit / Neural-LinUCB 思路，不使用 NSS 的监督 checkpoint，也不把 NSS 训练流程照搬过来。

---

## 4. 当前最终任务定义

当前任务单元是：

- 输入：
  - 一个实例 `sample`
  - 实例属于 `tsp` 或 `cvrp`
  - 实例带有原始节点张量、手工特征、problem one-hot、可行动作 mask、各方法离线 cost / reward / rank
- 动作：
  - 在当前问题可用的方法集合里选一个 initialization method
- 输出：
  - 选中的方法 `arm`
- 反馈：
  - 该方法在该实例上的 reward

这里是 **one-step contextual bandit**。

不是：

- 多步 MDP
- 迭代阶段 operator policy
- 完整 RL episode return 优化

---

## 5. 当前项目与 RL、与 NSS 监督学习的关系

### 5.1 它是不是 RL

严格说，它更准确是 **contextual bandit**，是 RL 的简化形式：

- 有状态上下文 `x`
- 有动作 `a`
- 有探索-利用权衡
- 但没有多步状态转移和长期回报 credit assignment

所以：

- 叫“bandit”比叫“完整强化学习”更准确
- 但它属于序列决策 / online decision 范式，而不是纯静态监督学习

### 5.2 当前实现是 online 还是 offline

语义上更接近：

- **online contextual bandit 训练流程**

因为训练时的主循环是：

1. 到来一个样本
2. 当前策略选择一个 arm
3. 读取这个 arm 的 reward
4. 立刻更新 bandit 统计量
5. 每隔若干步再触发表示层训练

但数据来源上又带有明显的 offline 特征：

- 当前样本是提前准备好的静态数据集
- 每个实例上所有方法的真实 cost 都已经离线可得

所以更准确的表述是：

> 当前是“用离线已知结果数据模拟 online contextual bandit 交互”的设置。

### 5.3 与 NSS 监督学习的根本区别

NSS：

- 直接看到完整标签
- 用监督损失训练 selector
- 本质上是实例级分类 / ranking

`2cmab2`：

- 训练主循环里只对“被选中的 arm”做 bandit 更新
- 表示层训练使用 bandit 历史缓存
- 线性头承担探索

一句话总结：

> NSS 是 supervised solver selection；`2cmab2` 是 bandit-style solver selection。

---

## 6. 代码主入口与文件地图

当前最关键的文件如下。

### 6.1 训练与评测入口

- `2cmab2/run_offline.py`
  - 当前主入口
  - 负责：
    - 构建数据集
    - 划分 train/val/test
    - 构造 selector
    - 训练
    - 周期性验证
    - checkpoint 保存
    - 最终 val/test 评测
    - baseline 对比
    - summary.json 输出

- `2cmab2/evaluate_nss_benchmarks.py`
  - 用训练好的 checkpoint 在 NSS 提供的 `TSPLIB / CVRPLIB` benchmark 上测试
  - 还会生成 benchmark 对比图和 CSV

### 6.2 数据相关

- `2cmab2/build_dataset.py`
  - 定义 `InstanceSample`
  - 支持两种数据源：
    - `easynco_results`
    - `nss`
  - 当前主线最重要的是：
    - `_build_nss_joint_dataset()`
    - `build_nss_benchmark_dataset()`

- `2cmab2/features.py`
  - 提取手工特征
  - 复用 NSS manual features 的思路

- `2cmab2/arm_config.py`
  - 定义动作空间
  - 同时保留：
    - EasyNCO 12 臂版本
    - NSS 9 臂版本

- `2cmab2/reward.py`
  - 定义 reward 转换
  - 当前默认 `linear_zero_one`

### 6.3 模型相关

- `2cmab2/encoders.py`
  - 复用 NSS encoder
  - 新增：
    - `DualNSSEncoder`
    - `HybridContextEncoder`

- `2cmab2/linucb.py`
  - 经典 disjoint LinUCB baseline

- `2cmab2/neural_linucb.py`
  - 当前主模型
  - 共享 action-conditioned 表示
  - problem-specific linear heads

- `2cmab2/neural_ucb_diag.py`
  - 参考 NeuralUCB 的 diagonal approximation 版本
  - 不是当前论文主结果，但属于本项目已实现算法

- `2cmab2/baselines.py`
  - `RandomSelector`
  - `SingleBestGlobalSelector`
  - `SingleBestPerProblemSelector`
  - `OracleSelector`

### 6.4 默认配置与分析

- `2cmab2/default_settings.py`
  - 所有默认参数集中定义

- `2cmab2/analysis.py`
  - 汇总多个 `summary.json`

- `2cmab2/analysis.ipynb`
  - 更丰富的 notebook 分析

- `2cmab2/run_output_analysis.ipynb`
  - 另一份运行结果分析 notebook

### 6.5 参考资料与外部代码归档

- `2cmab2/references/INDEX.md`
  - 统一入口

- `2cmab2/references/papers/xu2022/`
  - `Neural Contextual Bandits with Deep Representation and Shallow Exploration`

- `2cmab2/references/papers/zhou2020/`
  - `Neural Contextual Bandits with UCB-based Exploration`

- `2cmab2/references/code/neural_linucb_openreview/code/run_demo.py`
  - Neural-LinUCB 论文参考代码

- `2cmab2/references/code/neuralucb_official/`
  - NeuralUCB 官方 diag 参考代码

- `2cmab2/references/code/nss_snapshot/`
  - NSS 代码快照

---

## 7. 数据来源：当前主线到底在用什么数据

### 7.1 当前默认数据源

当前默认数据源是：

- `DATA_SOURCE = "nss"`

对应目录：

- `9nss论文/neural-solver-selection/datasets/`

### 7.2 当前主训练数据

当前主训练/验证/测试 split 是 NSS 自带的：

- `TSPtrain`
- `TSPval`
- `TSPtest`
- `CVRPtrain`
- `CVRPval`
- `CVRPtest`

每个目录都有：

- `dataset.pkl`
- `raw_label.pkl`
- `results/`

### 7.3 当前 synthetic split 规模

当前主线实际使用的样本数是：

- train:
  - TSP `10000`
  - CVRP `10000`
  - 总计 `20000`
- val:
  - TSP `1000`
  - CVRP `1000`
  - 总计 `2000`
- test:
  - TSP `1000`
  - CVRP `1000`
  - 总计 `2000`

这一点在主 run 的 `console.log` 和 `summary.json` 中都能看到。

### 7.4 当前 benchmark 数据

额外 benchmark 数据是 NSS 提供的：

- `TSPLIB`
- `CVRPLIB`

当前本地数量为：

- TSPLIB: `49`
- CVRPLIB: `100`

合计 benchmark 样本数：

- `149`

### 7.5 数据分布与实例规模

根据 NSS 数据生成代码 `9nss论文/neural-solver-selection/datasets/data_utils.py` 与 `data_config.yml`，当前 synthetic 数据的大背景是：

- 问题规模随机采样于 `[50, 500)`，也就是常说的 `50-499`
- TSP 坐标来自 Gaussian mixture 生成器
  - 当 mixture mode 数为 `0` 时，退化成近似 uniform 随机分布
  - 所以整体可以理解为“uniform-like + clustered Gaussian mixture”的混合
- CVRP 坐标生成继承 TSP 的坐标生成方式
  - 再额外加 depot 和 demand

### 7.6 CVRP 的 demand / capacity 背景

根据 NSS 原生成代码，CVRP 的 raw demand 生成方式包含多种模式：

- 可能来自 `1-10`
- 或 `5-10`
- 或 `1-100`
- 或 `50-100`

然后再按 capacity 做归一化，最终写入数据集的 `demand` 实际是：

- `raw_demand / capacity`

capacity 生成也有多种策略：

- `scale`
- `triangular`
- `random_triangular`

当前主线论文写作时，最稳妥的描述方式是：

> CVRP 数据包含不同规模、不同 demand 分布与不同 capacity 机制的合成实例，最终需求值经过按 capacity 的归一化处理。

### 7.7 benchmark 的意义

当前 benchmark 不是 synthetic split 内部测试，而是更接近跨分布测试：

- 训练在 NSS synthetic train/val
- 测试在 TSPLIB/CVRPLIB

所以 benchmark 更多反映：

- 泛化能力
- OOD 情况下 selector 的稳健性

---

## 8. `InstanceSample` 是什么

`build_dataset.py` 中定义的核心数据结构是：

- `InstanceSample`

每条样本对应“一个实例”，包含：

- `uid`
  - 如 `tsp:train:123`
- `problem`
  - `tsp` 或 `cvrp`
- `global_index`
  - 原数据集中的索引
- `nodes`
  - 原始节点张量
  - TSP: `[n, 2]`
  - CVRP: `[1+n, 3]`
- `problem_onehot`
  - TSP -> `[1, 0]`
  - CVRP -> `[0, 1]`
- `manual_raw`
  - 11 维手工特征
- `manual_with_scale`
  - 12 维手工特征，额外拼了 scale
- `linucb_context`
  - `problem_onehot + manual_with_scale`
  - 14 维
- `feasible_mask`
  - 当前问题可用方法 mask
- `costs`
  - 当前实例上所有 arm 的真实 cost
- `rewards`
  - 由 rank-based 规则转出来的 reward
- `ranks`
  - 各 arm 在该实例内的名次
- `best_arm`
  - 事后最优 arm
- `result_records`
  - 方便后续 trace / 复盘的结果记录
- `source_split`
  - `train` / `val` / `test` / `tsplib` / `cvrplib` 等

这意味着：

> 当前 bandit 的输入不只是一个简单特征向量，而是一条已经聚合好了原始实例、可行动作、离线效果信息的结构化样本。

---

## 9. 动作空间：当前有哪些 arm

### 9.1 当前主线 NSS 动作空间

当前论文主线的统一 arm 空间大小是 `9`。

定义见：

- `2cmab2/arm_config.py`

#### TSP 可选方法

- `bq`
- `elg`
- `lehd`
- `t2t`
- `t2t500`
- `difusco`
- `difusco500`

#### CVRP 可选方法

- `bq`
- `elg`
- `lehd`
- `omni`
- `mvmoe`

#### 统一 9 臂空间

- `bq`
- `elg`
- `lehd`
- `t2t`
- `t2t500`
- `difusco`
- `difusco500`
- `omni`
- `mvmoe`

### 9.2 可行动作约束

虽然用了统一 arm 空间，但不是所有问题都能选所有方法。

真正的决策规则是：

- 先进入统一 arm 空间
- 再按 `feasible_mask` 过滤

例如：

- TSP 无法选择 `omni`
- CVRP 无法选择 `t2t`

### 9.3 历史上的 EasyNCO 12 臂版本

代码里还保留了早期的 EasyNCO `12` 臂版本：

- `lehd`
- `elg`
- `invit`
- `icam`
- `dact`
- `lih`
- `udc`
- `omni`
- `pointerformer`
- `difusco`
- `t2t`
- `glop`

这一支仍然能跑，但不是当前论文主结果的数据版本。

写论文时默认应把 `9` 臂 NSS 版本作为主线，除非专门介绍项目演化。

---

## 10. reward 设计

### 10.1 当前默认 reward 模式

当前默认：

- `reward_mode = "linear_zero_one"`

定义在：

- `2cmab2/reward.py`

### 10.2 公式

当前 reward 由实例内排名得到：

\[
r = 1 - \frac{rank - 1}{K - 1}
\]

其中：

- `rank = 1` 表示该实例上最优方法
- `K` 是当前实例可行动作数

所以：

- 最优方法 reward = `1`
- 最差方法 reward = `0`
- 中间线性插值

### 10.3 ties 处理

如果多个方法 cost 完全相同：

- 使用 average ranks

例如：

- `[1, 1, 3] -> [1.5, 1.5, 3.0]`

### 10.4 为什么不用 raw cost 直接训练

原因是：

- 不同问题、不同实例、不同规模的 cost 量纲差异很大
- 直接回归 cost 会让训练目标不稳定
- 当前更关心“谁更好”，而不是绝对值差多少

### 10.5 reward 与论文主指标的潜在错位

这是当前项目一个非常重要的 caveat：

- 训练时优化的是 rank-based reward
- 最终最关心的论文指标往往是 `mean_cost`

这两者不完全等价。

这也是当前很多现象的重要背景，例如：

- 训练 loss 明显下降
- 但 `val/test mean_cost` 未必同步下降

后面“重要现象与解释”部分还会再次提到。

---

## 11. 手工特征 `manual_raw` 包含哪些信息

定义在：

- `2cmab2/features.py`

当前 `manual_raw` 是统一 `11` 维。

### 11.1 前 9 维几何特征

对应 NSS manual features 思路：

1. 所有点对距离的标准差
2. 质心 `x`
3. 质心 `y`
4. 平均半径
5. 距离离散值统计 `count_distinct`
6. 最近邻距离标准差
7. 聚类比例 `cluster_ratio`
8. 离群点比例 `outlier_ratio`
9. 簇半径 `cluster_radius`

### 11.2 后 2 维 demand 特征

仅 CVRP 有意义：

10. `demand_mean`
11. `demand_std`

对于 TSP：

- 这两维补 `0`

### 11.3 `manual_with_scale`

在 `manual_raw` 基础上再拼一个 `scale`：

- 共 `12` 维

这里的 `scale` 定义是节点数：

- TSP: `n`
- CVRP: `1 + n`
  - 因为当前节点张量里包含 depot

### 11.4 LinUCB 的输入维度

LinUCB 用的是：

- `problem_onehot (2)`
- `manual_with_scale (12)`

所以共：

- `14` 维

---

## 12. 图编码器与上下文编码器

### 12.1 总体结论

当前 `2cmab2` 的上下文编码不是一个通用黑盒 MLP，而是：

- 复用了 NSS 风格的图编码器
- 再拼手工特征和 problem one-hot

### 12.2 `encoders.py` 的来源

`2cmab2/encoders.py` 前半部分基本直接复制自：

- `1two_gate/paper_encoder.py`

而 `paper_encoder.py` 本身又来自 NSS encoder 结构。

这样做是用户明确要求：

> 复用 NSS encoder 时尽量直接复制，不要自己重写。

### 12.3 不是单一共享 encoder，而是双 encoder

当前不是一个 encoder 同时吃 TSP 和 CVRP。

而是：

- `tsp_encoder = Encoder_h(TSP)`
- `cvrp_encoder = Encoder_h(CVRP)`

原因非常关键：

- TSP 输入节点维度是 `2`
  - `[x, y]`
- CVRP 输入节点维度是 `3`
  - `[x, y, demand]`

所以第一层线性映射本来就不同。

### 12.4 `DualNSSEncoder`

`DualNSSEncoder` 的作用是：

- TSP 样本走 TSP encoder
- CVRP 样本走 CVRP encoder
- 输出统一拼成相同维度

### 12.5 `Encoder_h` 的输出

NSS 风格 `Encoder_h` 默认输出：

- `256` 维图嵌入

然后在当前实现里又额外拼一个 `scale`：

- 变成 `257` 维

即：

\[
graph\_context \in \mathbb{R}^{257}
\]

### 12.6 `HybridContextEncoder`

最终上下文是：

\[
270 = 257 + 11 + 2
\]

也就是：

- 图编码：`257`
- 手工特征：`11`
- 问题 one-hot：`2`

#### 对于一个 batch，维度变化可以写成

TSP:

- 原始节点：`[B, n, 2]`
- `Encoder_h(TSP)`：`[B, 256]`
- 拼 `scale`：`[B, 257]`
- 拼 `manual_raw`：`[B, 268]`
- 拼 `problem_onehot`：`[B, 270]`

CVRP:

- 原始节点：`[B, 1+n, 3]`
- `Encoder_h(CVRP)`：`[B, 256]`
- 拼 `scale`：`[B, 257]`
- 拼 `manual_raw`：`[B, 268]`
- 拼 `problem_onehot`：`[B, 270]`

### 12.7 为什么这里是“两个 encoder”

这是后来反复确认过的设计点：

- NSS 原始 encoder 也是按 problem type 分开处理第一层输入
- 当前实现保留两个 encoder 更符合原始结构
- 这并不妨碍后续在 bandit 层面统一处理 TSP 与 CVRP

---

## 13. 当前主模型：Neural-LinUCB

### 13.1 核心思想

当前主模型文件：

- `2cmab2/neural_linucb.py`

保留的是 Neural-LinUCB / Xu 2022 这条思想的核心：

1. 深度表征层
2. 浅层线性探索层

在当前实现里分别对应：

- 深度表征：
  - `HybridContextEncoder`
  - `arm_embedding`
  - `phi_net`
- 浅层探索：
  - 最后一层线性 UCB 头 `A_inv / b / theta`

### 13.2 当前版本不是原论文逐字复现

当前版本不是 OpenReview demo 的逐字搬运，而是结合当前“实例 + 方法选择”任务做的适配版。

最重要的适配有三个：

1. 输入不是标准 tabular context，而是图实例
2. 使用 `phi(x, a)` 的 action-conditioned 设计
3. 默认使用 `problem-specific linear heads`

### 13.3 当前网络结构

完整结构可以写成：

\[
sample \rightarrow context(x) \rightarrow [context(x), e(a)] \rightarrow \phi(x,a) \rightarrow score(a)
\]

其中：

- `context(x)` 是 `HybridContextEncoder` 输出
- `e(a)` 是 arm embedding
- `phi(x,a)` 是 `phi_net` 输出

### 13.4 维度变化

当前默认配置下：

- `context(x)` 维度：`270`
- `e(a)` 维度：`16`
- 拼接后：`286`
- `phi(x,a)` 维度：`64`

也就是：

\[
[270] + [16] \rightarrow [286] \rightarrow [128] \rightarrow [64]
\]

更具体地说，`phi_net` 是：

- `Linear(286, 128)`
- `GELU`
- `Linear(128, 64)`

### 13.5 为什么要加 arm embedding

因为当前不是“一个实例只对应一个固定特征向量然后在多个线性头里分别打分”，而是：

- 同一个实例要和多个候选方法分别组合
- 需要显式告诉网络“现在在评估哪个 arm”

所以使用：

- `arm_embedding(a)`

来表示方法身份。

这意味着：

- 对同一个实例 `x`
- 不同 arm 会形成不同的联合输入

例如对同一个样本同时评估 `M` 个候选 arm：

- `context(x)`: `[1, 270]`
- `arm_emb`: `[M, 16]`
- `context(x)` 扩展成 `[M, 270]`
- 拼接后变成 `[M, 286]`
- 再经过 `phi_net` 得到 `[M, 64]`

所以：

\[
\phi(x,a_1), \phi(x,a_2), \dots, \phi(x,a_M)
\]

不是相同的，因为：

- `context(x)` 虽然相同
- 但 `arm_embedding(a_i)` 不同
- 拼接后的输入不同

### 13.6 线性头为什么按 problem 分开

当前默认：

- `problem_specific_heads = True`

也就是：

- TSP 一套 `A_inv / b / theta`
- CVRP 一套 `A_inv / b / theta`

共享的是：

- encoder
- arm embedding
- phi_net

不共享的是：

- 最后线性 UCB 头

这么做的原因是：

- 同一个方法在 TSP 和 CVRP 上的偏好可能完全不同
- 如果最后一层头也共享，会把两类问题的偏好硬平均掉

所以当前设计是：

> 共享深度表征，分开最后线性探索头。

这是当前主实现非常重要的设计点。

---

## 14. 论文符号与当前代码的对应关系

对于 Neural-LinUCB，可以这样对照：

- \(x_{t,a}\)
  - 当前不是现成扁平特征，而是由 `sample + arm_id` 动态编码
- \(\phi(x_{t,a}; w)\)
  - `HybridContextEncoder + arm_embedding + phi_net` 的输出
- \(w\)
  - 所有深度表征网络参数
- \(\theta_t\)
  - 当前 problem 对应线性头的 `theta`
- \(A_t\)
  - 代码里不显式存 `A`，而是存 `A_inv`
- \(b_t\)
  - 奖励加权特征和

当前 `theta` 在代码里是：

- 一个 `numpy` 向量，维度 `[64]`

当前 `phi(x,a)` 也是：

- `[64]`

所以当前打分公式就是：

\[
mean(a) = \theta^\top \phi(x,a)
\]

\[
bonus(a) = \sqrt{\phi(x,a)^\top A^{-1}\phi(x,a)}
\]

\[
score(a) = mean(a) + \alpha \cdot bonus(a)
\]

---

## 15. 一个样本进来之后，Neural-LinUCB 如何选 arm

这一部分很关键，因为写论文方法部分时经常要讲清楚“前向到底发生了什么”。

### 15.1 第一步：取当前样本的可行动作

由 `sample.feasible_mask` 决定。

例如：

- 对 TSP，只保留 `bq/elg/lehd/t2t/t2t500/difusco/difusco500`
- 对 CVRP，只保留 `bq/elg/lehd/omni/mvmoe`

### 15.2 第二步：一次性编码所有候选动作

调用：

- `_encode_actions(sample, feasible_arms)`

具体过程：

1. `context_encoder.encode_samples([sample]) -> [1, 270]`
2. `arm_embedding(arm_ids) -> [M, 16]`
3. `base.expand(M, -1) -> [M, 270]`
4. 拼接 -> `[M, 286]`
5. `phi_net` -> `[M, 64]`

### 15.3 第三步：逐个计算 mean / bonus / score

对每个候选 arm：

- `mean = theta @ phi`
- `bonus = sqrt(phi^T A_inv phi)`
- `score = mean + alpha * bonus`

### 15.4 第四步：warm start / epsilon / score 选择

选择逻辑顺序是：

1. 如果有 arm 还没达到 `initial_pulls`
   - 优先选这些 arm
   - `selection_reason = warm_start`
2. 否则如果在训练阶段命中 `train_epsilon`
   - 随机探索
   - `selection_reason = epsilon_random`
3. 否则
   - 选 score 最大的 arm
   - `selection_reason = score`

### 15.5 选择阶段的日志细节

当前会记录每个候选 arm 的：

- `arm_id`
- `arm_name`
- `pull_count`
- `mean`
- `bonus`
- `score`

所以后续复盘时可以知道：

- 为什么选了这个 arm
- 当时其它 arm 分数分别是多少

---

## 16. 一个训练 step 之后，Neural-LinUCB 如何更新

### 16.1 在线 bandit 更新

收到 `(sample, arm, reward)` 后：

1. 先缓存当前被拉取时的 `theta_at_pull`
2. 编码当前 `(sample, arm)` 得到 `phi`
3. 用 Sherman-Morrison 更新 `A_inv`
4. 更新 `b += reward * phi`
5. 更新 `theta = A_inv @ b`

代码语义上就是标准的 last-layer LinUCB 更新。

### 16.2 同时更新统计量

还会更新：

- 全局 `arm_counts`
- 全局 `arm_reward_sums`
- 某个 problem 内部的 `problem_arm_counts`
- 某个 problem 内部的 `problem_arm_reward_sums`

### 16.3 同时写入两个历史缓存

这一点是后来反复讨论并修改过的设计。

当前有两个不同的 buffer：

1. `representation_history`
   - 给表示层训练使用
   - 默认 recent window
2. `linear_head_history`
   - 给线性头重建使用
   - 可以比表示层窗口更大

这么分开的原因是：

- 表示层训练代价更高，适合 recent buffer
- 线性头 rebuild 较便宜，可以利用更多历史

### 16.4 何时触发表征训练

当：

- `total_steps % train_every == 0`

时触发：

- `fit_representation()`

当前主 run 默认：

- `train_every = 100`

---

## 17. `fit_representation()` 具体在做什么

这是当前项目最重要、也最容易被误解的部分之一。

### 17.1 它更新的是哪些网络

在当前实现里，`fit_representation()` 更新的是：

- `HybridContextEncoder`
- `arm_embedding`
- `phi_net`

也就是：

- 整个深度表示层

它**不会**直接用梯度更新最后的线性 UCB 头。

最后线性头是通过：

- bandit 增量更新
- 或 rebuild linear head

来更新的。

### 17.2 当前表示层训练目标

当前 loss 目标是：

\[
\text{pred}_i = {\theta_i^{pull}}^\top \phi(x_i,a_i)
\]

\[
L = \frac{1}{B} \sum_{i=1}^{B} (\text{pred}_i - r_i)^2
\]

其中：

- \(x_i, a_i\)
  - 历史里第 `i` 条 bandit 记录
- \(\theta_i^{pull}\)
  - 当时选中这个 arm 时缓存下来的线性头参数
- \(r_i\)
  - 当时观察到的 reward

### 17.3 为什么 `pred = torch.sum(phi * theta_tensor, dim=1)`

因为：

- `phi`: `[B, 64]`
- `theta_tensor`: `[B, 64]`

二者逐元素相乘后还是 `[B, 64]`，再对最后一维求和，就得到每个样本自己的内积：

- `pred`: `[B]`

这不是“batch 内所有样本求总和”，而是：

- 每个样本一条预测值

### 17.4 为什么 loss 再做 `mean`

因为：

- `pred - target` 得到 `[B]`
- 平方后还是 `[B]`

最后 `torch.mean(...)` 是对 batch 内所有样本的 MSE 做平均，得到一个标量 loss。

### 17.5 为什么用 reward 做 target

因为当前表示层要学的是：

> 在当前 bandit 收集到的监督下，让 \(\phi(x,a)\) 更适合最后的线性头去预测 reward。

当前 reward 不是 raw cost，而是 rank-based `0~1` 值。

所以当前表示层训练的真正目标是：

> 让 \(\theta_i^{pull}\) 和当前 \(\phi(x_i,a_i)\) 的内积，逼近该 `(sample, arm)` 的 bandit reward。

### 17.6 为什么 current loss 降低，不一定代表 mean_cost 一定降低

因为当前 loss 只是在优化：

- 对 recent buffer 内已选 arm 的 reward 拟合能力

它没有直接优化：

- top1 accuracy
- mean_cost
- mean regret

而最终选择动作又是通过：

- argmax over score

完成的。

所以可能出现：

- 表示层越来越会拟合 bandit 历史监督
- 但最终 argmax 结果没有显著改善

这是当前项目已经观察到的重要现象之一。

### 17.7 mini-batch 的语义

当前主循环仍然是一条样本一步步 online 决策。

只有表示层训练部分才用 mini-batch。

也就是说：

- bandit 决策是 sample-wise online
- representation fitting 是 batch SGD

这和传统 RL 里把 replay buffer 按 batch 训练有点类似，但当前任务是 contextual bandit，不是多步 RL。

---

## 18. 表示层训练窗口与线性头重建窗口

这是一个后来明确做出的重要设计修正。

### 18.1 当前方案

当前实现把两个窗口分开：

- `representation_buffer_size`
  - 表示层 recent window
- `linear_head_buffer_size`
  - 线性头重建窗口

默认主 run 是：

- `representation_buffer_size = 5000`
- `linear_head_buffer_size = 10000`

### 18.2 为什么要分开

因为如果两个都只看最近很小窗口：

- 表示层可能忘掉早期模式
- 线性头也只保留最近经验，过于短视

但如果两个都用全历史：

- 表示层训练会越来越慢
- 容易被高频 arm / 高频问题样本主导

所以当前折中方案是：

- 表示层看较近窗口
- 线性头重建看更长窗口

### 18.3 当前实现不是“每次从头重训整个网络”

这一点很重要。

当前实现里：

- `fit_representation()` 是接着当前参数继续训练
- 不是每次都把 encoder/phi_net 重新初始化

这与论文中某些更偏理论分析的表述不完全一致，但从当前工程目标看更实用。

---

## 19. 表示层训练时的数据平衡

当前实现为了抑制高频 arm 主导，加入了平衡采样。

### 19.1 当前主 run 配置

- `representation_balance_mode = "problem_arm"`
- `representation_balance_power = 0.5`
- `representation_loss_reweight = True`

### 19.2 含义

按 `(problem, arm)` 分组，低频组获得更高采样概率和更高 loss 权重。

这样做的目的：

- 避免某个高频 arm 完全淹没低频 arm
- 避免某个问题类型主导表示层

这也是当前“anti-collapse / problem-heads”这批实验的重要组成部分。

---

## 20. 当前辅助算法：LinUCB 与 NeuralUCB-Diag

### 20.1 LinUCB

文件：

- `2cmab2/linucb.py`

特点：

- 经典 disjoint LinUCB
- 每个 arm 一套 `A_inv / b`
- 输入是 14 维 `linucb_context`
- 支持 action mask

它是当前最标准、最容易解释的 bandit baseline。

### 20.2 NeuralUCB-Diag

文件：

- `2cmab2/neural_ucb_diag.py`

特点：

- 参考 Zhou 2020 官方 diag 代码风格
- 共享 action-conditioned reward network
- 使用 diagonal uncertainty approximation
- 支持：
  - `arm_embedding`
  - `arm_onehot`
  - `arm_bias`
  - `problem-arm safety penalty`

### 20.3 当前论文主结果不是 NeuralUCB-Diag

虽然 `neural_ucb_diag` 已经实现并支持日志、checkpoint、周期性评估，但当前主论文主线与最佳结果主要围绕：

- `neural_linucb`

展开。

---

## 21. `run_offline.py` 的完整训练流程

当前最重要的主流程都在：

- `2cmab2/run_offline.py`

### 21.1 它做的事

1. 读取命令行参数
2. 构造输出目录
3. 构建共享数据集
4. 决定 train/val/test split
5. 构造 selector
6. 在 train 上按 online/bandit 语义训练
7. 按 step 做 checkpoint
8. 按 step 做周期性验证
9. 根据 val 指标选择 best checkpoint
10. 训练结束后恢复 best checkpoint
11. 在 val/test 上做 greedy / ucb 评测
12. 跑 baseline 对比
13. 输出 `summary.json`

### 21.2 当前 best checkpoint 规则

当前默认：

- `use_best_checkpoint_for_final_eval = True`
- `best_checkpoint_metric = "val_greedy.overall.mean_cost"`
- `best_checkpoint_mode = "min"`

所以最终 test 结果默认来自：

- val_greedy.mean_cost 最好的那个 checkpoint

不是简单地用最后一步参数。

### 21.3 当前 epoch 的含义

当前 `epochs` 的含义不是传统 supervised training 的大 epoch。

在当前 bandit 语义下，它表示：

- 把整个 train split 按 bandit 顺序完整跑几遍

当前主 run 一般是：

- `epochs = 1`

所以最合理的理解方式是：

> 一个长的 online contextual bandit 训练过程，内部每 `train_every` 步做一次表示层更新。

---

## 22. 当前日志与输出文件体系

这一部分非常重要，因为另一个 AI 模型如果要继续分析实验，最可能直接读这些输出文件。

### 22.1 主输出目录

当前主 run 目录是：

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/`

### 22.2 核心文件

#### `summary.json`

最终实验摘要。

包括：

- config
- splits
- checkpoint_selection
- selector 指标
- baselines 指标
- artifact 路径

#### `console.log`

阶段日志。

主要看：

- 数据集构建进度
- 训练开始/结束
- checkpoint 保存
- 周期性 val
- 最终评测摘要

#### `neural_linucb.log`

详细文本日志。

包括：

- 每一步 train 时选了哪个 arm
- 每个候选 arm 的 `mean / bonus / score`
- 每次 update 的 reward / regret / selected_cost / best_cost
- 每次 `fit_representation()` 的 loss 摘要

#### `periodic_val.jsonl`

周期性验证记录。

每个 step 点会记录：

- `val_greedy`
- `val_ucb`

的完整指标块。

#### `traces/*.jsonl`

结构化轨迹。

重要文件包括：

- `selector_train_ucb.jsonl`
- `selector_val_greedy.jsonl`
- `selector_test_greedy.jsonl`
- `selector_val_ucb.jsonl`
- `selector_test_ucb.jsonl`

每一行都记录一次 decision 的详细信息。

### 22.3 trace 里最重要的字段

对每个样本，trace 会写出：

- `selected_arm`
- `selected_arm_name`
- `selection_reason`
- `selected_mean`
- `selected_bonus`
- `selected_score`
- `selected_reward`
- `selected_cost`
- `selected_result`
- `best_arm`
- `best_cost`
- `regret`
- `top1`
- `arm_details`

其中 `selected_result` 是为了复盘方便特地做的紧凑结果记录，例如：

```json
{"global_index": 6, "name": null, "score": 19.5989}
```

### 22.4 checkpoint 文件

当前支持 step 级保存。

例如：

- `checkpoints/step_0001000.pt`
- `checkpoints/step_0002000.pt`
- ...
- `checkpoints/best.pt`
- `checkpoints/latest.pt`

### 22.5 监控文件

#### `monitor.log`

记录了训练时的：

- `pgrep`
- `nvidia-smi`
- console tail
- detail log tail
- periodic val tail

主要用于服务器上监控 detached 训练进程。

---

## 23. 当前主实验配置

主 run：

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server`

从 `summary.json` 可读出的关键配置如下。

### 23.1 算法与数据

- method: `neural_linucb`
- data_source: `nss`
- reward_mode: `linear_zero_one`

### 23.2 训练设置

- epochs: `1`
- seed: `0`
- alpha: `1.0`
- reg: `1.0`
- hidden_dim: `64`
- arm_embed_dim: `16`
- lr: `1e-3`
- initial_pulls: `2`
- train_every: `100`
- representation_steps: `10`
- representation_buffer_size: `5000`
- representation_batch_size: `32`
- representation_balance_mode: `problem_arm`
- representation_balance_power: `0.5`
- representation_loss_reweight: `true`
- linear_head_buffer_size: `10000`
- train_epsilon: `0.1`
- problem_specific_heads: `true`

### 23.3 评测与保存

- eval_every_steps: `1000`
- eval_log_every: `500`
- checkpoint_every_steps: `1000`
- use_best_checkpoint_for_final_eval: `true`
- best checkpoint metric: `val_greedy.overall.mean_cost`

### 23.4 实际运行命令

从 `monitor.log` 可恢复当前主 run 的启动命令近似为：

```bash
conda run --no-capture-output -n easynco_zhoucl \
python 2cmab2/run_offline.py \
  --data-source nss \
  --method neural_linucb \
  --epochs 1 \
  --representation-batch-size 32 \
  --eval-every-steps 1000 \
  --eval-log-every 500 \
  --checkpoint-every 0 \
  --checkpoint-every-steps 1000 \
  --disable-progress \
  --no-quiet \
  --output-dir /public/home/zhoucl/shiys/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server
```

其余超参数使用的是 `default_settings.py` 默认值。

---

## 24. 当前主实验结果：synthetic val/test

主结果见：

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/summary.json`

### 24.1 best checkpoint

- best checkpoint: `checkpoints/best.pt`
- best step: `19000`
- best metric:
  - `val_greedy.overall.mean_cost = 13.105003054421164`

### 24.2 val greedy

overall:

- count: `2000`
- mean_reward: `0.7695`
- top1_accuracy: `0.4525`
- mean_regret: `0.0810`
- mean_cost: `13.1050`

by problem:

- TSP:
  - top1: `0.332`
  - mean_cost: `7.9209`
- CVRP:
  - top1: `0.573`
  - mean_cost: `18.2891`

### 24.3 test greedy

overall:

- count: `2000`
- mean_reward: `0.7654583333333334`
- top1_accuracy: `0.441`
- mean_regret: `0.08856422259251262`
- mean_cost: `13.207625660925933`

by problem:

- TSP:
  - top1: `0.331`
  - mean_cost: `8.051783500253812`
- CVRP:
  - top1: `0.551`
  - mean_cost: `18.363467821598054`

### 24.4 test ucb

overall:

- top1_accuracy: `0.4315`
- mean_cost: `13.209832728511664`

因此当前主结果中：

- `test_greedy` 略优于 `test_ucb`

### 24.5 baseline 对比（synthetic test）

#### random

- top1: `0.1795`
- mean_cost: `13.4845`

#### single_best_global

- top1: `0.338`
- mean_cost: `13.3093`

#### single_best_per_problem

- top1: `0.338`
- mean_cost: `13.3093`

#### oracle

- top1: `1.0`
- mean_cost: `13.1191`

### 24.6 结论

在 synthetic test 上，当前 selector：

- 明显优于 random
- 优于 `single_best_global`
- 也优于 `single_best_per_problem`
- 但距离 oracle 仍有差距

---

## 25. 当前 synthetic test 上的选臂行为

根据：

- `traces/selector_test_greedy.jsonl`

当前 `test_greedy` 的选臂分布是：

### 25.1 TSP test

- 总数：`1000`
- top1: `0.331`
- mean_cost: `8.0518`

选臂分布：

- `bq`: `430`
- `t2t500`: `304`
- `elg`: `177`
- `difusco500`: `84`
- `lehd`: `4`
- `t2t`: `1`

### 25.2 CVRP test

- 总数：`1000`
- top1: `0.551`
- mean_cost: `18.3635`

选臂分布：

- `omni`: `515`
- `lehd`: `228`
- `bq`: `148`
- `elg`: `69`
- `mvmoe`: `40`

### 25.3 简要解释

当前 selector 没有坍缩成单一 arm，但也表现出明显偏好：

- TSP 偏向 `bq + t2t500 + elg`
- CVRP 偏向 `omni + lehd + bq`

这和 benchmark 上的行为有相似点，但不完全一致。

---

## 26. benchmark 测试：TSPLIB / CVRPLIB

benchmark 结果目录：

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/`

主摘要：

- `summary.json`
- `summary.txt`

### 26.1 benchmark 总体结果

selector:

- count: `149`
- mean_reward: `0.720917225950783`
- top1_accuracy: `0.40268456375838924`
- mean_regret: `3.660283321700183`
- mean_cost: `52.183436404141645`

single_best_global:

- top1_accuracy: `0.35570469798657717`
- mean_cost: `49.06518056728696`

single_best_per_problem:

- top1_accuracy: `0.35570469798657717`
- mean_cost: `49.06518056728696`

oracle:

- top1_accuracy: `1.0`
- mean_cost: `48.52315308244146`

### 26.2 TSPLIB

selector:

- count: `49`
- top1_accuracy: `0.4489795918367347`
- mean_cost: `8.21157299013478`

single_best:

- top1_accuracy: `0.4489795918367347`
- mean_cost: `8.27312845113326`

oracle:

- mean_cost: `8.142167115105343`

### 26.3 CVRPLIB

selector:

- count: `100`
- top1_accuracy: `0.38`
- mean_cost: `73.72964947700501`

single_best:

- top1_accuracy: `0.31`
- mean_cost: `69.05328610420227`

oracle:

- mean_cost: `68.30983620643616`

### 26.4 benchmark 的直接结论

TSPLIB:

- selector 的 top1 与 `single_best` 持平
- 但 selector 的 mean cost 更低

CVRPLIB:

- selector 的 top1 高于 `single_best`
- 但 selector 的 mean cost 更差

所以 benchmark 上非常明显地体现出：

> `top1 accuracy` 与 `mean cost` 不一定一致。

---

## 27. TSPLIB 上“selector 和 bq top1 完全一样”不是 bug

这是一个已经专门核查过的重要现象。

### 27.1 现象

在 TSPLIB 上：

- selector top1 = `22/49 = 0.448979...`
- `bq` 的 top1 也是 `22/49 = 0.448979...`

看起来像图画错了，但实际上不是。

### 27.2 核查来源

核查过的文件包括：

- `benchmark_eval_tsplib_cvrplib/tsplib_single_method_compare.csv`
- `benchmark_eval_tsplib_cvrplib/selector_benchmark.trace.jsonl`

### 27.3 selector 在 TSPLIB 上的真实选择分布

TSPLIB 共 `49` 个实例。

selector 实际选择：

- `bq`: `35`
- `elg`: `7`
- `t2t500`: `7`

其中 top1 命中数：

- `bq`: `17`
- `elg`: `3`
- `t2t500`: `2`

总 top1 命中：

- `22`

### 27.4 TSPLIB 上 oracle 最优臂分布

TSPLIB 上事后最优方法分布为：

- `bq`: `22`
- `elg`: `7`
- `t2t`: `6`
- `t2t500`: `5`
- `lehd`: `4`
- `difusco500`: `3`
- `difusco`: `2`

### 27.5 正确解释

正确解释不是“selector 完全等价于 bq”，而是：

- selector 在 top1 命中数上恰好与 `bq` 相同
- 但 selector 会在部分实例上选择 `elg` 或 `t2t500`
- 这些选择虽然没有改变 top1 总命中数，但降低了平均 cost

所以：

> TSPLIB 上 selector 与 `bq` 的 top1 一样，但 mean cost 更好，这不是作图错误，而是 top1 与 cost 指标不一致的真实体现。

---

## 28. benchmark 上单方法对比

### 28.1 TSPLIB

从 `tsplib_single_method_compare.csv` 可得：

selector 相比所有单一方法，在 mean cost 上都更好。

在 top1 上：

- 与 `bq` 持平
- 超过其它所有单方法

### 28.2 CVRPLIB

从 `cvrplib_single_method_compare.csv` 可得：

在 top1 上：

- selector 超过 `omni / bq / lehd / mvmoe`
- 但不如 `elg`

在 mean cost 上：

- selector 优于 `mvmoe / lehd`
- 但不如 `elg / omni / bq`

这说明 benchmark 尤其是 CVRPLIB 上存在明显分布偏移问题。

---

## 29. 当前观测到的重要现象与解释

### 29.1 训练 loss 明显下降，但 val/test mean cost 改善有限

这是目前最核心的现象之一。

当前最合理的解释包括：

1. 表示层 loss 目标和最终 mean cost 不完全一致
   - 当前优化的是
     - `theta_at_pull^T phi(x,a) -> reward`
   - 不是直接优化 `mean_cost`

2. reward 是 rank-based `0~1`
   - 它只关心实例内相对顺序
   - 不关心不同错误选择之间真实 cost 差多大

3. 当前表示层监督只来自“被选中的 arm”
   - 不是每一步都用全臂标签训练
   - 所以监督信号相对稀疏

4. 最终决策是 argmax over score
   - 一个很小的 score 排序变化就可能改变最终 arm
   - 也可能 loss 继续下降但 argmax 基本不变

5. benchmark 分布与 synthetic 分布不同
   - 这会进一步放大“训练目标与测试指标”的错位

### 29.2 top1 accuracy 与 mean cost 并不等价

当前无论 synthetic 还是 benchmark，都多次观察到：

- 更高的 top1 不一定 mean cost 更低
- 反过来也成立

原因是：

- top1 只看是否选到最优 arm
- mean cost 还关心“选错时错得有多离谱”

### 29.3 当前瓶颈可能不只是超参数

此前的 stage1 / stage2 tuning 已经表明：

- 在一批近邻超参数之间，性能差异很小
- 很多设置都只能接近 `single_best_per_problem`
- 很难靠小修小补取得大提升

这说明当前瓶颈更可能来自：

- reward 设计
- 表示层训练目标
- bandit 监督稀疏性
- synthetic -> benchmark 的分布偏移

而不只是：

- 学习率没调好
- alpha 没调好

---

## 30. 历史实验与调参脉络

当前项目不是一口气得到主 run 的，中间经历过几批实验。

### 30.1 早期 12 臂 / EasyNCO 路线

较早的输出目录，例如：

- `2cmab2/outputs/2026-03-23-23-07-19_neural_linucb`
- `2cmab2/outputs/2026-03-25-18-53-19_neural_linucb`

这批实验更多对应：

- `easynco_results` 路线
- 12 臂动作空间
- 较早期日志和分析

这些目录还值得保留作历史参考，但不是当前主线结果。

### 30.2 stage1 小规模调参

实验脚本目录：

- `2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/`

特点：

- `1000 / problem`
- 8 组超参数
- 主要为了判断：
  - 是否表示层训练过强
  - 是否探索不足
  - reward 口径 / 线性头窗口 / freeze encoder 是否敏感

### 30.3 stage2 5k/problem 调参

实验脚本目录：

- `2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/`

正式分析文档：

- `2cmab2/analysis/STAGE2_NEURAL_LINUCB_FORMAL_ANALYSIS_2026-04-02.md`

结论大意：

- 多组超参数差异非常小
- 默认配置已经是最优或近似最优
- selector 只能稳定接近 `single_best_per_problem`
- 当前性能瓶颈更可能来自方法本身而不是近邻超参数

### 30.4 problem-specific heads / anti-collapse 实验

还有一批目录体现了后续围绕“collapse、problem-heads、balanced replay”的修正，例如：

- `2cmab2/outputs/2026-04-02_neural_linucb_problem_heads_1k_seed0`
- `2cmab2/outputs/2026-04-02_neural_linucb_anticollapse_1k_seed0`
- `2cmab2/outputs/2026-04-02_nss_neural_linucb_ph_safe_bs64_seed0`
- `2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue/`

这些实验的价值主要在于：

- 验证 problem-specific heads 是否合理
- 缓解单一 arm 主导表示层训练的问题
- 测试 balanced replay 和 safety 机制

### 30.5 一个重要 caveat：部分分析文件是“中途分析”，不是最终结论

例如：

- `analysis_partial/`
- `analysis_from_existing_logs/`

里面有些结论对应的是：

- step=`6000`
- 或训练尚未完成时的 best checkpoint

而不是最终主 run 的 `best step = 19000`。

因此：

> 如果另一个 AI 模型在写论文时看到 `analysis_partial` 与 `summary.json` 不一致，应以主 run 的 `summary.json` 和 benchmark `summary.json` 为准。

---

## 31. 当前主 run 中，验证轨迹的变化

从：

- `periodic_val.jsonl`

可以看到当前主 run 的总体趋势。

### 31.1 早期

step `1000`：

- `val_greedy.top1 = 0.3545`
- `val_greedy.mean_cost = 13.1892`

### 31.2 中期

例如 step `5000`：

- `val_greedy.top1 = 0.4310`
- `val_greedy.mean_cost = 13.1120`

### 31.3 最优点

step `19000`：

- `val_greedy.top1 = 0.4525`
- `val_greedy.mean_cost = 13.1050`

### 31.4 最后一步

step `20000`：

- `val_greedy.top1 = 0.4395`
- `val_greedy.mean_cost = 13.1122`

所以：

- 最优点不是最后一步
- 使用 best checkpoint 恢复再做最终 test 是合理且必要的

---

## 32. 当前 `analysis.py` / notebook 的作用

### 32.1 `analysis.py`

作用是：

- 聚合多个 `summary.json`
- 生成：
  - `runs_manifest.csv`
  - `selector_test_greedy.csv`
  - `selector_test_ucb.csv`
  - `baseline_test_greedy.csv`
  - `selector_overall_ranking.csv`
  - `best_summary.json`
  - `report.md`

### 32.2 `analysis.ipynb` 与 `run_output_analysis.ipynb`

这些 notebook 用来做更丰富的图和交互式分析。

当前它们的价值主要是：

- 看 loss 曲线
- 看 train / val / test 选臂分布
- 看 selector vs baselines vs single methods
- 看 benchmark 图

如果另一个 AI 模型需要继续做结果图，先看这两个 notebook 即可。

---

## 33. 参考论文与代码归档情况

当前 `2cmab2/references/` 已经归档了论文与参考代码。

### 33.1 Neural-LinUCB

- 论文：
  - `2cmab2/references/papers/xu2022/xu2022.pdf`
  - `2cmab2/references/papers/xu2022/xu2022.txt`
- 参考代码：
  - `2cmab2/references/code/neural_linucb_openreview/code/run_demo.py`

此外，这个目录下还有两份讲解文档：

- `neural_linucb_论文与代码详解.md`
- `neural_linucb_详解.md`

### 33.2 NeuralUCB

- 论文：
  - `2cmab2/references/papers/zhou2020/zhou20a.pdf`
- 代码：
  - `2cmab2/references/code/neuralucb_official/`

### 33.3 NSS

- 论文资料：
  - `2cmab2/references/papers/nss/`
- 代码快照：
  - `2cmab2/references/code/nss_snapshot/`

### 33.4 AutoSAEA

- 主要作为 reward / TL-UCB 风格参考
- 已归档到：
  - `2cmab2/references/papers/autosaea/`
  - `2cmab2/references/code/autosaea_snapshot/`

---

## 34. 已有的说明文档与对比文档

当前仓库里还有一些直接相关的说明文档：

- `0label/实现讲解_模型设计与训练流程.md`
  - 重点讲 Neural-LinUCB 的结构和训练流程
- `0label/2cmab_vs_2cmab2_对比分析.md`
  - 对比旧版 `2cmab` 与当前 `2cmab2`
- `2cmab2/analysis/STAGE2_NEURAL_LINUCB_FORMAL_ANALYSIS_2026-04-02.md`
  - 5k/problem 调参分析

这些文档不是当前主结果本身，但对写方法和 related explanation 很有帮助。

---

## 35. 当前项目里最容易混淆的几点

### 35.1 不要把 `2cmab2` 和 NSS 原训练过程混为一谈

当前只复用了：

- NSS encoder 结构
- NSS 手工特征思路
- NSS synthetic / benchmark 数据集

没有直接使用：

- NSS 监督训练流程
- NSS 训练好的 selector checkpoint

### 35.2 不要把当前主结果和早期 12 臂结果混在一起

当前主论文结果对应：

- NSS 数据
- 9 臂空间

而不是：

- EasyNCO results/test
- 12 臂空间

### 35.3 不要把 `analysis_partial` 里的旧 best step 当成最终 best

有些中途分析文件显示：

- best step = `6000`

但当前最终主 run 的 best step 是：

- `19000`

### 35.4 不要只看 top1

当前多个阶段都说明：

- `top1_accuracy`
- `mean_cost`

不能互相替代。

如果只能选一个主指标，当前更稳妥的是：

- `mean_cost`

---

## 36. 如果要继续这个项目，另一个 AI 应该优先读哪些文件

建议阅读顺序如下。

### 第一优先级：先搞清当前主线

1. `2cmab2/default_settings.py`
2. `2cmab2/arm_config.py`
3. `2cmab2/build_dataset.py`
4. `2cmab2/features.py`
5. `2cmab2/encoders.py`
6. `2cmab2/neural_linucb.py`
7. `2cmab2/run_offline.py`
8. `2cmab2/evaluate_nss_benchmarks.py`

### 第二优先级：看当前结果

1. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/summary.json`
2. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/console.log`
3. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/neural_linucb.log`
4. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/periodic_val.jsonl`
5. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/traces/selector_test_greedy.jsonl`
6. `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/summary.json`

### 第三优先级：看参考与解释

1. `2cmab2/references/INDEX.md`
2. `2cmab2/references/code/neural_linucb_openreview/code/run_demo.py`
3. `2cmab2/references/code/nss_snapshot/model.py`
4. `0label/实现讲解_模型设计与训练流程.md`
5. `2cmab2/analysis/STAGE2_NEURAL_LINUCB_FORMAL_ANALYSIS_2026-04-02.md`

---

## 37. 当前最稳妥的论文表述建议

如果另一个 AI 要据此写论文方法与实验，当前最稳妥的说法是：

1. 任务定义
   - `TSP + CVRP` initialization method selection
   - one-shot contextual bandit

2. 输入表示
   - NSS-style graph encoder
   - manual features
   - problem indicator

3. 动作建模
   - 统一 arm 空间
   - feasible action mask

4. 主方法
   - Neural-LinUCB
   - shared action-conditioned representation
   - problem-specific linear exploration heads

5. 奖励
   - instance-wise rank-based reward

6. 训练方式
   - online bandit-style sample-by-sample interaction over offline prepared data
   - periodic representation fitting
   - best checkpoint selected by periodic validation mean cost

7. 结果
   - synthetic test 优于 random 与 single-best
   - benchmark 上 TSPLIB / CVRPLIB 呈现更复杂的 top1 vs mean cost 差异

---

## 38. 当前最大的问题和可继续挖的点

如果另一个 AI 后续要讨论“未来工作”或“当前不足”，最值得提的点包括：

1. reward 与最终指标存在错位
   - 训练优化 rank-based reward
   - 论文主指标是 mean cost

2. bandit 监督依然较稀疏
   - 训练更新只用被选中的 arm

3. synthetic -> benchmark 分布偏移明显
   - 尤其 CVRPLIB 上表现突出

4. 当前 action-conditioned `phi(x,a)` 是工程适配版
   - 有较强实用性
   - 但与最标准的 tabular contextual bandit 设定不同

5. 当前多问题统一方式是：
   - 共享表示
   - 分开线性头
   - 仍可继续探索更强的跨问题迁移结构

6. top1 和 mean cost 的矛盾值得专门分析
   - 尤其 benchmark 部分

---

## 39. 复现实验的最小命令清单

### 39.1 主训练

```bash
conda run --no-capture-output -n easynco_zhoucl \
python 2cmab2/run_offline.py \
  --data-source nss \
  --method neural_linucb \
  --epochs 1 \
  --representation-batch-size 32 \
  --eval-every-steps 1000 \
  --eval-log-every 500 \
  --checkpoint-every 0 \
  --checkpoint-every-steps 1000 \
  --disable-progress \
  --no-quiet \
  --output-dir 2cmab2/outputs/<your_run_name>
```

### 39.2 benchmark 测试

```bash
conda run --no-capture-output -n easynco_zhoucl \
python 2cmab2/evaluate_nss_benchmarks.py \
  --run-dir 2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server \
  --checkpoint best.pt \
  --device cuda
```

### 39.3 聚合 summary

```bash
conda run --no-capture-output -n easynco_zhoucl \
python 2cmab2/analysis.py \
  --input-root 2cmab2/outputs \
  --output-dir 2cmab2/analysis/latest_local
```

---

## 40. 当前最重要的结果文件路径总表

### 主 run

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/summary.json`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/console.log`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/neural_linucb.log`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/periodic_val.jsonl`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/checkpoints/best.pt`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/traces/selector_test_greedy.jsonl`

### benchmark

- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/summary.json`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/summary.txt`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/selector_benchmark.trace.jsonl`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/tsplib_single_method_compare.csv`
- `2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/cvrplib_single_method_compare.csv`

### 解释与分析

- `0label/实现讲解_模型设计与训练流程.md`
- `2cmab2/analysis/STAGE2_NEURAL_LINUCB_FORMAL_ANALYSIS_2026-04-02.md`
- `2cmab2/references/INDEX.md`

---

## 41. 最后一句总结

当前 `2cmab2` 已经形成了一个比较完整、可运行、可复现、可分析的研究原型：

- 任务定义清晰：
  - `TSP + CVRP` 的 initialization-only solver selection
- 方法路线清晰：
  - contextual bandit
  - 主方法是 Neural-LinUCB
- 代码结构清晰：
  - 数据、模型、训练、benchmark、分析已分开
- 结果也已经具备论文讨论价值：
  - synthetic 上优于 strong baselines
  - benchmark 上呈现出有研究价值的泛化现象与指标矛盾

如果另一个 AI 模型要继续写论文，最重要的是不要混淆：

- 当前主线是 `nss + 9 arms + neural_linucb + problem-specific heads`
- 不是早期 `EasyNCO 12-arm` 版本
- 也不是 NSS 原始监督学习 selector

