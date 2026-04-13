# `1online` 实现说明

本文档说明 `1online/` 这套单实例 online RL 框架的实现思路、代码结构与运行数据流。目标是：**文档不长，但把核心设计讲清楚**。

## 1. 目标与范围

`1online/` 只做一件事：

- 对 **一个 TSPLIB TSP 实例** 单独训练一个策略；
- 训练样本不是离线数据集，而是对这个实例**在线生成几何增强 batch**；
- 每次 rollout：
  - 先选一次初始化方法；
  - 再固定迭代步数，每一步选一个 operator；
- 用真实求解结果直接更新策略，不依赖离线标签。

当前范围固定为：

- 只做 `TSP`
- 只做“初始化 + 多步迭代算子选择”这一种 online RL 方案
- 不修改 `EasyNCO`
- 强调**细粒度日志与 trace**，方便后续分析是否 collapse

## 2. 整体数据流

训练时的数据流如下：

1. 读取一个 TSPLIB 基实例；
2. 将坐标归一化到单位棋盘；
3. 在线生成一个 augmentation batch；
4. 对 batch 中每个样本先选初始化器；
5. 得到初始解后，固定 `rollout_steps`，每一步选择一个 operator；
6. 记录每一步动作、概率、长度变化；
7. 用最终解质量构造 reward，并用 REINFORCE 更新 selector。

评测时复用同一套求解流程，但动作改成 `argmax`，并分别输出：

- `eval_original`：原始实例上的 greedy 结果；
- `eval_augmented`：固定 seed 生成的增强集上的 greedy 结果。

## 3. 状态、动作与奖励

### 3.1 动作空间

当前默认动作空间为：

- 初始化器：`lehd / elg / difusco`
- 迭代算子：`two_opt / lehd_rrc_step / dact_2opt_step`

策略结构是两个 head：

- `init head`：只在 rollout 开始时决策一次；
- `operator head`：在每一步迭代都决策一次。

对应实现见：`1online/policy.py`。

### 3.2 状态表示

状态编码分两部分：

- 初始化阶段：只看实例本身；
- 迭代阶段：看“实例 + 当前解 + 初始化动作”。

当前 `OperatorStateEncoder` 主要使用三类信息：

- 实例静态表示：对点集做 Transformer 图编码；
- 当前解表示：按当前 tour 顺序组织节点特征，再做 Transformer 编码；
- 全局统计：当前 tour 的边长统计量 + 初始化动作 embedding。

这里延续了之前你在 `1step/operator_policy` 中已经保留的那版设定：**不再使用迭代特征**（如 step index、stagnation、prev operator 等）作为最终编码输入。

对应实现见：`1online/state_encoder.py`。

### 3.3 reward 与 baseline

reward 直接定义为最终解质量：

```text
reward = -best_length
```

其中 `best_length` 是整个 rollout 结束时的历史最优长度。

这样定义的关键原因是：

- 如果 reward 用“相对初始化解的改进量”，初始化器只会通过“后续还有多少提升空间”被间接评价；
- 这会弱化“初始化本身好不好”的训练信号；
- 改成 `-best_length` 后，初始化器和后续 operator 都直接对最终解质量负责。

baseline 采用当前 augmentation batch 的 mean reward：

```text
baseline = mean(reward over current batch)
```

如果 batch 只有 1 个样本，则使用一个简单的 running baseline，避免 advantage 恒为 0。

同时，代码里仍然保留 `normalized_improvement = (init_length - final_length) / init_length`，但它现在只作为分析指标写入 trace，不再作为训练 reward。

因此训练/评测日志里的 `reward_mean`，应理解为 `episode_reward = -final_length` 的均值。

对应实现见：`1online/trainer.py`。

## 4. 关键模块说明

### 4.1 实例读取

`1online/instance_data.py`

职责：

- 支持 `instance_path`
- 支持 `tsplib_root + instance_name`
- 调用 EasyNCO 的 TSPLIB 读取逻辑
- 做单位棋盘归一化

输出统一为 `TSPInstance(name, path, problem_size, coords, ...)`。

### 4.2 在线增强

`1online/augmentations.py`

v1 只使用几何保真增强：

- `identity`
- `rotate`
- `reflect`
- `mix`

其中：

- 第 1 个样本固定为 `identity`
- 其余样本从启用增强中随机采样
- `mix` 表示在 `rotate / reflect` 之间随机选一种

这样做的目标是：围绕同一个实例 family 学习，而不是追求跨实例泛化。

### 4.3 环境与搜索状态

`1online/env.py`

环境内部维护：

- 当前解 `current`
- 历史最优解 `best_tour / best_length`
- 初始解长度 `init_length`
- 初始化动作 `init_action`

每执行一步 operator：

- 生成下一个解；
- 如果更优，则更新 `best_length`；
- 单步 `reward_by_step` 使用“best length 的减少量”，只用于日志与 trace；
- 最终训练 reward 在 `trainer` 中统一定义为 `episode_reward = -final_length`。

### 4.4 solver zoo 适配

`1online/solver_zoo.py`

职责：

- 把 EasyNCO 中的初始化器/迭代器包装成统一接口；
- 初始化器统一输出 `SolutionBatch(tour, length)`；
- 迭代器统一输入当前解并输出改进后的解。

这里做了一个重要实现选择：

- `two_opt` 会优先调用 `EasyNCO.neural_solvers.methods.difusco.util.two_opt_refine`；
- 如果当前环境中的原生依赖不可用，则退回 `1online` 本地 fallback；
- 目的是在尽量复用 EasyNCO 的同时，保留一个不被外部依赖卡住的后备路径。

### 4.5 EasyNCO bootstrap

`1online/easynco_bootstrap.py`

职责：

- 在不修改 `EasyNCO` 源码的前提下，保证本地导入可用；
- 做少量运行期 patch；
- 为当前环境提供轻量兼容 stub，减少 Lightning 训练栈的强依赖。

这里还遵循一个实现原则：**augmentation、初始化器、迭代器都优先调用 EasyNCO 原生实现**；
只有在当前环境中某个原生依赖不可用时，才退回 `1online` 本地 fallback。

这是 `1online` 能“只在自己目录内改动”并顺利调用 EasyNCO 推理组件的关键。

## 5. 训练与评测入口

### 5.1 训练入口

`1online/train_online_instance_policy.py`

主要工作：

- 解析 `TrainConfig`
- 读取单实例并写出 `config.json`
- 构建 initializer zoo / operator zoo / selector policy
- 周期性保存：
  - `selector_last.pt`
  - `selector_best.pt`
- 周期性评测：
  - `eval_original`
  - `eval_augmented`

其中 `best checkpoint` 的判定标准是：**原始实例上的 final length 更小**。

### 5.2 评测入口

`1online/evaluate_online_instance_policy.py`

主要工作：

- 读取 checkpoint 中保存的配置；
- 用命令行参数覆盖部分字段；
- 在原始实例和增强集上分别做 greedy 评测；
- 输出：
  - `eval_metrics.json`
  - `REPORT.md`
  - 全量 trace 文件。

## 6. 日志与 trace 设计

这是 `1online` 的重点。

### 6.1 聚合日志

训练阶段：

- `run.log`
- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `latest_metrics.json`

这些文件负责看整体趋势：

- 当前 loss / reward / entropy
- 初始化方法分布
- operator 分布
- 评测时 original / augmented 的表现

### 6.2 细粒度 trace

每个训练 step 都会写：

- `train_traces/step_xxxxxx.jsonl`
- `train_traces/step_xxxxxx.meta.json`
- `train_traces/step_xxxxxx_step_summary.json`

每次评测都会写：

- original trace
- augmented trace
- 对应的 meta 与 step summary

其中每个样本会记录：

- 初始化动作与概率；
- 每一步 operator 选择与概率；
- 每一步长度变化；
- 最终改进。

因此后续可以直接分析：

- 是否 collapse 到单一初始化器；
- 是否 collapse 到单一 operator；
- 是所有 step 都 collapse，还是只在特定 step collapse；
- 不同 augmentation 上动作是否一致。

## 7. 当前实现的边界

当前实现是一个 **instance-specialized online policy**，它的预期特点是：

- 对当前实例及其几何增强 family 有针对性；
- 出效果会比跨实例训练更快；
- 但泛化性通常较差，不应默认期待对别的 TSPLIB 实例也有效。

换句话说，`1online` 不是“跨实例通用 selector”，而更接近：

- 面向单实例 family 的在线自适应策略。

## 8. 推荐阅读顺序

如果你要继续改这个系统，建议按下面顺序看代码：

1. `1online/train_online_instance_policy.py`
2. `1online/trainer.py`
3. `1online/env.py`
4. `1online/policy.py`
5. `1online/state_encoder.py`
6. `1online/solver_zoo.py`
7. `1online/evaluate_online_instance_policy.py`

如果你要分析 collapse，优先看：

- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `*_step_summary.json`
- 对应的 `*.jsonl` 样本级 trace
