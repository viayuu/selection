# 0new `operator_policy` 实现说明

## 1. 这版实现到底在做什么

这版原型严格按照你后面明确下来的新思路来做，只关注 `TSP`：

1. 先从一个 **initializer zoo** 里选一个初始化方法；
2. 得到初始解 `sol_0`；
3. 然后进入固定长度为 `I` 的迭代过程；
4. 在每一步 `t`，根据 `实例 + 当前解 + 搜索阶段信息` 选择一个 operator；
5. operator 对 `sol_t` 做一次改进，得到 `sol_{t+1}`；
6. 重复直到 rollout 结束，最终用 best-so-far tour 计算奖励。

也就是说，这里学的不是“下一个点选谁”，而是：

> 在当前这条解的状态下，下一步该调用哪个改进算子。

因此它本质上是一个 **solution-based operator selection policy / learned hyper-heuristic**，而不是传统的构造式 TSP policy。

---

## 2. 代码结构

主要文件如下：

- `0new/operator_policy/easynco_bootstrap.py`
  - 运行时把本仓库里的 `EasyNCO/` 注册成可导入模块。
  - 这样就可以直接复用平台里的初始化器和迭代器，而不需要改平台源码。

- `0new/operator_policy/checkpoints.py`
  - 统一加载不同格式的 checkpoint。
  - 自动兼容 `state_dict`、`model_state_dict`、带 `policy.` / `module.` 前缀等情况。

- `0new/operator_policy/tsp_utils.py`
  - TSP tour 的基础工具。
  - 包括 tour 长度、succ/perm 相互转换、合法性检查等。

- `0new/operator_policy/solver_zoo.py`
  - 初始化器 zoo 和 operator zoo 的核心封装。
  - 负责把 `EasyNCO` 平台中的不同方法统一包装成：
    - `Initializer.solve(coords) -> SolutionBatch`
    - `Operator.apply(coords, solution) -> SolutionBatch`

- `0new/operator_policy/state_encoder.py`
  - selector 的状态编码器。
  - 这是这次实现里最关键的一部分，解特征参考了 **LIH + DACT**，但又不是简单照抄。

- `0new/operator_policy/policy.py`
  - selector policy。
  - 包含两个 head：
    - 初始化方法选择 head
    - 每步 operator 选择 head

- `0new/operator_policy/env.py`
  - 新定义的 `solution-improvement env`。
  - `reset`：选择初始化器并生成 `sol_0`
  - `step`：执行一个 operator，更新 `current/best/stagnation`

- `0new/operator_policy/trainer.py`
  - POMO 风格的 REINFORCE 训练器。
  - 记录初始化决策与每一步 operator 决策的 `log_prob`，统一用最终改进计算 policy gradient。

- `0new/operator_policy/train_tsp_operator_policy.py`
  - 训练入口。
  - 目前默认直接生成随机 TSP 实例做训练/评估。

---

## 2.1 训练日志与增量保存

现在训练过程不会只等到全部 `train_steps` 结束后才落盘。

每次运行都会在 `output_dir` 下持续写入：

- `run.log`
  - 终端里看到的主要日志会同步追加到这里。
  - 是实时写入的，训练过程中可以直接 `tail -f` 观察。

- `train_metrics.jsonl`
  - 每个 train step 追加一条 JSON 记录。
  - 包含 `loss / init_length / final_length / improvement / entropy / timing / action distribution` 等。

- `eval_metrics.jsonl`
  - 每次评估追加一条 JSON 记录。
  - 包含该次 eval 的长度、改进、initializer/operator 分布与耗时。

- `latest_metrics.json`
  - 每隔 `--save_every` step 覆盖写一次快照。
  - 便于你在训练中途快速打开看“最近一次 train / eval 的摘要”。

新增参数：

- `--save_every`
  - 控制 `latest_metrics.json` 的快照保存间隔。
  - 默认是 `20`。

如果你想边跑边看，最方便的是：

```bash
tail -f 0new/operator_policy/outputs/default/run.log
```

---

## 3. 当前支持的初始化方法

当前实现支持以下初始化器：

- learning-based
  - `lehd`
  - `elg`
  - `difusco`
  - `pomo`（若你自己提供 checkpoint，也可以启用）
  - `am`（若你自己提供 checkpoint，也可以启用）
- heuristic-based（新增，完全在 `0new/operator_policy` 内实现，不修改 `EasyNCO`）
  - `ins_nearest`
    - nearest insertion
    - 每一步先挑“离当前 tour 最近”的未访问点，再插入到增量最小的位置
  - `ins_random`
    - random insertion
    - 每一步随机挑一个未访问点，再插入到增量最小的位置
  - `ins_regret`
    - regret insertion
    - 对每个候选点比较最优与次优插入代价，选择 regret 最大者

默认配置仍然使用：

- `lehd`
- `elg`
- `difusco`

也就是说，启发式初始化器不会自动加入默认训练；需要你显式通过 `--init_zoo` 打开。

### 3.1 启发式初始化器的统一约定

这三个启发式初始化器有一致的实现约定：

- 只做 **构造初始化**，不做后续改进；
- 都直接返回 `SolutionBatch(tour, length)`；
- 默认使用 `4` 次重启（`restarts=4`），最终取最短 tour；
- 默认使用欧氏距离，即 `cartesian` + `p=2`；
- 随机性直接复用当前 run 的 PyTorch RNG；
- 当前只支持 `TSP`。

公开参数如下：

- `--heuristic_init_restarts`
- `--heuristic_metric_strategy`
- `--heuristic_metric_p`
- `--heuristic_regret_k`

默认值分别是：

- `4`
- `cartesian`
- `2`
- `1`

### 3.2 checkpoint 兼容性说明

如果你扩展了 `init_zoo`，例如从：

- `lehd elg difusco`

变成：

- `lehd elg difusco ins_nearest ins_random ins_regret`

那么旧 checkpoint **不能直接 resume**。

原因是：

- `init_head` 的输出维度变了；
- 状态编码里和初始化动作相关的 embedding 维度也会跟着变。

v1 的约定是：

- **只要初始化动作空间变了，就从头训练。**

---

## 4. 当前支持的迭代算子

当前实现支持以下 operator：

- `two_opt`
  - 调用 `EasyNCO.neural_solvers.methods.difusco.util.two_opt_refine`
  - 是一个 **普通启发式 2-opt** wrapper
  - 默认只跑 `1` 次 refinement 轮次，因此把它当作“单步 operator”

- `difusco_2opt_step`
  - 当前语义上与 `two_opt` 一致
  - 只是为了从实验命名上更明确地区分：这是走 `EasyNCO/DIFUSCO` 那套 `2-opt refine` 工具链

- `lehd_rrc_step`
  - 调用 `EasyNCO` 的 LEHD repair policy
  - 一次 operator 调用只做 **一步 RRC destroy-repair**

- `dact_2opt_step`
  - 调用 `EasyNCO` 的 DACT policy
  - 一次 operator 调用只做 **一步 policy-guided 2-opt**

- `lih_2opt_step`
  - 调用 `EasyNCO` 的 LIH policy
  - 一次 operator 调用只做 **一步 policy-guided 2-opt**
  - 默认 **不启用**，因为当前仓库里没有现成 LIH checkpoint，而且上游实现硬编码了 `.cuda()`

- `glop_sub100_step`
  - 调用 `EasyNCO` 的 GLOP reviser
  - 以 `revision_len = 100` 做一次子问题 revise pass
  - 当 `problem_size = 100` 时，它等价于对整条 tour 做一次大粒度 revise

- `glop_sub50_step`
  - 调用 `EasyNCO` 的 GLOP reviser
  - 以 `revision_len = 50` 做一次子问题 revise pass

- `glop_sub20_step`
  - 调用 `EasyNCO` 的 GLOP reviser
  - 以 `revision_len = 20` 做一次更细粒度的 revise pass

- `glop_pass_step`
  - 把 `glop_sub100_step -> glop_sub50_step -> glop_sub20_step` 串成一个单独 operator
  - 每个粒度默认各跑 `1` 次 revise
  - 它比单个 `glop_subXX_step` 更强，但动作粒度也更粗

默认配置仍然保守使用：

- `two_opt`
- `lehd_rrc_step`
- `dact_2opt_step`

原因是这三类的语义最清晰：

- 一个纯启发式局部搜索
- 一个 destroy-repair 类 operator
- 一个 learned pair-selection 类 operator

而新加的 GLOP 算子更适合后续做 richer zoo 对比实验。

### 4.1 `policy-guided 2-opt` 和普通 `2-opt` 有什么区别

这是这次实现里一个很容易混淆、但必须说清楚的点。

#### 普通 `2-opt`

普通 `2-opt` 的本质是：

1. 枚举或启发式挑选一对边 / 一个区间 `(i, j)`；
2. 把区间反转；
3. 如果 tour 变短，就接受这个 move。

这里的关键是：

- **move family 固定**：都是 2-opt move；
- **pair 的选择方式不是学出来的**：通常是 exhaustive、first-improvement、best-improvement、邻近筛选、随机采样等启发式。

也就是说，普通 `2-opt` 学到的东西是 **0**，它只是固定的局部搜索规则。

#### `policy-guided 2-opt`

`policy-guided 2-opt` 依然执行 2-opt move，本质 move family 没变，变的是：

- **哪一对位置 `(i, j)` 要被拿来做 2-opt**，由一个 learned policy 决定；
- 这个 policy 看到的是 **当前完整解的状态表示**；
- 所以它学到的是：
  - 当前这条 tour 在什么形态下，
  - 哪一类 segment reverse 更值得尝试。

也就是说：

> 普通 `2-opt` 是“固定规则选 pair，再执行 2-opt”；
> `policy-guided 2-opt` 是“先用神经策略选 pair，再执行同样的 2-opt”。

#### 在当前原型里的具体对应

- `two_opt` / `difusco_2opt_step`
  - 是 **普通 2-opt**
  - 用的是平台现有 heuristic refine 工具

- `dact_2opt_step`
  - 是 **DACT 的 policy-guided 2-opt**
  - DACT policy 先根据当前解给出交换对，再调用 `Iteration_tool.operate()` 执行 move

- `lih_2opt_step`
  - 是 **LIH 的 policy-guided 2-opt**
  - LIH policy 也是先预测 exchange pair，再执行固定 2-opt 风格反转

所以，对你这个“operator selection”任务来说：

- `two_opt` 更像一个 **纯 heuristic operator**；
- `dact_2opt_step` / `lih_2opt_step` 更像一个 **learning-based operator**；
- 它们都属于“2-opt family”，但策略来源不同，搜索偏好也不同。

---

## 5. 状态表示：为什么说它是 `LIH + DACT + 搜索阶段信息` 的混合设计

这部分是整个实现里最重要的地方。

你的任务不是：

- 给当前节点选下一个节点；
- 或给当前解直接选一对节点做 swap。

你的任务是：

> 给定当前实例、当前完整解、当前搜索阶段，判断下一步应该调用哪个 operator。

因此，状态表示必须同时回答三件事：

1. **这个实例本身长什么样**；
2. **当前 tour 的局部结构长什么样**；
3. **搜索已经进行到哪个阶段了**。

这也是为什么我没有直接照抄 POMO / LIH / DACT 任何一家，而是做了重新抽象。

### 5.1 初始化选择阶段：只编码实例，不编码解

初始化选择阶段还没有 `sol_0`，所以这里只需要编码实例本身。

`InitStateEncoder` 的输入很简单：

- 节点坐标 `coords`
- 图规模 `n`

对应代码在 `0new/operator_policy/state_encoder.py`：

```python
class InitStateEncoder(nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        graph_emb = self.graph_encoder(coords)
        scale = torch.full((coords.size(0), 1), float(coords.size(1)), ...)
        scale_emb = self.scale_proj(scale)
        return self.final_proj(torch.cat([graph_emb, scale_emb], dim=1))
```

这里的意思很直接：

- 用一个轻量 Transformer pooling 编码整张点集；
- 再拼一个规模 embedding；
- 输出给初始化器选择 head。

这部分和 LIH / DACT 的关系不大，因为它们重点都在“当前解的表示”，而不是“无解时的实例编码”。

### 5.2 每步 operator 选择阶段：这才是真正参考 LIH 和 DACT 的地方

`OperatorStateEncoder` 的核心不是“把一条解编码成一个向量”这么简单，而是把三类信息拼在一起：

- **static instance embedding**：实例几何本身
- **solution-conditioned embedding**：当前 tour 结构
- **search-stage embedding**：当前搜索阶段与历史动作

具体对应代码是：

```python
static_emb = self.static_encoder(coords)
solution_feats = _tour_node_features(coords, tour, self.cpe_dim)
solution_emb = self.solution_encoder(solution_feats)
edge_stats = _tour_edge_stats(coords, tour)
...
global_emb = self.global_proj(global_feats)
init_emb = self.init_embedding(init_action)
prev_emb = self.prev_op_embedding(prev_idx)
return self.final_proj(torch.cat([static_emb, solution_emb, global_emb, init_emb, prev_emb], dim=1))
```

这里可以直接看出：

- `static_emb` 是“这个实例本身”；
- `solution_emb` 是“当前解长什么样”；
- `global_emb` 是“当前搜索进展如何”；
- `init_emb` / `prev_emb` 是“这条解从哪里来、上一跳做了什么”。

下面分别解释它是如何借 LIH 和 DACT 的。

### 5.3 参考 LIH 的部分：`pre / self / next` 局部结构

LIH 的一个核心思想是：

> 节点的语义不只来自它的坐标，还来自它在当前解中的局部邻域结构。

在 LIH 的 TSP 实现里，`tsp_embedding(input, rec)` 会先根据当前解 `rec` 重新组织节点顺序，再去构造与当前位置相关的表示。

我这里没有直接照搬它的 actor 结构，而是保留了它最关键的“局部结构先验”：

```python
def _tour_node_features(coords, tour, cpe_dim):
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    prev_ordered = ordered.roll(shifts=1, dims=1)
    next_ordered = ordered.roll(shifts=-1, dims=1)
    rel_prev = ordered - prev_ordered
    rel_next = next_ordered - ordered
    dist_prev = rel_prev.norm(p=2, dim=-1, keepdim=True)
    dist_next = rel_next.norm(p=2, dim=-1, keepdim=True)
    ...
```

这段的含义是：对 tour 中的每个位置 `k`，都显式构造一个 token，里面包含：

- 当前节点坐标 `self`
- 前驱节点坐标 `prev`
- 后继节点坐标 `next`
- `self - prev`
- `next - self`
- 两条相邻边长度 `dist_prev` / `dist_next`

如果把维度展开，它实际上是：

- `ordered`：2 维
- `prev_ordered`：2 维
- `next_ordered`：2 维
- `rel_prev`：2 维
- `rel_next`：2 维
- `dist_prev`：1 维
- `dist_next`：1 维
- 再加 `pos`：1 维

一共是 `13` 维，再加后面的 CPE。

这就是为什么代码里 `solution_encoder = GraphPoolEncoder(13 + cpe_dim, ...)`。

#### 为什么这部分是 LIH 风格

因为它抓住了 LIH 最重要的一点：

- 不把 tour 看成“单纯的 permutation”；
- 而把每个 tour position 看成一个带有局部邻域信息的状态 token。

对于 operator selection 来说，这个信息非常重要，因为：

- `2-opt` 类 operator 是否值得做，很大程度取决于局部边结构；
- `RRC` 是否值得做，也和当前 tour 有没有明显坏边、局部扭曲有关；
- 即使最后输出的是“选 operator ID”，你也仍然需要知道当前解的局部几何状态。

### 5.4 参考 DACT 的部分：位置通道、完整解条件化、全局池化

DACT 的一个关键点是：

> 它不是只看节点几何，而是把“节点在当前解中的位置/拓扑角色”也编码进去。

DACT 原方法更偏向“为 pair selection 服务”的表示，所以它会把当前 solution 的结构信息嵌入到 actor 里，然后输出 node-pair logits。

你的任务不是输出 pair，而是输出 operator ID，所以我做了一个更适合 meta-control 的抽象：

```python
pos = torch.arange(tour.size(1), dtype=coords.dtype, device=coords.device)
pos = (pos / max(tour.size(1), 1)).view(1, -1, 1).expand(coords.size(0), -1, -1)
cpe = _cyclic_positional_encoding(tour.size(1), cpe_dim, coords.device)
...
return torch.cat([ordered, prev_ordered, next_ordered, rel_prev, rel_next,
                  dist_prev, dist_next, pos, cpe], dim=-1)
```

这里有两个 DACT 风格的点：

1. **显式位置比例 `pos / n`**
   - 同一个节点局部形状类似，但位于 tour 前段 / 中段 / 后段，搜索含义可能不同；
   - 对某些 operator，特别是基于 segment 的 operator，这种位置信息有帮助。

2. **Cyclic Positional Encoding (CPE)**
   - tour 是一个环，不是普通序列；
   - 直接用线性位置编码不够自然；
   - 用周期位置编码更贴合 tour 的循环结构。

#### 为什么没有直接用 DACT 的 successor representation

这是一个非常关键的设计取舍。

DACT 原生很适合“pair selection”，因为它直接在当前解结构上预测 `(i, j)`。

但你现在要解决的是 heterogeneous operator selection：

- 有的 operator 基于 pair（如 2-opt）
- 有的 operator 基于大区间（如 RRC）
- 有的 operator 基于子问题 reviser（如 GLOP）

因此，如果直接把状态完全绑死在 successor/pair 表示上，会让表示过于偏向某一类 operator。

所以我这里保留 DACT 的两点精华：

- **当前完整解条件化**
- **位置/顺序通道**

但把最终输出头从“pair logits”改成了“whole-solution embedding -> operator logits”。

#### 为什么需要 `GraphPoolEncoder`

DACT 原生是为 pair action 做的，不需要把整条 tour 压成一个全局向量再分类 operator。

而你现在的动作空间是 `m` 个 operator，因此必须把“整条当前解”池化成一个 global state embedding。

所以这里我做了：

```python
h = self.input_proj(x)
h = self.encoder(h)
pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
return self.output_proj(pooled)
```

也就是：

- 先在 ordered tour token 上做 Transformer 编码；
- 再做 `mean + max` pooling；
- 最终得到一个 whole-solution embedding。

这一步本质上是把 LIH / DACT 的 **node-level / pair-level representation**，变形成更适合 **operator-level decision** 的 `state embedding`。

### 5.5 当前版本的全局状态：只保留解统计，不再使用迭代特征

你最新要求里，已经明确把“迭代特征”去掉，所以当前版本不再把下面这些量送进 operator state encoder：

- `current_length / init_length`
- `best_length / init_length`
- `(current_length - best_length) / init_length`
- `(init_length - best_length) / init_length`
- `step_index / rollout_steps`
- `stagnation / rollout_steps`
- `prev_operator`

当前版本保留下来的全局特征只有：

- `edge_stats = mean / std / min / max`
  - 当前 tour 的边长统计量

也就是说，operator state 现在主要由三部分组成：

- 实例静态图表示
- 当前 tour 的局部 + 顺序表示
- 当前 tour 的全局边长统计

此外仍然保留一个条件信息：

- `init_action`
  - 表示这个解最初是由哪个 initializer 生成的

这样做的含义是：

- selector 不再依赖“现在是第几步 / 连续多久没提升 / 当前相对初始提升多少”这类 search-progress 信号；
- 而是更纯粹地根据“实例结构 + 当前解结构 + 初始化来源”来决定下一步 operator。

### 5.6 额外条件：`init_action`

当前版本额外保留的离散条件只有一个：

- `init_action`
  - 当前这条解最开始是由哪个 initializer 生成的
  - 这会影响后续 operator 偏好
  - 例如某些初始化器生成的 tour 可能更适合继续 2-opt 微调，另一些更适合大步 revise

也就是说，当前版本已经**不再**把 `prev_operator` 当作状态输入。 

对应代码：

```python
init_emb = self.init_embedding(init_action)
```

这一步的作用，是把“这个解最初来自哪个 initializer”作为一个轻量条件送给 policy。

### 5.7 一句话总结这套状态表示

如果用一句话概括现在的 `OperatorStateEncoder`：

> **LIH 给了我局部结构视角，DACT 给了我完整解的顺序/位置视角，而当前版本只保留与“实例结构 + 当前解结构 + 初始化来源”直接相关的部分。**

所以这并不是“把 LIH 和 DACT 拼接一下”，而是：

- 继承它们最适合 solution-improvement 的部分；
- 去掉 pair-action 特化太强的部分；
- 同时去掉显式的迭代进度/历史动作特征。

---

## 6. 训练框架：参考 POMO，但动作换成 operator

训练器在 `0new/operator_policy/trainer.py`。

核心思想是：

1. 用初始化选择器采样一个初始化器；
2. 调 EasyNCO 初始化器得到 `sol_0`；
3. 连续 `I=rollout_steps` 步，每一步：
   - 编码当前解状态；
   - 采样一个 operator；
   - 调 EasyNCO 对应 operator 执行一步；
4. 最终奖励用 `init_length - best_length`；
5. baseline 用 batch 内均值；
6. loss 为：

```text
loss = - E[(logp_init + Σ_t logp_op_t) * advantage] - entropy_bonus
```

这就是 POMO/REINFORCE 风格，但 action space 从“节点”换成了“方法/算子”。

---

## 7. 环境设计

`TSPImprovementEnv` 不再是构造式 env，而是 solution-improvement env。

### `reset(coords, init_actions)`

- 根据初始化动作调用 `run_initializers_by_action`
- 得到 `sol_0`
- 初始化：
  - `current = sol_0`
  - `best = sol_0`
  - `init_length = len(sol_0)`
  - `prev_operator = -1`
  - `step_index = 0`
  - `stagnation = 0`

### `step(state, operator_actions)`

- 根据 operator id 调 `run_operators_by_action`
- 得到新解
- 若新解非法，则退回旧解
- 更新 best-so-far
- reward 用 `prev_best - new_best`

这里 reward 用的是 **best-so-far 的增量改善**，而不是 `current -> next` 的瞬时差值。这样更稳，也更符合“高层调度器”的目标。

---

## 8. 为什么 operator 设计成“单步”

这次实现里，我特意把每个 operator 设计成“调用一次只做一个搜索步 / 一个 revise pass / 一个 repair 步”，而不是一次调用直接跑几十步。

这样更符合你最开始的想法：

> 总共走 `I` 步，每一步选一个算子。

也就是说，外层策略掌控“每一步用哪个方法”，而不是把整段迭代交给某一个方法内部自己跑完。

需要说明的是，像 `glop_pass_step` 这种 operator 虽然内部串了三个 revise 粒度，但它在外层仍然只算 **一个宏观动作**。这和 `glop_sub20_step` / `glop_sub50_step` / `glop_sub100_step` 的差异，刚好可以让策略自己学：

- 什么时候用更细粒度 revise；
- 什么时候直接用更强的多粒度 pass。

---

## 9. 默认配置与现实约束

当前默认配置建议：

- `problem_size=100`

原因是 `EasyNCO/pretrained` 下默认现成 checkpoint 基本围绕 `tsp100`。

另外：

- `LIH` 默认没开，因为当前仓库里没有直接可用的 `lih_tsp100.ckpt`
- `DACT` 默认放在 CPU 跑，因为 EasyNCO 里的 DACT 实现做 device 管理时偏脆弱，CPU 更稳
- `GLOP` 的 wrapper 已补上，但由于当前 shell 环境缺少 `torch`，这里只做了静态实现与语法检查，没在本环境里做真正 rollout 验证

`train_tsp_operator_policy.py` 还补了以下 GLOP 相关参数：

- `--glop_ckpt`
- `--glop_pomo_size`
- `--glop_aug_factor`
- `--glop_revision_iters`

默认会优先尝试自动发现：

- `EasyNCO/pretrained/glop_policy_tsp.pt`

---

## 10. 运行方式

### 10.1 训练一个基础版本

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py \
  --device cuda:0 \
  --problem_size 100 \
  --batch_size 16 \
  --train_steps 500 \
  --rollout_steps 20 \
  --init_zoo lehd elg difusco \
  --operator_zoo two_opt lehd_rrc_step dact_2opt_step \
  --output_dir 0new/operator_policy/outputs/run1
```

### 10.1B 加入启发式初始化器训练

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py \
  --device cuda:0 \
  --easynco_solver_device cuda:0 \
  --dact_solver_device cuda:0 \
  --problem_size 100 \
  --train_data_source easynco \
  --train_distribution uniform \
  --train_data_size 20000 \
  --eval_data_source easynco \
  --eval_distribution cluster \
  --init_zoo lehd elg difusco ins_nearest ins_random ins_regret \
  --operator_zoo two_opt lehd_rrc_step dact_2opt_step \
  --heuristic_init_restarts 4 \
  --output_dir 0new/operator_policy/outputs/heuristic_aug
```

这个实验对应“扩容 initializer action space 后，heuristic 是否会 dominate learning-based 初始化器”。

### 10.1C 单独 benchmark 初始化器

```bash
PYTHONPATH=. python 0new/operator_policy/benchmark_initializers.py \
  --problem_size 100 \
  --device cuda:0 \
  --easynco_solver_device cuda:0 \
  --batch_size 32 \
  --init_zoo lehd elg difusco ins_nearest ins_random ins_regret \
  --heuristic_init_restarts 4 \
  --output_dir 0new/operator_policy/benchmarks/init_tsp100_uniform
```

输出内容包括：

- `results.json`
- `REPORT.md`
- `benchmark.log`

核心指标包括：

- `avg_init_length`
- `win_rate`
- `avg_excess_to_best_init`
- `avg_init_seconds`

这个脚本的目的，是把“初始化器本身很强”与“selector 训练坍缩”区分开。

### 10.2 加上 GLOP reviser 系列 operator

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py \
  --device cuda:0 \
  --problem_size 100 \
  --init_zoo lehd elg difusco \
  --operator_zoo two_opt dact_2opt_step lehd_rrc_step glop_sub50_step glop_sub20_step glop_pass_step \
  --glop_ckpt EasyNCO/pretrained/glop_policy_tsp.pt \
  --output_dir 0new/operator_policy/outputs/run_glop
```

### 10.3 只评估

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py \
  --device cuda:0 \
  --eval_only \
  --load_path 0new/operator_policy/outputs/run1/selector_best.pt
```

### 10.4 若你后面拿到 LIH checkpoint

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py \
  --device cuda:0 \
  --operator_zoo two_opt lehd_rrc_step dact_2opt_step lih_2opt_step glop_sub20_step \
  --lih_ckpt /path/to/lih_tsp100.ckpt
```

---

## 11. 这版实现和你需求的对应关系

### 已满足的

- 只关注 `TSP`
- 初始化方法选择一次
- 后续每一步都重新选择算子
- 解特征明确参考 `LIH + DACT`
- 额外加入了搜索阶段特征与历史动作特征
- 训练框架参考 `POMO/REINFORCE`
- 初始化器与算子执行都直接调 `EasyNCO`
- 已经扩展到不止一种平台迭代方法：`2-opt / LEHD / DACT / GLOP`

### 当前没有做的

- 暂时没有接 `CVRP`
- 暂时没有做 operator 内部参数联合选择
- 暂时没有把 `swap / divide-and-conquer / quadtree` 这些都补成统一 wrapper
- 暂时没有做 PPO 版本
- 暂时没有把“按阶段统计 operator 使用偏好”的分析写进训练日志

---

## 12. 我建议你接下来优先推进什么

如果你下一步继续让我推进，我建议优先做这几件事：

1. 在 `trainer` 里加上 **按 step 段统计 operator 使用分布**；
2. 把 `glop_subXX_step` 和 `glop_pass_step` 真正拉进实验，观察不同阶段偏好；
3. 再补一个 PPO 版本，降低方差；
4. 最后再考虑“operator family + family 内参数”两层动作，而不是一开始就把动作做得过细。

对你当前研究问题来说，最值得先验证的不是“能不能再加更多模型”，而是：

> **同一个实例在不同搜索阶段，policy 是否真的会自发偏向不同 operator family。**

这会直接验证你当前这条研究路线的核心价值。


---

## 13. 数据集该怎么选：什么更适合训练这个 selector

### 13.1 最合理的主训练集：`EasyNCO` 风格的合成 TSP 数据

如果目标是训练一个“初始化 + 逐步 operator 选择”的 selector，**最合理的主训练集**不是直接拿 `TSPLIB`，而是：

- 以 `EasyNCO` 的 TSP 数据接口为主；
- 在和 solver zoo 预训练权重更匹配的尺度上训练（当前默认就是 `tsp100`）；
- 用大量合成实例覆盖不同几何分布。

原因有三点：

1. **数据量够大**
   - 你的 selector 不是在学一个静态分类器，而是在学一个多步决策策略；
   - 这种策略训练通常需要大量 rollout 才稳定；
   - 小数据集很容易过拟合到少数实例。

2. **和当前 operator zoo 更匹配**
   - 现在接进来的 `LEHD / ELG / DIFUSCO / DACT / GLOP` 预训练模型，大多是围绕 `tsp100` 这一类训练出来的；
   - 所以 selector 的主训练分布，最好不要一开始就偏离这些模型最擅长的尺度太远。

3. **更容易做 distribution-level 控制**
   - 你可以分别训练在 `uniform / cluster / diagonal / gaussian / explosion` 上；
   - 也可以自己离线生成混合 `.pt/.pkl` 数据，再喂给当前入口；
   - 这样更容易观察 selector 是否真的学会“问题性质 -> operator 偏好”。

### 13.2 `TSPLIB` 更适合做什么

`TSPLIB` 更适合做：

- OOD 泛化验证；
- small-data fine-tuning；
- case study；
- 逐实例分析 selector 是否学到与实例结构相关的 operator 偏好。

不太适合直接拿来当**唯一主训练集**，原因是：

- 数据量小；
- 规模分布离散；
- 很多尺寸只有极少数实例；
- 当前这版 REINFORCE 主要还是更适合在大量实例上学策略，而不是在几十个 benchmark 上硬记。

### 13.3 一个比较合理的实际训练流程

我建议你用下面的顺序：

1. **主训练**：`EasyNCO` 合成数据（`tsp100`，大量实例）
2. **验证/调参**：另一份 `EasyNCO` 合成验证集
3. **OOD 测试**：`TSPLIB`
4. **如果需要**：再在 `TSPLIB` 上做轻量 fine-tune 或 few-shot adaptation

也就是：

> **合成数据负责学“策略规律”，TSPLIB 负责检验“真实 benchmark 上是否成立”。**

### 13.4 这版代码现在已经支持的数据来源

当前训练入口 `0new/operator_policy/train_tsp_operator_policy.py` 已支持三种数据来源：

- `online`
  - 当前默认方式；
  - 直接 `torch.rand` 在线生成 `TSP` 坐标。

- `easynco`
  - 使用 `EasyNCO.data.TSPGenerator`；
  - 可用于：
    - 在线生成 `EasyNCO` 风格合成数据；
    - 读取 `EasyNCO` 风格 `.pt/.pkl` 数据文件。

- `tsplib`
  - 直接读取 `TSPLIB` 目录；
  - 不再要求路径名必须以 `tsplib` 结尾；
  - 适合做评估，也可以做小规模 fine-tune。

### 13.5 为什么我还补了“单样本 baseline 回退”

`TSPLIB` 常常只能按 `batch_size=1` 读，因为不同实例规模不同，没法直接堆成一个 tensor batch。

而原来训练器用的是：

- `advantage = reward - batch_mean_reward`

如果 batch 只有 1 个样本，那么 advantage 恒为 0，训练就完全没梯度。

所以现在 `trainer.py` 里补了一个很轻的回退机制：

- batch 大于 1：仍然用 batch mean baseline；
- batch 等于 1：改用一个 EMA running baseline。

这样 `TSPLIB` 的单样本流式训练至少是**可训练的**。

### 13.6 现在怎么用

#### 用 `EasyNCO` 合成数据训练

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py   --device cuda:0   --problem_size 100   --train_data_source easynco   --train_distribution uniform   --train_data_size 20000   --eval_data_source easynco   --eval_distribution cluster   --operator_zoo two_opt lehd_rrc_step dact_2opt_step
```

#### 用 `EasyNCO` 的 `.pt/.pkl` 数据训练

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py   --device cuda:0   --problem_size 100   --train_data_source easynco   --train_data_path path/to/train_dataset.pt   --eval_data_source easynco   --eval_data_path path/to/val_dataset.pt
```

#### 在 `TSPLIB` 上评估

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py   --device cuda:0   --problem_size 100   --eval_only   --eval_data_source tsplib   --eval_data_path tsplib/tsplib
```

#### 分析一个带启发式初始化器的输出目录

```bash
PYTHONPATH=. python 0new/operator_policy/analyze_outputs.py \
  0new/operator_policy/outputs/heuristic_aug
```

现在如果该目录下有 `config.json`，脚本会优先自动读取：

- `init_zoo`
- `operator_zoo`

因此当你把 `ins_nearest / ins_random / ins_regret` 加进训练后，不需要再手动传一遍名字。

#### 在 `TSPLIB` 上做小规模 fine-tune

```bash
PYTHONPATH=. python 0new/operator_policy/train_tsp_operator_policy.py   --device cuda:0   --problem_size 100   --train_data_source tsplib   --train_data_path tsplib/tsplib   --eval_data_source tsplib   --eval_data_path tsplib/tsplib   --single_sample_baseline_momentum 0.95
```

如果你后面要做得更严谨，我建议把 `TSPLIB` 再按规模区间切开，例如：

- `--train_scale_range 50 150`
- `--eval_scale_range 150 300`

这样可以更明确地测试尺度泛化。
