# 0new 阅读与方案报告：基于“初始化选择 + 逐步算子选择”的神经迭代求解框架

## 1. 报告目的与阅读范围

这份报告只关注 `0new/` 目录中的三套材料：

- `0new/pomo/`：POMO 论文与代码
- `0new/Learn-Improvement-Heuristics-for-Routing-main/`：LIH（Learning Improvement Heuristics）论文与代码
- `0new/dact/VRP-DACT-new_version/`：DACT 论文与代码

你的**新思路**不再是“初始化 gate + 迭代 gate 各选一次方法”，而是：

1. **先选择一种初始化方法**，得到初始解 `sol_0`
2. 然后进入一个固定长度为 `I` 的迭代过程（例如 `I=100`）
3. 在每一个迭代步 `t`，**根据当前解 `sol_t` 再选择一个算子**（例如 `rrc`、`2-opt`、`swap`、`divide-and-conquer / quadtree` 等）
4. 算子作用在当前解上，得到 `sol_{t+1}`
5. 重复直到达到最大步数或提前停止

并且你特别强调：

- **如果某个方法是基于热图（heatmap）的**，它只参与**初始化**，**不参与后续迭代**；
- 迭代阶段的动作空间不是节点，不是边，也不是 pair，而是**算子集合**；
- 你希望在一个现有学习框架上套这个思路，例如 POMO，但核心动作要从“选点”变成“选算子”。

这本质上把 routing 的学习问题从：

- **construction MDP**：状态是部分解，动作为“下一个节点”

改成了：

- **solution-improvement MDP**：状态是当前完整解，动作为“下一步使用哪个算子”

这是一个完全不同的建模方向。

---

## 2. 先给结论：你的新思路是可行的，而且比“双 gate 一次性选迭代器”更自然

### 2.1 为什么原来的双 gate 思路不够合适

原来的双 gate 思路是：

- Gate 1 选初始化器
- Gate 2 只选一次迭代器

这隐含了一个假设：

> 一个实例在整个改进过程中，始终适合用同一种迭代方法。

这个假设通常并不成立。因为 routing 的局部搜索过程往往有明显的**阶段性**：

- 初始阶段需要大步跳跃，适合破坏-修复类算子（如 RRC / destroy-repair）
- 中期需要较强但较局部的结构重排，适合 2-opt / 3-opt / segment-reverse / swap*
- 后期需要细修，适合 very local 的 swap / insert / relocate

所以，**“每一步都重新决定下一步用什么算子”**，比“在迭代开始前只选一个迭代器”更符合优化过程本身。

### 2.2 你的新思路更像什么

你的新框架更像一个**solution-based operator policy**：

- 输入：实例 `X` 和当前解 `sol_t`
- 输出：下一步要使用的算子 `a_t ∈ {op_1, ..., op_m}`
- 转移：`sol_{t+1} = op_{a_t}(X, sol_t)`
- 奖励：当前步或累计的长度改进

也就是说，你不是让模型“直接构造解”，而是让模型扮演一个**高层调度器（operator scheduler）**。

这个方向在思想上更接近：

- hyper-heuristic
- operator selection
- learned search control
- learned improvement policy

而不是传统的 node-by-node constructive policy。

### 2.3 三个参考框架中谁最接近你的需求

结论很明确：

- **POMO**：训练框架简单，适合借用 rollout / baseline / batch 组织方式，但它原生是 constructive，不是 improvement
- **LIH**：已经是 improvement learning，但动作仍是“选节点对 / 固定局部算子”，不是“选算子”
- **DACT**：和你的目标**最接近**，因为它已经是“当前完整解 → 迭代一步 → 得到新解”的 RL 框架

所以更准确地说：

- **如果只想借训练脚手架，POMO 最轻**；
- **如果想借状态建模和迭代接口，DACT 最接近**；
- **如果想借 solution feature 的 handcrafted/local 设计，LIH 很有参考价值**。

我的建议不是“纯套 POMO”，而是：

> **训练环路可以借 POMO 的简洁性；解表示和迭代环境要大量借 DACT；局部结构特征可借 LIH。**

---

## 3. POMO 分析：它适合拿来当训练骨架，但不适合原样复用为你的模型主体

### 3.1 POMO 论文核心思想

POMO（Policy Optimization with Multiple Optima）是一种针对组合优化构造式求解的 RL 框架。其核心点有两个：

1. **多起点 rollout（POMO rollout）**：对于同一个实例，利用解的对称性，从多个不同起点并行 rollout
2. **低方差 baseline**：用同一实例内部多个 rollout 的平均 reward 作为 baseline，提高训练稳定性

它天然面向的问题是：

- TSP / CVRP / KP 这类可以按序逐步构造解的问题
- 每一步动作就是“从还没选过的对象中选一个”

### 3.2 POMO 代码结构里最值得关注的部分

#### (1) 环境：`0new/pomo/NEW_py_ver/TSP/POMO/TSPEnv.py`

POMO 的环境非常干净：

- `reset()` 返回静态问题实例 `problems`
- `step(selected)` 记录当前选中的节点
- `ninf_mask` 负责屏蔽已访问节点
- 最终 reward 是完整 tour 的负长度

这说明 POMO 的环境是一个标准的：

- 状态：部分路径
- 动作：选下一个节点
- 终止：选满 `n` 个节点

#### (2) 模型：`0new/pomo/NEW_py_ver/TSP/POMO/TSPModel.py`

POMO 的模型分成：

- `encoder`：对所有节点做图编码
- `decoder`：根据当前状态输出下一个节点概率

关键点在于：

- 第一步动作由 `selected = torch.arange(pomo_size)` 强行设成多个不同起点
- 后续动作才由 decoder 产生
- decoder 的输出空间是 `problem_size = n`

也就是说，它的 action space 本质就是节点集 `{0,...,n-1}`。

#### (3) 训练：`0new/pomo/NEW_py_ver/TSP/POMO/TSPTrainer.py`

POMO 的 loss 很简单：

- `reward` 是每个 POMO rollout 的最终 reward
- `advantage = reward - reward.mean(dim=1)`
- `loss = - advantage * log_prob`

这部分非常适合你借用，因为它已经把：

- batch rollout
- baseline
- policy gradient

组织得非常清楚。

### 3.3 POMO 对你有什么可借的

#### 可以直接借的

1. **环境接口风格**
   - `load_problems`
   - `reset`
   - `pre_step`
   - `step`

2. **rollout 组织方式**
   - batch 内并行 rollout
   - 记录每一步 log-prob
   - episode 结束后统一回传 reward

3. **baseline 思想**
   - 可以继续用 batch mean / multi-start mean / multi-init mean

4. **代码复杂度低**
   - 比 DACT 的 PPO 更容易改造

#### 不能直接借的

1. **动作头不能复用**
   - 你的动作不是节点，而是算子 `op_id`
   - 所以 decoder 不应该输出 `n` 维，而应该输出 `m` 维

2. **状态不再是 partial tour**
   - 你的状态是当前完整解 `sol_t`
   - 所以不能继续用 `ninf_mask` 那种“已访问节点 mask”语义

3. **奖励不一定只在终点给**
   - 你的每一步都是完整解到完整解的改进，可以有 dense reward
   - 所以更适合每一步给 `delta_cost`

### 3.4 对你的意义

POMO 最适合的角色是：

> **训练外壳 / 框架壳子**

也就是：

- 用它的 `env-trainer-rollout-loss` 风格
- 但把“节点解码器”换成“算子分类头”
- 把“部分解状态”换成“完整解状态编码”

它不适合作为你最终方法的“状态表示参考”，因为它根本没有 solution-conditioned representation。

---

## 4. LIH 分析：它是 solution-based improvement learning，但动作是“节点对”，不是“算子 ID”

LIH 是你这里非常值得认真参考的一套工作，因为它已经不是 construction，而是 improvement。

### 4.1 LIH 论文要解决什么

LIH（Learning Improvement Heuristics for Solving Routing Problems）要学的是：

- 给定一个当前完整解
- 学会选择下一步最值得执行的局部修改
- 通过多步迭代逐步优化路径

这与 POMO 最大的区别是：

- POMO：从空解开始构造
- LIH：从已有解开始改进

这和你的新目标已经非常接近。

### 4.2 LIH 的代码形态

重点文件：

- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/attention_model.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/graph_encoder.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/train.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/problems/problem_tsp.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/CVRP/CVRP100/train.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/CVRP/CVRP100/attention_model.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/CVRP/CVRP100/graph_encoder.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/CVRP/CVRP100/problems/problem_vrp.py`

它的整体流程是：

1. 当前解 `rec`
2. 从 `rec` 里抽 solution-dependent features
3. 用 attention 网络打一个 pairwise score matrix
4. 从中采样/选最大 pair `(i,j)`
5. 用固定局部规则（TSP 里基本是 2-opt 反转）执行一步
6. 继续迭代

### 4.3 LIH 的核心点：它如何表示“当前解”

这是你最需要参考的地方。

#### 4.3.1 TSP 里的解表示

在 TSP 代码里，`rec` 是一个**排列序列**，表示 tour 的访问顺序。例如：

- `rec = [0, 5, 2, 7, ...]`

在 `problem_tsp.py` 的 `get_costs()` 中，`rec` 被当作排列来 gather 坐标并计算路径长度。

#### 4.3.2 TSP 的状态特征提取

TSP 的外部特征提取函数在：

- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/train.py`
- `0new/Learn-Improvement-Heuristics-for-Routing-main/TSP/tsp100/test_function.py`

函数名：`emded`（拼写上有点随意，但本质是 embedding 前的特征整理）

它做了两件事：

1. **对当前解的位置信息做 sinusoidal positional encoding**
2. **为每个节点取出它在当前排列中的坐标信息**

代码上：

- `cor = torch.nonzero(rec.long() == i)`：找到节点 `i` 在当前 tour 中的位置
- `single_pos`：取该位置对应的 sinusoidal position embedding
- `input_info`：TSP 版本里最终喂进 `init_embed` 的是 2 维，因此实际使用的是较简单的节点坐标表示

需要特别指出：

- `attention_model.py` 里有一个 `_emdedding()` 成员函数，里面构造了更丰富的 `(predecessor, self)` 局部结构特征
- 但训练/测试主流程实际上调用的是 `train.py`/`test_function.py` 里的外部 `emded()`
- 因此**当前代码真正跑起来时，TSP 的解特征并没有完全用上更复杂的局部 tuple，而是主要使用“节点坐标 + 当前序位 encoding”**

这是一个非常重要的代码层面观察：

> **LIH-TSP 的代码实现，比论文给人的印象更“轻”，更偏向“节点坐标 + 当前位置编码”而不是高度复杂的 solution local pattern。**

#### 4.3.3 CVRP 的状态特征提取

CVRP 的 `emdedding()` 在：

- `0new/Learn-Improvement-Heuristics-for-Routing-main/CVRP/CVRP100/train.py`

这里比 TSP 明显更强：

对于每个节点，它构造：

- `pre`：前驱节点
- `mid`：当前节点
- `las`：后继节点
- `dem`：当前节点需求

然后 gather 三个节点的坐标：

- `pre(x,y)`
- `self(x,y)`
- `next(x,y)`

再拼上 demand，得到：

- `6 + 1 = 7` 维局部特征

再加上该节点在当前序列中的**位置编码** `pos_enc`。

所以 LIH-CVRP 的每个节点状态，本质上是：

> **当前解上的局部三元组结构 + 当前位置 encoding**

这非常有参考价值，因为它说明：

- 对于 solution-based policy，
- “静态实例特征”不够，
- 必须把“这个节点当前在解中的前后关系”显式编码进去。

### 4.4 LIH 的动作空间是什么

这点必须讲清楚，因为它和你的目标并不一样。

#### 4.4.1 LIH 的动作不是“选算子”

LIH 的动作实际上是：

- 从所有节点对 `(i,j)` 中选一个 pair

也就是说，它的 policy 输出的是一个 `n × n` 的 pairwise 打分表。

在 `graph_encoder.py` 里：

- `att_s` 被 reshape 成 `(heads, batch, gs, gs)`
- 训练时从 flatten 后的 `gs * gs` 里采样 `multinomial(1)`
- 再还原为 `(row, col)` 对

这就是：

- 动作空间：pair of nodes
- 局部操作：由底层 problem logic 固定解释

#### 4.4.2 LIH 的 operator 是“隐含固定”的

例如在 TSP 中，`problem_tsp.py` 的 `get_costs()` 里：

- 给定 `exchange` 后，代码实际上执行的是一个**区间反转**
- 这本质上就是 2-opt 风格的修改

所以 LIH 做的不是：

- 从 `{2opt, swap, insert, ...}` 中选一个算子

而是：

- **固定使用某种局部 move family，再从该 family 的参数空间里选具体动作**

这和你的方案不同。

### 4.5 LIH 的训练方式

从代码看：

- TSP 版本更像 actor-critic / REINFORCE 风格：
  - `reinforce_loss = ((val_tru - val_est_det) * log_like).mean()`
- CVRP 版本明显更接近 PPO 式 clipped surrogate：
  - `ratios = exp(logprob - old_logprob)`
  - `surr1 / surr2` clipping

这也说明 LIH 的代码库内部并不完全统一，TSP 和 CVRP 分支存在一定程度的“各自演化”。

### 4.6 LIH 对你的真正启发

LIH 对你的帮助不在于“直接照搬动作定义”，而在于两点：

#### 启发 1：**当前解的局部结构要进入状态表示**

尤其是：

- 前驱 / 当前 / 后继
- 节点当前位置
- 当前 route 上的局部邻域

#### 启发 2：**如果后续你想把算子选得更细，可以把算子内部参数继续学习化**

你的当前目标是“每一步先只选算子 ID”，这是对的，因为更稳、更容易起步。

但如果以后你要更强，可以做成两层：

- 上层选算子 family（2-opt / swap / rrc / ...）
- 下层再选该算子的具体参数（例如 pair `(i,j)`）

而 LIH 就是这种“固定 family，下层选参数”的典型参考。

### 4.7 LIH 的局限（从你这个目标看）

1. **动作空间不是算子，而是 pair**
2. **TSP 实现中的解特征其实没想象中那么充分**
3. **不同 problem/size 目录代码重复较多，工程复用性一般**
4. **不适合直接扩展成 heterogeneous operator zoo**

所以 LIH 更适合作为：

> **solution feature 提取的参考来源**

而不是你的主框架。

---

## 5. DACT 分析：它是最接近你目标的参考原型

DACT 是这三者中与你目标最接近的一套。

### 5.1 DACT 论文要解决什么

DACT（Dual-Aspect Collaborative Transformer）要解决的是：

- 给定当前完整解
- 通过神经网络选择一个改进动作
- 反复迭代求解 routing 问题

它的关键创新有：

1. **双通道表示**：Node Feature Embedding (NFE) 与 Positional Feature Embedding (PFE) 分开建模
2. **Cyclic Positional Encoding (CPE)**：针对 tour/cycle 的循环对称结构设计 PE
3. **DAC-Att**：在 node aspect 和 position aspect 之间做协同注意力
4. **PPO + curriculum learning**：用于训练 improvement policy

### 5.2 DACT 的代码结构

关键文件：

- `0new/dact/VRP-DACT-new_version/run.py`
- `0new/dact/VRP-DACT-new_version/agent/ppo.py`
- `0new/dact/VRP-DACT-new_version/nets/actor_network.py`
- `0new/dact/VRP-DACT-new_version/nets/graph_layers.py`
- `0new/dact/VRP-DACT-new_version/problems/problem_tsp.py`
- `0new/dact/VRP-DACT-new_version/problems/problem_vrp.py`
- `0new/dact/VRP-DACT-new_version/options.py`

### 5.3 DACT 的核心优点：它已经是“当前解 -> 一步算子 -> 新解”的 MDP

#### 5.3.1 解表示：linked-list / successor representation

DACT 用的不是排列，而是**后继表**表示：

- `solution[i] = j` 表示当前解里节点 `i` 的后继是 `j`

例如 README 里的说明：

- 若边 `0->1`, `1->5`, `2->10` 在解里，则 `rec[0]=1, rec[1]=5, rec[2]=10`

这个表示非常适合迭代算子：

- `swap`
- `insert`
- `2-opt`

因为这些操作本质上都可以通过改写若干 successor 指针完成。

相比排列表示，这种 linked-list 表示在“局部修改一步解”时更自然。

#### 5.3.2 TSP 的状态特征：静态节点特征 + 循环位置特征

在 `actor_network.py` 中：

- TSP 直接把 `x_in`（节点坐标）送入 `EmbeddingNet`
- `EmbeddingNet.forward()` 输出：
  - `NFEs`：Node Feature Embeddings，来自静态输入特征
  - `PFEs`：Positional Feature Embeddings，来自当前解诱导出的 visited order

也就是说，TSP 的状态编码被拆成两部分：

1. **节点是谁**（坐标）
2. **它目前在当前 tour 的哪个循环位置**（CPE）

这是 DACT 相比 LIH 的一个非常强的点：

> LIH 更像“把当前解的局部结构硬拼到输入里”，而 DACT 是把“节点特征”和“当前解位置特征”明确拆成两个通道。

#### 5.3.3 CPE 的代码落地

在 `nets/graph_layers.py` 的 `EmbeddingNet` 中：

- `Cyclic_Positional_Encoding()` 预生成周期模式
- `position_encoding()` 根据当前 `solutions` 的 visited order 构造 `visited_time`
- 再按 `visited_time` 从 CPE 表中 gather 对应位置向量

这说明 DACT 不是普通 sinusoidal PE，而是：

- 专门针对 cycle 的编码
- 对 tour 的旋转对称更友好

这对于你的任务非常有价值，因为你的迭代状态也是“一个完整 tour / 完整 routing solution”。

#### 5.3.4 CVRP 的状态特征：非常值得借鉴

DACT 在 CVRP 中的状态设计明显比 LIH 更系统。

在 `actor_network.py` 中，它先构造：

- `loc = (x, y)`
- `pre`：当前节点到前驱的距离
- `post`：当前节点到后继的距离

然后从 `problem_vrp.py:get_real_mask()` 拿到 `to_actor`，其三维为：

1. **当前节点之前的累计载重** `cum_demand_before`
2. **当前节点需求** `demand`
3. **当前 route 剩余需求量 / route demand after current node`**

最后组合成 7 维：

1. `x`
2. `y`
3. `dist_to_pre`
4. `dist_to_post`
5. `cum_demand_before`
6. `self_demand`
7. `remaining_route_demand_after`

这 7 维正对应 `Actor.__init__()` 中：

- `problem_name == 'cvrp'` 时 `node_dim = 7`

这个设计非常好，因为它把：

- 静态几何信息
- 当前解的局部边结构
- 当前 route 的容量上下文

统一进了每个节点表示。

这比 LIH-CVRP 的“前驱/自身/后继坐标 + demand”更进一步，因为它显式加入了**可行性相关上下文**。

#### 5.3.5 可行性 mask：DACT 的另一个强项

在 `problem_vrp.py:get_real_mask()` 中，DACT 会根据当前解和容量上下文构造：

- 哪些 pair 是不允许的
- 哪些 pair 是允许的

尤其在 `2_opt` 模式下，它通过：

- route plan
- cumulative demand
- route demand totals

来判断一对节点之间的改动是否会违反容量约束。

这说明 DACT 不是简单“学一个动作打分器”，而是：

> **把当前解下的约束可行域信息，显式融合进 policy 的动作 mask。**

这是你未来做 CVRP operator selection 时一定要借鉴的思想。

### 5.4 DACT 的动作空间仍然不是“选算子”，但已经非常接近

DACT 的动作仍然是 pair `(i,j)`，不是 operator ID。

但是，它已经有了两个非常接近你需求的部分：

#### (1) `problem.step()` 已经是“算子接口”

在 `problem_tsp.py` / `problem_vrp.py` 中：

```python
if self.step_method == 'swap':
    next_state = self.swap(...)
elif self.step_method == '2_opt':
    next_state = self.two_opt(...)
elif self.step_method == 'insert':
    next_state = self.insert(...)
```

这说明 DACT 已经把“如何对当前解做一步修改”封装成了 operator interface。

只是当前实现里：

- `step_method` 在一次训练/推理过程中是固定的
- policy 学的是该 operator family 的参数 `(i,j)`

而你的目标是：

- `step_method` 本身也要成为动作

所以从 DACT 到你的方法，只差一步观念转换：

> 从“固定 operator family，学习 family 内参数”
>
> 变成“先学习选哪个 operator family，再执行该 operator”

#### (2) `problem.step()` 的 reward 已经是 improvement reward

DACT 的 reward 是：

- 当前 best-so-far 和新解之间的改进量
- `reward = pre_bsf - now_bsf`

这正是 improvement RL 最自然的 reward 形式之一。

对你的方法来说，也完全可以保留类似设计。

### 5.5 DACT 的训练 loop 也很适合作为你的主参考

在 `agent/ppo.py` 中：

- 先生成初始解 `solutions = problem.get_initial_solutions(batch)`
- 再循环 `T_train` 步
- 每步：
  - actor 根据当前解选 action
  - `problem.step(...)` 产生新解与奖励
  - 更新 best solution
- 最后做 PPO 更新

这和你的目标几乎就是同一个问题，只差：

- DACT action = pair
- 你要改成 action = operator ID

因此，从“迭代优化 RL 环境”的角度，DACT 比 POMO 更像你的直接前身。

### 5.6 DACT 的限制

1. **动作仍然是 pair，不是 operator**
2. **CVRP 的 mask 主要围绕 `2_opt` 写，扩更多 operator 要补接口**
3. **模型设计偏复杂（双流 + PPO + curriculum）**
4. **如果你只做 operator-level selection，DACT 当前 decoder 会显得过重**

### 5.7 DACT 对你的真正启发

DACT 对你的帮助主要有四点：

1. **当前解用 successor representation 表示非常合适**
2. **solution-conditioned position encoding 很重要**
3. **CVRP 必须显式编码 route/load/feasibility context**
4. **迭代式 PPO / rollout 组织已经基本现成**

---

## 6. LIH 与 DACT 对“解特征提取”的对比总结

这是最关键的一节，因为你明确提到：

> LIH 和 DACT 都是基于解的，请参考一下是如何提取解的特征的。

### 6.1 从表示方式看

#### LIH

- TSP：排列 `rec = [v_1, v_2, ..., v_n]`
- CVRP：特定顺序序列，附加 depot/dummy 处理
- 更偏向“从当前序列中抽局部 tuple 特征”

#### DACT

- TSP/CVRP：后继表 `solution[i] = next(i)`
- 更偏向“用 successor structure + visited_time 还原当前解”
- 更适合高频局部修改

**结论**：

- 如果你只做 constructive decoding，排列表示够用
- 如果你做 iterative operator application，**DACT 的 linked-list / successor 表示更适合**

### 6.2 从 TSP 特征看

#### LIH-TSP

更像：

- 节点坐标
- 当前排列位置的 sinusoidal PE

#### DACT-TSP

更像：

- 节点静态坐标（NFE）
- 当前 tour 中的循环位置编码（PFE / CPE）

**结论**：

- LIH-TSP 的启发是“当前位置信息确实重要”
- DACT-TSP 的启发是“位置特征应当和节点特征解耦”

对你的方法，我更推荐：

> **用 DACT 的“静态实例通道 + 解位置通道”的双通道思路。**

### 6.3 从 CVRP 特征看

#### LIH-CVRP

每个节点特征接近于：

- 前驱坐标
- 当前坐标
- 后继坐标
- 当前需求
- 再加一个位置编码

它强调的是：

- **局部几何三元组**

#### DACT-CVRP

每个节点特征接近于：

- 当前节点坐标 `(x,y)`
- 与前驱/后继的边长
- 当前节点之前的累计载重
- 当前节点需求
- 当前 route 剩余需求量
- 再加 solution-based 的循环位置编码

它强调的是：

- **局部边结构 + 约束上下文 + 位置上下文**

**结论**：

- LIH 更适合提供“局部拓扑三元组”的直觉
- DACT 更适合提供“可行性上下文 + 路由上下文”的系统设计

### 6.4 对你来说最值得保留的“混合特征模板”

如果让我为你的新方法设计 solution state feature，我会用一个**LIH + DACT 混合版**。

对于每个节点，建议特征包括：

#### 静态实例特征

- `x, y`
- 若是 CVRP，再加 `demand`
- 节点到 depot 的距离
- 最近邻距离统计（可选）

#### 当前解局部结构特征

- predecessor 节点坐标 / embedding
- successor 节点坐标 / embedding
- 边长 `dist(pre, self)`
- 边长 `dist(self, next)`
- 节点在当前 route 中的位置 index
- 节点在整个 solution 中的 visited time

#### CVRP 专属 route/context 特征

- 当前节点之前累计载重
- 当前 route 剩余容量
- 当前 route 总载重
- 当前 route 节点数
- 当前 route 的几何中心 / route compactness（可选）

#### 全局搜索状态特征

- 当前 cost
- best-so-far cost
- 最近一步 improvement
- 连续多少步没改进（stagnation counter）
- 当前迭代步 `t / I`
- 上一步用了哪个算子
- 最近 K 步 operator histogram（可选）

这比单纯的“节点坐标 + 当前解”更适合做 operator selection。

---

## 7. 你的新方法如何正式建模

现在我把你的想法形式化一下。

### 7.1 初始化阶段

给定实例 `X`，先从初始化方法集合中选一个：

- `a_init ∈ A_init = {pomo, am, heatmap_xxx, greedy, random, ...}`

得到初始解：

- `sol_0 = Init[a_init](X)`

如果某个方法是 heatmap-based：

- 它只在这里使用
- 不进入迭代算子集合

### 7.2 迭代阶段

设最大步数 `I=100`，那么对 `t = 0,1,...,I-1`：

- 状态：`s_t = (X, sol_t, best_t, hist_t)`
- 动作：`a_t ∈ A_op = {op_1, ..., op_m}`
- 转移：`sol_{t+1} = op_{a_t}(X, sol_t)`
- 更新 best：`best_{t+1} = min(best_t, cost(sol_{t+1}))`
- 奖励：由 `cost` 改变量定义

### 7.3 这和标准 TSP decoding 的对应关系

你给的类比非常准确：

#### 标准 constructive TSP

- 输入：实例
- 走 `n` 步
- 每一步选一个点
- 动作空间：节点 index

#### 你的 operator-based iterative TSP

- 输入：实例 + 当前完整解
- 走 `I` 步
- 每一步选一个算子
- 动作空间：算子 index

差异不在“是否是 RL”，而在：

- 状态从 partial solution 变成 full solution
- 动作从 node selection 变成 operator selection

这是完全可做的。

---

## 8. 最关键的建模决策：你的“动作”到底是算子 ID，还是算子 + 参数？

这是实现前必须先定清楚的。

### 8.1 我建议 MVP 先做“只选算子 ID”

也就是：

- policy 只输出：`2-opt`、`swap`、`rrc`、`divide&conquer`、...
- 每个算子内部自己完成具体搜索/执行

例如：

- `2-opt` 算子内部跑一次 best-improvement / first-improvement
- `swap` 算子内部跑一次候选对搜索
- `rrc` 算子内部执行一次固定参数的 destroy-repair

这样你的 policy 学的是：

> **在当前解状态下，哪类 operator 最值得用**

这是最清晰、最符合你现在目标的版本。

### 8.2 为什么不要一开始就做“算子 + 参数联合动作”

因为那会立刻把动作空间重新放大到非常复杂：

- 选 `2-opt` 还要选 `(i,j)`
- 选 `rrc` 还要选 seed/subproblem size/repair budget
- 选 divide-and-conquer 还要选 region / split granularity

这会让你的问题从：

- operator selection

变成：

- hierarchical action search

虽然最终会更强，但**不适合第一版**。

### 8.3 推荐分阶段路线

#### 阶段 1（MVP）

- 动作 = 算子 ID
- 每个算子内部固定参数、固定策略

#### 阶段 2

- 动作 = 算子 ID + 少量离散参数

#### 阶段 3

- 层次化：上层选算子，下层选参数 / 局部位置

而 LIH / DACT 可以作为阶段 3 的参考。

---

## 9. 我对你新方法的推荐实现方案

下面给出一个我认为最稳、最贴近你需求的实现方案。

## 9.1 总体框架

### 模块 1：初始化选择器（可学习，也可先固定）

输入：

- 实例 `X`

输出：

- 初始化方法 `a_init`
- 初始解 `sol_0`

第一版可以简化为两种实现：

#### 方案 A：先不学习初始化选择器

- 固定一个初始化器（例如 greedy / POMO / heatmap）
- 先把迭代 operator policy 跑通

#### 方案 B：初始化选择器做成单独小策略

- policy over init methods
- reward 与最终 best solution 绑定

我的建议：

> **先做 A，再做 B。**

因为真正难的是迭代 operator policy，而不是 init selector。

---

### 模块 2：迭代算子策略（主角）

输入：

- 实例编码 `enc(X)`
- 当前解编码 `enc(sol_t | X)`
- 历史状态特征 `hist_t`

输出：

- `m` 个算子的 logits / probs

然后选出：

- `a_t = argmax / sample(probs)`

再调用：

- `sol_{t+1} = operator_zoo[a_t].apply(X, sol_t)`

---

### 模块 3：算子库 `operator_zoo`

第一版建议只收少量但差异明显的算子：

- `2_opt`
- `swap`
- `insert`
- `rrc`（destroy-repair）
- `segment_reverse` / `relocate`（二选一）
- `divide_and_conquer` / `quadtree_subsolve`（如果你已经有现成实现）

重要的是：

- 每个 operator 都要统一接口
- 都输入当前解，输出新解和本步信息

建议接口：

```python
new_solution, op_info = operator.apply(instance, solution)
```

其中 `op_info` 可以包括：

- `delta_cost`
- `runtime`
- `feasible`
- `num_changed_edges`
- `operator_internal_stats`

这样后续你也可以把这些信息回馈给策略网络或日志系统。

---

## 9.2 状态编码器怎么设计

### 我最推荐的版本：DACT 主体 + LIH 局部特征补充

#### 节点级输入建议

对每个节点，构造：

1. **静态特征**
   - `x, y`
   - `demand`（CVRP）
   - `dist_to_depot`（CVRP/TSP 都可）

2. **当前解局部结构特征**
   - predecessor 节点坐标 / embedding
   - successor 节点坐标 / embedding
   - `dist(pre, self)`
   - `dist(self, succ)`

3. **当前解位置特征**
   - visited time
   - cyclic positional encoding（推荐）

4. **约束与 route 上下文（CVRP）**
   - cum_load_before
   - remaining_capacity
   - route_total_load
   - route_id / route index（可选）

#### 全局聚合特征建议

对整个当前解，再做一个 global pooling，得到：

- graph embedding
- current cost
- best cost
- relative gap to init
- step ratio `t/I`
- stagnation count
- 上一步 operator id embedding

最后用：

- global token + pooled node embeddings + history features

去预测 operator logits。

### 为什么不建议直接用 POMO 的 encoder 原样做

因为 POMO 的 encoder只编码：

- 节点坐标图

但你这里需要编码的是：

- 同一个实例上，不同当前解对应的不同状态

也就是说：

> **instance encoder 不够，你必须要有 solution encoder。**

而 DACT/LIH 的价值就在这里。

---

## 9.3 训练方式怎么选

### 方案 1：先用 REINFORCE / batch mean baseline（最容易起步）

如果你想尽量接近 POMO 风格，第一版可以这么做：

- rollout 长度：`I`
- 每一步采样一个 operator
- 累计 reward 或只看 final best improvement
- baseline 用 batch mean

#### 奖励建议

第一版推荐 dense reward：

- `r_t = cost(sol_t) - cost(sol_{t+1})`

如果使用 best-so-far 风格：

- `r_t = bsf_t - bsf_{t+1}`

其中第二种和 DACT 更像，更稳定。

#### loss

```text
R = sum_t r_t
adv = R - baseline
loss = - sum_t log pi(a_t | s_t) * adv
```

### 方案 2：用 PPO（更像 DACT）

如果你希望更稳，尤其算子执行有一定随机性时，可以直接上 PPO。

这时 DACT 的 `agent/ppo.py` 思路就很有参考价值：

- rollout 收集 `state / action / reward / logprob`
- n-step return / GAE
- clipped surrogate 更新

### 我的建议

#### 第一版：

- 如果你想快点看到结果：**REINFORCE + batch mean baseline**

#### 第二版：

- 如果你发现训练震荡或高方差：**切换 PPO**

也就是说：

> 可以先借 POMO 的 trainer 风格，再逐步过渡到 DACT 的 PPO。

---

## 10. 如果你坚持“套 POMO 框架”，我建议怎么改

你说“随便套一个现有框架（例如 POMO）”，这个是可以的，但要明确：

> 套的是**训练骨架**，不是套它的节点解码器。

### 10.1 可以保留的部分

- `Env` 风格：`reset / pre_step / step`
- batch rollout
- log-prob 累积
- advantage / baseline
- 多轨 rollout（如果你愿意）

### 10.2 需要彻底改掉的部分

#### 原来 POMO 的动作：

- 输出 `problem_size=n` 个节点 logits

#### 你的动作：

- 输出 `num_operators=m` 个算子 logits

所以新的 policy head 应该是：

```python
operator_logits = MLP(global_solution_state)
```

而不是：

```python
node_probs = decoder(encoded_nodes, current_partial_solution)
```

### 10.3 新环境应当长什么样

建议环境接口如下：

```python
reset():
    - 采样实例 X
    - 生成/选择初始化解 sol_0
    - best = cost(sol_0)
    - t = 0
    - 返回 state_0

step(op_id):
    - 执行 operator_zoo[op_id] on sol_t
    - 得到 sol_{t+1}
    - 更新 best
    - 计算 reward
    - t += 1
    - done = (t == I)
```

### 10.4 POMO 的“multiple optima”思想怎么迁移

POMO 的核心是多起点。

在你的问题里，可以迁移成：

#### 迁移方式 A：多初始解 rollout

- 同一个实例，用多个初始化器/多个初始化样本
- 每个初始化解作为一个 parallel rollout

#### 迁移方式 B：多 perturbation seeds

- 同一个初始解，但给不同随机扰动 or operator seeds

这会带来一个很自然的 baseline：

- 同一实例内多个 rollout 的均值

因此，POMO 的思想不是无用，而是应该改造成：

> **multiple-initial-solution optimization**，而不是 multiple-start node decoding。

---

## 11. 我最推荐的第一版（MVP）

如果现在真的开始做，我建议你这样分三步。

### 11.1 第一步：只做 TSP，固定初始化器

- 初始化器固定为 `greedy` 或 `POMO` 推理输出
- 迭代 operator 集合只放：
  - `2_opt`
  - `swap`
  - `insert`
  - `rrc`

目标：

- 跑通“当前解 -> 选 operator -> 新解”的训练闭环

### 11.2 第二步：把状态编码做强

从简单到复杂逐步叠加：

#### baseline-1

- 只用 global 特征：
  - current cost
  - best cost
  - step index
  - stagnation
  - 初始解 cost

#### baseline-2

- + DACT 风格 CPE + node pooling

#### baseline-3

- + LIH 风格局部结构 tuple

这样你可以清楚比较：

- solution features 到底有没有带来增益

### 11.3 第三步：再引入初始化选择器

等 operator policy 稳了以后，再加：

- 初始化方法选择 `a_init`

此时整体训练就是：

- 先选 init
- 再跑 I 步 operator policy
- 最终 reward 同时归因给 init policy 和 operator policy

这个时候再做 joint training 才比较稳。

---

## 12. 一个更进一步但很重要的观点：你要学的是“哪类搜索阶段该用哪类算子”

你的方法最有潜力的地方，不只是“实例到算子”的映射，而是：

> **实例 × 当前解阶段 × 当前停滞状态 -> 最优算子**

也就是说，真正重要的不只是问题实例本身，还包括当前搜索进度。

例如：

- 初始 gap 很大时，RRC 常常比 2-opt 更值钱
- 已经很接近局部最优时，2-opt / swap 更合适
- 连续若干步无改进时，应该切回更强破坏算子

因此，你的 policy 最好显式看到：

- 当前 cost 相对 init 的下降比例
- 最近 K 步平均改进
- 连续无改进步数
- 上一步 / 上几步用过哪些算子

这类“搜索阶段特征”在你的方法里会比纯静态实例特征更重要。

---

## 13. 代码实现建议（只从 0new 出发，不碰你原来的双 gate 原型）

如果你后面要正式写代码，我建议在 `0new/` 下新开一个独立目录，例如：

```text
0new/operator_policy/
  README.md
  env.py
  init_zoo.py
  operator_zoo.py
  state_encoder.py
  policy.py
  value_net.py
  trainer_reinforce.py
  trainer_ppo.py
  rollout.py
  tsp_operators.py
  cvrp_operators.py
  metrics.py
```

### 各模块职责

#### `env.py`

- 管理 `reset/step`
- 保存 `instance/current_solution/best_solution/step_count`

#### `init_zoo.py`

- 封装各种初始化器
- heatmap-based 方法只在这里出现

#### `operator_zoo.py`

- 注册所有算子
- 统一 `apply()` 接口

#### `state_encoder.py`

- 做 solution-dependent 特征提取
- 重点融合 DACT + LIH 设计

#### `policy.py`

- 输入 state embedding
- 输出 operator logits

#### `value_net.py`

- 如果上 PPO / actor-critic，需要 value baseline

#### `trainer_reinforce.py`

- 第一版建议从这里起步

#### `trainer_ppo.py`

- 第二版更稳定时再上

---

## 14. 最终建议：你该借谁，怎么借

最后给一个非常明确的结论。

### 14.1 如果只问“谁最接近你的问题”

答案是：

- **DACT 最接近**

因为它已经是：

- 当前解 -> 选一步动作 -> 新解 -> reward

### 14.2 如果只问“谁最适合做外壳”

答案是：

- **POMO 最容易改**

因为它：

- env/trainer 简洁
- REINFORCE baseline 简单
- 代码负担小

### 14.3 如果只问“解特征该怎么做”

答案是：

- **TSP：优先学 DACT 的 CPE + 双通道**
- **CVRP：优先学 DACT 的 route/load context**
- **局部结构补充：参考 LIH 的 pre/self/next tuple**

### 14.4 如果让我给你一句最实用的设计建议

我会建议：

> **不要把 POMO 当作模型模板，把它当训练壳；不要把 LIH/DACT 当作最终方法照抄，把它们当“解状态编码设计手册”。**

进一步地：

> **第一版先做“固定初始化器 + 选算子 ID”的 operator policy；不要一开始就学习 operator 内部参数。**

---

## 15. 一页式总结（供后续实现时快速回顾）

### 你现在要做的不是：

- 构造式 node policy
- 或一次性选一个迭代器

### 你现在要做的是：

- `X -> init -> sol_0`
- `sol_t -> operator policy -> op_t -> sol_{t+1}`

### 最合适的状态表示：

- DACT 风格 successor representation
- DACT 风格 CPE
- LIH 风格 local tuple
- 全局搜索进度特征

### 最合适的第一版动作：

- 只选 operator ID

### 最合适的第一版训练：

- REINFORCE / batch mean baseline
- 或 PPO（若训练不稳）

### 三套参考工作的角色分工：

- **POMO**：训练骨架参考
- **LIH**：局部 solution feature 参考
- **DACT**：迭代环境 / 解表示 / PPO 参考

---

## 16. 我对后续实现的建议顺序

如果你下一步要我开始动手实现，我建议顺序是：

1. 先在 `0new/` 新建独立原型目录
2. 先做 TSP-only
3. 先固定初始化器
4. 先实现 operator zoo + env
5. 再做 solution encoder
6. 再做 operator policy
7. 最后再把 initialization selection 加回来

这个顺序能最大限度降低工程复杂度。

---

如果后续你愿意，我下一步可以继续做两件事中的任意一件：

1. **把这份报告继续细化成“具体代码设计文档”**（类、接口、张量 shape、训练流程全部写清楚）
2. **直接在 `0new/` 下给你搭一个第一版原型代码骨架**


---

## 17. 补充：落到当前实现时，状态表示是如何具体参考 LIH 和 DACT 的

前面的报告已经从论文层面解释了 LIH 和 DACT 的思想。这里专门补一段“落到当前原型代码里，到底怎么借的”。

### 17.1 不是直接照抄，而是“抽象后重组”

当前原型的动作不是：

- 选下一个节点；
- 或选一对节点 `(i, j)`。

而是：

- 在一个 heterogeneous operator zoo 中选一个 operator ID。

因此，状态表示不能完全绑死在某一个 operator family 上。我的做法是：

- 保留 **LIH** 中最重要的“局部结构视角”；
- 保留 **DACT** 中最重要的“完整解条件化 + 位置通道”；
- 再补上 **搜索阶段 / 历史动作** 这类 hyper-heuristic 特别需要的信息。

### 17.2 对应 LIH 的部分：`pre / self / next` 局部结构 token

当前实现 `0new/operator_policy/state_encoder.py` 里，`_tour_node_features()` 会先把坐标按当前 tour 排序，再构造前驱、后继和相邻边长度：

```python
ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
prev_ordered = ordered.roll(shifts=1, dims=1)
next_ordered = ordered.roll(shifts=-1, dims=1)
rel_prev = ordered - prev_ordered
rel_next = next_ordered - ordered
dist_prev = rel_prev.norm(p=2, dim=-1, keepdim=True)
dist_next = rel_next.norm(p=2, dim=-1, keepdim=True)
```

这正对应了 LIH 最值得借的那部分：

- 一个点的状态不只是自己的坐标；
- 还包括它在当前解中的局部邻域关系；
- 对“当前这条 tour 是否存在明显坏边 / 扭曲区段”特别敏感。

换句话说，当前实现借的不是 LIH 的“pair-action 输出头”，而是它的 **solution-locality inductive bias**。

### 17.3 对应 DACT 的部分：位置比例 + 周期位置编码 + 完整解条件化

在同一个 `_tour_node_features()` 里，我还额外加入了：

```python
pos = torch.arange(tour.size(1), dtype=coords.dtype, device=coords.device)
pos = (pos / max(tour.size(1), 1)).view(1, -1, 1).expand(coords.size(0), -1, -1)
cpe = _cyclic_positional_encoding(tour.size(1), cpe_dim, coords.device)
```

这部分对应的是 DACT 的思想：

- 当前解不是无序集合，而是一个有顺序、有环结构的对象；
- 节点在当前解中的“位置角色”需要被编码进去；
- 这种表示应该是 **solution-conditioned** 的，而不是单纯实例静态编码。

与 DACT 的差别在于：

- DACT 原方法最终输出的是 pair logits；
- 当前实现最终输出的是 operator logits；
- 所以我保留了它的顺序/位置通道，但把输出头改成了 `whole-solution embedding -> operator classifier`。

### 17.4 为什么没直接用 DACT 的 successor 表示

因为 successor 表示非常适合“预测下一次 2-opt pair”，但不一定最适合 heterogeneous operator selection。

你的 operator zoo 里同时有：

- `2-opt` 家族
- `RRC destroy-repair`
- `GLOP subproblem reviser`

如果状态完全围绕 pair 结构设计，就会天然偏向 `2-opt` 类动作。

因此我这里采用的是更“中性”的 ordered-tour token 方案：

- 仍然利用当前解顺序；
- 仍然保留 DACT 风格位置编码；
- 但最终压成 whole-solution embedding，服务于 operator-level 决策。

### 17.5 新增的、专门为“operator scheduler”设计的部分

这部分是 LIH 和 DACT 都没有直接帮你解决的，需要单独补：

```python
progress = torch.stack([
    current_length / safe_init,
    best_length / safe_init,
    (current_length - best_length) / safe_init,
    (init_length - best_length) / safe_init,
    step_index.float() / max(rollout_steps, 1),
    stagnation.float() / max(rollout_steps, 1),
], dim=1)
```

以及：

- `edge_stats = mean / std / min / max`
- `init_action embedding`
- `prev_operator embedding`

这些特征解决的是：

- 当前已经搜索到第几步；
- 是否出现停滞；
- 当前边长分布是否还很粗糙；
- 这条解最初来自哪个 initializer；
- 上一步刚用了哪个 operator。

这正是上层 operator selection 与普通“单一方法内部迭代”最大的区别。

### 17.6 一句话总结这套表示

如果用一句话概括当前实现的状态表示：

> **LIH 提供局部几何结构，DACT 提供完整解的顺序/位置视角，而 operator selection 这个新任务额外要求加入搜索阶段与历史动作信息。**

也就是说，它不是简单拼接，而是：

- 从 LIH / DACT 提炼最适合 solution-improvement 的表征成分；
- 再把它们重组为适合 operator-level action 的全局状态编码。

---

## 18. 补充：`policy-guided 2-opt` 与普通 `2-opt` 的区别

这个问题必须单独强调，因为它关系到你后面做 operator zoo 的语义边界。

### 18.1 普通 `2-opt`

普通 `2-opt` 的 family 是固定的：

- 选一对位置或两条边；
- 把中间 segment reverse；
- 如果变好就接受。

它的差异通常来自：

- best-improvement / first-improvement
- 邻域裁剪
- 候选对的启发式筛选
- 是否随机重启

但这些都还是 **启发式选择 pair**。

### 18.2 `policy-guided 2-opt`

`policy-guided 2-opt` 不改变 move family，改变的是：

- 哪一对 `(i, j)` 最值得试；
- 由一个 conditioned on current solution 的神经策略来选。

因此：

- 普通 `2-opt` 更像 heuristic operator；
- `policy-guided 2-opt` 更像 learning-based operator；
- 两者都属于 `2-opt family`，但“pair selection mechanism” 完全不同。

### 18.3 对应到当前原型

- `two_opt` / `difusco_2opt_step`
  - 普通 2-opt
  - 依赖平台已有 refine 工具

- `dact_2opt_step`
  - DACT 的 policy-guided 2-opt
  - 先预测 pair，再执行 2-opt move

- `lih_2opt_step`
  - LIH 的 policy-guided 2-opt
  - 同样是 learned pair selection + 固定 move family

这意味着，在你的 operator selection 里，`two_opt` 和 `dact_2opt_step` 不应该被视作“重复动作”。

它们虽然都属于 2-opt family，但代表了两种不同的控制方式：

- 一个是 heuristic search control；
- 一个是 learned search control。

---

## 19. 补充：这次我实际扩展了哪些 EasyNCO 迭代方法

为了回应“尝试支持更多种平台支持的迭代方法”，当前原型已经从最早版本的：

- `two_opt`
- `lehd_rrc_step`
- `dact_2opt_step`

扩展到了：

- `two_opt`
- `difusco_2opt_step`
- `lehd_rrc_step`
- `dact_2opt_step`
- `lih_2opt_step`
- `glop_sub100_step`
- `glop_sub50_step`
- `glop_sub20_step`
- `glop_pass_step`

其中最重要的新扩展是 **GLOP reviser family**。

### 19.1 为什么优先接 GLOP

因为从 TSP 角度看，GLOP 很自然地对应一种和 `2-opt` / `RRC` 不同的 operator family：

- 它不是简单 pair move；
- 也不是单次 destroy-repair；
- 它更像“基于子问题分解的 learned reviser”。

这对于你的研究问题非常有价值，因为它让 operator zoo 真正开始具有 family diversity。

### 19.2 为什么暂时没把 UDC / DeepACO 也直接塞进来

主要是工程性原因，不是原理上不可能：

- `UDC` 在平台里的 TSP eval 路径更复杂，单步 wrapper 容易比较脆；
- `DeepACOIteration` 当前更偏分数输出，不像 `LEHD / DACT / GLOP` 这样容易抽成“给当前解做一步改进然后回传新 tour”的接口。

所以当前这次扩展优先选了：

- 平台里已有 checkpoint；
- 容易抽成单步 operator；
- 和已有 operator family 语义差异足够大的方法。

如果你后面要继续扩 operator zoo，我建议优先顺序是：

1. 把 GLOP 家族真正跑进实验；
2. 再尝试把 UDC TSP 单步化；
3. 最后再考虑更复杂的 multi-step reviser 或 family 内参数动作。
