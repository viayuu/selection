# GLOP & LEHD 论文/代码阅读笔记与“初始化-迭代”可组合性分析

本文档基于仓库 `papers/` 中的两篇论文全文（`papers/glop.md`、`papers/lehd.md`）以及作者给出的源代码（`papers/GLOP/`、`papers/LEHD/`）整理而成，目标是：

1. 详细梳理 **GLOP** 与 **LEHD** 的论文内容与实现方法（重点放在“算法管线/训练方式/推理方式/代码落点”）。
2. 从论文原理与代码接口层面分析：是否能把两者拆成 **初始化（init）** 与 **迭代改进（iter）** 两阶段，并进行交叉组合：
   - `GLOP-init + LEHD-iter`
   - `LEHD-init + GLOP-iter`

> 注：你当前实验主要在 TSP/TSPLIB 上验证“init/iter 可分离并可组合”是否带来实例级收益，因此本文讨论以 TSP 为主，最后补充 CVRP 的注意事项。

---

## 0. 两个方法的共同抽象：可拆成 init + iter

虽然两篇论文的叙事不同，但它们都可以抽象成同一类“先构造、再改进”的范式：

- **初始化（init）**：从实例 `x` 生成一个可行解 `s0`（TSP 即一个 Hamiltonian cycle 的排列）。
- **迭代改进（iter）**：从 `(x, s)` 出发做若干次局部重构/局部求解，得到更优 `s*`。

这类抽象在传统启发式里也很常见（构造式启发式 + 局部搜索），所以“分阶段组合”在原理上是自然的，但是否“有效”取决于：

- `iter` 是否**只依赖**“实例 + 当前解”而不依赖初始化过程的隐藏状态；
- `iter` 的局部算子/重构器是否对“不同分布的初始解”仍有改进能力（分布偏移问题）；
- 工程上能否在解表示上对齐（节点序列、端点约束、归一化方式等）。

### 0.1 形式化：把 init / iter 写成算子（以 TSP 为例）

为了把“能否组合”说清楚，先把问题写成更接近算法/代码接口的形式。

- **实例**：`x = (v_0, ..., v_{n-1})`，其中 `v_i ∈ R^2` 是城市坐标（欧氏 TSP）。
- **解空间**：`S_n`（所有长度为 `n` 的排列）。一个具体 tour 可以表示为排列 `π`，对应的闭环路径为 `v_{π_0} → v_{π_1} → ... → v_{π_{n-1}} → v_{π_0}`。
- **目标函数**：`C(x, π) = Σ_{t=0}^{n-1} ||v_{π_t} - v_{π_{(t+1) mod n}}||_2`（或在 TSPLIB 上用对应 edge weight 规则；见第 4 节提醒）。

于是两种方法都可写成：

- **初始化算子**：`I(x) -> π0`（输出一个可行 tour）。
- **迭代改进算子**：`F(x, π; budget) -> π*`（在预算约束下对当前 tour 做局部改进，输出更好的 tour）。

你要验证的“组合”就是把 `I` 和 `F` 交叉配对：

- 纯方法：`π = F_LEHD(x, I_LEHD(x))` 或 `π = F_GLOP(x, I_GLOP(x))`
- 交叉组合：`π = F_GLOP(x, I_LEHD(x))` 或 `π = F_LEHD(x, I_GLOP(x))`

### 0.2 组合可行性的“硬条件”（TSP）

在 TSP 上，一个 `iter` 想要能和“任何 init”组合，至少需要满足：

1. **定义域条件**：`F(x, π)` 对任意合法排列 `π` 都能运行（不依赖 init 的隐藏状态/内部中间量）。
2. **端点/拼接一致性**：`F` 做局部重构时若需要“固定端点”，它必须明确端点是谁，并保证重构后端点保持不变，否则会破坏全局 tour 的拼接关系。
3. **可比代价**：改进判定使用的代价（子段/局部/全局）必须与全局目标 `C` 一致，或者至少在端点固定时仅差一个常数（这样“局部更优”就等价于“全局更优/不劣”）。
4. **单调接受（可选但很重要）**：如果 `F` 采用“只接受更优”的策略，则组合更安全：最坏情况是“改不动”，而不是系统性变差。

后文会看到：GLOP 的 revisions 和 LEHD 的 RRC 都非常接近满足这四点，这也是你在实验里能看到“分开 init/iter 仍然能跑且偶尔更好”的根本原因。

---

## 1. LEHD（NeurIPS 2023）论文内容与实现

### 1.1 动机与贡献

论文 `papers/lehd.md` 解决的核心痛点是：多数 constructive NCO（如 AM/POMO）在小规模训练后，**对大规模实例泛化很差**。作者提出一个结构性解释：

- 传统构造式模型常用 **Heavy Encoder + Light Decoder (HELD)**：编码器一次性得到静态节点 embedding，解码器轻量逐步选点。
- 当规模变大，静态 embedding 很难“持续表达”全局/动态关系；模型容易学习到与规模强相关的特征，导致 OOD/大规模泛化失败。

因此作者提出：

- **LEHD：Light Encoder + Heavy Decoder**
  - 编码器很轻（1 层 attention），避免把“规模相关的模式”固化成静态 embedding。
  - 解码器很重（多层 attention），在每一步用“当前部分解 + 可选节点集合”**动态重嵌入**，更像“在线重算关系”。
- 为了能训练 heavy decoder（RL 太贵且稀疏奖励），提出 **监督式的数据增强训练**：Learn to Construct Partial Solution。
- 推理时提出 **RRC（Random Re-Construct）**：与训练同构的“随机抽子路径-重构-若更优则替换”的在线改进机制。

### 1.2 模型结构：Light Encoder + Heavy Decoder

论文第 3 节（`papers/lehd.md#L51` 起）给出结构公式。概念上可总结为：

- **Encoder（轻）**：节点特征（TSP 为 2D 坐标）线性投影后，过 **1 层 attention layer** 得到初始节点 embedding。
  - 关键实现：去掉 normalization（论文 Appendix A 讨论其对泛化的影响）。
  - 代码对应：
    - TSP：`papers/LEHD/TSP/TSPModel.py` 中 `TSP_Encoder`（`encoder_layer_num = 1`）
    - CVRP：`papers/LEHD/CVRP/VRPModel.py` 中 `CVRP_Encoder`

- **Decoder（重）**：每一步构造时，把输入组织成：
  - 起点（starting node）embedding
  - 终点（destination node）embedding（训练/修复时固定）
  - 当前可选节点集合 embeddings（未访问节点）
  然后过 **L 层 attention layer（默认 6 层）** 输出对每个可选节点的打分，softmax 得到选择概率。
  - 关键实现点（论文 Appendix C 也强调）：不是在 logits 上做 mask，而是**直接从 decoder 输入中剔除 irrelevant nodes**（已选节点），以节省计算并强化“动态集合”的建模。
  - 代码对应（TSP）：
    - `papers/LEHD/TSP/TSPModel.py` 中 `TSP_Decoder._get_new_data()` 会把未选节点抽出来形成 `left_encoded_node`
    - `TSP_Decoder.forward()` 将 `(first, left, last)` 拼接后过 `DecoderLayer` 堆叠

### 1.3 训练：Learn to Construct Partial Solution（监督 + 数据增强）

论文第 4 节（`papers/lehd.md#L99` 起）核心是：利用 TSP/CVRP 的“最优性不变性（optimality invariance）”做增强。

- 给定一个实例的**最优 tour** `x`，其任意“连续子序列” `x_sub` 也应是该子问题（固定端点的 open path/局部结构）的最优（或至少是很强的监督信号）。
- 因此对每个训练实例，从最优 tour 中随机采样：
  - 长度 `w ~ Unif([4, n])`
  - 随机方向（正向/反向）
  - 随机起点位置
  得到大量 `x_sub`，把它们当作监督数据。

训练时：
- `x_sub` 的首/尾节点作为固定端点（start/dest），中间节点作为“可选集合”；
- 逐步预测下一个节点，用 cross-entropy（teacher forcing）训练。

代码对应（TSP）：
- 数据与子路径采样：
  - `papers/LEHD/TSP/TSPEnv.py:sampling_subpaths()`：从完整 tour 里截取连续片段，并构造“子问题坐标 + rank 映射”
- 训练 loop：
  - `papers/LEHD/TSP/TSPTrainer.py:_train_one_batch()`：直接用 `env.solution`（标签 tour）做 teacher forcing；loss 是 `-log(prob)` 的均值
- 模型 forward：
  - `papers/LEHD/TSP/TSPModel.py:TSPModel.forward()`：训练模式下用 `solution[:, current_step-1]` 作为 teacher

CVRP 版本的关键差异（论文 Appendix D）：
- 由于 CVRP 的 depot 分段导致“序列对齐”困难，作者用“节点序列 + via-depot 二值标记”的形式对齐（论文式子 (4)）。
- 代码对应：`papers/LEHD/CVRP/VRPModel.py` 内 `selected_flag_*` 相关逻辑（把“经 depot”编码成扩展动作空间）。

### 1.4 推理：Greedy 构造 + RRC 迭代改进

#### 1.4.1 Greedy 构造完整解

论文第 4.3 节给出推理描述：固定一个 destination，并让 starting node 随每步选择动态移动，直到构造出全序列。

实现上（TSP）：
- `papers/LEHD/TSP/TSPTester.py:_test_one_batch()` 首先用模型 greedy 生成一条完整 tour（`best_select_node_list`）。
- 模型每一步的“构造式决策”由 heavy decoder 完成：`papers/LEHD/TSP/TSPModel.py:TSP_Decoder.forward()` 会把
  - 当前 tour 的起点 embedding（first）
  - 当前节点/上一步节点 embedding（last）
  - 未访问节点 embeddings（left）
  拼成一个动态集合，再过多层 `DecoderLayer` 输出 logits → softmax → `argmax` 选下一个节点。

这点在 “init/iter 可组合性” 上很关键：**LEHD-init 只是返回一个排列 `π0`**，它不携带其它“必须给后续用”的隐状态，因此天然可被其它 iter 接管。

> 备注：代码里起始点常固定为 0（并不影响 TSP 环的长度），论文描述为随机选择 destination；两者都符合“只要生成一个合法 tour 就行”的层面。

#### 1.4.2 Random Re-Construct（RRC）

论文第 5 节：RRC 是一个“主动改进”机制，和训练时的 partial-solution 构造方式同构：

1. 从当前解 `s` 中随机采样一段连续子路径（随机长度/方向）。
2. 固定子路径端点，调用同一个 LEHD 模型重构中间节点顺序，得到新子路径。
3. 若新子路径更短，则用它替换原子路径；重复直到预算耗尽。

代码对应（TSP）：
- 抽子路径/构造子问题：
  - `papers/LEHD/TSP/TSPEnv.py:destroy_solution()` → 内部调用 `sampling_subpaths(..., repair=True)`
- 接受准则与替换：
  - `papers/LEHD/TSP/TSPTester.py:decide_whether_to_repair_solution()`：`before_reward > after_reward` 才替换（单调改进，不接受更差片段）
- 完整 RRC loop：
  - `papers/LEHD/TSP/TSPTester.py:_test_one_batch()` 中 `for bbbb in range(budget): ...`

这里补充两个“从原理上解释 RRC 为什么可组合”的关键点（也对应到代码细节）：

**(a) RRC 本质是“端点固定的子段重构”**

虽然论文用“partial solution”叙事，但从算子角度，RRC 更像一个 learned 的“k-opt / segment reordering”操作：

- 从当前 tour `π` 上抽取一段连续子序列（环上连续，所以用 `double_solution` 处理 wrap-around，见 `papers/LEHD/TSP/TSPEnv.py:sampling_subpaths()`）
- 这段子序列的两端节点在被重构时是“锚点”（start/dest），中间节点重新排列
- 若重排后的子段更短则替换回全局 tour（`decide_whether_to_repair_solution` 做映射与替换）

这意味着：RRC 的输入只需要 `(x, π)`，对 `π` 的来源没有要求，因此它天然可以当成“迭代组件”接在任意 init 后面。

**(b) 为什么它用“闭环长度”也能正确比较子段优劣（常数项抵消）**

LEHD 的 TSP 训练/修复在实现上把 “start/dest 固定的 open path” 转成一个“闭环 tour”来算长度：

- 在训练/修复时，decoder 的上下文包含 start/dest（论文公式；代码里通过把 start/dest 放在输入集合的特殊位置实现）。
- 在 `papers/LEHD/TSP/TSPTrainer.py:_train_one_batch()` 里，前两步会先固定 dest / start（`selected_teacher = solution[:, -1]` 与 `solution[:, 0]`），使得子段的两个端点在序列中相邻。
- 因此子段“闭环 tour”的长度 = `||dest-start||`（常数） + “从 start 到 dest 的 open path 长度”。

端点固定时，`||dest-start||` 是不变常数，所以比较两次重构的“闭环长度”与比较 open path 长度等价。这解释了为什么 RRC 作为局部改进算子能成立，并且“只接受更优”的策略可以保证非劣改进。

（你如果要从代码上验证这点，可以从 `TSPEnv._get_travel_distance()` 的 roll 计算入手：它本质算的是闭环长度，但端点被固定/相邻后，差异项是常数。）

### 1.5 论文实验结论（与本项目关注点相关的）

与“可组合性”最相关的点是：LEHD 不是只靠一次 greedy 构造，而是依赖 **RRC 作为可控预算的迭代改进算子**（`papers/lehd.md` 表 1/2/5）。

论文 Appendix B 进一步强调：
- RRC 能让模型性能**突破标签质量的上限**（用 OR-Tools 的次优标签训练，RRC 仍能超过 OR-Tools）。
- 这从原理上支持一个观点：**RRC 更像一个可插拔的改进算子**，它不“绑定”某一种初始化分布（当然效果会受分布偏移影响）。

---

## 2. GLOP（AAAI 2024）论文内容与实现

### 2.1 动机与贡献

`papers/glop.md` 的核心目标是：让神经求解器在 **大规模（上万/十万节点）** 路由问题上仍能实时给出可用解。

作者提出 **GLOP（Global and Local Optimization Policies）** 的分层框架：

- 对大规模 routing：先做 **全局划分/聚类/子集选择（global policy, heatmap, NAR）**；
- 对（子）TSP：用 **局部重构（local policy, AR）** 在小规模子问题上精细求解；
并宣称这是首次有效地“混合 NAR 与 AR”以兼顾可扩展性与精细构造能力。

### 2.2 (Sub-)TSP Solver：Insert → Divide → Conquer → Compose

论文第 4 节给出了 TSP 子求解器管线（这部分与你当前的 init/iter 拆分最直接相关）：

#### 2.2.1 Initialization：Random Insertion（RI）

- 用一个简单通用的启发式 **Random Insertion** 生成初始 tour。
- 直觉：不依赖学习、极快、对规模不敏感；为后续“局部重构”提供一个可行起点。

代码对应：
- 初始 tour 生成在 `papers/GLOP/main.py:_eval_dataset()`：
  - 通过 `utils.insertion.random_insertion_parallel(...)` 得到 `pi_batch`（本质是一个排列）
  - 再用 `batch.gather(...)` 得到按 tour 顺序排列的坐标序列 `seed`
- RI 的封装在 `papers/GLOP/utils/insertion.py`（底层调用 `random_insertion` 库）

从 init/iter 组合视角看，RI 的意义是：

- 它只需要坐标 `x`，输出一个可行 tour `π0`（或在代码里表现为按 tour 排好的 `seed`）。
- 它不携带任何“后续必须使用”的隐状态，所以同样天然可被其它 iter 替换/接管。

#### 2.2.2 Revisions：用 SHPP revisers 反复改进 tour

核心思想：
- 把完整 tour 随机切成长度为 `n` 的多个子段（subtour），每段对应一个固定端点的 **SHPP（open-loop TSP）**；
- 用一个训练好的 reviser（AR 模型）去重构每段内部节点顺序；
- 若该段变好则替换，否则保留原段；
- 拼回完整 tour；重复多次（revision iterations），并可用多个 `revision_lens` 从不同尺度改进（如 100/50/20）。

代码对应（这是 GLOP “iter”的核心实现）：
- 总入口：`papers/GLOP/utils/functions.py:reconnect()`
  - 对每个 `revision_len` 调用一次 `LCP_TSP(...)`
  - 可选 `no_prune`：第一轮后把 width 个候选中明显差的丢掉（`min` over width）
- 分解与滑动切点：`papers/GLOP/utils/functions.py:decomposition()`
  - 通过 `seed = torch.cat([seed[:, shift_len:], seed[:, :shift_len]], 1)` 改变分解起点（相当于随机/周期性 shift）
  - 之后 reshape 成 `(-1, revision_len, 2)` 得到子段 batch
- 坐标变换与增强：`papers/GLOP/utils/functions.py:coordinate_transformation()` + `revision()`
  - min-max 归一化到 (0,1) 并视情况交换坐标轴（让输入更同质）
  - 4 种翻转增强（可关闭 `--no_aug`）
- 用 reviser 求解并做“只接受改进”：`papers/GLOP/utils/functions.py:revision()`
  - `reviser(...)` 会输出两种方向的解（forward/backward），挑更优者
  - `reduced_cost = init_cost - cost_revised`
  - 若 `reduced_cost < 0`（变差）则回退到原子段（`sub_tour = identity`）

补充关键原理点（这决定了它为什么能作为“独立 iter”去组合）：

**(a) 子段的目标函数就是“端点固定的 open path”**

GLOP 在 revision 中使用的是 **路径长度（不闭环）**：

- `papers/GLOP/problems/tsp/problem_tsp.py:TSP.get_costs(..., return_local=True)` 返回 `Σ ||v_{t+1}-v_t||`（没有最后回到起点）
- `papers/GLOP/problems/local/problem_local.py:LOCAL.get_costs()` 也是 path length

这对应论文里 SHPP 的定义：固定端点的 Hamiltonian path。

**(b) 端点固定在代码里是“硬约束”**

local reviser 的 decoding 通过 mask 强制端点：

- `papers/GLOP/nets/attention_local.py:_get_log_p()`（你可以直接看 mask 的逻辑）
  - forward：第 1 步只允许选 node 0；最后一步只允许选 node (n-1)
  - backward：反过来（先选 n-1，最后选 0）

因此 reviser 输出的子段排列天然保持端点不变，这保证了把子段拼回全局 tour 时不会破坏“段与段之间的连接关系”。

**(c) 为什么“子段更优”就意味着“全局 tour 更优/不劣”**

在 decomposition 的语境下，全局 tour 可以看成：

`... -> a | [segment nodes] | b -> ...`

其中 `a` 与 `b` 是 segment 的两端节点，它们与外部的连接边保持不变。若 segment 内部的 open path 长度下降了 `Δ < 0`，则全局闭环 tour 的长度也下降同样的 `Δ`（外部边不变），所以接受准则可以只看子段代价。

这解释了为什么 `revision()` 里的 “只接受改进” 可以保证单调不劣，并且该 iter 算子对初始 tour 的来源没有依赖。

这使得 GLOP 的迭代算子具备一个非常关键的性质：

- **对任意初始 tour 都定义良好**
- **单调不劣（每个子段“变差就不接受”）**

这两个性质正是“可组合性”讨论的核心前提。

### 2.3 Local policy（Reviser）如何训练

论文第 4.3 节：reviser 是一个针对 SHPP-n 的 AR 策略，目标是最小化路径长度，训练用 REINFORCE，并把 forward/backward 两个方向都用于训练与 baseline 估计（降低方差）。

代码对应（训练部分在 `papers/GLOP/local_construction/`）：
- 训练入口：`papers/GLOP/local_construction/train.py`
  - `train_batch()` 中对 `cost/log_likelihood` 用 REINFORCE：`((cost - baseline) * log_likelihood).mean()`
  - 模型会输出两条方向的解（代码里是 `cost`/`cost2`）
- 模型结构：`papers/GLOP/nets/attention_local.py`（Attention Model 变体）
  - forward 返回 `cost, pi, cost2, flipped(pi2)` 用于两方向解

论文还提到一个 “two-stage curriculum learning”（让训练输入与推理中的变换/上游输出更一致），代码层面在 local_construction 训练设置中体现（此处不展开每个超参）。

### 2.4 General routing solver（CVRP/PCTSP）：global partition heatmap

论文第 5 节：在 CVRP/PCTSP 等需要“先分组/选子集”的问题上，引入 global policy：

- 用 GNN 输出一个 **partition heatmap** `H_phi`，表示节点对“同属一组”的相对概率；
- 通过约束 mask 的顺序采样生成 partition；
- 训练目标是“分出来的子问题交给 sub-TSP solver 后，总长度最小”，仍然用 REINFORCE。

代码对应：
- GNN 模型：`papers/GLOP/nets/partition_net.py`
- CVRP heatmap：`papers/GLOP/heatmap/cvrp/*`
  - `papers/GLOP/heatmap/cvrp/infer.py:load_partitioner()` 加载 partitioner
- PCTSP heatmap：`papers/GLOP/heatmap/pctsp/*`

### 2.5 GLOP 的“工程化可扩展性”来源

从实现上看，GLOP 能扩展到超大规模的关键是：

- 最重的 AR 模型只在 **固定小尺度（revision_len）** 的子段上跑；
- 大问题通过“切段 + 并行 batch”转化成大量小 SHPP；
- 迭代次数/尺度/width 都是显式可调的预算旋钮。

这也意味着：它的 iter 阶段本质上是一个“可插拔的局部改进器”，而不是必须绑定某个 learned initializer。

---

## 3. 从论文原理分析：LEHD-init + GLOP-iter / GLOP-init + LEHD-iter 是否可行？

下面把问题精确化为：两种组合是否能在“算法定义上成立”，以及“为何可能有效/何处可能失效”。

### 3.0 先把两者都写成 init/iter 的“接口”

**LEHD（TSP）**

- `I_LEHD(x)`：`papers/LEHD/TSP/TSPTester.py:_test_one_batch()` 中 greedy 构造得到 `best_select_node_list`（一个排列）
- `F_LEHD(x, π; budget)`：RRC 循环（随机抽子段 + 重构 + 接受），budget 即 `RRC_budget`

**GLOP（TSP）**

- `I_GLOP(x)`：`papers/GLOP/main.py:_eval_dataset()` 中 Random Insertion 产生 `pi_batch`（排列）；代码里常立即转成 `seed`（按排列 gather 出来的坐标序列）
- `F_GLOP(x, π; params)`：`reconnect()`/`LCP_TSP()`/`revision()` 反复对子段做 SHPP 重构；params 即 `revision_lens`/`revision_iters`/`width`/`no_aug` 等

只要我们能把两者的输出都统一成同一种“tour 表示”，组合就成立。

### 3.0.1 工程对齐：tour 用 indices 还是 coords？

在 `papers/GLOP/` 原始实现里，TSP tour 往往直接表示为 **按 tour 顺序排列的坐标序列 `seed`**（不显式保留节点 id）。
而 `papers/LEHD/` 更偏向保留 **节点 index 的排列**。

这不影响原理可行性，但会影响你“如何把 A 的 init 喂给 B 的 iter”：

- indices → coords：最简单，`seed = coords.gather(1, pi[...,None].repeat(...))`（GLOP 已经这么做了）。
- coords → indices：需要维护一个“坐标到节点 id”的映射（注意浮点误差/重复点），更建议在工程里始终保留 indices。

你的 EasyNCO 实验代码（`EasyNCO/neural_solvers/methods/glop/*`）基本已经把 GLOP 的迭代阶段改成了“indices 版本”，因此组合更直接。

### 3.1 必要条件（TSP 场景）

对 TSP，`init` 与 `iter` 只需要在以下层面兼容即可：

1. **解表示兼容**：都是一个长度为 `n` 的节点排列（Hamiltonian cycle 的某个起点展开），无重复、无遗漏。
2. **iter 的算子定义域**：iter 能接受“任意可行 tour”，而不是只对某类特殊 tour 有定义（例如依赖初始化时的隐藏状态）。
3. **目标一致**：都在优化同一个目标（欧氏 tour length 或同等度量）。

从代码与论文描述看：

- **LEHD 的 RRC**：仅依赖“当前 tour + 坐标”，随机抽连续子路径并重构，且“变差不接受”（`papers/LEHD/TSP/TSPTester.py:decide_whether_to_repair_solution()`）。
- **GLOP 的 revision**：仅依赖“当前 tour 顺序下的坐标序列”，切段、重构、变差回退（`papers/GLOP/utils/functions.py:revision()`）。

因此在“定义域/接口”层面，二者都满足“可插拔 iter”的必要条件。

### 3.2 LEHD-init + GLOP-iter：可行性分析

**为什么原理上可行：**

- GLOP 的 iter 阶段本质是 `seed(coords ordered by tour)` 上的局部重构算子，不关心 seed 是 RI 得到的还是别的方法得到的；
- 且每次 revision 子段若变差直接回退，理论上不会“把你一个好的 init 弄坏”（单调不劣性质）。

**可能有效的原因（为什么会出现“组合 win 的实例”）：**

- LEHD 的 greedy init 可能更擅长捕获某些“全局结构/长程关系”，给出一个不错的整体骨架；
- GLOP 的 iter 是多尺度（例如 100/50/20）+ 多次 shift 的局部重构，更像强力的局部搜索算子；
- 两者优化的邻域结构不同：LEHD 的 RRC 是“随机长度子路径重构”，GLOP 是“固定长度段 SHPP 最优化 + 多尺度叠加”，存在互补空间。

**潜在风险（分布偏移）：**

- GLOP 的 reviser 是用其训练分布/坐标变换/上游输出形成的 SHPP 进行训练的；若 LEHD-init 产生的 tour 在局部结构统计上与 RI seeds 不同，reviser 的改进幅度可能下降。
- 但因为“变差不接受”，最坏情况通常是“改不动/改得少”，而不是系统性变差。

### 3.3 GLOP-init + LEHD-iter：可行性分析

**为什么原理上可行：**

- LEHD 的 RRC 只需要一个当前 tour 来抽子路径，并对该子路径做“固定端点的重构”；这对任意 tour 都成立。
- 接受准则也是“变好才替换”，保证单调不劣。

**可能有效的原因：**

- RI 生成的初始 tour 质量通常一般，但它提供了一个合法解；
- LEHD 的模型（在论文设定下）学到的是“最优 tour 的连续子段如何构造”，RRC 相当于把这些学到的局部模式反复施加到当前解的随机子段上；
- 因此在某些实例上，LEHD 的局部重构可能能显著修补 RI tour 的坏段。

**潜在风险（训练信号的“子段分布”）：**

- LEHD 的监督训练子段来自（近）最优 tour；而 RI tour 的子段可能离最优较远，导致“模型重构出的子段未必更优”的概率上升。
- 但依然由于“只接受更优”，不会累积负面影响；更多表现为“RRC 需要更多预算才显著变好”或“改进不稳定”。

### 3.4 更进一步：两种 iter 的“邻域”互补性（为什么组合会 win）

你在 TSPLIB61 的日志里看到“有些实例组合更好，有些实例反而更差”的现象，用“邻域搜索”的语言很好解释：

- **GLOP-iter 的邻域**：把 tour 切成固定长度 `k` 的块（可多尺度），对每块做“端点固定的全排列重构”（由 learned reviser 近似求解），再滚动切点重复。
  - 这更像一种“结构化、固定尺度”的大邻域操作（并且可以多尺度叠加）。
- **LEHD-RRC 的邻域**：随机长度 `w ~ Unif([4,n])` 的连续子段重构，长度分布更宽、采样更随机。
  - 更像“随机化、可变尺度”的大邻域操作。

两者覆盖的“可达重排模式”并不相同：某些实例可能更受益于固定尺度的强力重排（GLOP），某些实例可能更受益于可变尺度的随机修补（LEHD）。

从理论上讲，把两个 iter 串起来（或让 gate 去选）相当于使用 **邻域的并集**，这正是 solver selection / pipeline composition 能带来收益的经典理由。

### 3.5 一个务实结论：TSP 上“能组合”，且组合的最坏情况通常可控

基于以上原理与代码实现，可以给一个更强的结论：

- **能组合（算法定义成立）**：两者 iter 都只依赖 `(instance, current tour)`，且端点固定/代价可比。
- **最坏情况可控**：两者都采用“只接受改进”的策略，分布偏移导致的最坏情况通常是“改不动/收益变小”，而不是累积性变差。
- **是否更好取决于实例**：这正是你要证明的“组合有效性/偏好性”——不同实例偏好不同 init/iter 组件。

> 注：上面的“单调不劣”讨论依赖于子段代价与全局目标的一致性（或常数差异）。若你在 TSPLIB 上用的是特殊边权（如 `CEIL_2D`）但 solver 用欧氏距离，gap 会异常，这属于**评测指标不一致**而非组合本身的问题（见第 4 节）。

### 3.6 结论：从原理上“分开两个阶段”是可行的

综合两篇论文的表述与代码实现，可以给出一个清晰结论：

- **可行性（定义层面）**：两种组合都成立，因为两者的 iter 阶段都只依赖 `instance + current solution`，且使用了“变差不接受”的单调改进策略。
- **有效性（性能层面）**：是否在某个实例上更优取决于“初始解分布”与“局部改进算子”的匹配程度；组合能赢的直观来源是两种局部重构邻域的互补。

这与你想证明的主张（“组合是有效的：不同实例偏好不同 init/iter 组件”）在理论上是一致的。

---

## 4. CVRP 场景的额外注意事项（如果你要把组合推广到 CVRPLIB）

TSP 的组合主要是“tour permutation”的对接；CVRP 会更复杂一些：

- LEHD 的 CVRP 解表示包含“via depot 的二值标记”（论文 Appendix D，代码 `papers/LEHD/CVRP/VRPModel.py`），属于更强约束的序列表示。
- GLOP 的 CVRP 解通常是“先 partition 再把每个子 route 变成 sub-TSP 求解”，表示为多个子序列拼接（代码 `papers/GLOP/problems/cvrp.py` + `heatmap/cvrp/*`）。

因此若做 `init/iter` 交叉组合，需要先明确“iter 的局部算子”作用在：

- 单条 route 内的 TSP 子序列？
- 还是跨 route 的节点移动/重新分配（这会涉及容量约束，远不止 TSP 的排列）？

在不改动算法本体的情况下，一个更稳妥的落地方式是：

- 先仅在 **“把 CVRP 解拆成若干子 TSP”** 的表示上做局部改进（保持 route 分配不变，只优化每条 route 内顺序）。
- 如果要做跨 route 的改进，需要额外的可行性维护与代价评估，这已经超出两篇论文当前的主要算子定义。

---

## 5. 面向你当前实验的“可写进论文的论证点”

如果你的目标是论证“init/iter 可分离并且组合有效”，结合两篇论文可以写出更有说服力的理论支撑：

- **模块化/可组合的证据来自算法定义本身**：LEHD 的 RRC 与 GLOP 的 revision 都是对“当前解”的局部重构算子，天然可插拔。
- **两者都具备单调改进机制**：变差回退/不接受，从而组合的风险主要是“改不动”，而非系统性恶化。
- **邻域结构不同 → 互补性**：GLOP 是固定尺度 SHPP 重构 + 多尺度叠加；LEHD 是随机长度子段重构（RRC）。不同邻域对不同实例结构可能更有效 → 你在 TSPLIB61 看到的实例级 win/loss 现象有合理解释。

如果你想进一步强化论证，下一步可考虑：

- 在实例级别统计“组合相对纯方法的提升/退化”并按 `n`/分布特征分桶（你已经在 notebooks 里做了类似分析）。
- 从“局部片段长度/尺度偏好”的角度解释：例如某些实例更受益于 `revision_len=100` 的结构性重排，某些实例更受益于 RRC 的随机长度修补。
