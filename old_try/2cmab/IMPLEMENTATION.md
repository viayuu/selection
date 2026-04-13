# 2cmab 实现讲解：模型设计、架构与训练流程

这份文档的目标不是讲所有代码细节，而是帮助我把当前 `2cmab` 的实现讲清楚，尤其是：

- 我到底在解决什么问题
- 数据是怎么组织成 bandit 任务的
- 模型是怎么设计的
- `train` 到底是怎么跑的
- 为什么这样设计

当前实现的主线是：

- 问题范围：`TSP + CVRP`
- 任务形式：`initialization-only` 方法选择
- 学习范式：`shared contextual bandit`
- 主模型：`Online Neural-LinUCB`
- 附加基线：`LinUCB`、`random`、`single-best-global`、`single-best-per-problem`、`oracle`

---

## 1. 任务定义

当前任务不是"生成解"，也不是"每一步迭代都选算子"，而是更前面的一个决策问题：

- 输入：一个组合优化实例（TSP 或 CVRP 的节点坐标）
- 动作：在当前问题可用的初始化方法中选一个
- 输出：被选中的方法
- 反馈：只能观察到被选中方法的 cost（不知道其他方法表现如何）
- 目标：尽量选到在该实例上表现最好的方法

所以这个任务本质上是一个 **instance-level solver selection**，不用监督分类来做，而是用 **contextual bandit** 的方式来做。

更具体地说：

- 对每个实例，系统只做一次决策
- 决策后只拿到被选中方法的 reward 来更新
- 不涉及多步状态转移，所以它不是一般的 MDP 强化学习
- 它是"单步决策"，也就是 contextual bandit

---

## 2. 动作空间是怎么设计的

### 2.1 统一共享动作空间

当前动作空间不是把 TSP 和 CVRP 分开做两套 bandit，而是用一个共享动作空间：

- 总共 `12` 个 arm
- `TSP` 可选全部 12 个
- `CVRP` 可选前 8 个

具体定义在 `arm_config.py`：

```python
ALL_METHODS = [
    "lehd", "elg", "invit", "icam",     # 0-3: TSP + CVRP 共有
    "dact", "lih", "udc", "omni",       # 4-7: TSP + CVRP 共有
    "pointerformer",                      # 8:   TSP only
    "difusco", "t2t", "glop",           # 9-11: TSP only
]

TSP_MASK  = [1,1,1,1, 1,1,1,1, 1, 1,1,1]   # 12 个全可选
CVRP_MASK = [1,1,1,1, 1,1,1,1, 0, 0,0,0]   # 前 8 个可选
```

### 2.2 为什么用共享 arm

- 同名方法在两个问题上共享"方法身份"，例如 `lehd` 在 TSP 和 CVRP 中映射到同一个 arm id
- 共有方法可以跨问题类型共享经验
- 通过 `action_mask` 保证动作合法性

所以当前系统是：

> 共享模型 + 共享 arm 空间 + 按问题类型过滤非法动作

---

## 3. reward 是怎么定义的

### 3.1 为什么不用原始 cost 直接训练

原始 cost 有几个问题：

1. 不同问题量纲不同（TSP cost ~8，CVRP cost ~16）
2. 不同实例难度不同
3. bandit 真正关心的是"谁更好"，而不是绝对数值大小

所以当前实现把 cost 转成了 **实例内排名 reward**。

### 3.2 当前 reward 公式

定义在 `reward.py` 的 `rank_reward` 函数：

$$r = \frac{K + 1 - rank}{K}$$

其中：
- `K` = 当前实例可用方法数（TSP: 12，CVRP: 8）
- `rank` = 该方法在所有可用方法中的排名（1 = cost 最低 = 最好）

因此：
- 最好方法 rank=1 → reward = K/K = 1.0
- 最差方法 rank=K → reward = 1/K

例如 CVRP 实例（K=8），某方法排名第 2 → r = (8+1-2)/8 = 0.875。

### 3.3 ties 怎么处理

如果多个方法 cost 一样，代码用 `scipy.stats.rankdata(method='average')`，即 **平均名次**。

例如：costs = `[7.5, 7.5, 8.0]`，排名会变成 `[1.5, 1.5, 3]`，不会随便打散。

---

## 4. 为什么需要 Neural-LinUCB

在讲模型之前，先理解一下背景。

### 4.1 最简单的情况：线性 LinUCB

假设 reward 和 context 之间是线性关系。对每个臂 k，假设存在未知的 $\theta_k^*$：

$$E[r \mid x, k] = {\theta_k^*}^\top x$$

给定历史数据，用岭回归估计 $\theta_k$：

$$\hat\theta_k = Z_k^{-1} b_k$$

其中：
- $Z_k = \lambda I + \sum_{s: a_s = k} x_s x_s^\top$ — 所有选过臂 k 的特征外积累加
- $b_k = \sum_{s: a_s = k} r_s \cdot x_s$ — reward 加权特征累加
- $\lambda$ 是正则化系数

然后用 UCB 公式选臂：

$$UCB_k = \hat\theta_k^\top x + \alpha \sqrt{x^\top Z_k^{-1} x}$$

第一项是 exploitation（预测 reward），第二项是 exploration（不确定性）。

### 4.2 LinUCB 的局限

LinUCB 假设 reward 是 context 的 **线性函数**。但在当前场景中：

- context 是原始坐标 `[100, 2]`，维度极高
- 不同求解器在不同图结构上的表现是高度非线性的

线性假设太弱了。

### 4.3 Neural-LinUCB 的核心思想

Neural-LinUCB 的做法是：**用神经网络把原始 context 映射到一个低维特征空间，在这个特征空间上做线性 UCB**。

$$\phi = g(x; \theta) \in \mathbb{R}^{64}$$

然后假设：

$$E[r \mid x, k] = {\theta_k^*}^\top g(x; \theta)$$

非线性由神经网络 $g$ 负责，线性部分由 LinUCB 负责。

但这带来一个问题：$\phi = g(x; \theta)$ 依赖于网络参数 $\theta$，如果 $\theta$ 更新了，$\phi$ 就变了，之前累积的 $Z_k$ 和 $b_k$ 就全部失效。所以 **每次网络更新后必须重建统计量**。

---

## 5. 上下文特征是怎么设计的

当前模型的上下文不是手工特征，而是用图编码器从原始坐标中学习。

### 5.1 为什么需要 DualEncoder

有一个关键问题：

- TSP 输入是 `[x, y]`，2 维
- CVRP 输入是 `[x, y, demand]`，3 维

一个 encoder 的第一层 embedding 维度不同，不能直接统一处理两类问题。

所以当前实现在 `encoder.py` 中采用了 `DualEncoder`：

- TSP 用一个 `Encoder_h`（输入层 `Linear(2, 128)`）
- CVRP 用另一个 `Encoder_h`（输入层 `Linear(3, 128)`）

根据 `problem_type` 选择走哪一路：

```python
def forward(self, coords, mask, problem_type):
    if problem_type.lower() == "tsp":
        return self.tsp_encoder(coords, mask)      # → [B, 256]
    else:
        return self.cvrp_encoder(coords, mask)      # → [B, 256]
```

两个 encoder 输出都是 256 维，后续流程统一。

### 5.2 Encoder_h 的内部结构

`Encoder_h` 来自 NSS 论文，是一个层次注意力编码器，定义在 `nss_encoder.py`。

它的结构是：

```
输入: [B, 100, 2/3]  (100 个节点的坐标)
  ↓
Embedding: Linear(2→128) 或 Linear(3→128)
  ↓ [B, 100, 128]
  ↓
Block_1:
  │  2 层 EncoderLayer (多头自注意力 + FFN + ReZero)
  │  下采样: 学习打分 → top-80% 节点保留
  ↓ [B, 80, 128]
  ↓
Block_2:
  │  2 层 EncoderLayer + 下采样
  ↓ [B, 64, 128]
  ↓
最终池化: GELU(cat(masked_mean, masked_max))
  ↓ [B, 256]
```

关键设计点：

- **层次下采样**：每个 Block 用一个额外的注意力层给节点打分，选 top-k（k = 当前节点数 × 0.8），逐步压缩图规模 100 → 80 → 64
- **双池化**：对保留节点同时取 mean 和 max，拼接得到 `2 × 128 = 256` 维
- **ReZero 归一化**：残差连接的缩放系数初始化为 0，训练初期等价于恒等映射

每个 `EncoderLayer` 的参数是：8 头注意力，`qkv_dim=16`，FFN 隐藏层 512。

### 5.3 Context 拼接

Encoder 输出 256 维图嵌入后，在 `_build_context` 中拼接 3 个额外特征：

```python
context = cat([graph_emb,    # [B, 256]  图结构嵌入
               scale,         # [B, 1]    节点规模 (100)
               one_hot])      # [B, 2]    问题类型 ([1,0]=TSP, [0,1]=CVRP)
# context: [B, 259]
```

所以模型看到的上下文是 259 维。

---

## 6. 主模型 Neural-LinUCB 的设计

这是当前最重要的部分。

### 6.1 总体思想

Neural-LinUCB 的核心思想可以概括成两句话：

1. 用神经网络学表示
2. 用线性 UCB 做探索

也就是论文里说的：

- deep representation
- shallow exploration

### 6.2 网络结构

当前实现的完整结构在 `neural_linucb.py` 中：

```
实例坐标 [B, 100, 2/3]
    ↓
DualEncoder (TSP/CVRP 各一个 Encoder_h)
    ↓ [B, 256]
    ↓
cat(graph_emb, scale, one_hot)
    ↓ [B, 259]
    ↓
ContextProj: Linear(259→128) → GELU → Linear(128→64) → GELU
    ↓ [B, 64]  ← 这就是 φ(x)
    ↓
    ├── (训练时) reward_heads[k]: Linear(64→1, no bias) × 12
    │   预测 reward，提供梯度信号
    │
    └── (选臂时) LinUCB
        UCB_k = θ̂_kᵀ φ + α√(φᵀ Z_k⁻¹ φ)
```

### 6.3 为什么当前不是 action-conditioned

当前实现 **不是** 把 arm 编码后和实例特征拼在一起。而是：

- 先编码实例得到 φ(x) ∈ R⁶⁴
- 每个臂维护自己独立的 Z_k、b_k 统计量
- 选臂时用同一个 φ(x) 对所有可行臂分别算 UCB

也就是说：φ 只取决于实例，不取决于 arm。不同 arm 的区分完全由 LinUCB 的线性部分（per-arm 的 Z_k 和 b_k）来做。

### 6.4 每个臂维护什么

对每个臂 k，维护的统计量在 `_reset_linucb_stats` 中初始化：

```python
Z[k]     = λI           # 64×64 精度矩阵，初始化为 λ 倍单位矩阵
b_vec[k] = 0            # 64 维零向量
Z_inv[k] = I/λ          # Z 的逆，缓存起来避免重复计算
_z_dirty[k] = False     # Z 是否被更新过（lazy 求逆标记）
```

`Z_inv` 不是每次 update 都重算的。代码里用了 lazy 策略：update 时只设 `_z_dirty[k] = True`，等下次 select 需要用到 `Z_inv[k]` 时才真正调 `np.linalg.inv`。

### 6.5 reward_heads 是什么，为什么需要

`reward_heads` 是 12 个线性层（每个臂一个），定义在 `__init__` 里：

```python
self.reward_heads = nn.ModuleList([
    nn.Linear(phi_dim, 1, bias=False) for _ in range(n_arms)
])
```

它的作用 **不是选臂**（选臂靠 LinUCB），而是 **给网络训练提供梯度信号**。

具体来说，网络训练时的 loss 是：

$$L = (w_{a_s}^\top \phi(x_s) - r_s)^2$$

其中 $w_{a_s}$ 就是 `reward_heads[a_s]` 的权重。这个 loss 的梯度会回传到 φ → ContextProj → Encoder，驱动整个网络学到"对预测 reward 有用的特征"。

**关键**：`reward_heads` 是持久组件，不是每次训练都重建的。它和 encoder、context_proj 放在同一个 optimizer 里，在上一轮参数基础上继续优化。

### 6.6 打分公式

对一个可行臂 k，`select` 方法计算：

```python
theta = self.Z_inv[k] @ self.b_vec[k]                             # 岭回归系数
mu = float(theta @ phi_np)                                         # exploitation
bonus = self.alpha * np.sqrt(float(phi_np @ self.Z_inv[k] @ phi_np))  # exploration
ucb = mu + bonus                                                   # 总分
```

选 UCB 最大的可行臂：

```python
if ucb > best_ucb:
    best_ucb, best_arm = ucb, k
```

---

## 7. 两类学习对象，为什么要分开处理

Neural-LinUCB 实际上同时在学两件事，它们的更新频率和方式完全不同。

### 7.1 线性头（Z_k、b_k）

- **更新频率**：每一步都更新
- **更新方式**：直接累加，无需梯度
- **计算量**：O(d²) = O(4096)，很快

```python
# update() 中，每次都执行：
self.Z[arm] += np.outer(phi_np, phi_np)    # 64×64
self.b_vec[arm] += reward * phi_np         # 64
```

### 7.2 表示层（encoder + context_proj + reward_heads）

- **更新频率**：每 `train_every`（默认 50）步才更新一次
- **更新方式**：用历史数据做梯度下降
- **计算量**：重，需要对历史样本做前向 + 反向传播

### 7.3 为什么不每步都训网络

表示层训练很贵（一次 encoder 前向就是多层 Transformer），而且每步只新增一条样本，频繁训练意义不大。所以当前实现故意把它们分开：

- 线性头：每步都更新，保持即时响应
- 表示层：攒够 q 步后批量训一次，保证效率

这是一个非常重要的工程折中，也是论文原始设计的一部分（Algorithm 1 中的参数 q）。

---

## 8. train 是怎么实现的

### 8.1 先强调：这是离线数据上的 online-style bandit 训练

当前训练并不是完全真实的 online interaction，而是：

> 用离线完整结果数据，模拟 online contextual bandit 的决策与更新过程

也就是说：

- 离线其实知道所有方法在每个实例上的 cost
- 但训练时仍然只让 bandit "看到自己选中的那个 arm 的 reward"
- 没选中的 arm 的 cost 不参与训练

这样做的目的，是保留 bandit 的学习方式，而不是退化成监督标签学习。

### 8.2 run_offline.py 的主流程

训练主入口在 `run_offline.py` 的 `main()` 中，对于 neural_linucb 方法：

1. 加载 cost 矩阵 + 实例坐标
2. 70/15/15 分层划分 train/val/test
3. 构造 `NeuralLinUCB` selector
4. 在 train 上做在线 bandit 训练
5. 在 test 上做 greedy 评测（网络不再更新）
6. 与 baselines 对比
7. 保存结果 + checkpoints

### 8.3 train split 上的一轮训练怎么跑

代码在 `run_offline.py` 第 396-426 行。对训练集中的每个实例：

```python
for step, i in enumerate(order):           # order 是打乱后的 train_idx
    ptype = all_types[i]
    c = get_coords(i)
    arm = nl.select(c, None, ptype, 100, all_masks[i])     # UCB 选臂
    reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])  # 只看选中臂的 cost
    nl.update(arm, c, None, ptype, 100, reward)             # 更新
```

注意这里的关键点：

- `select` 时用 UCB 决策（有探索）
- `rank_reward` 只计算被选中 arm 的排名奖励（bandit feedback，不是全量监督）
- `update` 时只更新被选中 arm 的 Z_k 和 b_k

### 8.4 选臂之后到底更新了什么

每次 `update(arm, coords, ...)` 调用时，`neural_linucb.py` 第 179-217 行做了三件事：

#### 第一件事：立刻更新线性头统计量

```python
phi_np = self._extract_phi(coords_t, mask_t, problem_type, scale)  # 无梯度
self.Z[arm] += np.outer(phi_np, phi_np)
self.b_vec[arm] += reward * phi_np
self._z_dirty[arm] = True
```

这一步 **每个样本都做一次**，O(d²) 很快。

#### 第二件事：把记录写入 history

```python
self.history.append((coords, attn_mask, problem_type, scale, arm, reward))
```

这个 history 后面会被用来训练表示层和重建统计量。

#### 第三件事：判断是否触发网络重训

```python
if self.t % self.train_every == 0 and len(self.history) > 0:
    self._fit_representation()    # 训练网络
    self._rebuild_stats()         # 重建 Z/b
```

每累计 `train_every`（默认 50）次后，触发一次网络训练 + 统计量重建。

### 8.5 表示层是怎么训练的

这是 `_fit_representation()` 方法，在 `neural_linucb.py` 第 231-285 行。

#### 8.5.1 训练目标

训练的目标是让网络能预测 reward。对于历史样本 s（选了臂 $a_s$，观察到 reward $r_s$）：

$$L = \sum_{s} (w_{a_s}^\top \phi(x_s) - r_s)^2$$

其中：
- $\phi(x_s) = \text{ContextProj}(\text{build\_context}(\text{Encoder}(x_s)))$，有梯度
- $w_{a_s}$ = `reward_heads[a_s]` 的权重

注意：每条样本只用 **被选中臂** 的 reward_head，其他臂的 head 不参与。但 encoder 和 context_proj 对所有样本都更新。

#### 8.5.2 滑动窗口

```python
train_history = self.history
if len(train_history) > self.max_history:
    train_history = train_history[-self.max_history:]
```

随着 t 增大，历史越来越长。代码只取最近 `max_history`（默认 5000）条训练，保证每次重训耗时恒定。

#### 8.5.3 梯度累积

因为 TSP 输入 `[N,2]` 和 CVRP 输入 `[N,3]` 维度不同，无法拼成一个 batch tensor。所以用梯度累积等价实现 mini-batch：

```python
for step in range(self.train_steps):       # J=10 个 epoch
    np.random.shuffle(indices)
    self.optimizer.zero_grad()

    for i, idx in enumerate(indices):
        # 逐条前向（有梯度）
        context = self._build_context(coords_t, mask_t, ptype, scale)
        phi = self.context_proj(context)
        pred = self.reward_heads[arm](phi).squeeze()
        target = torch.tensor(float(reward), ...)
        loss = (pred - target) ** 2 / bs      # 除以 batch_size
        loss.backward()                        # 梯度累积，不清零

        if (i + 1) % bs == 0:
            self.optimizer.step()              # 每 bs 条做一次参数更新
            self.optimizer.zero_grad()

    # 末尾不足 bs 的剩余
    if n % bs != 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
```

除以 `bs` 是为了：累积 bs 条的梯度 = 平均梯度，与真正的 mini-batch 数学等价。

#### 8.5.4 分组学习率

```python
self.optimizer = optim.Adam([
    {"params": self.encoder.parameters(),       "lr": 1e-4},
    {"params": self.context_proj.parameters(),  "lr": 1e-3},
    {"params": self.reward_heads.parameters(),  "lr": 1e-3},
])
```

encoder 有约 50 万参数（多层 Transformer），而训练数据只有几千条，所以 encoder 用小学习率防过拟合。

#### 8.5.5 持久性

所有网络组件和优化器都是持久的。每次 `_fit_representation` 是在上一轮参数和 Adam 动量（m, v）基础上继续优化，不会重新初始化。

### 8.6 为什么表示层训练后要 rebuild 线性头

这是当前实现中非常关键的一点。

问题在于：

- 线性头原来记录的 Z_k、b_k
- 是基于旧的 φ(x) 空间建立的

但一旦表示层更新了，φ(x) 就变了。

如果不处理，就会出现：

> 表示层已经换了新空间，但线性头还停留在旧空间

这会导致 UCB 计算完全错误——$\hat\theta_k$ 是在旧特征空间拟合的系数，拿去和新特征做内积没有意义。

所以当前实现会在每次表示层训练后调用 `_rebuild_stats()`，在 `neural_linucb.py` 第 287-304 行：

```python
def _rebuild_stats(self):
    self._reset_linucb_stats()   # Z_k = λI, b_k = 0

    for coords, attn_mask, ptype, scale, arm, reward in self.history:
        phi_np = self._extract_phi(...)      # 用新网络算特征，无梯度
        self.Z[arm] += np.outer(phi_np, phi_np)
        self.b_vec[arm] += reward * phi_np
```

这一步会：

1. 清空当前所有臂的统计量
2. 从 **全部历史** 中取样本（不截取，和训练不同）
3. 用 **当前最新的网络** 重新算 φ
4. 重建每个臂的 Z_k 和 b_k

重建后，线性头和表示层就重新对齐了。

为什么 rebuild 用全部历史而不是滑动窗口：因为 LinUCB 的 Z_k 和 b_k 是所有历史样本的累加和，如果只用部分历史，统计量会丢失信息，不确定性估计不准确。rebuild 只做无梯度前向传播，比训练快得多。

---

## 9. train / test 是怎么区分的

### 9.1 train

在 train 阶段（`run_offline.py` 第 396-426 行）：

- 用 UCB 决策（有探索）
- 决策后调用 `update` 更新线性头 + 记录历史
- 每 q 步触发网络重训 + 统计量重建

### 9.2 test

在 test 阶段（`run_offline.py` 第 432-457 行）：

- 仍然用 UCB 公式计算分数（但此时 bonus 很小，几乎纯 exploitation）
- **不再调用 update**，不更新任何东西
- 只记录选择结果和 oracle 对比

---

## 10. 一轮完整交互的时间线

以 `train_every=50` 为例，展示 t=1 到 t=100 发生了什么：

```
t=1:   select (所有臂 mu=0, bonus≈7.8, 选第一个) → update Z_0, b_0
t=2:   select (arm 0 有经验了 bonus 降到 1.0, 未见过的臂 bonus≈7.8) → 选 arm 1
t=3:   select → 选 arm 2  (继续探索未见过的臂)
...
t=12:  select → 选 arm 11 (所有臂至少被拉过一次)
t=13:  select (开始真正比较 UCB 分数) → 选 UCB 最高的
...
t=49:  select → update
t=50:  select → update
       ↓
       触发网络重训！
       (1) _fit_representation():
           从 history 取最近 50 条
           10 个 epoch × 梯度累积
           encoder + context_proj + reward_heads 权重更新
           → φ(x) 变了！
       (2) _rebuild_stats():
           Z_k = λI, b_k = 0 (重置)
           遍历全部 50 条历史
           用新网络算 φ_s^new
           重新累加 Z_{a_s} += φ_s^new (φ_s^new)ᵀ
           → Z/b 现在和新 φ 对齐了
t=51:  select (用更新后的网络 + 重建后的统计量) → update
...
t=100: 再次触发网络重训 + 统计量重建
...
```

### exploration 的演化

```
早期 (t < 12):
  所有臂都没被拉过，Z_k = λI → bonus ≈ 7.8（很大）
  mu = 0（没有数据）
  → 纯探索：依次轮询每个臂

中期 (t ≈ 50):
  每个臂被拉了几次，bonus 降到 0.3 ~ 1.2
  mu 开始分化：好臂 mu ≈ 0.85，差臂 mu ≈ 0.15
  → 探索利用平衡：主要选好臂，偶尔探索不确定的臂

后期 (t > 500):
  好臂被拉了几十次，bonus 极小 (< 0.3)
  mu 估计很准确
  → 几乎纯利用：总是选 mu 最高的臂

网络重训的效果：
  每 50 步重训后，φ 变得更好 → mu 预测更准 → 更快收敛到最优臂
```

---

## 11. 这套实现最值得讲的设计亮点

### 11.1 它不是普通监督分类，而是 bandit 化的方法选择

虽然离线知道所有方法结果，但训练时仍然只用"被选中 arm 的 reward"更新，这一点保留了 contextual bandit 的核心味道。

### 11.2 它不是单问题模型，而是共享的 TSP+CVRP 模型

通过：

- 共享 arm 空间
- 问题 one-hot 编码
- feasible mask
- 双 NSS 编码器

实现了一个统一框架处理两个问题。

### 11.3 它不是"整个网络做 UCB"

而是：

- 神经网络学表示 φ(x)
- 最后一层线性 bandit 做探索

这正是 Neural-LinUCB 这类方法最重要的思想：deep representation + shallow exploration。

### 11.4 线性头每步更新，表示层周期性更新

两类学习对象更新频率不同：

- 线性头（Z_k, b_k）：每步都更新，O(d²)，保持即时响应
- 表示层（网络权重）：每 q 步更新，训练量大，但频率低

这是论文原始设计的核心折中。

### 11.5 表示层训练后必须 rebuild 线性头

φ 变了旧统计量就失效。代码在每次 `_fit_representation` 后立刻调用 `_rebuild_stats`，用新 φ 重新累加全部历史的 Z 和 b。

这是最容易出 bug 也最容易讲出亮点的设计点。

### 11.6 训练用滑动窗口，rebuild 用全部历史

两者的历史范围故意不同：

- 表示层训练：看最近 `max_history` 条（防止训练变慢）
- 统计量重建：看全部历史（保证 LinUCB 统计量完整性）

### 11.7 梯度累积而不是真 batch

TSP/CVRP 输入维度不同无法拼 batch，用梯度累积等价实现，数学上完全等价。

### 11.8 不只追求能跑，还强调可复盘

额外保存了：

- 每次选臂的所有臂 UCB 分数（mu, bonus, ucb）
- 每次选择与 oracle 的对比（cost, reward, match）
- 网络重训的 loss 曲线
- 周期性 checkpoint

所以后续分析空间很大。
