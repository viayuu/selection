# 2cmab vs 2cmab2 实现对比

本文档详细比较 `2cmab`（我的实现）和 `2cmab2`（参考实现）的 Neural-LinUCB 实现差异，并分析为什么 `2cmab` 的训练速度慢、GPU 利用率低。

---

## 1. φ(x) vs φ(x, a)：最核心的设计差异

这是两套实现最根本的不同，后续所有差异几乎都由此衍生。

### 2cmab：φ 只取决于实例

```python
# neural_linucb.py, _build_context + ContextProj
context = cat([encoder(coords), scale, one_hot])   # [1, 259]
phi = context_proj(context)                         # [1, 64]
```

φ(x) 是一个 **纯实例表示**，和 arm 无关。不同 arm 的区分完全由 LinUCB 的 per-arm 统计量（Z_k, b_k）来做。

**影响**：

- 需要维护 12 组独立的 Z_k, b_k（每组 64×64 + 64）
- reward_heads 也是 12 个独立的 Linear(64→1)
- 选臂时，用同一个 φ 对 12 个臂分别算 UCB

### 2cmab2：φ 同时编码实例和动作

```python
# neural_linucb.py, _encode_actions
base = context_encoder.encode_samples([sample])     # [1, 270]
arm_emb = arm_embedding(arm_tensor)                 # [1, 16]
phi = phi_net(cat([base, arm_emb], dim=1))          # [1, 64]
```

φ(x, a) 把 arm 信息通过 arm_embedding（16维可学习嵌入）拼入 phi_net 的输入，让网络学到"这个实例和这个方法组合起来的适配程度"。

**影响**：

- 只需维护 1 组全局共享的 A_inv, b, theta
- 没有 reward_heads，训练时直接用 `pred = φ(x,a)ᵀ θ_at_pull`
- 选臂时，对每个可行 arm 各算一次 φ(x, a_k)，再用同一个 theta 打分

### 对比表

| | 2cmab | 2cmab2 |
|---|---|---|
| φ 的输入 | 只有实例 x | 实例 x + 动作 a |
| φ 的维度来源 | 259 → 128 → 64 | (270 + 16) = 286 → 128 → 64 |
| arm 区分方式 | 12 组独立 Z_k, b_k | 1 组共享 A_inv, b + arm_embedding |
| 预测公式 | `w_k^T φ(x)` | `θ^T φ(x, a)` |
| reward_heads | 12 个 Linear(64→1, no bias) | 无 |

---

## 2. 上下文特征的差异

### 2cmab：259 维

```python
# _build_context()
context = cat([graph_emb,    # 256, DualEncoder 输出
               scale,         # 1,   节点规模 (100)
               one_hot])      # 2,   问题类型
# 总计 259
```

没有手工统计特征。

### 2cmab2：270 维

```python
# HybridContextEncoder.encode_samples()
context = cat([graph_emb,    # 257, DualNSSEncoder 输出 (256图嵌入 + 1 scale)
               manual_raw,    # 11,  手工特征 (9维几何 + 2维demand统计)
               onehot])       # 2,   问题类型
# 总计 270
```

多了 11 维手工统计特征（距离标准差、质心坐标、半径、聚类系数等），而且 scale 是在 graph encoder 内部拼接的（所以 graph 部分是 257 而不是 256）。

### 对比表

| 特征来源 | 2cmab | 2cmab2 |
|----------|-------|--------|
| 图编码 | 256 | 256 |
| scale | 1 (外部拼) | 1 (encoder内部拼) |
| 手工几何特征 | 无 | 9 |
| demand 统计 | 无 | 2 |
| 问题 one-hot | 2 | 2 |
| **总维度** | **259** | **270** |

---

## 3. LinUCB 统计量的维护方式

### 2cmab：per-arm 独立统计量，直接矩阵求逆

```python
# _reset_linucb_stats()
self.Z = [np.eye(64) * reg for _ in range(12)]      # 12 组 64×64
self.b_vec = [np.zeros(64) for _ in range(12)]       # 12 组 64
self.Z_inv = [np.eye(64) / reg for _ in range(12)]   # 12 组 64×64 (缓存)

# update() 中
self.Z[arm] += np.outer(phi_np, phi_np)              # Z_k += φφᵀ
self.b_vec[arm] += reward * phi_np                    # b_k += rφ
self._z_dirty[arm] = True                             # 标记需要重算逆

# select() 中，lazy 求逆
if self._z_dirty[k]:
    self.Z_inv[k] = np.linalg.inv(self.Z[k])         # 直接求逆 O(d³)
```

- 维护 12 组独立的统计量
- Z_inv 用 lazy 策略：update 时只标记 dirty，select 时需要用到才求逆
- 求逆方式：`np.linalg.inv(Z_k)`，O(d³)

### 2cmab2：全局共享统计量，Sherman-Morrison 增量更新

```python
# __init__()
self.A_inv = np.eye(64) / reg     # 1 组 64×64
self.b = np.zeros(64)             # 1 组 64
self.theta = np.zeros(64)         # 1 组 64

# update() 中，Sherman-Morrison 公式增量更新 A_inv
phi = self._encode_actions(sample, [arm])[0].cpu().numpy()
a_inv_phi = self.A_inv @ phi                              # O(d²)
denom = 1.0 + float(phi @ a_inv_phi)                      # O(d)
self.A_inv = self.A_inv - np.outer(a_inv_phi, a_inv_phi) / denom  # O(d²)
self.b = self.b + float(reward) * phi
self.theta = self.A_inv @ self.b                           # O(d²)
```

- 只维护 1 组全局统计量（因为 φ(x,a) 已经编码了 arm 信息）
- 不存储 Z，直接维护 A_inv（用 Sherman-Morrison 公式增量更新，避免 O(d³) 的矩阵求逆）
- 每步都更新 theta = A_inv @ b

### Sherman-Morrison 公式

$$ A_{new}^{-1} = A^{-1} - \frac{A^{-1}\phi\phi^T A^{-1}}{1 + \phi^T A^{-1}\phi} $$

这个公式让 A_inv 的更新只需 O(d²) 而不是 O(d³)，数值上也更稳定。

### 对比表

| | 2cmab | 2cmab2 |
|---|---|---|
| 统计量组数 | 12 组 (per-arm) | 1 组 (全局) |
| 存储的量 | Z_k, b_k, Z_inv_k | A_inv, b, theta |
| 逆矩阵更新 | `np.linalg.inv(Z)`, lazy, O(d³) | Sherman-Morrison 增量, O(d²) |
| 内存占用 | 12 × (64×64 + 64×64 + 64) | 1 × (64×64 + 64 + 64) |
| theta 计算 | select 时算 `Z_inv @ b` | update 时直接维护 |

---

## 4. 训练目标的差异

### 2cmab：用 reward_head 预测

```python
# _fit_representation()
phi = self.context_proj(context)          # φ(x), 有梯度
pred = self.reward_heads[arm](phi)        # w_a^T φ(x)
loss = (pred - reward) ** 2               # MSE
```

loss = $(w_{a_s}^T \phi(x_s) - r_s)^2$

其中 $w_{a_s}$ 是 `reward_heads[a_s]` 的可学习权重。每条样本只通过被选中臂的 head。

### 2cmab2：用 θ_at_pull 快照预测

```python
# fit_representation()
phi = self._encode_record_batch(batch_records)        # φ(x, a), 有梯度
theta_tensor = torch.as_tensor(np.stack([theta_at_pull for ...]))
pred = torch.sum(phi * theta_tensor, dim=1)           # φ(x,a)^T θ_at_pull
loss = torch.mean((pred - target) ** 2)               # MSE
```

loss = $\frac{1}{B}\sum (φ(x_s, a_s)^T \theta_{at\_pull} - r_s)^2$

其中 $\theta_{at\_pull}$ 是 **做出选择那一刻的 theta 快照**（在 update 时保存的），是固定的常量，不参与梯度计算。

### 关键区别

| | 2cmab | 2cmab2 |
|---|---|---|
| 预测公式 | $w_a^T \phi(x)$ | $\theta_{snapshot}^T \phi(x, a)$ |
| 梯度来源 | $w_a$ 和 φ 都有梯度 | 只有 φ 有梯度，θ 是常量 |
| 什么在学 | reward_heads 学"哪个臂好" + encoder 学特征 | encoder 和 phi_net 学"让 φ 适配线性头" |
| 保存的历史 | (coords, arm, reward) | (sample, arm, reward, **theta_at_pull**) |

2cmab2 保存 theta_at_pull 的原因：让表示层学到的 φ 能被**当时的线性参数**正确使用，这比用一个独立的可学习 head 更直接地对齐了 LinUCB 的实际使用方式。

---

## 5. 训练实现：逐样本 vs 真 batch

### 这是 2cmab 训练慢的根本原因

### 2cmab：逐样本前向 + 梯度累积

```python
# _fit_representation()
for i, idx in enumerate(indices):                    # Python 循环
    coords_t = self._to_tensor(coords)               # [1, 100, 2/3] → GPU
    mask_t = torch.zeros(1, coords.shape[0])          # [1, 100] → GPU
    context = self._build_context(coords_t, mask_t, ptype, scale)  # encoder 前向 [1,259]
    phi = self.context_proj(context)                   # [1, 64]
    pred = self.reward_heads[arm](phi).squeeze()       # 标量
    loss = (pred - target) ** 2 / bs
    loss.backward()                                    # 反向传播
    if (i + 1) % bs == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
```

每条样本：
1. Python 创建 tensor + CPU→GPU 传输
2. Encoder 前向（多层 Transformer），batch_size=1
3. ContextProj 前向
4. reward_head 前向
5. loss.backward()

5000 条历史 × 10 epoch = **50000 次 encoder 前向 + 反向**，每次 batch_size=1。

### 2cmab2：真正的 mini-batch

```python
# fit_representation()
for start in range(0, len(history_records), actual_bs):
    batch_records = [history_records[int(idx)] for idx in batch_indices]

    phi = self._encode_record_batch(batch_records)     # 内部批量编码！

    # _encode_record_batch 内部：
    #   base = context_encoder.encode_samples(samples)  # [B, 270], 一次 encoder 前向
    #   arm_emb = arm_embedding(arm_tensor)             # [B, 16]
    #   phi = phi_net(cat(base, arm_emb))               # [B, 64]

    theta_tensor = torch.as_tensor(np.stack([...]))    # [B, 64]
    target = torch.as_tensor([...])                    # [B]
    pred = torch.sum(phi * theta_tensor, dim=1)        # [B], 向量化
    loss = torch.mean((pred - target) ** 2)            # 标量
    loss.backward()                                    # 一次反向
    self.optimizer.step()
```

**关键**：`encode_samples()` 内部按问题类型分组后，同类型实例拼成一个 batch tensor 一次过 encoder：

```python
# HybridContextEncoder.encode_samples()
for problem in ("tsp", "cvrp"):
    idxs = [idx for idx, s in enumerate(samples) if s.problem == problem]
    batch_nodes = torch.stack([samples[idx].nodes for idx in idxs])  # [B_type, 100, 2/3]
    encoded = self.graph_encoder.encode_problem(batch_nodes, problem)  # 一次前向 [B_type, 257]
```

1000 条 / batch_size=32 × 10 epoch = **约 312 次 encoder 前向 + 反向**，每次 batch_size ≈ 32。

### 效率对比

| | 2cmab | 2cmab2 |
|---|---|---|
| encoder 输入形状 | `[1, 100, 128]` | `[32, 100, 128]` |
| 注意力矩阵形状 | `[1, 8, 100, 100]` | `[32, 8, 100, 100]` |
| encoder 调用次数 (1000条×10epoch) | 10000 | ~312 |
| optimizer.step() 次数 | 10000/64 ≈ 156 | ~312 |
| 前向+反向对数 | 10000 (每次B=1) | ~312 (每次B=32) |
| Python 循环开销 | 10000 次 tensor 创建 + GPU 传输 | 312 次 |

---

## 6. 为什么 GPU 显存和占用低

### 直接原因：batch_size=1

GPU 的计算单元（CUDA cores）设计为并行处理大量数据。当 batch_size=1 时：

1. **显存低**：Encoder 的中间激活只有 `[1, 100, 128]` 而不是 `[32, 100, 128]`，显存占用约为正常的 1/32
2. **GPU 利用率低**：多头注意力的矩阵乘法只有 `[1, 8, 100, 100]`，数千个 CUDA cores 大部分闲置
3. **Kernel launch 开销占比高**：每次 CUDA kernel 启动有固定开销（~10μs），batch=1 时计算量太小，启动开销反而成为瓶颈
4. **CPU-GPU 数据传输频繁**：每条样本都要 `_to_tensor(coords)` 创建新 tensor 并传到 GPU

### 时间花在哪

实际运行时，绝大部分时间不是在 GPU 计算，而是在：

1. **Python 循环**：50000 次 Python 级别的 for 循环迭代
2. **Tensor 创建**：每次 `torch.tensor(coords)` + `.unsqueeze(0)` + `.to(device)`
3. **Kernel launch**：50000 次小 kernel 的启动/同步开销
4. **CPU-GPU 同步**：每次 `.backward()` 后的隐式同步

本质上，2cmab 的训练瓶颈在 CPU 端的调度开销，而不是 GPU 端的计算。

---

## 7. 历史 buffer 管理方式

### 2cmab：单一 list

```python
self.history = []   # 一个 list 存所有记录

# 训练时截取
if len(self.history) > self.max_history:
    train_history = self.history[-self.max_history:]   # 只取最近 H 条训练

# rebuild 时用全部
for record in self.history:    # 遍历全部历史
    ...
```

一个 history 同时服务于训练和 rebuild，通过不同的截取策略区分。

### 2cmab2：两个独立 deque

```python
self.representation_history = deque(maxlen=rep_maxlen)   # 默认 1000
self.linear_head_history = deque(maxlen=linear_maxlen)   # 可独立配置

# update 时同时写入两个 buffer
self.representation_history.append(record)
self.linear_head_history.append(record)
```

两个 buffer 可以独立配置大小：

- `representation_history`：给网络训练用，窗口较小（默认 1000），保证训练快
- `linear_head_history`：给线性头 rebuild 用，窗口可以更大，保证统计量质量

使用 Python `deque(maxlen=...)` 自动丢弃最旧的记录，不需要手动截取。

### 对比

| | 2cmab | 2cmab2 |
|---|---|---|
| buffer 数量 | 1 个 list | 2 个 deque |
| 训练数据范围 | `history[-max_history:]` | `representation_history` (固定 maxlen) |
| rebuild 数据范围 | `history` (全部) | `linear_head_history` (独立 maxlen) |
| 自动截断 | 手动切片 | deque 自动 |
| 内存管理 | history 无限增长 | deque 固定上限 |

2cmab 的 history 是普通 list，即使训练只用最近 5000 条，全部历史仍然保留在内存中（因为 rebuild 需要全部）。当 t=14000 时，内存中存了 14000 条完整的坐标数组。

---

## 8. rebuild 线性头的差异

### 2cmab：全量重算 + np.linalg.inv

```python
# _rebuild_stats()
self._reset_linucb_stats()                     # Z_k = λI, b_k = 0
for coords, attn_mask, ptype, scale, arm, reward in self.history:   # 全部历史
    phi_np = self._extract_phi(...)             # 无梯度前向，B=1
    self.Z[arm] += np.outer(phi_np, phi_np)     # 累加外积
    self.b_vec[arm] += reward * phi_np
    self._z_dirty[arm] = True                   # select 时再算逆
```

rebuild 后，Z_inv 不是立刻算的，而是等 select 时 lazy 求逆。

### 2cmab2：增量 Sherman-Morrison

```python
# rebuild_linear_head_from_history()
self.A_inv = np.eye(self.hidden_dim) / self.reg   # 重置
self.b = np.zeros(self.hidden_dim)
for sample, arm, reward, _ in history_records:     # linear_head_history
    phi = self._encode_actions(sample, [arm])[0].cpu().numpy()
    a_inv_phi = self.A_inv @ phi
    denom = 1.0 + float(phi @ a_inv_phi)
    self.A_inv -= np.outer(a_inv_phi, a_inv_phi) / denom
    self.b += float(reward) * phi
self.theta = self.A_inv @ self.b
```

rebuild 后，A_inv 和 theta 都是最新的，可以直接用。

### 对比

| | 2cmab | 2cmab2 |
|---|---|---|
| 遍历范围 | 全部历史 (可能 14000 条) | linear_head_history (有上限) |
| 更新方式 | 累加 Z，select 时求逆 | Sherman-Morrison 逐条更新 A_inv |
| rebuild 后状态 | Z_inv 标记 dirty | A_inv + theta 直接可用 |

---

## 9. reward 函数的差异

### 2cmab

```python
# reward.py
r = (K + 1 - rank) / K
```

- 最好 rank=1 → r = 1.0
- 最差 rank=K → r = 1/K（TSP: 1/12 ≈ 0.083, CVRP: 1/8 = 0.125）
- 最差方法仍有正 reward

### 2cmab2（默认 mode="linear_zero_one"）

```python
# reward.py
r = 1 - (rank - 1) / (K - 1)
```

- 最好 rank=1 → r = 1.0
- 最差 rank=K → r = 0.0
- 最差方法 reward 为 0

### 对比

| | 2cmab | 2cmab2 (linear_zero_one) |
|---|---|---|
| 公式 | (K+1-rank)/K | 1-(rank-1)/(K-1) |
| 最好 | 1.0 | 1.0 |
| 最差 | 1/K > 0 | 0.0 |
| 区分度 | 较低 (差臂也有正 reward) | 较高 (差臂 reward 为 0) |
| K=1 时 | r=1.0 | r=1.0 (特殊处理) |

---

## 10. 选臂时的差异

### 2cmab：一次 encoder 前向 + 12 次 numpy 运算

```python
# select()
phi_np = self._extract_phi(coords_t, mask_t, problem_type, scale)  # 1 次 encoder

for k in range(self.n_arms):
    if action_mask[k] < 0.5: continue
    theta = self.Z_inv[k] @ self.b_vec[k]               # numpy 矩阵乘
    mu = float(theta @ phi_np)                            # numpy 内积
    bonus = self.alpha * np.sqrt(float(phi_np @ self.Z_inv[k] @ phi_np))
    ucb = mu + bonus
```

同一个 φ(x) 对所有臂分别计算。

### 2cmab2：1 次 encoder + K 次 phi_net 前向

```python
# decision_details()
phi = self._encode_actions(sample, feasible_arms)    # 1 次 encoder + K 次 phi_net

# 内部：
# base = context_encoder.encode_samples([sample])    # [1, 270], 1 次 encoder
# base.expand(K, -1)                                 # [K, 270], 复制 K 份
# arm_emb = arm_embedding(arm_tensor)                # [K, 16], K 个 arm 的 embedding
# phi = phi_net(cat(base, arm_emb))                  # [K, 64], K 次 phi_net

for idx, arm in enumerate(feasible_arms):
    mean = float(self.theta @ phi_np[idx])
    bonus = np.sqrt(float(phi_np[idx] @ self.A_inv @ phi_np[idx]))
```

每个 arm 得到不同的 φ(x, a_k)，因为 arm_embedding 不同。

### 2cmab2 还有 warm-start 机制

```python
# decision_details()
if self.initial_pulls > 0:
    undersampled = [arm for arm in feasible_arms if self.arm_counts[arm] < self.initial_pulls]
    if undersampled:
        return min(undersampled, key=lambda a: self.arm_counts[a])   # 优先选欠采样臂
```

2cmab 没有这个机制，纯靠 UCB bonus 驱动早期探索。

---

## 11. 优化器与学习率

### 2cmab：分组学习率

```python
self.optimizer = optim.Adam([
    {"params": self.encoder.parameters(),       "lr": 1e-4},   # encoder 小学习率
    {"params": self.context_proj.parameters(),  "lr": 1e-3},
    {"params": self.reward_heads.parameters(),  "lr": 1e-3},
])
```

encoder 用较小学习率防过拟合。

### 2cmab2：统一学习率

```python
self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
```

所有参数（encoder + arm_embedding + phi_net）用同一个学习率 1e-3。

---

## 12. 训练调度参数对比

| 参数 | 2cmab 默认值 | 2cmab2 默认值 | 说明 |
|------|-------------|--------------|------|
| train_every | 50 | 20 | 2cmab2 更新更频繁 |
| train_steps / representation_steps | 10 | 10 | epoch 数相同 |
| batch_size | 64 (梯度累积) | 32 (真 batch) | 2cmab 的 64 是伪 batch |
| max_history / representation_buffer_size | 5000 | 1000 | 2cmab2 窗口更小 |
| linear_head_buffer_size | 同上 (共享) | 独立配置 | 2cmab2 可分开 |
| initial_pulls | 无 | 1 | 2cmab2 有冷启动保护 |

---

## 13. 完整差异总结表

| 维度 | 2cmab | 2cmab2 |
|------|-------|--------|
| **φ 含义** | φ(x) 纯实例表示 | φ(x, a) 实例+动作联合表示 |
| **上下文维度** | 259 (无手工特征) | 270 (含 11 维手工特征) |
| **arm 区分** | 12 组 per-arm 统计量 + 12 个 head | arm_embedding 16 维 + 全局共享统计量 |
| **训练 batch** | 逐样本前向 + 梯度累积 (GPU batch=1) | 真 mini-batch (GPU batch=32) |
| **训练目标** | `(w_a^T φ(x) - r)²` | `(θ_snapshot^T φ(x,a) - r)²` |
| **A/Z 逆矩阵** | 直接 `np.linalg.inv`, lazy | Sherman-Morrison 增量 |
| **历史 buffer** | 1 个 list (无限增长) | 2 个 deque (独立上限) |
| **rebuild 范围** | 全部历史 | linear_head_history |
| **reward 公式** | (K+1-rank)/K | 1-(rank-1)/(K-1) |
| **学习率** | 分组 (encoder 1e-4, 其他 1e-3) | 统一 1e-3 |
| **冷启动** | 纯 UCB bonus | initial_pulls 保护 |
| **早停** | 无 | loss ≤ 1e-4 时早停 |
| **GPU 利用率** | 极低 (batch=1) | 正常 (batch=32) |
| **训练速度** | 慢 (50000 次 B=1 前向) | 快 (~312 次 B=32 前向) |

---

## 14. 对照原论文的合理性分析

下面逐一分析两套实现中的关键设计，对照原论文（Xu et al., ICLR 2022, "[Neural Contextual Bandits with Deep Representation and Shallow Exploration](https://arxiv.org/abs/2012.01780)"）判断哪个更合理。

### 14.1 per-arm 统计量 vs 全局共享统计量

**原论文的做法**：Algorithm 1 中只维护 **一个共享的** $A_t \in \mathbb{R}^{d \times d}$，不是 per-arm 的。原文明确写道：

> $A_t = \lambda I + \sum_{s=1}^{t-1} \phi(x_{s,a_s}; w_{t-1}) \phi(x_{s,a_s}; w_{t-1})^\top$

这里的 $\phi(x_{s,a_s}; w)$ 是网络最后一层隐藏层的输出，它接受的是 **context-action pair 的原始特征**，即 arm 信息已经编码在输入 $x_{t,a}$ 中。所以不需要 per-arm 矩阵——不同 arm 通过不同的输入特征 $x_{t,a}$ 自然得到不同的 $\phi$。

**2cmab 的做法**：维护 12 组独立的 $Z_k, b_k$（per-arm disjoint 模式）。这实际上对应的是 [Li et al., 2010](https://arxiv.org/abs/1003.0146) 的 **LinUCB Disjoint** 模型，而不是 Neural-LinUCB。

**2cmab2 的做法**：维护 1 组共享的 $A_{inv}, b, \theta$，通过 arm_embedding 将 arm 信息编码进 $\phi(x, a)$。这更接近原论文。

**结论**：

- **2cmab 的 per-arm 设计偏离了原论文**。原论文是 joint model + shared matrix。
- **2cmab2 更接近原论文**，但用 arm_embedding 替代了原论文中的"arm 信息编码在原始特征中"的做法，这是一个合理的工程适配——因为在 solver selection 场景中，arm 没有天然的原始特征，需要用可学习嵌入来表示。
- 2cmab 的 per-arm 设计不一定"错"，但它本质上是 **LinUCB Disjoint + 神经网络特征提取**，不是严格意义上的 Neural-LinUCB。它的缺点是：12 组独立统计量无法跨 arm 共享信息，样本效率较低。

### 14.2 φ(x) vs φ(x, a)：是否应该编码 arm

**原论文的做法**：$\phi(x_{t,a}; w)$ 的输入是 context-action pair 的 raw feature。在论文的实验中（classification-to-bandit 转换），每个 arm 对应一个类别，context 是原始特征 $x$，通过 zero-padding 将 $x$ 扩展为 $x^{(a)} = (0, ..., x, ..., 0) \in \mathbb{R}^{Kd}$ 来区分不同 arm。这说明 **原论文的 φ 确实编码了 arm 信息**。

**2cmab**：φ(x) 只编码实例，不含 arm 信息。arm 的区分完全由 per-arm 的线性统计量和 reward_heads 来做。

**2cmab2**：φ(x, a) 通过 arm_embedding 编码 arm，和实例特征拼接后一起过 phi_net。

**结论**：

- **2cmab2 更符合原论文精神**。原论文的 φ 就是 arm-conditioned 的。
- 2cmab 把"arm 区分"完全推给了线性层（12 个独立的 Z_k + reward_heads），这增加了参数量但减少了 arm 间的信息共享。
- 在 solver selection 场景中，arm_embedding 是合理的替代——求解器没有天然的向量特征，可学习嵌入让模型自动学习"方法间的相似性"。

### 14.3 训练目标：reward_heads vs θ_at_pull

**原论文的做法**：Algorithm 1 中网络训练的损失函数是：

$$L(w) = \sum_{s=1}^{t} (f(x_{s,a_s}; \theta) - r_s)^2 + m \|\theta\|^2$$

其中 $f(x; \theta) = \phi(x; w)^\top \mu$ 是完整网络的输出，$\theta = (w, \mu)$ 包含隐藏层参数 $w$ 和最后一层参数 $\mu$。也就是说，**最后一层的线性参数 $\mu$ 是网络的一部分，参与梯度训练**。

**2cmab**：用 `reward_heads[a]`（每臂一个可学习的 Linear(64→1)）做预测，梯度回传到 encoder + context_proj + reward_head。这 **和原论文一致**——reward_head 就是最后一层 $\mu$，参与训练。

**2cmab2**：用 `θ_at_pull`（做出选择那一刻的 theta 快照，固定常量）做预测，只有 φ 有梯度。这 **和原论文不同**——原论文中最后一层参数也参与优化，而 2cmab2 中 theta 是冻结的快照。

**结论**：

- **2cmab 的训练目标更接近原论文**。reward_heads 就是可学习的最后一层。
- 2cmab2 用 θ_at_pull 快照的做法是一种工程变体。它的好处是：让表示层的训练目标和 LinUCB 的实际使用方式更直接对齐（"学一个 φ 使得当时的 theta 能正确预测 reward"）。但它和原论文的损失函数不同。
- 两种做法各有道理，但严格按论文来说，2cmab 在这一点上更忠实。

### 14.4 训练频率与方式

**原论文的做法**：Algorithm 1 中，网络每 $H$ 步更新一次（Line 9-10）。训练时对 $L(w)$ 做 $J$ 步梯度下降。这里 $H$ 和 $J$ 是超参数。

**2cmab**：`train_every=50`（每 50 步触发），`train_steps=10`（10 个 epoch），但因为是逐样本前向，实际效率很低。

**2cmab2**：`train_every=20`（每 20 步触发），`representation_steps=10`（10 个 epoch），真 mini-batch 训练。

**结论**：
- 两者都实现了"周期性重训"这一核心设计，和原论文一致。
- 2cmab2 更新更频繁（q=20 vs q=50），理论上能更快适应新数据。
- 2cmab 的逐样本训练不影响算法正确性，但 **严重影响训练效率**。

### 14.5 统计量重建

**原论文的做法**：Algorithm 1 的 Line 11 明确写道，网络更新后要重建 $A_t$：

> $A_t \leftarrow \lambda I + \sum_{s=1}^{t-1} \phi(x_{s,a_s}; w_t) \phi(x_{s,a_s}; w_t)^\top$

也就是说，用 **全部历史** + **最新的网络** 重建 $A_t$。

**2cmab**：`_rebuild_stats()` 遍历 `self.history`（全部历史），用新网络重算所有 φ，重建 Z_k 和 b_k。**和原论文一致。**

**2cmab2**：`rebuild_linear_head_from_history()` 遍历 `linear_head_history`（deque，有上限）。如果历史超过 buffer 大小，早期记录会被丢弃。**和原论文有偏差**——原论文要求用全部历史。

**结论**：
- **2cmab 在 rebuild 上更忠实于原论文**（用全部历史）。
- 2cmab2 用有限 buffer 是工程折中——完整 rebuild 太慢时可以牺牲统计量精度换速度。
- 但 2cmab 的问题是：全部历史 rebuild 也是逐样本前向（batch=1），t=14000 时非常慢。

### 14.6 综合评估

| 设计维度 | 原论文 | 2cmab | 2cmab2 | 更接近原论文的是 |
|---------|--------|-------|--------|----------------|
| 统计量模式 | 全局共享 A_t | per-arm Z_k (**disjoint**) | 全局共享 A_inv | **2cmab2** |
| φ 是否含 arm | 是 (context-action) | 否 (只有 context) | 是 (arm_embedding) | **2cmab2** |
| 训练目标 | 网络最后一层参与训练 | reward_heads 参与训练 | θ 快照不参与训练 | **2cmab** |
| rebuild 范围 | 全部历史 | 全部历史 | 有限 buffer | **2cmab** |
| 周期性重训 | 每 H 步，J 步梯度下降 | 每 50 步，10 epoch | 每 20 步，10 epoch | 都符合 |
| batch 训练 | 论文未指定 | 逐样本 (B=1) | mini-batch (B=32) | 不涉及 |

**总结**：两套实现各有取舍。2cmab2 在核心架构（shared matrix + action-conditioned φ）上更接近原论文；2cmab 在训练目标和 rebuild 完整性上更接近原论文。但 2cmab 最大的问题不是算法设计，而是 **工程实现上的逐样本训练导致 GPU 效率极低**。

---

## 参考文献

- Xu, P., Wen, Z., Zhao, H., & Gu, Q. (2022). [Neural Contextual Bandits with Deep Representation and Shallow Exploration](https://arxiv.org/abs/2012.01780). ICLR 2022.
- Zhou, D., Li, L., & Gu, Q. (2020). [Neural Contextual Bandits with UCB-based Exploration](http://proceedings.mlr.press/v119/zhou20a/zhou20a.pdf). ICML 2020.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146). WWW 2010.
