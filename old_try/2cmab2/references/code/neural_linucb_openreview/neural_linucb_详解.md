# Neural-LinUCB 论文与源码详解

## 论文信息

- **标题**: Neural Contextual Bandits with Deep Representation and Shallow Exploration
- **作者**: Pan Xu, Zheng Wen, Handong Zhao, Quanquan Gu
- **会议**: ICLR 2022
- **核心贡献**: 提出 Neural-LinUCB，将神经网络的表示学习能力和 LinUCB 的探索效率结合
- **源码来源**: OpenReview supplementary material（`code/run_demo.py`）

---

## 一、问题背景：Contextual Bandit

### 1.1 标准设定

每一轮 t = 1, 2, ..., T：
1. 环境给出一个**上下文** x_t（如实例特征）
2. 玩家从 K 个**臂**（arm）中选一个 a_t
3. 环境返回**奖励** r_t

目标：最小化**累积遗憾（cumulative regret）**：
```
R_T = Σ_{t=1}^{T} [ r*(x_t) - r_t ]
```
其中 r*(x_t) 是在上下文 x_t 下选最优臂能得到的奖励。

### 1.2 奖励模型假设

论文假设奖励函数 h* 属于一个**再生核希尔伯特空间（RKHS）**，即：
```
E[r_t | x_t, a_t] = h*(x_t, a_t)
```
其中 h* 可能是非线性的。这比 LinUCB（假设线性）更一般。

### 1.3 现有方法的问题

| 方法 | 特征 | 探索 | 问题 |
|------|------|------|------|
| LinUCB | 线性 | UCB | 只能处理线性奖励 |
| NeuralUCB | 全网络 | 全网络 UCB | 计算太慢（需要对所有参数维护置信矩阵） |
| Neural-Linear | 神经网络 | 贝叶斯线性 | 只有启发式，无理论保证 |

---

## 二、Neural-LinUCB 核心思想

### 2.1 一句话概括

> **用深层神经网络学习好的特征表示（deep representation），但只在最后一层做 UCB 探索（shallow exploration）**

### 2.2 为什么这样设计？

直觉：
- **NeuralUCB** 对整个网络的所有参数维护置信矩阵 → 参数量 P 可能上万 → P×P 矩阵太大
- **Neural-LinUCB** 只对最后一层的参数维护置信矩阵 → 最后一层维度 m 通常很小（如 64） → m×m 矩阵很小
- 核心洞察：**探索的不确定性主要来自最后一层**，前面的层只负责特征提取

### 2.3 两阶段算法

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: 深层表示学习 (Deep Representation)         │
│                                                       │
│  每隔 q 轮, 用累积的数据做梯度下降训练前面的网络层:     │
│    输入: context x → 多层ReLU网络 → 特征 φ(x)         │
│    损失: MSE(θ^T φ(x) - r)                           │
│    注意: 只更新前面的层(特征提取), 不更新最后一层 θ     │
│                                                       │
│  目的: 学到好的非线性特征映射 φ(x)                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Stage 2: 浅层 UCB 探索 (Shallow Exploration)        │
│                                                       │
│  在冻结的特征 φ(x) 上, 运行标准 LinUCB:              │
│    θ_t = Λ_t^{-1} b_t                               │
│    UCB(x,a) = θ_t^T φ(x,a) + β √(φ^T Λ_t^{-1} φ)  │
│    选 argmax UCB                                      │
│    更新: Λ_t += φ φ^T, b_t += r·φ                   │
│                                                       │
│  目的: 在已学好的特征空间中做有理论保证的探索           │
└─────────────────────────────────────────────────────┘
```

### 2.4 与标准 LinUCB 的关系

Neural-LinUCB 可以理解为：
```
标准 LinUCB:     手工特征 x → LinUCB 选择
Neural-LinUCB:   原始输入 x → 神经网络 → 学习特征 φ(x) → LinUCB 选择
```

最后一层的 LinUCB 和标准 LinUCB **完全一样**，只是输入的特征换成了神经网络自动学到的。

---

## 三、网络架构详解（对齐源码）

### 3.1 输入表示

源码中，每个 (context, action) 对被转换为一个统一的输入向量：

```python
def TRANS(c, a, arm_size):
    """将 context + action 编码为统一输入"""
    dim = len(c)                    # 上下文维度
    action = np.zeros(arm_size)     # one-hot 编码动作
    action[a] = 1
    c_final = np.append(c, action)  # 拼接: [context, action_onehot]
    c_final = c_final.view((len(c_final), 1))
    c_final = c_final.repeat(2, 1)  # 复制一份: [x; x]（维度 2d）
    return c_final  # 形状: [2*(context_dim + arm_size), 1]
```

**为什么要 repeat(2, 1)？** 这与论文中的理论分析有关。论文使用的是一种特殊的网络初始化（见下文 INI 函数），需要输入是 [x; x] 的形式，以满足 NTK（Neural Tangent Kernel）分析的条件。

### 3.2 网络初始化（INI 函数）

```python
def INI(dim):
    """
    特殊初始化: 确保网络满足理论分析的条件

    dim = [input_dim, hidden1, hidden2, ..., 1]
    对于中间层: W_i 初始化为 kron(I_2, random) (块对角)
    对于最后层: W_L 初始化为 kron([1,-1], random)
    """
    w = []
    total_dim = 0
    for i in range(len(dim) - 1):
        if i < len(dim) - 2:
            # 中间层: Kronecker 积 → 块对角矩阵
            temp = np.random.randn(dim[i+1], dim[i]) / np.sqrt(dim[i+1])
            temp = np.kron(np.eye(2), temp)    # [2*out, 2*in] 块对角
            total_dim += dim[i+1] * dim[i] * 4  # 参数量 ×4 (因为 kron(I_2,...))
        else:
            # 最后层: Kronecker 积 → [1, -1] 模式
            temp = np.random.randn(dim[i+1], dim[i]) / np.sqrt(dim[i])
            temp = np.kron([[1, -1]], temp)    # [out, 2*in]
            total_dim += dim[i+1] * dim[i] * 2  # 参数量 ×2
        w.append(torch.from_numpy(temp))
    return w, total_dim
```

**理论意义**: 这种初始化保证了网络在训练初期的行为接近线性化（NTK regime），使得理论遗憾界成立。具体来说：
- `kron(I_2, W)` 使得网络的正半部分和负半部分对称
- `kron([1,-1], W)` 使得最后一层做差（正-负），消除偏置项

**实际维度示例** (默认配置: hidden=[1000,1000]):
```
输入: [context+action]  dim = context_size + arm_size (如 8+7=15)
repeat(2,1) → 2 * 15 = 30

W0: kron(I_2, random(1000,15)) → [2000, 30]
W1: kron(I_2, random(1000,1000)) → [2000, 2000]
W2: kron([1,-1], random(1,1000)) → [1, 2000]  ← 最后一层
```

### 3.3 前向传播（FUNC_FE）

```python
def FUNC_FE(x, W):
    """
    特征提取器: 多层 ReLU 网络
    x: [2d, 1] 输入
    W: 权重列表 [W0, W1, ..., WL]

    注意: 只做到倒数第二层! 最后一层不在这里。
    最后一层由 θ (LinUCB 的参数) 负责。
    """
    depth = len(W)
    output = x
    for i in range(depth - 1):      # 注意: depth-1, 跳过最后一层
        output = torch.mm(W[i], output)
        output = output.clamp(min=0)  # ReLU 激活
    # 缩放（理论需要）
    output = output * math.sqrt(W[depth-1].size()[1])
    return output  # 形状: [dim_second_last, 1] = [2000, 1]
```

**关键点**: `FUNC_FE` 输出的是**倒数第二层的特征** φ(x)，而不是最终预测值。最终预测值由 `θ^T φ(x)` 给出，其中 θ 是 LinUCB 维护的参数。

### 3.4 完整前向流程图

```
原始输入: context c (如 8维) + action a (如 7个中选1个)
    │
    ▼
TRANS: 拼接 + repeat → [2*(8+7), 1] = [30, 1]
    │
    ▼
W0 (块对角, 2000×30) + ReLU → [2000, 1]
    │
    ▼
W1 (块对角, 2000×2000) + ReLU → [2000, 1]
    │                                          ← FUNC_FE 到这里为止
    ▼                                            输出 φ(x,a) ∈ R^2000
θ^T φ(x,a)  → 标量预测值                       ← LinUCB 的线性层
    │
    ▼
UCB = θ^T φ + β √(φ^T Λ^{-1} φ)              ← UCB 探索项
```

---

## 四、主循环详解（main 函数）

### 4.1 初始化

```python
# 网络维度: [input, hidden1, hidden2, output]
dim_for_init = [context_size + arm_size] + hidden_dims + [1]
# 例如: [15, 1000, 1000, 1]

W0, total_dim = INI(dim_for_init)  # 初始网络权重

# LinUCB 的统计量 (在 dim_second_last = 2000 维上)
LAMBDA = λ * I_{2000}   # 精度矩阵 Λ_0 = λI
bb = 0                   # 奖励累积向量 b_0 = 0
theta = random(2000, 1)  # LinUCB 参数 θ_0
```

### 4.2 每轮选择

```python
for t in range(T):
    context = CONTEXT[t]  # 当前上下文

    # 1. 对每个臂计算 UCB 值
    for a in range(arm_size):
        temp = TRANS(context, a, arm_size)  # 编码 (context, action)
        feat = FUNC_FE(temp, W)              # 提取特征 φ(x,a)
        # UCB = θ^T φ + β √(φ^T Λ^{-1} φ)
        #       ─────   ─────────────────
        #       利用项      探索项
        ucb[a] = θ^T @ feat + β * UCB(LAMBDA, feat)

    # 2. 选臂 (前 3*K 轮用 round-robin 保证每个臂至少选 3 次)
    if t < 3 * arm_size:
        a_choose = t % arm_size      # 轮流选
    else:
        a_choose = argmax(ucb)        # UCB 最大的

    # 3. 获取奖励并累积 regret
    reward = REWARD[t][a_choose]
    regret += max(REWARD[t]) - reward
```

### 4.3 UCB 计算（UCB 函数）

```python
def UCB(A, phi):
    """
    计算 UCB 的探索项: √(φ^T Λ^{-1} φ)

    A: Λ (精度矩阵, dim_second_last × dim_second_last)
    phi: φ(x,a) (特征向量, dim_second_last × 1)

    通过解线性方程组 Λ·tmp = φ 来避免显式求逆:
      tmp = Λ^{-1} φ
      UCB = √(φ^T · tmp) = √(φ^T Λ^{-1} φ)
    """
    tmp = torch.solve(phi, A)  # 解 Λ·tmp = φ → tmp = Λ^{-1}·φ
    return torch.sqrt(phi.T @ tmp)
```

**与 LinUCB 标准公式的对应**:
```
LinUCB:        UCB_a = θ^T x + α √(x^T A^{-1} x)
Neural-LinUCB: UCB_a = θ^T φ(x,a) + β √(φ^T Λ^{-1} φ)
                       ─────────────   ─────────────────
                       完全一样         完全一样，只是 x 换成了 φ(x,a)
```

### 4.4 统计量更新（浅层探索的核心）

```python
# 更新精度矩阵和奖励向量 (标准 LinUCB 更新)
feat = FUNC_FE(bphi[a_choose], W)  # 所选臂的特征
LAMBDA += feat @ feat.T              # Λ_{t+1} = Λ_t + φ φ^T
bb += reward * feat                   # b_{t+1} = b_t + r·φ
theta = solve(LAMBDA, bb)            # θ_{t+1} = Λ_{t+1}^{-1} b_{t+1}
```

这与标准 LinUCB 的更新**完全一致**:
- `A += x x^T`  →  `Λ += φ φ^T`
- `b += r·x`     →  `b += r·φ`
- `θ = A^{-1}b`  →  `θ = Λ^{-1}b`

### 4.5 表示学习（深层表示的更新）

```python
# 每隔 q 轮更新一次网络权重（深层表示学习）
if t % H_q == H_q - 1:
    W = TRAIN_SE(CONTEXT_action, REWARD_action, W0, interT, et, THETA_action, H_q)
```

TRAIN_SE 函数的核心逻辑：

```python
def TRAIN_SE(X, Y, W_start, T, et, THETA, H):
    """
    用梯度下降训练前面的网络层 (Stage 1)

    关键: 只更新 W[0], W[1], ..., W[L-2] (前面的层)
          不更新 W[L-1] (最后一层，由 LinUCB 的 θ 负责)

    X: 累积的 context 数据
    Y: 累积的 reward 数据
    W_start: 初始权重 W0 (每次从 W0 开始训练，不是从上次结果继续)
    THETA: 对应每个样本的 θ 值（因为 θ 在每轮都更新了）
    """
    W = copy.deepcopy(W_start)  # 从初始权重 W0 重新开始！

    for i in range(T):  # T = interT = 1000 步梯度下降
        grad = GRAD_LOSS(X, Y, W, THETA)
        for j in range(len(W) - 1):    # 只更新前面的层
            W[j] = W[j] - et * grad[j]  # SGD
        # 最后一层 W[-1] 不更新!
```

**重要细节**:
1. **从 W0 重新开始**: 每次重新训练都从初始权重 W0 出发，不是从上次结果继续。这是理论分析要求的（保证收敛性）。
2. **只更新前面的层**: 循环 `for j in range(len(W)-1)` 跳过了最后一层。最后一层由 LinUCB 的 θ 负责。
3. **训练目标**: 最小化 `Σ (θ_t^T φ_W(x_t, a_t) - r_t)^2`，其中 θ_t 是 LinUCB 在第 t 轮给出的参数值。

---

## 五、与 NeuralUCB 的关键对比

```
                    NeuralUCB                Neural-LinUCB
                    ─────────                ─────────────
探索对象:          整个网络所有参数           只有最后一层参数
置信矩阵大小:      P × P (P=总参数量)       m × m (m=最后一层维度)
                   P 可能 > 10万             m 通常 = 2000 或 64
计算量:            O(P²) 每轮               O(m²) 每轮
特征更新频率:      每轮都可以                每隔 q 轮批量更新
理论遗憾界:        Õ(√T)                    Õ(√T) (一样!)
实际速度:          statlog 上 19028 秒       statlog 上 783 秒 (快24倍)
```

**为什么遗憾界一样但速度差 24 倍?**
- NeuralUCB 在每轮都要: (1) 计算所有参数的梯度 (2) 更新 P×P 矩阵
- Neural-LinUCB 只在最后一层做 UCB，前面的层每隔 q 轮才更新一次
- 论文证明：只要表示学习足够好，在最后一层做 UCB 就足够了

---

## 六、源码中的关键超参数

| 参数 | 默认值 | 含义 | 对应论文 |
|------|--------|------|---------|
| `lambd` | 1.0 | LinUCB 正则化参数 λ | Λ_0 = λI |
| `beta` | 0.02 | UCB 探索系数 β | UCB = θ^T φ + **β** √(φ^T Λ^{-1} φ) |
| `et` | 0.0001 | 梯度下降学习率 | Stage 1 的 SGD 步长 |
| `H_q` | 100 | 每隔多少轮更新一次网络 | 论文的 q |
| `interT` | 1000 | 每次训练的 SGD 步数 | Stage 1 的内部迭代次数 |
| `hidden_dim` | [1000,1000] | 隐藏层维度 | 网络深度和宽度 |
| `K` | 15000 | 总轮数 | 实验长度 T |

---

## 七、源码流程总结（时序图）

```
t=0 ──────────────────────────────────────────────────────── t=T
│                                                             │
│  初始化: W0, Λ=λI, b=0, θ=random                           │
│                                                             │
│  ┌─── 每轮 t ──────────────────────────────────────────┐   │
│  │ 1. 对每个臂 a: 算 φ(x,a) = FUNC_FE(TRANS(x,a), W) │   │
│  │ 2. 算 UCB: θ^T φ + β √(φ^T Λ^{-1} φ)              │   │
│  │ 3. 选 argmax UCB (前 3K 轮用 round-robin)           │   │
│  │ 4. 观察 reward                                      │   │
│  │ 5. 更新 Λ, b, θ (浅层 LinUCB 更新)                 │   │
│  │ 6. 累积 (x, r, θ) 到 buffer                        │   │
│  │ 7. 如果 t % q == q-1:                               │   │
│  │      TRAIN_SE: 用 buffer 重新训练 W (深层更新)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  输出: 累积 regret 曲线                                     │
```

---

## 八、对我们 2cmab 项目的启示

### 8.1 我们的 Neural-LinUCB 与源码的对应

| 源码 | 我们的实现 |
|------|-----------|
| `TRANS(context, action)` → [2d, 1] | DualEncoder(coords) + cat(scale, one_hot) → [259] |
| `FUNC_FE(x, W)` → [2000, 1] | `context_proj(259→128→64)` → [64] |
| `LAMBDA (2000×2000)` | `Z[k] (64×64)` 每个臂一个 |
| `theta (2000,)` | `θ_k = Z[k]^{-1} b[k]` |
| `UCB(LAMBDA, feat)` | `α √(φ^T Z_k^{-1} φ)` |
| `TRAIN_SE(...W0...)` | `train_representation(Stage 1)` |
| 从 W0 重新训练 | 我们在 Stage 1 训练后冻结 |
| `H_q = 100` (每100轮更新) | 我们用 epoch 训练后一次性冻结 |

### 8.2 我们做的简化

1. **输入表示**: 源码用 `[context; action_onehot]` 然后 repeat(2,1)。我们用 DualEncoder 直接提取图特征，不需要 repeat。
2. **初始化**: 源码用 Kronecker 积的特殊初始化。我们用标准 PyTorch 初始化（因为我们不需要 NTK 理论的严格条件）。
3. **每个臂独立 vs 共享**: 源码对所有臂用同一个 Λ 矩阵。我们对每个臂维护独立的 Z_k。两种做法都合理，独立版更灵活。
4. **网络更新频率**: 源码每 q=100 轮从 W0 重新训练。我们在离线阶段一次性训练完就冻结，更简单也更适合离线 setting。

### 8.3 源码告诉我们的重要设计选择

1. **β (UCB 探索系数) 设得很小**: 默认 0.02，远小于 LinUCB 的典型 α=1.0。因为神经网络特征已经很有信息量，不需要太多探索。
2. **从 W0 重新训练**: 源码每次重新训练都从初始权重 W0 开始，不是继续训练。但在我们的离线场景中不需要这个（一次训练即可）。
3. **Round-robin 热身**: 前 3×K 轮（K=臂数）强制轮流选每个臂至少 3 次，避免初始 LinUCB 因数据不足做出不合理选择。我们也应该这样做。

---

## 九、我们的实现 vs 原论文实现：完整对比

以下对比基于：
- **原论文源码**: `2cmab2/references/code/neural_linucb_openreview/code/run_demo.py`
- **我们的实现**: `2cmab/neural_linucb.py`

### 9.1 设计层面对比

#### (1) 输入表示

```
原论文:
  context (如 8维) + action_onehot (如 7维) = 15维
  → repeat(2, 1) → 30维列向量
  → 所有臂共用一个网络，action 通过 one-hot 区分

我们:
  坐标 [B, N, 2/3]
  → DualEncoder (TSP/CVRP 各一个 Encoder_h) → 256维图嵌入
  → cat(scale(1), problem_onehot(2)) → 259维
  → 所有臂共用特征φ，但每个臂有独立的 reward_head
```

**关键差异**: 原论文把 action 编码进输入（一个统一网络处理所有 (context, action) 对），我们把 action 分离到独立的 reward heads。两种做法都是合法的 Neural-LinUCB 实现：

```
原论文方式:  f(x, a) = θ^T · φ_W(x, a)      ← 一个共享θ，action嵌在φ里
我们的方式:  f(x, a) = θ_a^T · φ_W(x)        ← 每个arm一个θ_a，φ与arm无关
```

原论文方式让不同 arm 共享特征提取器的容量（可能更参数高效），但需要 K 次前向传播。我们的方式只需 1 次前向传播就得到 φ(x)，然后每个 arm 只是一个线性层。

#### (2) 网络架构

```
原论文:
  W0: [2000, 30] ─ReLU→ W1: [2000, 2000] ─ReLU→ (最后层由 θ 负责)
  特殊初始化: Kronecker积 kron(I_2, random) (为NTK理论需要)
  特征维度: dim_second_last = 2000
  激活函数: ReLU

我们:
  DualEncoder: 坐标 → 多层注意力 → 256维 (冻结, 不参与Stage 1训练)
  ContextProj: 259→128→64 (两层MLP)
  特征维度: phi_dim = 64
  激活函数: GELU
```

**影响**: 原论文的特征维度 2000 远大于我们的 64。这意味着：
- 原论文的 Λ 矩阵是 2000×2000（约 30MB），我们的 Z 矩阵是 64×64（约 32KB）
- 我们的版本计算更快，但表达能力依赖 DualEncoder（256维图嵌入）是否足够好
- 如果 DualEncoder 的预训练权重好，64维完全够用

#### (3) Λ/Z 矩阵的管理方式

```
原论文:
  所有臂共享一个 Λ (2000×2000)
  一个共享的 θ (2000,)
  Λ += φ(x,a) · φ(x,a)^T    ← 不同 arm 的 φ 不同 (因为 action 编码进了输入)
  θ = Λ^{-1} b               ← 用 torch.solve 解方程

我们:
  每个臂独立维护 Z[k] (64×64) 和 b_vec[k] (64,)
  θ_k = Z[k]^{-1} · b_vec[k]  ← 用 np.linalg.inv 预计算逆
  Z[k] += φ(x) · φ(x)^T       ← 注意: φ 与 arm 无关, 因为 action 不在输入里
```

**差异解释**: 原论文的 action 编码进了输入，所以不同 arm 的 φ(x, a) 不同，一个 Λ 就能区分。我们的 φ(x) 与 arm 无关，所以必须给每个 arm 独立的 Z_k。这两种方式在数学上等价：

```
原论文: 一个大 Λ 包含所有 arm 的信息
我们:   K 个小 Z_k 分别包含各自 arm 的信息

等价关系:
  原论文的 Λ 可以理解为 block-diagonal 的近似
  只是 off-diagonal blocks 允许 arm 之间共享信息
```

#### (4) 表示学习 (Stage 1) 的训练方式

```
原论文 (TRAIN_SE):
  - 每隔 q=100 轮执行一次
  - 每次从初始权重 W0 重新开始训练!
  - 用 SGD，步长 et=0.0001
  - 训练 interT=1000 步
  - 只更新前面的层 W[0], W[1]，不更新最后层
  - 训练目标: min Σ (θ_t^T φ_W(x_t) - r_t)²
    其中 θ_t 是每个样本对应时刻的 LinUCB 参数（随时间变化的！）

我们 (train_representation):
  - 离线一次性训练完
  - 从随机初始化开始
  - 用 Adam，步长 lr=0.001
  - 训练 n_epochs=50 个 epoch
  - 同时更新 context_proj 和 reward_heads
  - 训练目标: min Σ_k (reward_head_k(φ(x)) - r_k)²
    对每个实例的所有可用 arm 同时拟合 reward
```

**这是最大的设计差异**。详细对比：

| 维度 | 原论文 | 我们 |
|------|--------|------|
| 训练时机 | 在线，每隔q轮 | 离线，一次性 |
| 起始权重 | 每次从W0重新开始 | 从随机初始化连续训练 |
| 优化器 | SGD (手动写梯度) | Adam (PyTorch autograd) |
| 学习率 | 0.0001 (很小) | 0.001 |
| 训练步数 | 1000步/次 | 50 epochs × N样本 |
| θ 值 | 每个样本用对应时刻的θ_t | 统一用最终的reward head |
| 最后层 | 不更新 (由LinUCB管) | 也更新 (reward_heads参与训练) |

**原论文从 W0 重新开始的原因**: 理论分析要求网络参数不要偏离初始化太远（NTK regime）。但在实际实验中，这个约束可以放松，直接连续训练通常效果更好。

**我们同时训练 reward_heads 的原因**: 在离线设定中，我们有所有 arm 的 reward 数据，可以直接做监督学习。这比原论文的纯在线设定信息更丰富。Stage 1 训练完后，reward_heads 被冻结，Stage 2 中 LinUCB 的 θ_k 会覆盖它们。

#### (5) 在线/离线 设定的差异

```
原论文 (纯在线):
  t=1: 来一个实例 → 选臂 → 观察reward → 更新Λ,θ → (每100轮更新W)
  t=2: 来下一个实例 → ...
  没有预先知道所有数据

我们 (离线模拟在线):
  Stage 1: 用全部训练数据训练表示 (知道所有arm的reward)
  Stage 2: 模拟在线过程 (假装不知道reward, 用UCB选臂)
```

**影响**:
- 原论文的 Stage 1 只能用历史数据中**被选中的 arm** 的 reward 训练
- 我们的 Stage 1 可以用**所有 arm** 的 reward 训练（因为离线数据都有）
- 这使得我们的表示学习更高效，但也意味着 UCB 探索在 Stage 2 中的重要性降低

### 9.2 代码层面逐函数对比

#### UCB 计算

```python
# === 原论文 (run_demo.py:230-237) ===
def UCB(A, phi):
    # 解线性方程 Λ·tmp = φ (避免显式求逆)
    tmp, LU = torch.solve(phi, A)
    return torch.sqrt(torch.mm(phi.T, tmp))
    # 所有臂共享一个 Λ

# === 我们 (neural_linucb.py:161-173) ===
# select 方法中:
for k in range(self.n_arms):
    if action_mask[k] < 0.5:
        continue
    if self._z_dirty[k]:
        self.Z_inv[k] = np.linalg.inv(self.Z[k])   # 显式求逆 + 缓存
        self._z_dirty[k] = False
    theta = self.Z_inv[k] @ self.b_vec[k]
    mu = float(theta @ phi_np)
    bonus = self.alpha * np.sqrt(float(phi_np @ self.Z_inv[k] @ phi_np))
    ucb = mu + bonus
    # 每个臂独立的 Z_k
```

**差异**:
- 原论文用 `torch.solve` 避免显式求逆 → 数值更稳定
- 我们用 `np.linalg.inv` 显式求逆 + 脏标记缓存 → 多次 select 之间不用重复求逆
- 原论文一次 solve 同时得到 θ 和 UCB bonus，我们分开计算

#### 统计量更新

```python
# === 原论文 (run_demo.py:309-311) ===
LAMBDA += torch.mm(FUNC_FE(bphi[a_choose], W),
                   FUNC_FE(bphi[a_choose], W).t())  # Λ += φ·φ^T
bb += reward * FUNC_FE(bphi[a_choose], W)            # b += r·φ
theta, LU = torch.solve(bb, LAMBDA)                   # θ = Λ^{-1}·b

# === 我们 (neural_linucb.py:176-185) ===
def update(self, arm, coords, attn_mask, problem_type, scale, reward):
    with torch.no_grad():
        phi = self._extract_phi(coords_t, mask_t, problem_type, scale)
    phi_np = phi.squeeze(0).cpu().numpy()
    self.Z[arm] += np.outer(phi_np, phi_np)    # Z_k += φ·φ^T
    self.b_vec[arm] += reward * phi_np          # b_k += r·φ
    self._z_dirty[arm] = True                   # 标记需要重新求逆
```

**差异**:
- 原论文每次 update 都立即 solve 得到新 θ → 计算量略大但 θ 始终是最新的
- 我们延迟求逆到下次 select 时 → 避免不必要的求逆

#### 特征提取

```python
# === 原论文 (run_demo.py:126-136) ===
def FUNC_FE(x, W):
    depth = len(W)
    output = x                           # x: [30, 1], 包含 context+action
    for i in range(depth - 1):           # 只到倒数第二层
        output = torch.mm(W[i], output)  # 矩阵乘法
        output = output.clamp(min=0)     # ReLU
    output = output * math.sqrt(W[depth-1].size()[1])  # NTK 缩放
    return output                        # [2000, 1]

# === 我们 (neural_linucb.py:79-89) ===
def _extract_phi(self, coords, attn_mask, problem_type, scale_val):
    with torch.no_grad():
        graph_emb = self.encoder(coords, attn_mask, problem_type)  # DualEncoder → [B, 256]
    # 拼接 scale + one-hot
    context = torch.cat([graph_emb, scale_t, is_tsp], dim=-1)     # [B, 259]
    phi = self.context_proj(context)                                # MLP → [B, 64]
    return phi
```

**差异**:
- 原论文的特征提取器是手写的矩阵乘法 + ReLU，输入包含 action one-hot
- 我们用 DualEncoder（注意力机制）提取图嵌入，然后用 MLP 投影到低维
- 原论文输出 2000 维，我们输出 64 维
- 我们的 encoder 是冻结的（不参与 Stage 1 的梯度更新），原论文的前面层 W 参与更新

### 9.3 总结表

| 维度 | 原论文 | 我们 | 评价 |
|------|--------|------|------|
| **输入** | 向量 context + action_onehot | 图坐标 → DualEncoder | 我们更适合图结构数据 |
| **action 编码** | 编入输入 (一个网络) | 独立 reward_heads (K个) | 我们1次前向即可 |
| **特征维度** | 2000 | 64 | 我们更快，依赖encoder质量 |
| **Λ/Z 管理** | 1个共享 Λ | K个独立 Z_k | 原论文: arm间共享; 我们: arm间独立 |
| **Stage 1 训练** | 在线, 每100轮从W0重来 | 离线, 一次性 | 我们更简单, 适合离线setting |
| **Stage 1 监督** | 只用被选arm的reward | 用所有arm的reward | 我们信息更丰富 |
| **初始化** | Kronecker积 (NTK理论) | 标准PyTorch | 原论文更严格但不必要 |
| **优化器** | 手写SGD + 手写反向传播 | Adam + autograd | 我们更现代 |
| **逆矩阵** | torch.solve(不求逆) | np.linalg.inv + 缓存 | 原论文数值更稳定 |
| **UCB 参数** | β=0.02 (很小) | α=1.0 (标准) | 需要在实验中调参 |
| **热身** | round-robin 3×K轮 | 无 | 应该加上 |
| **NTK 缩放** | √m 缩放 | 无 | 原论文为理论需要 |

### 9.4 我们的实现可以改进的地方

基于对比分析，以下是可以考虑的改进：

1. **加 round-robin 热身**: 原论文前 3×K 轮强制轮流选每个臂。我们应该在 Stage 2 开始时对每个臂至少有几次观测。

2. **α 调小**: 原论文用 β=0.02，远小于 LinUCB 典型的 α=1.0。因为神经网络特征信息量更大，不需要那么多探索。建议我们也尝试 α=0.01~0.1。

3. **用 solve 替代 inv**: `np.linalg.solve(Z, phi)` 比 `np.linalg.inv(Z) @ phi` 数值更稳定，尤其是 Z 接近奇异时。

4. **Stage 1 中不更新 reward_heads**: 原论文严格只更新前面的层，最后层完全交给 LinUCB。我们当前 Stage 1 同时训练 reward_heads 和 context_proj。如果想更严格对齐论文，可以只训练 context_proj，让 reward_heads 在 Stage 2 由 LinUCB 的 θ_k 接管。
