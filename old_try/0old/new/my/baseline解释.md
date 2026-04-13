# `baseline=critic` 训练流程详解（Two-Gate Actor-Critic）

本文只解释你当前 `my/train_two_gate_tsp.py` 里 **`--baseline critic`** 时的训练流程；`batch_mean` / `critic_batch_mean` 不展开。

> 适用代码位置（便于你对照阅读）：  
> - 主训练循环：`my/train_two_gate_tsp.py:569`  
> - `baseline=critic` 分支：`my/train_two_gate_tsp.py:613`  
> - gate/value 网络构建：`my/train_two_gate_tsp.py:438`  
> - value MLP 定义：`my/gates.py:29`

---

## 1. 你的问题在代码里被建模成什么 RL 过程？

你现在的训练过程可以视作一个“**两步决策、终止奖励**”的 episodic RL：

- **状态 `s0`**：仅由实例 `data` 决定（TSP 就是坐标 `coords`）。
- **动作 `a1`（Gate1）**：从 initializer zoo 里选一个初始化器（例如 LEHD / POMO / AM / …）。
- 执行动作后得到 **中间结果 `sol0`**（初始 tour 与其长度）。
- **状态 `s1`**：由 `data + sol0` 决定（你的实现里就是 “实例特征 + 初始解条件特征”，例如 `len0` 等）。
- **动作 `a2`（Gate2）**：从 iterator zoo 里选一个迭代器（例如 RRC-LEHD / 2-opt / …）。
- 执行动作后得到最终解 **`sol1`**，并得到终止奖励
  - `reward = -length(sol1)`（你约定的符号）。

注意：你没有在训练求解器本身，求解器是“环境/执行器”的一部分；训练的参数只在 gate/encoder/critic 上。

更形式化一点（对应你当前实现的概率分解）：

- Gate1 的策略：\(\pi_1(a_1\mid s_0;\theta_1)\)
- Gate2 的策略：\(\pi_2(a_2\mid s_1;\theta_2)\)
- 联合策略可以写成：
  \[
  \pi(a_1,a_2\mid s_0)=\pi_1(a_1\mid s_0)\cdot \pi_2(a_2\mid s_1)
  \]
  这里 \(s_1\) 由 \((s_0,a_1)\) 以及 initializer 的执行结果 `sol0` 决定。

在代码里，这个分解对应：
- `dist1 = Categorical(logits=policy.gate1(feat1))`（`my/train_two_gate_tsp.py:574`）
- `dist2 = Categorical(logits=policy.gate2(feat2))`（`my/train_two_gate_tsp.py:582`）
- `logp1/logp2 = dist*.log_prob(a*)`（`my/train_two_gate_tsp.py:576` / `my/train_two_gate_tsp.py:584`）

---

## 2. 变量/张量对照表（代码里这些量分别是什么）

以下名字基本对应 `my/train_two_gate_tsp.py` 训练循环里的变量：

| 变量 | 形状（TSP） | 含义 |
|---|---:|---|
| `coords` | `[B, N, 2]` | 一批 TSP 实例坐标 |
| `feat1` | `[B, d1]` | Gate1/Value1 输入特征（只由实例决定） |
| `logits1` | `[B, K1]` | Gate1 对每个 initializer 的打分（logits） |
| `dist1` | Categorical | `Categorical(logits=logits1)` |
| `a1` | `[B]` | Gate1 采样得到的 initializer id |
| `logp1` | `[B]` | `log π1(a1 | s0)`（每个实例一项） |
| `sol0.tour` | `[B, N]` | 被选 initializer 解出的 tour |
| `sol0.length` | `[B]` | `sol0` 的长度（正数） |
| `feat2` | `[B, d2]` | Gate2/Value2 输入特征（实例 + sol0 条件） |
| `logits2` | `[B, K2]` | Gate2 对每个 iterator 的打分（logits） |
| `a2` | `[B]` | Gate2 采样得到的 iterator id |
| `logp2` | `[B]` | `log π2(a2 | s1)` |
| `sol1.length` | `[B]` | 迭代后最终 tour 长度 |
| `reward` | `[B]` | `-sol1.length` |
| `baseline1` | `[B]` | Critic1 预测的 `V1(s0)` |
| `baseline2` | `[B]` | Critic2 预测的 `V2(s1)` |
| `adv1` | `[B]` | `reward - baseline1` |
| `adv2` | `[B]` | `reward - baseline2` |

其中：
- `B = batch_size`，`N = problem_size`
- `K1 = len(init_zoo)`，`K2 = len(iter_zoo)`

### 2.1 `baseline=critic` 时到底在学什么参数？

在 `baseline=critic` 模式下，训练会更新的参数块可以分成三类（概念上）：

1) **Actor（策略）参数**
   - Gate1：`policy.gate1`（MLP 输出 logits）
   - Gate2：`policy.gate2`（MLP 输出 logits）

2) **Critic（值函数）参数**
   - Critic1：`policy.value1`（MLP 输出标量 baseline1）
   - Critic2：`policy.value2`（MLP 输出标量 baseline2）

3) **Feature extraction（特征提取）参数**
   - `policy.encoder`（默认是论文同款 encoder：`my/features.py:build_paper_tsp_encoder`）

求解器本身（`final/` 里的 policy/iteration）在你的训练里是“环境执行器”，被显式冻结（`requires_grad_(False)` + `no_grad`），不会被更新。

### 2.2 Gate 与 Critic 的“输入/输出”一眼看懂版

- Gate1：`feat1 -> logits1 -> Categorical -> a1, logp1`
- Gate2：`feat2 -> logits2 -> Categorical -> a2, logp2`
- Critic1：`feat1 -> baseline1 = V1(s0)`
- Critic2：`feat2 -> baseline2 = V2(s1)`

这里的 `feat1/feat2` 来自 `my/features.py`：
- `feat1 = tsp_instance_features(coords, ...)`：只看实例（+可选 `scale=N`）
- `feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, ...)`：实例特征 + `sol0` 条件（默认至少包含 `length0`，还可包含 tour 统计量）

---

## 3. 一次训练 step 的完整流程（`baseline=critic`）

下面按 **实际代码的执行顺序**解释一次训练 step（省略日志与 eval）：

### 3.1 采样一批实例（环境输入）

训练时实例来自实时生成的数据流（或你指定的分布生成器）。得到：

- `coords: [B,N,2]`

对应代码：`coords = train_gen.sample(...)`（`my/train_two_gate_tsp.py:570`）。

### 3.2 Gate1（选初始化器）— actor 前向

1) 特征提取（状态 `s0 → feat1`）  
`feat1 = tsp_instance_features(coords, encoder=policy.encoder, ...)`

2) 形成策略分布  
`dist1 = Categorical(logits=policy.gate1(feat1))`

3) 采样动作并得到 logprob  
`a1 = dist1.sample()`  
`logp1 = dist1.log_prob(a1)`

对应代码：`my/train_two_gate_tsp.py:573-576`。

### 3.3 执行初始化器（环境转移）

根据每个样本的 `a1`，把 batch 分发给对应 initializer 求解：

- `sol0 = run_initializers_by_action(coords, a1, init_zoo)`
- 得到 `sol0.tour: [B,N]` 与 `sol0.length: [B]`

这一步是 “环境执行”，你的实现里 solver 基本都在 `no_grad/inference_mode` 下运行，所以这里 **不会有梯度**回传到 solver。

对应代码：`sol0 = run_initializers_by_action(...)`（`my/train_two_gate_tsp.py:578`）。

### 3.4 Gate2（选迭代器）— actor 前向

1) 特征提取（状态 `s1=(data,sol0) → feat2`）  
`feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, encoder=policy.encoder, ...)`

2) 形成策略分布并采样  
`dist2 = Categorical(logits=policy.gate2(feat2))`  
`a2 = dist2.sample()`  
`logp2 = dist2.log_prob(a2)`

对应代码：`my/train_two_gate_tsp.py:581-584`。

### 3.5 执行迭代器并得到 reward（终止）

把 batch 分发给对应 iterator 做固定步数/固定迭代预算的改进：

- `sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)`
- `length = sol1.length`
- `reward = -length`

到这里，一条 episode 结束（终止奖励）。

对应代码：`sol1 = run_iterators_by_action(...)` + `reward=-length`（`my/train_two_gate_tsp.py:586-589`）。

### 3.6 Critic 前向：预测 baseline（`baseline=critic` 的关键）

此时你已经拿到了最终 `reward`，critic 做两次预测：

- `baseline1 = value1(feat1)`（只看 `s0`）
- `baseline2 = value2(feat2)`（看 `s1=data+sol0`）

并计算 critic 回归损失（MSE）：

- `critic_loss1 = (reward - baseline1)^2`
- `critic_loss2 = (reward - baseline2)^2`
- `critic_loss = 0.5 * (critic_loss1 + critic_loss2)`

对应代码：`my/train_two_gate_tsp.py:613-620`。

### 3.7 计算 advantage、总 loss，并更新参数

1) advantage：
- `adv1 = reward - baseline1`
- `adv2 = reward - baseline2`

2) actor loss（注意 `detach()`）：
- `actor_loss = -(logp1 * adv1.detach() + logp2 * adv2.detach()).mean()`

3) 总 loss（actor + critic）：
- `loss = actor_loss + critic_coef * critic_loss`

4) 反向传播与参数更新：
- `optimizer.zero_grad(); loss.backward(); optimizer.step()`

对应代码：`my/train_two_gate_tsp.py:624-632`。

---

## 4. Critic 的数学原理：为什么它不会“改变”梯度的期望，但能降方差？

你在做的目标是最大化期望奖励：

\[
J(\theta)=\mathbb{E}[R]
\]

在 REINFORCE/策略梯度里，有经典恒等式：

\[
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)\; R \right]
\]

引入 baseline \(b(s)\) 后：

\[
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)\; (R-b(s)) \right]
\]

成立的关键是：只要 \(b(s)\) **不依赖于动作 a**，就有

\[
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)\; b(s)\right] = 0
\]

因此 baseline 不会改变梯度的期望（无偏），但可以显著降低方差。  
在所有仅依赖 \(s\) 的 baseline 中，理论上最优（最小方差）的选择是：

\[
b^\*(s)=\mathbb{E}[R|s] = V(s)
\]

这就是 “critic” 的意义：用一个可学习函数 \(V_\phi(s)\) 去近似 \(V(s)\)，从而让 \(R-V_\phi(s)\)（优势/advantage）更稳定。

### 4.1 在你这个“两步决策 + 单终止奖励”里，策略梯度长什么样？

你的回报 \(R\) 只在最后拿到（`reward = -length(sol1)`），但它依赖两次动作：

\[
R = R(s_0, a_1, s_1, a_2)
\]

目标函数（最大化期望回报）：

\[
J(\theta_1,\theta_2) = \mathbb{E}_{a_1\sim \pi_1,\, a_2\sim \pi_2}[R]
\]

使用 log-derivative trick，对 Gate1 参数 \(\theta_1\)：

\[
\nabla_{\theta_1}J
=\mathbb{E}\left[\nabla_{\theta_1}\log \pi_1(a_1|s_0)\;R\right]
\]

对 Gate2 参数 \(\theta_2\)：

\[
\nabla_{\theta_2}J
=\mathbb{E}\left[\nabla_{\theta_2}\log \pi_2(a_2|s_1)\;R\right]
\]

所以在实现里把两项加在一起（`logp1*adv1 + logp2*adv2`）是完全合理的（`my/train_two_gate_tsp.py:627`）。

### 4.2 为什么可以给 Gate1/Gate2 用不同的 baseline？

对 Gate1 来说，只要 baseline \(b_1\) 不依赖 \(a_1\)（只依赖 \(s_0\)），就不引入偏差：

\[
\mathbb{E}\big[\nabla_{\theta_1}\log \pi_1(a_1|s_0)\; b_1(s_0)\big]=0
\]

同理对 Gate2，只要 \(b_2\) 不依赖 \(a_2\)（只依赖 \(s_1\)），就不引入偏差：

\[
\mathbb{E}\big[\nabla_{\theta_2}\log \pi_2(a_2|s_1)\; b_2(s_1)\big]=0
\]

因此你现在用 `baseline1=V1(s0)`、`baseline2=V2(s1)` 来分别降低两项的方差，在数学上是成立的。

---

## 5. `baseline=critic` 时 Critic 更新的数学形式（当前代码就是这样做的）

### 5.1 你当前有 **两个 critic**

因为你有两次决策、两个状态：

- Critic1：拟合 \(V_1(s_0)\)，输入是 `feat1`
- Critic2：拟合 \(V_2(s_1)\)，输入是 `feat2`

对应代码里的：

- `baseline1 = policy.value1(feat1).squeeze(-1)`
- `baseline2 = policy.value2(feat2).squeeze(-1)`

### 5.2 Critic 的监督信号是什么？

你这里没有用 TD(0)/bootstrapping，而是 **直接用 Monte Carlo 终止回报**当标签：

- 回报 `R` 就是当前 step 的 `reward = -length(sol1)`（每个样本一个标量）

因此 critic 的训练就是一个回归问题：

\[
\min_\phi \; \mathbb{E}\left[(R - V_\phi(s))^2\right]
\]

对应代码：

- `critic_loss1 = ((reward - baseline1) ** 2).mean()`
- `critic_loss2 = ((reward - baseline2) ** 2).mean()`
- `critic_loss = 0.5 * (critic_loss1 + critic_loss2)`

这就是 “Critic 更新的数学原理”：**对 value 网络参数做 MSE 回归的梯度下降**。

更细地写出梯度（以 Critic1 为例）：

令 \(V_1(s_0;\phi_1)\) 是 `policy.value1(feat1)` 的输出。定义损失：

\[
L_{\text{critic1}}(\phi_1)=\mathbb{E}\big[(R - V_1(s_0;\phi_1))^2\big]
\]

对参数求导：

\[
\nabla_{\phi_1} L_{\text{critic1}}
=\mathbb{E}\big[ -2(R - V_1(s_0;\phi_1))\;\nabla_{\phi_1}V_1(s_0;\phi_1)\big]
\]

直观含义：
- 如果当前预测 \(V_1\) **比真实 reward 更小**（更负），则梯度会推动 \(V_1\) 往上抬；
- 如果预测 \(V_1\) **比真实 reward 更大**，则梯度会把它压下来；
- 训练收敛时，\(V_1(s_0)\) 会逼近在该状态下的条件期望 \(\mathbb{E}[R|s_0]\)。

这就是 “Critic 更新的数学原理” 在你当前实现里的具体落地：用 MSE 回归把 `value1/value2` 拟合到 `reward`。

### 5.3 Actor 的 loss 是什么？

你对两个动作都用同一个终止奖励归因，但 baseline 不同：

\[
L_{\text{actor}}
= -\mathbb{E}\left[\log \pi_1(a_1|s_0)\cdot (R-V_1(s_0))
+ \log \pi_2(a_2|s_1)\cdot (R-V_2(s_1))\right]
\]

对应代码（注意 `adv.detach()`）：

- `adv1 = reward - baseline1`
- `adv2 = reward - baseline2`
- `actor_loss = -(logp1 * adv1.detach() + logp2 * adv2.detach()).mean()`

`detach()` 的含义：actor loss 不会把梯度传回 critic 分支（避免“为了让 advantage 变大而修改 V”这种耦合）。

如果不 `detach()`，actor loss 里含有 `baseline1/baseline2`，那么 gate 的 loss 会通过链式法则把梯度传到 value 网络上，
这会产生“critic 为了让 actor loss 更小而改变自身输出”的耦合，常见后果是训练不稳定、value 发散或被 gate 牵着跑。
你现在的做法是标准做法：**actor 用 advantage 的数值，但不把它当成可学习图的一部分**。

### 5.4 总 loss 与参数更新

总 loss：

\[
L = L_{\text{actor}} + \lambda \cdot L_{\text{critic}}
\]

其中 \(\lambda = \texttt{critic\_coef}\)（你命令行参数 `--critic_coef`）。

代码里：

- `loss = actor_loss + cfg.critic_coef * critic_loss`
- `loss.backward(); optimizer.step()`

并且优化器是 `Adam(list(policy.parameters()), lr=cfg.lr)`，所以：
- Gate1/Gate2 的 MLP 参数会被更新（actor 梯度）
- Value1/Value2 的 MLP 参数会被更新（critic 梯度）
- 如果你启用了 `policy.encoder`（论文同款 encoder），它同时参与 `feat1/feat2`，因此 **encoder 会同时收到 actor 与 critic 的梯度**（这是当前实现的设计）。

把“谁更新谁”写成表更直观：

| 参数块 | 来自 `actor_loss` 的梯度 | 来自 `critic_loss` 的梯度 |
|---|---|---|
| `gate1` | ✅ | ❌ |
| `gate2` | ✅ | ❌ |
| `value1` | ❌（因为 `adv1.detach()`） | ✅ |
| `value2` | ❌（因为 `adv2.detach()`） | ✅ |
| `encoder` | ✅（影响 `feat1/feat2`） | ✅（影响 `feat1/feat2`） |
| `final/` 求解器参数 | ❌（`no_grad` + `requires_grad_(False)`） | ❌ |

---

## 6. Critic 是如何预测的？输入是什么？输出是什么？

### 6.1 输入

- Critic1 输入：`feat1 = tsp_instance_features(coords, encoder=policy.encoder, ...)`
  - 只由实例决定（状态 `s0`）
- Critic2 输入：`feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, encoder=policy.encoder, ...)`
  - 由实例 + 初始解条件决定（状态 `s1`）

这些 `feat` 本质上是“特征提取模块（encoder + 手工拼接）”的输出向量。

### 6.2 输出

Critic 输出是一个标量（每个样本一个）：

- `baseline1[b] ≈ E[reward | s0(b)]`
- `baseline2[b] ≈ E[reward | s1(b)]`

---

## 7. Critic 的结构是什么？是 MLP 吗？

是的，当前 critic 是 **MLP（全连接前馈网络）**，定义在：

- `my/gates.py:build_value_mlp`

默认结构（不考虑 dropout）：

```
Linear(in_dim -> 128) + GELU
Linear(128 -> 128)    + GELU
Linear(128 -> 1)
```

你现在有两份独立的 value MLP：

- `policy.value1`: 用于 `feat1 -> V1(s0)`
- `policy.value2`: 用于 `feat2 -> V2(s1)`

---

## 8. “当前情况下各个值分别是什么？”（一句话总结版）

在你这个 TSP 两层 gate 场景里，`baseline=critic` 的核心就是：

- **reward**：每个实例最终 tour 的负长度 `-len1`（`sol1` 来自 “选中的 initializer + 选中的 iterator”）
- **critic 预测**：`value1(feat1)` 与 `value2(feat2)` 分别预测 “在只看实例/看实例+sol0 时，最终 reward 的期望”
- **advantage**：`reward - V(s)`（每个实例一个标量）
- **actor 更新**：让 Gate1/Gate2 更倾向于采样到“带来更大 advantage（更短 length）”的动作
- **critic 更新**：用 MSE 把 `V(s)` 往真实 `reward` 拟合，使 advantage 方差更小、训练更稳

---

## 9. `baseline=critic` 时，critic 在你这个问题里“到底在拟合什么”？

你这里的真实回报是：

\[
R = -\text{len1}
\]

但这个 \(R\) 不只由 `coords` 决定，它依赖：

1) Gate1 选的 initializer（`a1`）  
2) initializer 在该实例上生成的 `sol0`（部分方法内部也可能随机）  
3) Gate2 选的 iterator（`a2`）  
4) iterator 的固定步数改进过程（例如 RRC destroy/repair 的随机性）  

因此：

- \(V_1(s_0)\) 逼近的是 **只看实例信息**时的条件期望：
  \[
  V_1(s_0)\approx \mathbb{E}[R\mid s_0]
  \]
- \(V_2(s_1)\) 逼近的是 **看实例 + 初始解信息**时的条件期望：
  \[
  V_2(s_1)\approx \mathbb{E}[R\mid s_1]
  \]

其中 “期望” 包含了两类随机性：
- 你的策略采样随机性（`Categorical.sample()`）
- 求解器/迭代器内部的随机性（如果有）

这也是为什么 Gate2 更应该用 `feat2` 做 critic：它拿到了 `sol0` 的条件信息，更接近 “最优 baseline”。

---

## 10. 为什么你这里用的是 Monte Carlo critic（MSE 回归），而不是 TD / bootstrap？

因为你的 episode 很短（两步）且 reward 在最后一步立刻可得，所以可以直接用终止回报 \(R\) 当作监督信号：

- 不需要写 TD 误差 \(\delta = r + \gamma V(s') - V(s)\)
- 不需要维护 target network / rollout baseline 等

这就是你当前实现的形式：`critic_loss = (reward - V(s))^2`（`my/train_two_gate_tsp.py:618-620`）。

换句话说：虽然叫 actor-critic，但你这里的 critic 更准确地说是一个 **value regression baseline**（这在两步/终止奖励场景非常常见）。

---

## 11. 用一个具体数值例子理解 advantage 的符号

假设某个样本最终：

- `len1 = 7.80`
- `reward = -7.80`

如果 Critic2 预测：

- `baseline2 = V2(s1) = -8.10`

则：

- `adv2 = reward - baseline2 = (-7.80) - (-8.10) = +0.30`

直觉解释：
- 实际结果比 critic 预期更好（长度更短 → reward 更大/更不负）
- 因此这次选择的 `a2`（以及导致该 `s1` 的 `a1`）会被“鼓励”（梯度会提高对应动作概率）

反过来如果 `baseline2 = -7.50`，则 `adv2 = -0.30`，表示结果比预期差，应降低概率。

---

## 12. baseline=critic 下，你应该看哪些日志来判断“在正确学习”？

你现在的训练日志里通常会打印（`my/train_two_gate_tsp.py:634-655`）：

- `len0 / len1`：初始化长度与迭代后长度（直接反映 solver 效果）
- `adv`（你打印的是 Gate2 的 advantage 均值/方差）：`adv2.mean()` / `adv2.std()`
  - 如果 critic 拟合越来越好，通常 advantage 的波动会变小（方差下降）
- `critic_mse`：`critic_loss`（MSE 回归误差）
  - 不是越小越好到 0（因为策略在变、回报也在变），但应该在一个合理范围内波动，不要爆炸到很大
- `entropy`：`dist1.entropy()+dist2.entropy()`
  - 训练早期较大表示探索；后期变小表示策略开始“更确定”

如果你看到 `critic_mse` 持续爆炸、`adv` 极端大/极端小，往往意味着：
- reward 尺度不合适（长度太大或分布变了）
- 求解器返回的长度/奖励异常
- 或者某个 iterator/initializer 在该数据上不稳定（产生非法 tour 或极端长度）

---

## 13. 逐行级别：`baseline=critic` 在代码里到底做了什么？

下面严格按你当前代码的关键行（`my/train_two_gate_tsp.py:569-632`）解释：每一行在“数学上是什么”、张量形状是什么、梯度会流向哪里。

> 说明：你这里的环境（initializer/iterator）是**不可微**的黑盒（no_grad），但策略梯度只需要 `log_prob`，不需要对环境求导。

### 13.1 采样实例（输入状态 `s0`）

- `coords = train_gen.sample(cfg.batch_size, device=cfg.device)`（:570）
  - 数学上：从训练分布 \(p(\text{instance})\) 采样一批实例，得到状态 \(s_0\)。
  - 形状：`coords: [B,N,2]`。
  - 梯度：`coords` 是环境输入，不需要梯度。

### 13.2 Gate1：计算分布、采样动作、得到 `logp1`

- `feat1 = tsp_instance_features(coords, ...)`（:573）
  - 数学上：把状态 \(s_0\) 映射到特征向量 \(\phi_1(s_0)\)（由 encoder + scale 拼接）。
  - 形状：`feat1: [B,d1]`。
  - 梯度：`feat1` 对 `policy.encoder` 有梯度（因为 encoder 是可学习的）。

- `dist1 = Categorical(logits=policy.gate1(feat1))`（:574）
  - 数学上：`gate1` 输出 logits \(z_1 \in \mathbb{R}^{K_1}\)，定义策略
    \[
    \pi_1(a_1=i\mid s_0)=\text{softmax}(z_1)_i
    \]
  - 形状：`policy.gate1(feat1): [B,K1]`。
  - 梯度：logits 对 `gate1` 和 encoder 都有梯度。

- `a1 = dist1.sample()`（:575）
  - 数学上：按 \(\pi_1(\cdot|s_0)\) 采样动作 \(a_1\)。
  - 形状：`a1: [B]`（每个样本一个离散 id）。
  - 梯度：采样本身不可微；策略梯度通过下一行 `log_prob` 进入。

- `logp1 = dist1.log_prob(a1)`（:576）
  - 数学上：得到 \(\log \pi_1(a_1|s_0)\)（每个样本一个标量）。
  - 形状：`logp1: [B]`。
  - 梯度：`logp1` 对 `gate1`（以及 encoder）可导，是 actor loss 的“入口”。

### 13.3 环境执行：initializer 得到 `sol0`

- `sol0 = run_initializers_by_action(coords, a1, init_zoo)`（:578）
  - 数学上：执行环境转移 \(s_1 \leftarrow f(s_0, a_1)\)，并得到中间解 `sol0`。
  - 输出：`sol0.tour: [B,N]`、`sol0.length: [B]`。
  - 梯度：这里 solver 被 `no_grad` 冻结，`sol0` 不参与反向传播（但它作为“条件信息”进入 Gate2 特征）。

### 13.4 Gate2：计算分布、采样动作、得到 `logp2`

- `feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, ...)`（:581）
  - 数学上：把状态 \(s_1=(s_0,\text{sol0})\) 映射为特征 \(\phi_2(s_1)\)。
  - 形状：`feat2: [B,d2]`。
  - 梯度：`feat2` 的 base 部分来自 encoder，因此对 encoder 可导；`sol0.length/tour` 这类条件是环境输出，不会把梯度回传进 solver。

- `dist2 = Categorical(logits=policy.gate2(feat2))`（:582）
  - 数学上：`gate2` 输出 logits \(z_2\in\mathbb{R}^{K_2}\)，定义
    \[
    \pi_2(a_2=j\mid s_1)=\text{softmax}(z_2)_j
    \]
  - 形状：`policy.gate2(feat2): [B,K2]`。

- `a2 = dist2.sample()`（:583）
  - 形状：`a2: [B]`。

- `logp2 = dist2.log_prob(a2)`（:584）
  - 数学上：\(\log \pi_2(a_2|s_1)\)。
  - 形状：`logp2: [B]`。
  - 梯度：对 `gate2`（以及 encoder）可导。

### 13.5 环境执行：iterator 得到最终解 `sol1`，并定义 reward

- `sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)`（:586）
  - 数学上：执行最终转移与改进，得到终止结果。
  - 输出：`sol1.tour: [B,N]`、`sol1.length: [B]`。
  - 梯度：同样是黑盒执行，不参与反传。

- `reward = -length`（:588-589）
  - 数学上：终止奖励 \(R=-\text{len1}\)。
  - 形状：`reward: [B]`。
  - 梯度：`reward` 不需要梯度（来自黑盒），这不影响 REINFORCE/AC。

### 13.6 `baseline=critic`：critic 前向与 critic loss

当 `cfg.baseline == "critic"` 时（:613）：

- `baseline1 = policy.value1(feat1).squeeze(-1)`（:616）
  - 数学上：预测 \(V_1(s_0)\approx\mathbb{E}[R|s_0]\)。
  - 形状：`policy.value1(feat1): [B,1]`，`squeeze(-1)` 后是 `[B]`。
  - 梯度：对 `value1` 参数可导；同时对 `feat1`（encoder）可导。

- `baseline2 = policy.value2(feat2).squeeze(-1)`（:617）
  - 数学上：预测 \(V_2(s_1)\approx\mathbb{E}[R|s_1]\)。
  - 形状：`[B]`。
  - 梯度：对 `value2` 与 encoder 可导。

- `critic_loss1 = ((reward - baseline1) ** 2).mean()`（:618）
- `critic_loss2 = ((reward - baseline2) ** 2).mean()`（:619）
- `critic_loss = 0.5 * (critic_loss1 + critic_loss2)`（:620）
  - 数学上：对两个 value 回归做平均：
    \[
    L_{\text{critic}}=\frac12\mathbb{E}[(R-V_1(s_0))^2]+\frac12\mathbb{E}[(R-V_2(s_1))^2]
    \]
  - 梯度：会更新 `value1/value2`，也会更新 `encoder`（因为 feat1/feat2 依赖 encoder）。

### 13.7 advantage、actor loss、总 loss

- `adv1 = reward - baseline1`（:624）
- `adv2 = reward - baseline2`（:625）
  - 数学上：用 Monte Carlo return 构造优势 \(A=R-V(s)\)。
  - 形状：`adv1/adv2: [B]`。

- `actor_loss = -(logp1 * adv1.detach() + logp2 * adv2.detach()).mean()`（:627）
  - 数学上（完全对应代码）：
    \[
    L_{\text{actor}}
    =-\mathbb{E}\big[\log\pi_1(a_1|s_0)\cdot \text{stopgrad}(R-V_1(s_0))
    +\log\pi_2(a_2|s_1)\cdot \text{stopgrad}(R-V_2(s_1))\big]
    \]
  - 关键点：`detach()` 让 actor 梯度**不会**通过 \(V\) 回流进 critic（避免耦合）。

- `loss = actor_loss + cfg.critic_coef * critic_loss`（:628）
  - 数学上：总目标
    \[
    L = L_{\text{actor}} + \lambda L_{\text{critic}}
    \]
    其中 \(\lambda=\texttt{critic\_coef}\)。

- `optimizer.zero_grad(); loss.backward(); optimizer.step()`（:630-632）
  - 数学上：对总 loss 做一次 SGD/Adam 步。
  - 结果：一次训练 step 同时更新 actor（gate1/gate2）和 critic（value1/value2），以及共享 encoder。

---

## 14. 你经常“看得见但不理解”的部分：`Categorical(logits=...)` 到底在做什么？

你这里所有 actor 的随机性、以及策略梯度的入口，都在这四行里：

- `dist = Categorical(logits=logits)`
- `a = dist.sample()`
- `logp = dist.log_prob(a)`
- `entropy = dist.entropy()`

更具体地说（以 Gate1 为例）：

### 14.1 logits → 概率

对每个样本 \(b\)，gate 输出一组 logits：
\[
z^{(b)}\in\mathbb{R}^{K_1}
\]
`Categorical(logits=z)` 内部会做 softmax：
\[
p^{(b)}_i=\frac{e^{z^{(b)}_i}}{\sum_j e^{z^{(b)}_j}}
\]

### 14.2 sample 是“从离散分布抽签”

\[
a^{(b)}\sim \text{Cat}(p^{(b)})
\]
这一步不可微，但你不需要它可微；REINFORCE 用的是 \(\nabla \log \pi\)。

### 14.3 log_prob 是策略梯度真正用到的量

`log_prob(a)` 等价于：
\[
\log p^{(b)}_{a^{(b)}}
\]
因此 actor loss 的基本形状就是：
\[
-(\log p)\cdot A
\]

### 14.4 entropy 是“探索程度”的数值

\[
H(p^{(b)})=-\sum_i p^{(b)}_i\log p^{(b)}_i
\]
你现在打印的 entropy 仅用于观察，没有加到 loss 里（所以不会显式鼓励探索）。

---

## 15. 更严谨一点：为什么用 MSE 回归，critic 会逼近 \(\mathbb{E}[R|s]\)？

你这里 critic 的训练目标（对任意一个 critic）是：
\[
\min_\phi\; \mathbb{E}\left[(R - V_\phi(s))^2\right]
\]

固定某个状态 \(s\)，只看条件分布 \(R\mid s\)。对该状态下的目标函数：
\[
g(v)=\mathbb{E}[(R-v)^2\mid s]
\]
对 \(v\) 求导并令其为 0：
\[
g'(v)=\mathbb{E}[-2(R-v)\mid s]= -2(\mathbb{E}[R\mid s]-v)=0
\]
得到最优解：
\[
v^\*(s)=\mathbb{E}[R\mid s]
\]

这就是“critic 更新的数学原理”更本质的一句话：**MSE 的最优回归函数就是条件期望**，因此训练足够充分时，`value1/value2` 会分别逼近 \(V_1(s_0),V_2(s_1)\)。

---

## 16. `baseline=critic` 下：哪些量参与梯度？哪些量不参与？

这部分非常关键，因为你这里同时包含“黑盒求解器”（不可微）和“可学习 gate/critic”（可微）。

### 16.1 不参与梯度的量（环境输出/采样结果）

- `a1/a2`：离散采样结果不可微（但会进入 `log_prob` 的索引计算）。
- `sol0/sol1`：由 `final/` 求解器得到，处在 `no_grad` 下，等价于常量。
- `reward`：由 `sol1.length` 计算得到，同样等价于常量。

### 16.2 参与梯度的量（可学习部分）

- `logp1/logp2`：对 `gate1/gate2`（以及 encoder）可导，是 actor 梯度入口。
- `baseline1/baseline2`：对 `value1/value2`（以及 encoder）可导，是 critic 梯度入口。
- `feat1/feat2`：是 encoder 的输出，因此 encoder 同时收到 actor 与 critic 的梯度（你当前实现就是这样设计的）。

### 16.3 为什么 actor 要 `adv.detach()`？

从计算图角度，`adv = reward - baseline` 包含 `baseline`，而 baseline 又来自 critic 网络。

如果不 `detach()`：
- actor loss 会给 critic 传梯度，critic 会“为了让 actor loss 变小”而改变输出；
- 这会破坏我们希望的分工：critic 只负责拟合 reward 的期望，actor 只负责根据 advantage 调整策略。

你当前的 `detach()` 正是标准 AC 写法。

---

## 17. “当前情况下各个值分别是什么？”（用一个具体 batch 来把抽象量落地）

假设你训练时：

- `problem_size=N=50`
- `batch_size=B=32`
- `init_zoo=[pomo, am, lehd, elg]` → `K1=4`
- `iter_zoo=[none, rrc_lehd_3, two_opt_200]` → `K2=3`

那么一次 step 中：

- `coords`：`[32,50,2]` 的坐标（0~1 范围内）。
- `a1`：长度 32 的整数向量（0~3），比如 `[1,1,0,2,...]`。
- `sol0.length`：32 个正数（比如 5~15 的范围，取决于坐标分布、初始化器能力）。
- `a2`：长度 32 的整数向量（0~2）。
- `sol1.length`：32 个正数（通常 ≤ `sol0.length`，因为迭代可能改进）。
- `reward = -sol1.length`：32 个负数。
- `baseline1/baseline2`：32 个负数（critic 的预测，刚开始可能很不准）。
- `adv1/adv2 = reward - baseline`：32 个数，有正有负：
  - `adv>0` 表示“比 critic 预期更好”（更短 length）
  - `adv<0` 表示“比 critic 预期更差”

最终 actor loss 就是在做两件事：

- 若某样本 `adv1>0`，则增大该样本选到的 `a1` 的概率；
- 若某样本 `adv2>0`，则增大该样本选到的 `a2` 的概率；

因为 reward 是负长度，所以“更好”意味着长度更小 → reward 更大（更接近 0）。

---

## 18. Critic 网络是如何“被构建出来”的？输入维度是多少？

你运行 `my/train_two_gate_tsp.py` 时，gate/critic 的创建发生在 `_build_gates(...)`（`my/train_two_gate_tsp.py:438`）：

### 18.1 为什么代码要先做一次 dummy forward？

因为 `feat1/feat2` 的维度 `d1/d2` 取决于你选用的特征提取方式（`FeatureConfig`）：

- 你现在默认 `extractor="paper"`（`my/features.py:16`），即用论文同款 encoder 得到 `graph_emb`；
- 并且还会拼接一些额外维度（例如 `scale=N`，Gate2 还会拼接 `length0`、tour 统计）。

所以代码用 dummy 数据算一次，来自动得到：

- `d1 = tsp_instance_features(dummy, ...).size(1)`（`my/train_two_gate_tsp.py:452`）
- `d2 = tsp_gate2_features(dummy, dummy_tour, ...).size(1)`（`my/train_two_gate_tsp.py:454-460`）

### 18.2 在默认配置下，`feat1/feat2` 具体由哪些维度组成？

在 `extractor="paper"` 且 `paper_include_scale=True`、`use_length0=True`、`use_tour_stats=True` 时：

- `feat1 = [graph_emb, scale]`
  - `graph_emb`：encoder 输出的图级 embedding（默认 `paper_embedding_dim=128`）
  - `scale`：一个常数特征（`N`），形状 `[B,1]`
  - 所以通常 `d1 = 128 + 1 = 129`

- `feat2 = [feat1, length0, tour_stats]`
  - `length0`：初始化 tour 长度，形状 `[B,1]`
  - `tour_stats`：`tsp_tour_features` 返回 4 个统计量（mean/std/min/max），形状 `[B,4]`
  - 所以通常 `d2 = d1 + 1 + 4 = 134`

这些拼接在 `my/features.py:tsp_instance_features`（:77-99）和 `my/features.py:tsp_gate2_features`（:154-177）里能一行一行对上。

### 18.3 Critic MLP 是怎么实例化的？

当且仅当你设置 `--baseline critic` 时（`cfg.baseline in ("critic", "critic_batch_mean")`），才会创建 value 网络：

- `value1 = build_value_mlp(in_dim=d1)`（`my/train_two_gate_tsp.py:467`）
- `value2 = build_value_mlp(in_dim=d2)`（`my/train_two_gate_tsp.py:468`）

而 `build_value_mlp` 的默认 hidden dims 就是 `(128,128)`（`my/gates.py:29-44`）。

---

## 19. Critic “如何预测”？把 MLP 前向写成明确的数学形式

以 Critic1 为例，你的 `policy.value1` 是一个三层前馈网络：

1) `h1 = GELU(W1 * feat1 + b1)`，其中 `W1: [128,d1]`
2) `h2 = GELU(W2 * h1 + b2)`，其中 `W2: [128,128]`
3) `v  = W3 * h2 + b3`，其中 `W3: [1,128]`

输出 `v` 就是 `baseline1`（再 `squeeze(-1)` 变成 `[B]`）。

对应代码实现就是：

- `nn.Linear -> nn.GELU -> nn.Linear -> nn.GELU -> nn.Linear(out=1)`（`my/gates.py:35-44`）

这里没有对输出做 `tanh/sigmoid` 之类的约束，这是合理的：

- 因为你的 reward 是负长度，数值范围不是固定区间；
- value 的最优回归目标也是一个实数（\(\mathbb{E}[R|s]\)），不应被强行压缩到 [-1,1] 等区间。

---

## 20. EasyNCO 平台里 LEHD 的 RRC（Random Re-Construction）迭代：具体实现过程

本节基于 EasyNCO 平台的实现文件：

- `final/neural_solvers/methods/lehd/iteration.py:34`（TSP 的 RRC：`TSPLEHDIteration`）
- `final/neural_solvers/envs/TSPEnv.py:62`（`TSPEnv.solution` 字段给 RRC 用）

这里只解释 **TSP** 分支（你目前关注 TSP），CVRP 的 RRC 在同文件的 `CVRPLEHDIteration`（实现更复杂，且含 depot/flag 逻辑）。

### 20.1 RRC 的输入输出是什么？

在平台里，RRC 是“初始化之后的迭代改进”阶段：

- 输入：
  - 当前解（来自 initialization 跑完后环境里的 `selected_node_list`），在 `td['selected_node_list']` 或 `td['next']['selected_node_list']`（`final/neural_solvers/methods/lehd/iteration.py:56`）
  - 当前解的 reward（TSPEnv 里 reward = -tour_length），在 `td['reward']`（`final/neural_solvers/envs/TSPEnv.py:222`）
  - 原始实例坐标 `problems`（在 `LEHDIteration.run(..., problems=...)` 里通过 `kwargs` 传入，见 `final/neural_solvers/methods/lehd/iteration.py:23-25`）

- 输出：
  - `out = {'no_aug_score': mean_length, 'aug_score': mean_length}`（`final/neural_solvers/methods/lehd/iteration.py:126-129`）
  - 这里 `mean_length` 是 RRC 最终得到的完整 tour 的平均长度（注意这里是“长度”，不是 reward）。

### 20.2 迭代次数与随机性

RRC 外层循环是 `for bbbb in range(max_steps):`（`final/neural_solvers/methods/lehd/iteration.py:65`），每一步都会引入随机性：

1) **随机反转 tour（50% 概率）**（`final/neural_solvers/methods/lehd/iteration.py:70-76`）  
   目的：同一条环形 tour 反向等价，但后续“截取子路径”的结果会变，增加搜索多样性。

2) **随机选子路径起点与长度**（`first_node_index` 与 `subpath_length`）  
   - `first_node_index ~ Uniform{0..N-1}`（`final/neural_solvers/methods/lehd/iteration.py:181`）
   - `subpath_length ~ Uniform{4..N}`（`final/neural_solvers/methods/lehd/iteration.py:185`）

### 20.3 Destroy（破坏）：随机截取一段子路径并构造子问题

核心调用是：

- `partial_solution_length, first_node_index, subpath_length, solution_copy = self.destroy_solution(problems, current_node_list)`（`final/neural_solvers/methods/lehd/iteration.py:78-80`）

它做了两件事：

1) **从完整 tour 中截取子路径**  
   - 把 tour 复制成 `solution_copy = cat([solution, solution])`（`final/neural_solvers/methods/lehd/iteration.py:193`）  
     这样截取 `[first_node_index:first_node_index+subpath_length]` 时能自然处理“跨越末尾”的情况。
   - `new_solution = solution_copy[..., first:first+len]` 得到子路径（`final/neural_solvers/methods/lehd/iteration.py:195`）

2) **把子路径对应的节点坐标提取出来，形成一个“子 TSP 实例”**  
   为了让子问题的节点索引从 `0..subpath_length-1` 连续，代码会：
   - 对子路径节点 id 做排序，得到一个“子问题节点集合的排序”（`final/neural_solvers/methods/lehd/iteration.py:198-206`）
   - 用高级索引把这些节点的 `(x,y)` 坐标取出来，reshape 成 `new_problem: [batch, subpath_length, 2]`（`final/neural_solvers/methods/lehd/iteration.py:209-217`）
   - 同时计算一个映射 `new_solution_rank`，把“原 tour 子路径的节点 id”映射成“子问题里的局部索引 0..L-1”（`final/neural_solvers/methods/lehd/iteration.py:198-200`）

destroy 的返回里：

- `destroyed_problem = new_problem`（子问题坐标）
- `destroyed_solution = new_solution_rank`（子问题上的“旧解”）

随后 `destroy_solution` 会把它们塞进环境里：

- `self.env.solution = destroyed_solution`（`final/neural_solvers/methods/lehd/iteration.py:168`）
- `self.env.problems = destroyed_problem`（`final/neural_solvers/methods/lehd/iteration.py:169`）

并计算“旧子路径解”的长度 `partial_solution_length` 作为对比基线（`final/neural_solvers/methods/lehd/iteration.py:171`）。

### 20.4 Repair（修复）：在子问题上重新构造一条 tour（由 policy 生成）

destroy 后，代码会在子问题上跑一个完整 episode 来生成一条新 tour：

- `reset_td = env.reset(); self.policy.pre_forward(reset_td); state_td = env.pre_step()`（`final/neural_solvers/methods/lehd/iteration.py:85-88`）
- `while not done: ... state_td = env.step(next_td)`（`final/neural_solvers/methods/lehd/iteration.py:92-105`）

注意这里有一个“小技巧”：

- 第 0、1 步动作不是 policy 生成的，而是强行喂给环境（`final/neural_solvers/methods/lehd/iteration.py:93-99`）：
  - step0：`action = self.env.solution[:, :, -1]`（旧子路径的最后一个点）
  - step1：`action = self.env.solution[:, :, 0]`（旧子路径的第一个点）
- 从 step2 开始才调用 `next_td = self.policy(state_td)`（`final/neural_solvers/methods/lehd/iteration.py:100`）

这意味着：修复阶段不是“纯从零开始”的解码，而是给定了两个起始节点（相当于对解码轨迹做了一点条件化）。

episode 结束后：

- 子问题上的新解在 `env.selected_node_list` 里
- 子问题的长度对应 `after_repair_length = -state_td['reward']`（reward 是负长度，`final/neural_solvers/methods/lehd/iteration.py:108`）

### 20.5 Accept（接受）：仅当子路径变短才替换进完整 tour

接受逻辑在 `accept_repaired_solution(...)`（`final/neural_solvers/methods/lehd/iteration.py:134`）：

1) 把“子问题局部索引 0..L-1 的新解”映射回“原问题全局节点 id”：
   - `repaired_solution_index = sorted_solution_index[..., repaired_solution]`（`final/neural_solvers/methods/lehd/iteration.py:150`）

2) 仅当新子路径更短才接受：
   - `if_repair = pre_length > after_repair_length`（`final/neural_solvers/methods/lehd/iteration.py:152`）

3) 接受则替换掉完整 tour 的这段片段：
   - `solution_copy[if_repair] = cat(part1, repaired_segment, part2)`（`final/neural_solvers/methods/lehd/iteration.py:154-156`）
   - 再从双倍 tour 里切回长度 `N` 的完整 tour（`final/neural_solvers/methods/lehd/iteration.py:158`）

所以本质上：**RRC = 随机选一段子路径 → 在子问题上“重解码” → 若改进则替换**。它是一个“随机大邻域搜索/重构式局部搜索”的实现形态。

---

## 21. RRC 是否需要加载权重？

分两层回答：

### 21.1 “能不能跑起来”层面：不加载也能跑，但效果不可控

RRC 代码本身不负责加载权重；它只是调用 `self.policy(...)` 来产生修复动作（`final/neural_solvers/methods/lehd/iteration.py:100`）。

因此：

- 只要你给它一个“实现了 `pre_forward` + `forward`”的 `policy`，就能执行完整 RRC 流程；
- 即使这个 policy 是随机初始化的，RRC 仍能跑完（只是修复质量通常很差，基本靠“偶然改进 + 接受准则”在做事）。

### 21.2 “作为 LEHD 方法的一部分”层面：需要加载 LEHD 的训练权重

在 LEHD 的设计里，repair 阶段依赖神经网络产生较高质量的重构，因此实际使用时：

- 会加载 **LEHDPolicy 的 checkpoint**；
- 并且 initialization 和 iteration 通常复用同一个 LEHDPolicy（平台配置 `final/settings/lehd_settings.yaml` 里 `model: LEHDPolicy` + `iteration: LEHDIteration`，权重来源在 `test_loader.model_filename`）。

换句话说：**RRC 不需要“单独的一份迭代权重”，但它需要一个有意义的 policy 权重来指导修复。**

---

## 22. 能否“执行 RRC 时加载 POMO 权重”？能否正常执行？

这里存在两种不同理解，结论不同：

### 22.1 如果你的意思是“把 POMO checkpoint 加载进 LEHDPolicy 里”

基本不行：

- POMO 和 LEHD 的网络结构不同（encoder/decoder 的模块名与张量形状都不一致）；
- 你强行 `load_state_dict` 会出现大量 `missing_keys/unexpected_keys` 或 shape mismatch；
- 即使你选择 `strict=False` 让它“部分加载”，大部分参数仍是随机的，repair 质量不可控。

所以这种做法不推荐。

### 22.2 如果你的意思是“用 POMOPolicy + POMO 权重来当 RRC 的 repair policy”

从 **代码接口** 上看，这是可行的，因为在 EasyNCO 里：

- `final/neural_solvers/methods/pomo/policy.py:11` 的 `POMOPolicy` 也实现了：
  - `decoder_strategy` 字段
  - `set_decoder_strategy(...)`
  - `pre_forward(td)`
  - `forward(td)` 并在 td 里写入 `action/prob`

而 RRC 的 repair 阶段对 policy 的依赖基本就是这些（`final/neural_solvers/methods/lehd/iteration.py:63-105`）。

因此：**把 repair policy 换成 POMOPolicy（并加载 POMO 权重）后，RRC 这套 “destroy → repair → accept” 的步骤大概率能正常执行，不会因为接口缺失而直接报错。**

但要注意两点：

1) 这已经不再是“LEHD 的 RRC（用 LEHDPolicy 修复）”，而是“RRC 框架 + POMO 修复器”的混搭方法。  
2) RRC 的 repair 阶段会强行指定前两步 action（`final/neural_solvers/methods/lehd/iteration.py:93-99`）；POMO 虽然能继续解码，但它并不是专门按这种条件化方式训练的，因此效果是否好需要实验验证。

---

## 23. EasyNCO 平台里 LIH 的“迭代（Iteration）”到底在做什么？（以 TSP 为主）

LIH 在 EasyNCO 里本质是一个 **“神经网络指导的 2-opt 局部搜索”**：

- 你始终维护一条当前 tour（`td['solution']`），每一步：
  1) 用当前 tour 构造输入特征（把“几何信息 + 当前 tour 顺序”喂给网络）
  2) 网络输出一个 2-opt 交换点对 `exchange=(u,v)`
  3) 对当前 tour 执行一次 2-opt（反转子段）
  4) 计算新 tour 的长度与改进 reward
  5) 重复固定步数 `max_steps`

关键实现文件：

- 初始化：`final/neural_solvers/methods/lih/initialization.py:88`（`LIHInitialization`）
- 迭代：`final/neural_solvers/methods/lih/iteration.py:241`（`LIHIteration` + `Iteration_tool`）
- 策略网络：`final/neural_solvers/methods/lih/policy.py:12`（`LIHPolicy`）
- 选 2-opt 点对的 actor：`final/neural_solvers/methods/lih/actor.py:12`（`LIH_Actor`）

下面把 **一次 iteration step** 拆成“和代码完全一致”的过程。

### 23.1 初始化阶段给 iteration 准备了什么状态？

平台里的 LIH 初始化并不是“走 env.reset() 然后 rollout 一个解”，而是直接在 `Initialization_tool.initial()` 里构造一个初始解：

- TSP 初始 tour：随机生成一个 permutation（实现方式比较绕，但最终 `td['solution']` 是一个长度为 `N` 的排列）（`final/neural_solvers/methods/lih/initialization.py:13-31`）
- 初始长度：`td['current_length'] = get_costs(locs, solution)`（`final/neural_solvers/methods/lih/initialization.py:48-49`）
- `td['exchange']`：初始化为全 0 的占位点对（`final/neural_solvers/methods/lih/initialization.py:51`），用于后续 mask “不要重复同一对交换”

最终初始化输出的 td 关键字段是（`final/neural_solvers/methods/lih/initialization.py:79-86`）：

- `solution`: 当前 tour（TSP 是节点排列）
- `current_length`: 当前长度
- `pre_length`: 上一步长度（初始时等于 current_length）
- `exchange`: 上一次选的交换点对（初始为 0）
- `locs`: 节点坐标

### 23.2 LIH 每一步如何把“当前 tour”变成网络输入？（get_input_and_pe）

LIHPolicy 每一步都会调用：

- `input_info, pos_enc = tsp_embedding(td['locs'], td['solution'])`（`final/neural_solvers/methods/lih/policy.py:104-110`）

这里的设计意图是：**网络既要看到每个节点的几何坐标，也要知道该节点当前在 tour 里的“位置/顺序”**。

对 TSP：

- `input_info`：本质上就是节点坐标（按 node_id 顺序组织成 `[B,N,2]`），用于给每个节点做初始 embedding（`final/neural_solvers/methods/lih/policy.py:140`）
- `pos_enc`：一个 `[B,N,128]` 的位置编码，它不是按“节点 id 的自然顺序”给的，而是用 `td['solution']` 的顺序去索引位置编码（`final/neural_solvers/methods/lih/policy.py:134-145`）

这样 actor 看到的是：

- 节点几何：`init_embed = Linear(input_info)`（`final/neural_solvers/methods/lih/policy.py:59`）
- 节点在当前 tour 的相对位置：`pos_enc`
- 并把两者相加送入 Transformer：`td['init_embed'] + td['pos_enc']`（`final/neural_solvers/methods/lih/actor.py:63`）

### 23.3 LIH 如何“选一个 2-opt 点对”？

核心在 `LIH_Actor.forward(...)`（`final/neural_solvers/methods/lih/actor.py:59`）：

1) 用 Transformer 编码每个节点（带上 tour 位置编码）：
   - `out = TransformerNet(td['init_embed'] + td['pos_enc'])`（`final/neural_solvers/methods/lih/actor.py:63`）

2) 构造“图级 context”（max pooling）并融合到每个节点：
   - `graph_embed = out.max(1)[0]`（`final/neural_solvers/methods/lih/actor.py:64`）
   - `fixed_context = Linear(graph_embed)[:,None,:]`（`final/neural_solvers/methods/lih/actor.py:67`）
   - `fusion = project_node(out) + fixed_context`（`final/neural_solvers/methods/lih/actor.py:69-72`）

3) 计算每对节点 (i,j) 的兼容性分数（类似 attention 的 `qk^T`），并做 mask：
   - `score = q @ k^T` → 得到 `[B, N, N]`（`final/neural_solvers/methods/lih/actor.py:130-135`）
   - mask 对角线（禁止 i=j）：`score_scaled[diag] = -inf`（`final/neural_solvers/methods/lih/actor.py:138-154`）
   - mask “不要重复上一步的 exchange”：如果 `td['exchange']` 不是全 0，就把那一对以及对称位置置为 `-inf`（`final/neural_solvers/methods/lih/actor.py:155-162`）

4) 把 `[B,N,N]` 展平为 `[B,N*N]`，对每个样本得到一个 **对所有点对的分布**：
   - `im_s = softmax(score_scaled.view(..., N*N))`（`final/neural_solvers/methods/lih/actor.py:166-170`）

5) 从该分布里采样一个点对（不是 argmax）：
   - `inde = att_s.squeeze().multinomial(1)`（`final/neural_solvers/methods/lih/actor.py:90`）
   - `row = inde // N; col = inde % N`（`final/neural_solvers/methods/lih/actor.py:92-94`）
   - 得到 `exchange=(row,col)`（`final/neural_solvers/methods/lih/actor.py:94-101`）

随后 `LIHPolicy.forward` 把结果写回 `td`：

- `td['exchange'] = exchange[0]`（`final/neural_solvers/methods/lih/policy.py:66-71`）
- `td['log_likelihood'] = ll`（这里保存的是被采样点对对应的 logprob，供训练用；`final/neural_solvers/methods/lih/policy.py:63-69`）

### 23.4 LIH 如何执行一次 2-opt（真正改变 tour）？

在 TSP 的 eval/test 循环里，每一步是：

- `td['input_info'], td['pos_enc'] = policy.get_input_and_pe(td)`（`final/neural_solvers/methods/lih/iteration.py:413`）
- `td = policy(td)` → 得到 `exchange`（`final/neural_solvers/methods/lih/iteration.py:414`）
- `td = tool.operate(td)` → 执行 2-opt 并更新长度、reward（`final/neural_solvers/methods/lih/iteration.py:415`）

`Iteration_tool.operate` 的核心就是：

1) 从 `exchange=(u,v)` 得到两个节点 id（`final/neural_solvers/methods/lih/iteration.py:186-190`）
2) 对当前 `td['solution']` 做 2-opt：
   - 找到 u、v 在 tour 中的位置，然后把这段子序列反转（`final/neural_solvers/methods/lih/iteration.py:166-183`）
3) 计算新 tour 的长度：
   - TSP：`length = ||x_{t+1}-x_t||` 累加（`final/neural_solvers/methods/lih/iteration.py:230-237`）
4) 定义 reward（只奖励改进）：
   - `reward = max(prev_length - new_length, 0)`（等价于 `pre_bsf - now_bsf`，`final/neural_solvers/methods/lih/iteration.py:190-195`）
5) 更新 td：
   - `td['solution']` 变成新 tour（无论是否改进都会更新）（`final/neural_solvers/methods/lih/iteration.py:206-213`）

最后，test/eval 会把每一步的 `current_length` 都记录下来，返回整个轨迹的 best（取 min）：

- `current_m = torch.stack(length_current, 0); best = current_m.min(0)[0]`（`final/neural_solvers/methods/lih/iteration.py:438-440`）

因此：LIH 在实现上更像“**学习到的 proposal + 轨迹里取最优**”，而不是严格的“只接受改进的贪心爬山”。

> 额外说明：你之前在 `my` 里发现 LIH 必须 CUDA，原因也在这里：很多地方硬编码了 `.cuda()`（例如 `tsp_embedding(...).cuda()`：`final/neural_solvers/methods/lih/policy.py:127`，以及 `two_opt` 里 `rec = rec.cuda()`：`final/neural_solvers/methods/lih/iteration.py:182`）。

---

## 24. LIH vs “普通 2-opt”：相同点与不同点（结合 EasyNCO 的实现）

### 24.1 相同点（本质都是 2-opt 邻域）

- **同一个 move 定义**：选两个位置/节点，把中间子段反转（EasyNCO 的实现见 `final/neural_solvers/methods/lih/iteration.py:166-183`）。
- **目标相同**：尽量让 tour 变短（长度计算同样是欧氏距离累加）。
- **可迭代应用**：重复做多步局部改进，最终取最好结果。

### 24.2 不同点（核心差异在“怎么选 move / 怎么组织搜索”）

1) **move 选择方式不同**
   - 普通 2-opt：通常会“枚举很多候选 (i,j)”，计算真实的 length delta，然后选最优/第一个改进；要么确定性、要么随机但仍依赖显式 delta 评估。
   - LIH：用 Transformer 给所有点对打分，形成一个分布，然后 **直接采样一个点对**（`final/neural_solvers/methods/lih/actor.py:90-100`），不显式计算每个候选的真实 delta 来做 argmin。

2) **接受准则/搜索轨迹不同**
   - 普通 2-opt（经典爬山）：只接受改进 move；因此序列单调下降，最终停在 2-opt 局部最优。
   - EasyNCO 的 LIH：每一步都更新 `solution`（`final/neural_solvers/methods/lih/iteration.py:206-213`），reward 只在改进时为正（`final/neural_solvers/methods/lih/iteration.py:190-195`），最终在整条轨迹里取最优（`final/neural_solvers/methods/lih/iteration.py:438-440`）。
     - 这更像“策略引导的随机游走 + best-of-trajectory”。

3) **是否需要训练/权重**
   - 普通 2-opt：纯算法，无需权重。
   - LIH：核心优势来自训练好的 `LIHPolicy`（以及训练时用到的 critic）；没权重也能跑，但选点对近似随机，效果通常接近随机 2-opt。

4) **计算开销结构不同**
   - 普通 2-opt：如果全枚举是 \(O(N^2)\) 次候选评估；工程上常用增量 delta 和剪枝。
   - LIH：每一步前向里也会形成 \(N\times N\) 的 attention score（`qk^T`，`final/neural_solvers/methods/lih/actor.py:130-135`），所以同样带有 \(O(N^2)\) 的结构；区别在于它把“评估函数”换成了神经打分，并用采样产生 proposal。
