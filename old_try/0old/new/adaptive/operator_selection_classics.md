# 相关领域的“算子选择（Operator Selection / Heuristic Selection / Adaptive Operator Selection）”经典方法调研（跨领域）

> 目标：给你一个“可复用的工具箱”，把 **算子选择** 放进统一框架里理解：  
> **选择什么（operator / neighborhood / parameter / subproblem）**、**依据什么反馈（credit / reward）**、**怎么更新（policy / weights）**、**如何处理非平稳（non-stationary）**。  
> 重点偏“经典方法 + 可落地公式/伪代码 + 适用条件 + 常见坑”，不局限于组合优化。

---

## 0. 基本定义：什么叫“算子选择”？

在各种搜索/优化算法中，都会遇到一个共同结构：

- 你有一个 **算子集合** `O = {o1, o2, ..., ok}`（例如：不同邻域移动、不同破坏-修复算子、不同交叉/变异、不同局部搜索模块、不同温度/扰动强度等）。
- 在迭代 `t=1..T` 中，每一步要做决策：
  1) 选一个算子 `a_t ∈ O`（或者选一个“算子+参数”）；  
  2) 把它作用到当前状态/解 `x_t`，得到新解 `x_{t+1}`；  
  3) 观测到某种回报/代价变化 `r_t`（例如改进幅度、是否可行、耗时、是否进入新 best、解质量曲线等）。

**算子选择（operator selection）**就是设计一个策略 `π`：

`a_t ~ π( state_t ; θ )`

其中 `state_t` 可以很简单（只包含“历史每个算子表现”），也可以很复杂（包含“当前解结构特征 + 搜索进度 + 实例特征”）。

> 常见别名：  
> - Adaptive Operator Selection (AOS)（进化算法/元启发式文献）  
> - Heuristic Selection / Selection Hyper-heuristic（超启发式）  
> - Operator Allocation / Operator Scheduling  
> - 也可被看作参数控制（operator 概率就是参数）

---

## 1. 把问题拆成 3 个核心子问题（几乎所有方法都在回答这三件事）

### 1.1 Credit Assignment（怎么给算子“记功/记过”？）

你需要把“这一步用了算子 `a_t`”映射成一个可学习/可统计的 **信用值（credit）**：

- 最简单：`credit = Δf = f(x_t) - f(x_{t+1})`（最小化时为正表示改进）
- 常见改造：
  - **归一化**：`Δf / f(x_t)`、`Δf / n`、`Δf / time`（强调性价比）
  - **截断/鲁棒**：`clip(Δf, -c, c)`（避免极端值把策略带歪）
  - **分档奖励**：新全局最优给 5 分，改进给 3 分，被接受但不改进给 1 分，被拒绝给 0 分（ALNS 常见）
  - **延迟奖励/窗口奖励**：把一个算子触发的一段“连锁改进”累计到同一 credit（尤其在 memetic/ILS/复杂 pipeline）

关键难点：
- 信用往往 **噪声大**（同一算子在不同阶段效果不同）
- 信用往往 **非平稳**（早期和后期，最有效的算子可能完全不同）
- 多算子组合时会出现 **归因问题**（credit assignment problem：到底是谁带来的改进？）

补充：credit assignment 本身也有一条专门研究线，尤其在 **多目标 AOS** 场景更明显。  
例如 Hitomi & Selva 在 TEVC 对多目标自适应算子选择的 credit 策略做过系统分类比较：DOI `10.1109/TEVC.2016.2602348`（你做“gap+time”等多指标时很有参考价值）。

### 1.2 Selection Policy（怎么从 credit 得到“下一步选谁”？）

给定每个算子的估计价值 `Q_i` 或权重 `w_i`，你要把它转成选择概率/规则：

- “选最大”太贪婪会早熟
- “纯随机”太浪费
- 经典问题：**exploration vs exploitation**

### 1.3 Non-stationarity（如何“遗忘旧信息”，跟上阶段变化？）

算子表现随时间变化非常普遍：

- 早期需要强探索/大扰动
- 后期需要小邻域精修
- 或者不同实例结构导致偏好不同

因此你通常需要：

- 滑动窗口、指数衰减、分阶段建模、变点检测等机制
- 或者采用对非平稳更鲁棒的策略（如 adversarial bandit / multiplicative weights）

---

## 2. 经典“结构化”框架：从 OR/元启发式到 EA/ML 的共通范式

我先列出几类“经典框架”，你会发现后面很多策略都可以嵌进去当模块。

### 2.1 ALNS / ILS 里的权重自适应（Operator pool + score + exponential smoothing）

典型代表：

- Ropke & Pisinger (2006) “An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows” DOI `10.1287/trsc.1050.0135`
- Pisinger & Ropke (2007) “A general heuristic for vehicle routing problems” DOI `10.1016/j.cor.2005.09.012`

**核心思想**：

1. 有一组破坏算子 `R`（removal）和修复算子 `I`（insertion）
2. 每轮随机抽取 `(r ∈ R, i ∈ I)` 组合生成候选解
3. 根据本轮结果给组合打分 `score`
4. 用指数平滑更新权重：

`w_j ← (1-ρ) w_j + ρ · score_j`

5. 下次选择时按 `w` 做 roulette / softmax

**为什么经典**：  
它是“超简单、超稳健、工程上效果很好”的 baseline；很多后来 bandit/RL 方法本质上是它的理论化升级。

### 2.2 超启发式（Hyper-heuristics）：把“选启发式/选算子”当作一等公民

代表性综述：

- Burke et al. (2013) “Hyper-heuristics: a survey of the state of the art” DOI `10.1057/jors.2013.71`

经典早期工作（概念源头）：

- Cowling, Kendall, Soubeiga (2002) “Hyperheuristics: A Tool for Rapid Prototyping in Scheduling and Optimisation” DOI `10.1007/3-540-46004-7_1`

框架上常分两类：

- **Selection hyper-heuristic**：在给定的一组 low-level heuristics 中做选择（你关心的算子选择就是这类）
- **Generation hyper-heuristic**：生成新的启发式（如 GP 生成规则）

Selection HH 的典型结构：

`(heuristic selection policy) + (move acceptance)`

其中：
- selection policy：choice function、tabu-based selection、RL、bandit…
- acceptance：SA / threshold acceptance / great deluge / late acceptance / only-improve…

HyFlex 是一个跨领域 HH benchmark 框架：
- Ochoa et al. (2012) “HyFlex: A Benchmark Framework for Cross-Domain Heuristic Search” DOI `10.1007/978-3-642-29124-1_12`

### 2.3 Choice Function（超启发式里最经典的启发式选择器之一）

> 这块值得单独拎出来，因为它几乎就是“早期算子选择的工程范式”：  
> **把“近期表现 + 改进能力 + 多样性贡献”线性组合成一个分数，然后选分数最高的启发式。**

相关源头通常追溯到 Cowling/Kendall/Soubeiga 的超启发式体系（见上面的 2002 章节 DOI `10.1007/3-540-46004-7_1`）。

一个典型（常见于后续 HH 文献）的 choice function 结构是：

`CF(h) = α·f1(h) + β·f2(h) + γ·f3(h)`

其中 `h` 是某个 low-level heuristic / operator，三项常被解释为：

- `f1(h)`：**recency / time since last use**  
  让很久没用过的启发式分数变高，避免“饿死”，是一种结构化探索。
- `f2(h)`：**performance / improvement**  
  近期使用 `h` 带来的平均改进（或累计改进）。
- `f3(h)`：**pairwise synergy / diversity / sequence effect**  
  有些启发式“单用一般，但接在另一个后面特别好”；或它能把搜索带到与当前轨迹差异更大的区域。  
  `f3` 就是在捕捉这种“组合效应/多样性贡献”。

实现细节的常见做法：

- `f1` 用一个随时间增长的函数（例如距离上次使用的步数、或其归一化版本）
- `f2` 用滑动窗口平均/指数衰减平均的 `Δf` 或分档 score
- `f3` 用“上一次使用的启发式”作为条件：  
  维护一个矩阵 `S[i,j]` 表示 “先用 i 再用 j” 的历史收益，然后 `f3(h=j)=S[last,h]`

> 你会发现 choice function 本质上已经在做 **contextual selection** 了：context=“上一步用了什么 + 最近是否用过”。  
> 它是 bandit/RL 的一个非常“轻量工程实现”。

### 2.4 Tabu-based Hyper-heuristic（把“启发式选择”本身做 Tabu Search）

代表性早期工作之一：
- Kendall & Hussin (2005) “An Investigation of a Tabu-Search-Based Hyper-Heuristic for Examination Timetabling” DOI `10.1007/0-387-27744-7_15`

核心想法非常直观：

- 在低层你有一组启发式 `H={h1..hk}`
- 高层把“选哪个启发式”当作搜索空间
- 用 tabu list 记住最近用过/最近失败的启发式，暂时禁止重复选择

一个极简伪代码：

```
tabu = Queue(maxlen=L)
while not stop:
  candidates = [h for h in H if h not in tabu]
  pick h with best predicted score (or random among top-m)
  apply h -> get new solution
  if accepted:
    update stats of h
  tabu.push(h)
```

这种方法的特点：

- 机制上非常简单，几乎零数学假设
- 适合“启发式之间强相关、容易陷入重复模式”的场景（tabu 在高层打破模式）
- 常作为 choice function / bandit 的替代 baseline（对照组很有价值）

---

## 3. 经典方法族 A：确定性/半确定性“算子调度”（无需学习，但非常有用）

这类方法看起来“不像在学习”，但它们是很多自适应方法的思想祖先：**通过切换邻域/参数让搜索不断改变尺度**。

### 3.1 Variable Neighborhood Search (VNS) / VND

经典论文：
- Mladenović & Hansen (1997) “Variable neighborhood search” DOI `10.1016/S0305-0548(97)00031-2`

核心机制：

- 准备多个邻域 `N1, N2, ..., Nk`
- 在一个邻域里做到局部最优后，切换到更大/不同结构的邻域，若改进则回到小邻域精修

它给“算子选择”的启发：

- **阶段性偏好**是必然的：不同阶段适合不同邻域
- 即使不做复杂学习，**固定的邻域层级**也能显著增稳（先小后大、或小-大-小循环）

### 3.2 Reactive Tabu Search（参数的自适应控制 = 隐式算子选择）

经典论文：
- Battiti & Tecchiolli (1994) “The Reactive Tabu Search” DOI `10.1287/ijoc.6.2.126`

思想：tabu tenure（禁忌长度）不是固定的，而是根据“是否出现循环/重复访问”自适应调整。

你可以把它看成：

- action：调整 tabu tenure（相当于改变“可用邻域/可接受 move 集”）
- feedback：搜索是否在循环、重复率、停滞程度
- update：reactive 调参规则

> 这类“反应式参数控制（reactive search optimization）”在很多领域都很常见，是 AOS 的一条主线：  
> 与其直接选算子，不如选“让哪些 move 合法/更容易被接受”的参数。

---

## 4. 经典方法族 B：概率匹配（Probability Matching）与 Adaptive Pursuit（超经典、超好用）

### 4.1 Probability Matching (PM)

设每个算子有一个价值估计 `Q_i`，则选择概率：

`P(i) = (1 - k·p_min) · Q_i / Σ_j Q_j + p_min`

其中 `p_min` 是每个算子的最小探索概率，防止被永久饿死。

更新 `Q_i` 常用指数平滑：

`Q_i ← (1-α) Q_i + α · reward`

特点：
- 简单稳定
- 但适应速度可能偏慢（当最优算子随阶段快速切换时）

### 4.2 Adaptive Pursuit（Thierens 的经典公式）

经典论文：
- Thierens (2005) “An adaptive pursuit strategy for allocating operator probabilities” DOI `10.1145/1068009.1068251`

直觉：每一步先判定当前“最好”的算子 `b = argmax Q_i`，然后把它的概率 **快速推向** `p_max`，其余推向 `p_min`：

- `P_b ← P_b + β (p_max - P_b)`
- `P_i ← P_i + β (p_min - P_i)`  for `i ≠ b`

其中 `β` 控制追逐速度，`p_max` 常取 `1 - (k-1)p_min`。

为什么经典：
- 在非平稳场景，AP 往往比 PM 更快跟上阶段变化
- 又比纯贪婪稳定（因为仍保留 `p_min` 探索）

适用建议：
- 你有少量（10~50）算子池，并且希望“快速锁定当前阶段最佳算子”
- reward 噪声不是特别极端（否则 `argmax Q_i` 会抖动，可加窗口平滑）

---

## 5. 经典方法族 C：Multi-Armed Bandit（把算子选择正规化为探索-利用理论问题）

把每个算子看成一个 arm：

- 选择 arm `i` 得到随机 reward `r ~ D_i(t)`
- `D_i(t)` 可随时间变化（non-stationary）

### 5.1 经典 UCB1（上界置信）

经典论文（bandit 基石）：
- Auer, Cesa-Bianchi, Fischer (2002) “Finite-time Analysis of the Multiarmed Bandit Problem” DOI `10.1023/a:1013689704352`

UCB1 选择规则：

```text
i_t = argmax_i [ rbar_i + c * sqrt( ln(t) / n_i ) ]
```

其中：
- `rbar_i`：arm i 的平均 reward
- `n_i`：arm i 被选次数
- `c`：探索系数（理论版常为 `sqrt(2)`，工程上可调）

特点：
- 理论清晰
- 对非平稳不够鲁棒（需要加衰减/窗口）

### 5.2 Adversarial bandit / EXP3（对非平稳/对抗更鲁棒）

经典论文：
- Auer et al. (2002) “The Nonstochastic Multiarmed Bandit Problem” DOI `10.1137/S0097539701398375`

核心是 multiplicative weights：

- 维护权重 `w_i`
- 采样概率 `p_i ∝ w_i`
- 观察 reward 后对被选 arm 做指数更新：`w_i ← w_i · exp(η · r_hat_i)`

特点：
- 不假设 reward 稳定/独立同分布
- 在你怀疑“算子收益会频繁变”的场景很有用

### 5.3 AOS in EA 的经典：Dynamic MAB + Extreme value credit

代表性工作（把 bandit 明确用于 AOS）：

- Da Costa, Fialho, Schoenauer, Sebag (2008) “Adaptive operator selection with dynamic multi-armed bandits” DOI `10.1145/1389095.1389272`
- Fialho et al. (2009) “Dynamic Multi-Armed Bandits and Extreme Value-Based Rewards for Adaptive Operator Selection...” DOI `10.1007/978-3-642-11169-3_13`
-（同系列会议版本）DOI `10.1109/CEC.2009.4982970`

这里有两个关键点特别“经典”：

#### (1) 非平稳：dynamic bandit

不是用“全历史平均”，而是用滑动窗口/衰减，保证策略能快速响应阶段变化。

#### (2) credit 不是用 mean，而是用 extreme value（例如窗口内最大改进）

直觉：很多算子属于“偶尔爆发一次大改进，但平均很一般”。  
如果只看平均，容易把这些“救命算子”权重压到很低。  
用极值/上分位数做 credit，更符合“跳坑算子”的真实价值。

这对组合优化非常贴合：  
比如大扰动算子（ruin-recreate）可能 95% 时间没用，但那 5% 能跳出深坑。

### 5.4 非平稳（non-stationary）下的 bandit 经典处理套路

算子选择场景里，“reward 分布随时间变”几乎是常态（阶段切换）。因此很多 bandit 在工程上都会加“遗忘”机制。

常见套路（从简单到复杂）：

1. **指数衰减均值（Discounted / EMA）**  
   只保留近期信息：`Q ← (1-α)Q + α·r`  
   - 优点：实现最简单，几乎总能提升  
   - 缺点：α 的选择敏感；α 太小跟不上变化，太大就抖动

2. **滑动窗口（Sliding-window）**  
   `Q_i` 用最近 `W` 次该算子的 reward 统计（mean / max / quantile）  
   - 优点：更“硬”的遗忘，适合明显阶段切换  
   - 缺点：W 太小噪声大，太大反应慢

3. **Discounted-UCB / Sliding-window UCB**  
   把 UCB 的均值和计数都做衰减或只用窗口内样本（经典思路：在 UCB 里把 `n_i` 替换成“有效样本量”）。

4. **变点检测（Change-point detection）+ 重置**  
   检测到 reward 统计显著变化时，重置该 arm 的统计量（或提高探索）。  
   - 优点：对“突然变坏/突然变好”的阶段切换很灵敏  
   - 缺点：实现更复杂，需要统计检验/阈值

5. **Adversarial bandit（EXP3/Hedge）**  
   当你对 reward 的生成机制非常不确定时，用 multiplicative weights 的鲁棒性兜底。  
   - 直觉：不要估计“均值”，而是把选择当作在线学习问题，直接更新权重。

经验建议：
- 如果你第一次把 bandit 引入某个新领域，先做 (1)/(2) + `p_min` 下限，通常就足够把系统稳定下来。

### 5.5 Credit / Reward 设计的“经典坑”（算子选择里最常见的失败原因）

算子选择失败，很多时候不是 policy 不够高级，而是 reward 不可比或太噪声：

1. **算子耗时差异极大**：只用 `Δf` 会偏向慢算子（它有更多时间做事）。  
   - 经典修正：`Δf / time` 或在同等预算下比较（每步给相同 CPU 秒）

2. **强算子很“稀有爆发”**：均值奖励会压制“偶尔救命”的大扰动。  
   - 经典修正：窗口极值/上分位数（Fialho 系列）

3. **不同阶段 reward 尺度不同**：早期 Δf 大，后期 Δf 小，会导致策略在后期失真。  
   - 经典修正：用相对改进 `Δf / f(x_t)` 或对 reward 做阶段归一化

4. **多目标（质量+时间+可行率）**：单一标量 reward 很难兼顾。  
   - 工程常用：加权和、或分层（先保证可行率，再比质量，再比时间）

---

## 6. 经典方法族 D：Learning Automata（比 bandit 更“老派”，但概念很干净）

Learning Automata（LA）是更早期的随机决策自适应框架：维护一个动作概率向量 `p`，每次选动作并根据环境反馈更新 `p`。

常见更新族：

- **Linear Reward-Inaction (LRI)**：成功时提高概率，失败时不动
- **Linear Reward-Penalty (LRP)**：成功提高，失败降低

一个简化形式（选到动作 i 后）：

若成功（reward=1）：
- `p_i ← p_i + a (1 - p_i)`
- `p_j ← (1 - a) p_j` for `j ≠ i`

若失败（reward=0）：
- `p_i ← (1 - b) p_i`
- `p_j ← b/(k-1) + (1 - b)p_j`

LA 与 bandit 的关系：
- LA 可以看作 bandit 的一种在线概率更新算法（尤其在二值反馈场景）
- 工程上你会在很多“学习型超启发式”里看到它的影子

---

## 7. 经典方法族 E：Reinforcement Learning（从 bandit 升级到“有状态”的算子选择）

bandit 是 stateless：只用“哪个算子过去表现好”。  
但很多场景算子是否有效，强烈依赖于：

- 当前解的结构（比如 VRP 路线紧绷程度、TSP tour 的交叉情况）
- 当前阶段（早期/后期）
- 实例特征（规模、分布、约束紧张）

这时就自然进入 RL / contextual bandit。

### 7.1 Q-learning 选择算子（典型 selection hyper-heuristic）

MDP 形式：
- state `s_t`：由“搜索状态特征”组成（如：当前 best 与 current 的差、停滞步数、可行率、多样性、解结构统计…）
- action `a_t`：选择算子/邻域/扰动强度
- reward `r_t`：Δcost、Δcost/time、新 best 标志等

Q-learning 更新：

`Q(s_t,a_t) ← (1-α)Q(s_t,a_t) + α [ r_t + γ max_a Q(s_{t+1},a) ]`

优点：
- 能学到“不同状态用不同算子”的条件策略

缺点：
- 需要定义好 state features（否则学不动/泛化差）
- reward 设计和非平稳同样棘手

> 你现在做的“两层 gate”本质上就是一种 RL/conditional selection，只是动作空间在“初始化器/迭代器层级”。

### 7.2 Contextual Bandit（介于 bandit 和 RL 之间）

如果你认为“动作对未来状态影响很弱/可以忽略”（只关心单步回报），可以用 contextual bandit：

- 输入 context `c_t`（实例特征 + 当前解特征）
- 输出动作 `a_t`
- 只用即时 reward 更新（不回传 γ）

优点：更简单稳定；在很多 heuristic selection 上够用。

---

## 8. Parameter Control 与 Operator Selection 的统一（EA 经典视角）

经典综述：
- Eiben, Hinterding, Michalewicz (1999) “Parameter control in evolutionary algorithms” DOI `10.1109/4235.771166`

它把参数控制分为三类：

1. **Deterministic**：按时间表改参数（如退火温度曲线）
2. **Adaptive**：用在线反馈调参（如成功率驱动、可行率驱动）
3. **Self-adaptive**：把参数编码进个体/解里，由进化过程自调

算子选择可以被看作“对算子概率参数 `{p_i}` 的控制”：

- PM/AP/ALNS 权重更新：属于 adaptive parameter control
- 自适应交叉率/变异率：属于参数控制，但与算子选择完全同构

这个统一视角很有用：  
你在两阶段 pipeline 里做的“选哪个初始化器/迭代器”，也可以看成对“组件概率/组件预算”的控制问题。

---

## 9. 关联但更“外层”的经典：Algorithm Selection / Portfolio（Rice 框架）

虽然你问的是算子选择，但有一条相邻主线是 **算法选择（algorithm selection）**：

- Rice (1976) “The Algorithm Selection Problem” DOI `10.1016/S0065-2458(08)60520-3`

它关注的是：给定实例特征 `f(instance)`，选算法 `A`（或配置）使得性能最好。

与算子选择的关系：

- 把“算子”当成“微算法”，你就得到一个更细粒度、更在线的算法选择问题
- 你的神经 solver selection（实例级别选择 solver）是 Rice 框架的直接后裔
- 两者可以组合：外层选 solver / pipeline，内层再做 operator selection（算子自适应）

### 9.1 进一步关联：Algorithm Configuration / Tuning（离线“配方选择”，与算子选择互补）

很多时候你会发现：

- 算子选择解决的是 **在线** 决策（运行时根据反馈切换算子/参数）
- 但在运行前，你也想知道“哪些算子池/参数范围/权重更新规则”本身就更好

这就进入自动化算法配置（Auto Algorithm Configuration）：

- Hutter et al. (2009) “ParamILS: An Automatic Algorithm Configuration Framework” DOI `10.1613/jair.2861`
- Hutter et al. (2011) “Sequential Model-Based Optimization for General Algorithm Configuration”（SMAC）DOI `10.1007/978-3-642-25566-3_40`
- López-Ibáñez et al. (2016) “The irace package: Iterated racing for automatic algorithm configuration” DOI `10.1016/j.orp.2016.09.002`

它们与你的算子选择如何组合？

1. **离线配置 + 在线选择**：  
   - 先用 ParamILS/SMAC/irace 搜一个“好的默认算子池 + 更新超参（ρ/α/W 等）”  
   - 再在部署时用 bandit/HH/RL 做在线自适应

2. **把“算子选择策略本身”当成配置对象**：  
   例如比较 ALNS 权重更新 vs AP vs UCB，在离线层面做系统性选择（而不是手调）。

---

## 10. 工程落地指南（把“经典方法”变成你可以直接用的模块）

下面给一个“从零做算子选择系统”的最实用 checklist（不局限 CO）：

### 10.1 先把动作空间做对：算子要“可比较、可安全、可控预算”

1. **接口一致**：`operator(x) -> x'`（或 `-> (x', meta)`）
2. **可行性约束一致**：算子不能悄悄改变问题定义；要么保证可行，要么明确修复路径
3. **预算可控**：每个算子耗时差别大时，reward 必须纳入 time，或做 time-normalized credit

### 10.2 Credit 设计（最影响成败的部分）

推荐从简单到复杂：

1. `reward = max(0, f(x_t)-f(x_{t+1}))`（只奖励改进）
2. 加入 time：`reward = Δf / (time + ε)`
3. 加入“新 best bonus”：如果 `x_{t+1}` 打破历史 best，额外加常数
4. 对 heavy-tail 算子：用窗口极值/上分位数统计（Fialho 系列思想）

### 10.3 Selection policy 选型建议（给你一个很实用的经验排序）

如果你想要“少调参、稳健、可解释”：

1. **ALNS 风格权重更新（score + smoothing）**（最工程化）
2. **Adaptive Pursuit**（更快切换阶段）
3. **UCB / Sliding-window UCB**（想要更理论化的探索）
4. **EXP3 / multiplicative weights**（非平稳很强、reward 噪声大）
5. **Contextual bandit / RL**（你有好的 state 特征，且希望学条件策略）

### 10.4 非平稳处理（不要忽视）

至少做一个：

- 指数衰减：`Q ← (1-α)Q + α·reward`
- 或滑动窗口均值/极值

并且建议：
- 把“搜索阶段特征”显式放进 state（如停滞步数、当前温度、距终止剩余时间）

### 10.5 诊断与可视化（非常关键）

建议你每次跑实验都记录：

- 每个算子被选次数 `n_i(t)` 的曲线
- `P_i(t)` 的曲线（是否塌缩到单一算子）
- `reward` 分布（是否极端值主导）
- “按规模 n 分桶”的胜率/收益（你已经在 notebooks 里做了 win-bucket，这个思想完全可以迁移到 operator level）

### 10.6 必做对照组（否则你很难判断“选算子”到底有没有带来收益）

算子选择研究里，一个非常常见的陷阱是：只报告“最优策略很好”，但不知道提升来自哪里。建议至少包含：

1. **Uniform random**：每轮均匀随机选算子（最强 sanity baseline）
2. **Round-robin**：按固定顺序轮换算子（检验“仅靠覆盖算子池”能提升多少）
3. **Greedy best-so-far**：永远选历史平均 reward 最好的算子（检验探索是否必要）
4. **ALNS-score** vs **AP** vs **UCB/EXP3**：三类代表性方法（工程、追逐、理论 bandit）

并建议做 2 个 ablation：

- 去掉 time-normalization（看是否出现“慢算子霸权”）
- 去掉 `p_min`（看是否出现“早熟塌缩”）

### 10.7 评估指标（operator selection 的“正确打开方式”）

很多时候只看最终 best value 会误导，因为 operator selection 影响的是 **anytime 行为**：

1. **Anytime curve / AUC**：`best-so-far(t)` 曲线的面积（越低越好，最小化问题）
2. **Time-to-target**：达到某个质量阈值所需时间（更像工程 KPI）
3. **Δquality per second**：单位时间平均改进（特别适合比较快慢算子）
4. **稳定性**：多随机种子下的方差/分位数（bandit/RL 常有高方差）

统计检验（可选但推荐）：

- **paired test**：同一实例/同一随机种子配对比较（例如 Wilcoxon signed-rank、sign test）
- **多实例胜率**：win/tie/loss（你现在做的 per-instance win count 就是很好的形式）

---

## 11. 推荐阅读清单（按“经典程度/实用性”排序）

### 11.1 算子选择 / 超启发式（强相关）

- Burke et al. 2013, Hyper-heuristics survey, DOI `10.1057/jors.2013.71`
- Ropke & Pisinger 2006, ALNS, DOI `10.1287/trsc.1050.0135`
- Pisinger & Ropke 2007, VRP general heuristic, DOI `10.1016/j.cor.2005.09.012`
- Thierens 2005, Adaptive Pursuit, DOI `10.1145/1068009.1068251`
- Ochoa et al. 2012, HyFlex framework, DOI `10.1007/978-3-642-29124-1_12`

### 11.2 Bandit / online learning（提供理论与更鲁棒策略）

- Auer et al. 2002, UCB1 analysis, DOI `10.1023/a:1013689704352`
- Auer et al. 2002, EXP3 (nonstochastic bandit), DOI `10.1137/S0097539701398375`
- Da Costa et al. 2008, Dynamic MAB for AOS, DOI `10.1145/1389095.1389272`
- Fialho et al. 2009, Dynamic MAB + extreme value credit, DOI `10.1007/978-3-642-11169-3_13`

### 11.3 参数控制/外层算法选择（概念邻居）

- Eiben et al. 1999, Parameter control in EA, DOI `10.1109/4235.771166`
- Rice 1976, Algorithm selection problem, DOI `10.1016/S0065-2458(08)60520-3`
- Mladenović & Hansen 1997, VNS, DOI `10.1016/S0305-0548(97)00031-2`
- Battiti & Tecchiolli 1994, Reactive tabu search, DOI `10.1287/ijoc.6.2.126`

---

## 12. 和你当前研究（两阶段 pipeline + gate）怎么接起来？

你现在的视角是：

- Gate1：选择初始化器（init）
- Gate2：选择迭代器（iter）

算子选择文献给你的两条“可直接落地”的增强路径：

1. **把 Gate2 的动作从“选 iter”细化为“iter 内部算子选择”**：  
   例如把 LEHD/GLOP/传统算子拆成算子池，使用 ALNS/AP/UCB 做在线选择（reward = Δlength / time）。

2. **把 Gate2 做成 contextual bandit**：  
   context = `(instance features, sol0 features, n, slack indicators, stagnation)`  
   action = 选择邻域/扰动强度  
   reward = 最终改进幅度（或 fixed steps 后的改进）

你在 notebooks 里做的 **n 分桶 win 分析**，可以直接复用：  
把“方法”换成“算子”，就能看出算子对规模/结构的偏好性，从而指导你的动作空间设计。
