# 05) A POPMUSIC matheuristic for the capacitated vehicle routing problem（Queiroga, Sadykov, Uchoa, 2021）

- 论文：Eduardo Queiroga, Ruslan Sadykov, Eduardo Uchoa, *A POPMUSIC matheuristic for the capacitated vehicle routing problem*, Computers & Operations Research 136 (2021) 105475
- DOI：`10.1016/j.cor.2021.105475`
- 本地 PDF：`adaptive/pdfs/05_popmusic_matheuristic_cvrp_2021.pdf`
- 预印本（HAL）：`https://hal.inria.fr/hal-02994210/file/main-clear.pdf`

## 0.5 代码可获得性（论文是否给出开源链接）

- 在论文正文中未检索到官方开源仓库/代码下载链接（GitHub/GitLab/Bitbucket 等）。
- 该方法实现强依赖 VRPSolver/BCP/CPLEX 等组件；就算复现，也更像“系统工程”（子问题生成 + BCP 参数化 + 求解器集成），而不是单个启发式文件即可完成。

### 0.5.1 相关开源组件（论文未提供 POP 代码，但给出了可用的 BCP 基座）

论文的子问题求解器是 “BCP H”，并明确是基于 VRPSolver（Pessoa et al.）的 CVRP demo 做参数化的。因此虽然 POP 本身没开源，但你想落地一个“POPMUSIC + BCP”基线，最现实的路线通常是：

- **VRPSolver（BCP 框架）**：论文里给了官网入口 `https://vrpsolver.math.u-bordeaux.fr/`（实际工程通常还需要 CPLEX/商业 MILP 求解器许可）
- **VRPSolverEasy（Python 简化接口）**：`https://github.com/inria-UFF/VRPSolverEasy`（不是论文 POP 的实现，但能让你快速把“子问题=某类 VRP”跑通，理解 BCP 的输入/输出与时间限制方式）

> 这两者更像“你自己实现 POPMUSIC 外层循环时要调用的黑盒/基座”，而不是 POPMUSIC 本身的代码。

## 0. 一句话总结（你读完应当留下的印象）

这篇做了一个非常“硬核但干净”的 iteration 组件：  
**把 CVRP 的“邻域搜索”直接升级成“解的局部区域 → 子问题 → 用 BCP（branch-cut-and-price）当强启发式深搜 → 回填”**。  
它不是 ALNS 那种轻量 ruin&recreate，而是 POPMUSIC：**在特定条件下反复触发“局部精确/半精确优化”**，从而在长时间预算下持续挤出改进。

> 在你的两阶段视角里：POPMUSIC 基本可以视作一个“超强迭代器（iteration）”，前提是你能提供一份不错的初解，并愿意支付“子问题求解器”的开销。

## 1. 论文贡献点（按“你复现/移植时最有价值的点”整理）

1. **子问题定义方式很特别（Algorithm 1 的 Lines 10–19）**：不是按“最近邻客户集合”取子问题，而是按“与 seed 距离最近的 routes”逐条加入，直到达到目标规模 `dimsp`。
2. **用缓存 Π 避免重复求解（Line 21）**：如果当前子问题 `Vsp` 已被某个更大子问题 `V'` 覆盖（`Vsp ⊆ V'`），就跳过。
3. **“小子问题先解”是为了加速后续大子问题**：因为 BCP 的很多加速（边删除、路径枚举）都强依赖 upper bound（gap 小 → 更快）。
4. **把 BCP 变成启发式（BCP H）的方法非常具体**：  
   - 限制 B&B 节点数与时间  
   - 引入 false gap（人为缩小 gap，用于 edge elimination 与 route enumeration）  
   - 引入 restricted master heuristic（把枚举出的 10k 条 route 做成 IP 给 CPLEX）
5. **可复现性强**：作者直接基于 VRPSolver 的 CVRP demo（Pessoa et al. 系列 BCP）做参数化，并报告了 `α, δ` 等关键参数的校准结果。

## 2. POPMUSIC 的整体框架（Algorithm 1, 文中简称 POP）

### 2.1 输入/输出与核心状态

输入：

- 一个初始解 `S`（包含 routes）
- 一个子问题求解器 `A`（本文用 BCP H）
- 两个控制子问题规模的参数：`α`（初始 target dimension）与 `δ`（步进）

输出：

- 改进后的解 `S`（或原解，若没有找到更优子解）

内部关键状态：

- `dimsp`：当前目标子问题规模上限（子问题客户数 |Vsp| ≤ dimsp）
- `Π`：缓存集合，存放已经求解过的子问题及其最佳子解：`Π = {(V', S')}`

### 2.2 外层循环：dimsp 从小到大（Line 6, 31）

主循环逻辑是：

1. 从 `dimsp = α` 开始
2. 在固定 dimsp 下，尝试很多 seeds（遍历所有客户，只是顺序随机）
3. 只要找到一次改进，就 **回到同一个 dimsp 重新开始**（intensification）
4. 当“所有 seed 都不能改进”时，才让 `dimsp += δ`（扩大邻域）

这非常像一种“以子问题规模为层级的 VND”：

- 小 dimsp：便宜、容易解、频繁改进
- 大 dimsp：贵，但能跳出更深局部最优

#### 2.2.1 按论文 Algorithm 1 写成“可直接实现”的伪代码

下面这段和论文 Algorithm 1 一一对应（但我用更工程化的变量名复述），你照着写基本不会偏：

1. `dimsp = α`
2. `Π = ∅`（缓存：已求解的子问题与其最好子解）
3. while `time_left > 0` 且 `dimsp ≤ |V+|`：
   - `L = random_permutation(V+)`（客户顺序随机洗牌）
   - for `i in L`（逐 seed）：
     - 构造子问题：
       - `Vsp = ∅; R_used = ∅`
       - while `|Vsp| < dimsp`：
         - `r_hat = argmin_{r ∈ S, r ∉ R_used} min_{j∈C(r)} c(i,j)`（seed 到 route 的最近距离）
         - 若 `|Vsp| + |C(r_hat)| ≤ dimsp`：`Vsp ← Vsp ∪ C(r_hat)`；`R_used ← R_used ∪ {r_hat}`
         - 否则 break（不允许“半条 route”进子问题）
     - 若 `Vsp` 不被任何缓存子问题覆盖（`∀(V',·)∈Π: Vsp ⊄ V'`）：
       - `Ssp = Projection(S, Vsp)`（见 §2.5.1：在本文子问题构造下，Projection 实际上就是抽取若干条完整 route）
       - `Π.add((Vsp, Ssp))`
       - 用子问题求解器 A（BCP H）求解 `Vsp`，并把 `UB_init = cost(Ssp)` 作为初始上界输入
       - 若得到更优子解 `Ssp'`：
         - 更新缓存：用 `(Vsp,Ssp')` 覆盖 `(Vsp,Ssp)`
         - 回填全局解：`S ← ReplaceSubsolution(S, Vsp, Ssp')`
         - **restart：回到本轮 while 的开头（重新洗牌 L，且 dimsp 不变）**
   - 若这轮 for seeds 完全没有改进：`dimsp += δ`

POP 的“特殊 intensification 条件”就是这个 restart：**只要某个子问题改进了，就立刻在相同 dimsp 上重新扫一遍全部 seed**，直到这个 dimsp 下完全榨不出改进。

### 2.3 单个 seed 如何构造子问题（Line 10–19）

给定 seed 客户 `i`：

- 初始化 `Vsp=∅`，`R=∅`（R 用来记录已经加入的 routes，避免重复）
- while `|Vsp| < dimsp`：
  1. 在当前解 S 的 routes 中选一条 “离 seed i 最近且未用过”的 route `r̂`：  
     距离定义为 `min_{j∈C(r)} c_{ij}`（seed 到该 route 上任意客户的最短边）
  2. 如果 `|Vsp| + |C(r̂)| ≤ dimsp`：把这条 route 的客户集合并进 `Vsp`，并把 r̂ 放进 R
  3. 否则：停止（不再添加更多 routes）

因此，子问题不是“一个球形邻域”，而是“若干条与 seed 几何接近的 route 的并集”。  
这会让子问题天然带有“当前解的 route 结构”，更像是在优化“局部 route 群落”。

### 2.4 什么时候真的去求解这个子问题（Line 21）

只有当：

- 对所有 `(V', S') ∈ Π` 都有 `Vsp ⊄ V'`

才求解 `Vsp`。

直觉：如果更大的子问题 V' 已经被求解并缓存过，那么 Vsp 作为其子集通常“改进空间被覆盖”，再解一遍收益很低，直接跳过省时间。

### 2.5 如何把子解回填到全局解（Line 22–28）

记：

- `Ssp`：当前解 S 在子图 `{0}∪Vsp` 上的“投影子解”（即取出 Vsp 中客户在 S 中形成的 routes）
- 用算法 A（BCP H）在子问题上找一个更好的子解 `Ssp'`

若 `cost(Ssp') < cost(Ssp)`：

1. 更新缓存 Π 中该子问题的最好子解
2. 用 `Ssp'` 替换全局解 S 中对应的子结构（其它不在 Vsp 的 routes 保持不变）
3. **立刻回到 Line 7 重新生成 seed 顺序**（同一个 dimsp 再来一轮）

这其实隐含了一个策略：  
一旦找到局部改进，就重新洗牌 seeds，让搜索在当前规模下更充分地“挤干”改进空间。

#### 2.5.1 一个关键实现简化：Vsp 是“若干条完整 routes 的并集”

注意 Algorithm 1 的构造规则（Line 15–19）：只有当 `|Vsp| + |C(r_hat)| ≤ dimsp` 才把整条 route 加进来；否则直接停止，不会截断 route。

因此在 POP 的定义下：

- `Vsp = ⋃_{r∈R_used} C(r)`（严格是 route-union）
- `Ssp`（子解投影）可以直接实现为：**从全局解 S 中抽出 `R_used` 这几条 route**（然后把它们当作子问题的初始解给 BCP H）
- 回填也可以直接实现为：**从 S 删除 `R_used` 这几条 route，把 `Ssp'` 的 routes 插回去**（其它 routes 原样保留）

这点非常重要：它避免了“从一条 route 中切掉部分客户”带来的复杂一致性问题，使得 POP 的子问题接口非常干净。

#### 2.5.2 缓存 Π 的工程实现（Line 21）

论文的缓存条件是：如果 `Vsp ⊆ V'` 且 `(V',·) ∈ Π`，就跳过 `Vsp`。要让它在工程上不成为瓶颈，通常需要：

- **集合表示**：用 bitset（|V+|≤1000 时很合适）或排序数组（稀疏 set）表示 `Vsp`
- **快速子集判断**：
  - bitset：`Vsp ⊆ V'` 等价于 `(Vsp & ~V') == 0`
  - 排序数组：双指针线性扫
- **缓存条目数控制**：Π 会随着运行不断增长；实践里可以按 `dimsp` 分桶、或只保留“较大 V'”（因为它们能覆盖更多 Vsp）

论文没有强制给出 Π 的淘汰策略，但从原理上讲：Π 越大越省“重复求解”，也越贵；这正是一个典型的“记忆预算”权衡点。

## 3. 子问题求解器：把 BCP 变成启发式 BCP H（Section 3）

核心动机：POPMUSIC 不追求证明最优，而追求“把子问题解得很深，但时间可控”。  
作者把 Pessoa et al. 系列的 BCP（VRPSolver demo）改造成启发式，关键改动有三类：

### 3.0 你需要知道的 BCP “最小知识集”（为什么它适合作为子问题强求解器）

论文假定读者熟悉 VRPSolver/BCP，但对复现 POP 来说，你只需要抓住这几点：

- **Master（RMP）**：CVRP 常用 set partitioning 形式，变量是一条可行 route（覆盖一组客户）；目标最小化 route 成本；约束确保每个客户恰好被覆盖一次。
- **Pricing**：在当前对偶价格下，找 reduced cost 为负的可行 route（本质是带资源约束的最短路/ESPPRC）。
- **Cutting + Branching**：在 B&B 结点里做 column+cut generation，必要时加 cut（论文提到 rank-1 cuts、rounded capacity cuts 等），再分支。

POPMUSIC 的关键是：它不需要 exact，因此可以用 false gap 与节点/时间限制把 BCP “拉成一个很强的深度局部搜索器”。

### 3.1 硬预算限制（最直观）

- B&B 节点数上限：10
- 单个子问题时间上限：3600 秒

这保证 POP 不会被一个子问题卡死。

### 3.2 False gap 机制（最关键的加速点）

BCP 的两个重型加速：

- **edge elimination（删边）**：用 reduced cost + gap 做安全删边
- **path/route enumeration（路径枚举）**：枚举 reduced cost 小于 gap 的路径，之后用“检查表”加速 pricing，甚至构建 IP

它们都依赖 “gap = UB − LB”。

作者提出：用一个 **人为缩小的 gap** 来更激进地删边/枚举：

- `FG = (UB − LB) / FGF`
- 经验上用 `FGF = 3`

这会带来风险：可能删掉其实需要的边/路径（因此不再是 exact）。  
但作者实证显示：用温和的 FGF（例如 3）很少导致错过改进解，却能显著提速。

### 3.3 Restricted master heuristic（把枚举 route 做成 IP）

在每个 B&B 节点，当 column+cut generation 收敛后：

1. 逐步进一步减小 false gap（每次除以 2），直到 path enumeration 能完成
2. 从枚举出的 routes 里取 reduced cost 最小的 10,000 条
3. 用这些 routes 建一个 set partitioning / master IP，交给 CPLEX 求解

直觉：  
“不用等 BCP 把所有事做完”，只要把有希望的 route 候选集捞出来，让 MIP 在一个可控规模的候选集上做一次强组合优化即可。

### 3.4 作为 POP 子求解器时，BCP H 的输入输出接口（实现视角）

从 POP 的角度看，子求解器 `A`（BCP H）只需要满足很简单的接口：

- 输入：
  - 子问题客户集合 `Vsp`（以及它们的 demand）
  - `Q`、`c(i,j)`（距离/成本）
  - 初始上界 `UB_init = cost(Ssp)`（论文强调：给一个好的 UB 会显著提速）
  - 预算：`time_limit`、`node_limit`、`FGF`（false gap factor）等
- 输出：
  - 若在预算内找到改进：返回 `Ssp'`（routes 覆盖 Vsp）及其 cost
  - 否则：返回 “无改进”（POP 继续换 seed）

实现上，你甚至不需要让 BCP H “返回证明信息”，因为 POP 从不使用最优性证明，只比较 `cost(Ssp') < cost(Ssp)`。

## 4. 为什么 POP 要“先解小子问题再解大子问题”（一个很重要的因果链）

BCP 的很多加速都依赖 UB：

- UB 越好 → gap 越小
- gap 越小 → edge elimination 更 aggressive、route enumeration 更容易成功
- enumeration 成功 → pricing 变快、甚至能直接构 IP → 求解速度更快

因此 POP 的策略是：

1. 先用小 dimsp（容易求解）不断改进 UB
2. 再逐渐放大 dimsp，让更大的子问题“吃到更好的 UB 红利”

这就是 POPMUSIC 的“special intensification conditions”在工程上的体现：  
不是随便求子问题，而是把子问题求解器放在一个“更容易发挥”的环境里。

## 5. 关键参数与作者给出的推荐

### 5.1 POP 的子问题规模参数（Section 4.4）

作者对 `α, δ` 做了实验校准（8 小时预算），结论是：

- `α = 50`
- `δ = 40`

在 2/4/8 小时点的平均 gap 最好，因此后续实验固定用该组合。

### 5.2 BCP H 的关键参数（Section 3 / 4.3）

- `FGF = 3`（false gap factor）
- `MaxNbOfBBtreeNodeTreated = 10`
- `GlobalTimeLimit = 3600s`

以及 VRPSolver demo 的“修改过的参数集”（论文文字说明它比默认 exact 参数更适合频繁求小子问题）。

## 6. 这篇里的 “adaptive / algorithm selection” 应该怎么理解？

它不是传统意义的“学习型算子选择”，但确实存在“动态调度”：

1. **子问题选择是 solution-dependent 的**：同一个 seed i，在不同的当前解 S 下会得到不同的子问题（因为 routes 变了）。
2. **子问题规模是自适应推进的**：只有当当前 dimsp 下完全无改进，才扩大 dimsp。
3. **缓存 Π 是一种记忆机制**：避免重复解已经被“覆盖”过的局部区域。

如果你要把它纳入“可上手 baseline”的算子选择框架，你可以把它拆成几个可被选择的动作：

- 选 `dimsp`（或选 α/δ/schedule）
- 选 seed 策略（随机、按潜力排序、按困难度排序）
- 选“子问题构造规则”（按 route 距离加入 vs 按客户聚类加入）
- 选子问题求解预算（time/node/false-gap factor）

这些都能用 bandit/RL 做在线选择，而不需要离线枚举全部组合标注。

## 7. 和你的“两阶段选择（init→iter）”如何对齐

### 7.1 POPMUSIC 更像哪个阶段？

几乎完全是 **iteration**：

- 它假设你已经有一个完整解 S（来源可以是任何初始化器或其它求解器）
- 它做的是“在局部区域上深度优化并回填”

### 7.2 能否和不同初始化器组合？

原理上可行，接口要求：

- 你能把任意初解表示成 routes，并能提取某个 `Vsp` 的子解 `Ssp`
- 你能把子解 `Ssp'` 无冲突地回填到全局解（保持其它客户覆盖不变）

### 7.3 对你后续“组合证明”有什么帮助？

POPMUSIC 是一个很强的“后处理迭代器”，天然会出现：

- 某些初解（来自某个 init）更容易让 POP 找到可改进子问题（更“可优化”）
- 某些初解虽然 gap 小，但结构偏某类局部最优，POP 的局部精确优化反而“吃不进去”

这类现象非常适合用你要做的“实例特征 + 两阶段选择”来解释与利用。
