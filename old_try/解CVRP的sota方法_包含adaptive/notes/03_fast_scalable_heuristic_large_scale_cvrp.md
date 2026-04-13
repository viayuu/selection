# 03) A Fast and Scalable Heuristic for the Solution of Large-Scale CVRP（FILO, Accorsi & Vigo, 2021）

- 论文：Luca Accorsi, Daniele Vigo, *A Fast and Scalable Heuristic for the Solution of Large-Scale Capacitated Vehicle Routing Problems*, Transportation Science 55(4), 832–856 (2021)
- DOI：`10.1287/trsc.2021.1059`
- 本地 PDF：`adaptive/pdfs/03_fast_scalable_heuristic_large_scale_cvrp_2021.pdf`

## 0.5 代码（开源链接 + 本地下载）与论文-代码对照

- 论文 Section 3.1 明确给出下载入口：`https://acco93.github.io/filo/`，并指向两个仓库：
  - FILO 主程序：`https://github.com/acco93/filo`（本地：`adaptive/repos/03_filo_acco93_filo/`，commit `d4b68db98d7ab1ac7afb4d7706bbe3aa4f4bc774`）
  - COBRA 可复用组件库：`https://github.com/acco93/cobra`（本地：`adaptive/repos/03_filo_acco93_cobra/`，commit `58513f3d4ddda98815a3899b43111c2025fde14a`）
- 代码入口与组件映射（最关键的几个文件）：
  - `adaptive/repos/03_filo_acco93_filo/main.cpp`：完整 pipeline（Construction → RouteMin → CoreOpt/SA）
  - `adaptive/repos/03_filo_acco93_filo/arg_parser.hpp`：参数默认值与 CLI tokens（`--granular-gamma-base/--granular-delta/--shaking-*/--coreopt-iterations` 等）
  - `adaptive/repos/03_filo_acco93_filo/routemin.hpp`：ROUTEMIN（压 route 数）阶段
  - `adaptive/repos/03_filo_acco93_filo/RuinAndRecreate.hpp`：COREOPT 的 per-customer `ω` 破坏（walk-based ruin + greedy recreate）
  - `adaptive/repos/03_filo_acco93_cobra/include/cobra/Solution.hpp` + `.../LRUCache.hpp`：SVC（“recently touched vertices”缓存，`get_cache()/clear_cache()`）
  - `adaptive/repos/03_filo_acco93_cobra/include/cobra/MoveGenerators.hpp`：KNN granularity / move generators
  - `adaptive/repos/03_filo_acco93_cobra/include/cobra/LocalSearch.hpp`：RVND/HVND 与 move types 枚举（FILO 通过接口组装）

### 0.5.1 从源码看：FILO 的几处“关键自适应”到底怎么写

- **HRVND 两层**：代码里是两个 `RandomizedVariableNeighborhoodDescent`（`main.cpp`）：
  - `rvnd0`：大量 `E11/E10/TAILS/SPLIT/TWOPT/RE..` 等二次规模邻域
  - `rvnd1`：只含 `EJCH`（ejection chain）
  - 再用 `cobra::HierarchicalVariableNeighborhoodDescent.append()` 串成“tier1→tier2”的 hVND。
- **per-vertex 稀疏化 `γ_i`**：`gamma` 向量初始化为 `gamma_base`（默认 0.25），并为每个“近期被触及顶点”维护 `gamma_counter[i]`。当 `gamma_counter[i] >= max_non_improving_iterations` 时：
  - `gamma[i] = min(2*gamma[i], 1.0)`（逐步放宽到全邻域）
  - 仅对该点即时 `move_generators.set_active_percentage(gamma, {i})`（避免全局重建）。
- **`max_non_improving_iterations` 的计算**：源码用
  - `ceil(delta * total_budget * avg(|cache|) / n)`（`avg(|cache|)` 用 `cobra::Welford` 对 `neighbor.get_cache().size()` 的在线均值估计）
  - 直觉：SVC 区域越大（cache 越大）→ 允许更久不改进才放宽 γ。
- **per-customer 破坏强度 `ω_i`**：初始化 `omega_base = ceil(log(n))`，每次 `RuinAndRecreate::apply(neighbor, omega)`：
  - 以 seed 的 `ω_seed` 做“随机游走式 removal”（路内走或跳到相邻 route）
  - 再 greedy reinsert
  - 根据本轮 local optima `neighbor` 相对 `solution` 的质量落在 `[solution_cost + lb, solution_cost + ub]` 的哪一段，对本轮“被触及（cache）”客户的 `ω` 做 `±1` 微调（过弱→增大、过强→减小、中间→随机微调）。
- **SA 接受准则与强度重标定**：`cobra::SimulatedAnnealing` 的温度从 `mean_arc_cost/10` 衰减到 `/100`；只要 SA 接受邻居，就把 `solution ← neighbor`，并据此用“当前解的 mean arc cost”重标定 `shaking_lb_factor/shaking_ub_factor`（跟论文“自适应强度随解尺度变化”一致）。

### 0.5.2 关键源码片段（直接对应论文的自适应闭环）

**(1) γ 的 per-vertex 放宽**（`adaptive/repos/03_filo_acco93_filo/main.cpp`）：只对 `neighbor.get_cache()` 里“最近被触及”的顶点计数；超过阈值就把该点的 `γ_i` 翻倍（上限 1.0），并只对这个点更新 move generator 的活跃比例（避免全局重建）。

```cpp
for (auto i = neighbor.get_cache().begin(); i!=cobra::LRUCache::Entry::dummy_vertex; i = neighbor.get_cache().get_next(i)) {
    gamma_counter[i]++;
    if (gamma_counter[i] >= max_non_improving_iterations) {
        gamma[i] = std::min(gamma[i] * 2.0f, 1.0f);
        gamma_counter[i] = 0;
        gamma_vertices = {i};
        move_generators.set_active_percentage(gamma, gamma_vertices);
    }
}
```

**(2) ω 的“过强/过弱/居中”反馈微调**：根据 shaken solution `neighbor` 相对当前 `solution` 的质量落在不同区间，对本轮被 ruin 的客户集合 `ruined_customers` 做 `±1` 微调（围绕 seed 的 `ω_seed` 做夹紧）。

```cpp
if (neighbor.get_cost() > shaking_ub_factor + solution.get_cost()) {
    for (auto i : ruined_customers) if (omega[i] > omega[walk_seed] - 1) omega[i]--;
} else if (neighbor.get_cost() >= solution.get_cost() && neighbor.get_cost() < solution.get_cost() + shaking_lb_factor) {
    for (auto i : ruined_customers) if (omega[i] < omega[walk_seed] + 1) omega[i]++;
} else {
    // 中间区间：随机增/减
}
```

## 0. 一句话总结（你读完应当留下的印象）

FILO 的核心不是“又发明了一个邻域”，而是把 **大规模 CVRP 的工程约束**（每一步必须极快、而且还要能跳出局部最优）系统化成一套可复现的 ILS 框架：  
**受限邻域（GN）+ 静态 move 描述（SMD）+ 选择性缓存（SVC）+ 动态稀疏化（vertex-wise γ）+ 自适应破坏强度（per-customer ω）+ SA 接受准则**，最终让“强局部搜索”在 500–1000（乃至更大）规模上还能跑得动。

> 你做两阶段组合（init + iter）时：FILO 绝大部分创新都属于 **迭代阶段（iteration）** 的“改进器设计与自适应调度”，而不是初始化。

## 1. 贡献点（按“你复现/移植时最有价值的点”整理）

1. **局部搜索架构：HRVND（Hierarchical RVND）两层**  
   - tier1：大量“二次规模”的 inter/intra 操作混合（randomized order）  
   - tier2：只放一个最贵但强的 `ejch`（ejection chain）  
   整体像“VND 的层级 + 每层内部 RVND”，兼顾速度与跳坑能力。
2. **GN + SMD 的组合**：把“枚举邻域”替换为“维护可改进 move 的数据结构”，将局部搜索的 *每轮成本* 控制住。
3. **SVC（Selective Vertex Caching）**：不是全局做 LS，而是“只对近期被打乱的区域做深优化”，形成一种隐式分解（localized optimization）。
4. **更细粒度的动态稀疏化**：不是一个全局稀疏阈值，而是 **每个顶点一个稀疏因子 γᵢ**；配合“哪些顶点长期无改进就放宽候选”这种局部自适应。
5. **自适应 shaking 强度**：每个客户 i 一个 ωᵢ（被选为 seed 时决定破坏步数），用“破坏太弱/太强/刚好”的反馈闭环调整。
6. **可解释的速度-质量折中**：文中对关键参数（缓存大小 C、GN 是否包含 depot 边、γ 的管理、ω 的调度）都有组件级实验分析，适合作为你做选择/组合时的 OR baseline 参考点。

## 2. FILO 整体框架（Algorithm 1）

高层流程非常干净：

1. `S ← Construction()`：构造一份可行（或近可行）初解（restricted savings）。
2. `k ← GreedyRoutesEstimate(I)`：用需求做 bin packing 下界估计“理想 route 数 k”。
3. `if |S| > k: S ← RouteMin(S)`：可选的“压车”阶段（route minimization）。
4. `S ← CoreOpt(S)`：核心 ILS：自适应 ruin&recreate + HRVND + SA 接受准则。

你可以把它拆成两块（非常贴合你的两阶段视角）：

- **Initialization（初始化器）**：`Construction` +（可选）`RouteMin`
- **Iteration（迭代改进器）**：`CoreOpt`

### 2.1 Construction：受限 Clarke–Wright savings（Section 2.1）

论文把 Construction 明确当作“只跑一次”的轻量组件，但仍给出了 **如何把二次 savings 线性化** 的实现细节：

1. 为每个客户 `i` 预先计算其最近邻集合 `N_{n_cw}(i)`（Euclidean 距离，或实例给定距离矩阵）。
2. 只对受限 pair 计算 savings（避免 i↔j 对称重复）：
   - 论文做法：只考虑 `j ∈ N_{n_cw}(i)` 且 `i < j` 的 pair（用 lexicographic order 破除对称）
3. 对每个候选 pair `(i,j)` 计算 Clarke–Wright savings：
   - `s(i,j) = c(0,i) + c(0,j) - c(i,j)`（0 为 depot）
4. 按 `s(i,j)` 降序遍历，尝试把包含 i 与 j 的两条当前 route 合并（经典 CW：只有当 i/j 分别处在各自路线的端点时才允许无交叉合并），并保持容量可行。

实现时的“工程点”：

- `N_{n_cw}(i)` 可以复用后续 GN 需要的近邻表（只是 `n_cw` 往往比 `ngs` 大）。
- 论文正文中曾提到过 `n_cw=100` 的经验设置，但 **最终 Table 1 的 tuned 参数给的是 `n_cw=50`**；你复现时建议以 Table 1 为准。

## 3. 局部搜索引擎：HRVND（Algorithm 2–3）——“分层 + 每层随机”

### 3.1 HRVND 的组织方式

- 把算子集合拆成多个 **tier（层）**，tier 按“计算成本/规模”排序（像 VND）。
- 每个 tier 内部是 RVND：每次执行前随机打乱算子顺序；每个算子“彻底搜一遍直到局部最优”；一旦找到改进，就继续在同一 tier 内推进（本质是 RVND 的循环）。
- 只有当解对前面 tiers 都局部最优，才进入下一 tier；如果高 tier 找到改进，回到第一 tier（像 VND 的 restart）。

### 3.2 FILO 使用的 tiers（非常重要：它解释了“为什么快”）

论文把算子组织成 **两层**：

- **Tier 1（大量二次规模邻域，执行时间相近）**：  
  `10ex, 11ex, split, tail, twopt, 20ex, 21ex, 22ex,`  
  以及各种 `rex` / `rex*`（CROSS-exchange 路径反转变体），还包含 `30ex,31ex,32ex,33ex` 等更大交换。  
  直觉：用“很多便宜而互补”的 inter-route + intra-route 动作把局部最优推深。
- **Tier 2（最贵但强）**：  
  `ejch`（ejection chain：本质是“用 10ex 组成的一串连锁搬移”，探索受限树，参数 `nEC` 控制探索规模）。

> 你要做“算子选择/组合”的时候：HRVND 已经是一个手工设计的“算子调度策略”。你可以把 HRVND 里的每个算子当成 Gate2 的动作候选，也可以把 Tier 当成 coarse action（先选 tier，再在 tier 内选算子）。

### 3.3 按论文伪代码复现 HRVND（Algorithm 2–3 的实现等价形式）

论文把每个 tier 的算子存在一个“循环 list”里，并用两个指针 `e`（最近一次改进发生在哪个算子）与 `c`（当前算子 index）实现“只要还在改进就继续扫”。

**TierApplication（Algorithm 2）** 的实现骨架（我用更直白的变量名复述）：

1. 将 tier 内算子列表 `O` shuffle 一次。
2. `e = 0; c = 0`
3. repeat：
   - 对算子 `O[c]` 做“一次完整 neighborhood search”（注意：这一步内部是 SMD 驱动的，见 §4.3）
   - 若得到改进解：更新当前解，并令 `e = c`（记录“改进来自哪个算子”）
   - `c = (c + 1) mod len(O)`
4. until `c == e`（意味着已经绕一圈回到最后一次产生改进的算子后，期间再也没有改进）

**hrvnd（Algorithm 3）** 的实现骨架：

1. `tier_idx = 0`
2. while `tier_idx < len(T)`（T 为 tiers 列表）：
   - `S' = TierApplication(S, T[tier_idx])`
   - 若 `S'` 改进且 `tier_idx > 0`：`S = S'` 且 `tier_idx = 0`（VND 风格 restart）
   - 否则：`tier_idx += 1`

这个写法对复现很关键：它说明 HRVND 的随机性只发生在 **tier 内 shuffle**，tier 之间是固定顺序；并且每个 tier 是“彻底改到该 tier 局部最优”才退出。

### 3.4 FILO 的邻域算子如何定义（Section 2.2，Appendix B 有更多 Δ 公式）

论文把 “CROSS-exchange” 统一记为 `nmex`（n≥m）：

- **`nmex`（CROSS-exchange）**：从两条 route（可同一条）上各取一段连续 path（长度 n 与 m）并交换；论文实现 `n,m ∈ {0,1,2,3}` 且 `n ≥ m`，并认为 `nmex` 与 `mnex` 等价。
- **`nmrex`**：交换前把第一段长度 n 的 path 反转后再交换（论文只对若干 `nmex` 的变体启用，例如 `20/21/22/30/31/32/33`）。
- **`nmrex*`**：两段 path 都反转后再交换（论文只对 `22/32/33` 的变体启用）。
- **`twopt`**：route 内 2-opt（TSP 风格），剪掉两条边并反转中间段。
- **`tail`（inter-route 2-opt 变体）**：两条 route 在各自某个切点之后的“尾巴”互换。
- **`split`（inter-route 2-opt + reverse）**：两条 route 在切点处切开后，用“第二条的 head（反转）”替换第一条 tail、用“第一条的 tail（反转）”替换第二条 head。
- **`ejch`（ejection chain）**：从一次 `10ex`（等价于 relocate 1 个客户）出发，构建一个最多 `nEC` 个节点的搜索树：
  - 树节点表示“部分 relocation 序列”（部分路径），按当前最可能带来改进的节点优先扩展（best-first）
  - relocation 可能让目标 route 暂时 infeasible，于是再“eject”一些客户搬走来恢复可行（因此是 chain）
  - 允许重复访问同一路线；不显式限制 chain 长度
  - 一旦找到一条能恢复目标 route 可行性的序列，就执行整串 relocation（论文细节指向 Appendix B.3.8）

如果你要把这些算子“模块化”为 Gate2 的动作：`nmex/nmrex/nmrex*` 是一族参数化算子（n,m,是否反转），`ejch` 是强但贵的“高阶复合 move”。

## 4. GN（Granular Neighborhood）+ SMD：把邻域探索改造成“可维护的数据结构”

### 4.1 Move generator / GN 的定义（Section 2.2.2）

FILO 先为每个顶点 i 取 `ngs=25` 个最近邻：

- 基础候选集合（静态）：对每个 i，候选 arcs 指向其 `N_ngs(i)`  
- 注意论文把 move generator 用 **有向 arc (i,j)** 表示（因为很多 move 是非对称的）

这一步把“全邻域 O(n²)”压到“候选邻域 O(n·ngs)”，是 large-scale 的第一关键。

更形式化一点（按论文符号）：

- 顶点集 `V = {0} ∪ V_c`（0 为 depot，`V_c` 为客户）
- 对每个 `i ∈ V`，取其 `ngs` 个最近邻 `N_{ngs}(i) ⊆ V \\ {i}`
- GN 的 move generator 集合（论文用“弧”而非“边”）：
  - `T = ⋃_{i∈V} {(i,j),(j,i) ∈ E : j ∈ N_{ngs}(i)}`

并且论文强调：除 `twopt` 与 `split` 外，大多数算子诱导的是 **asymmetric GN**（`(i,j)` 与 `(j,i)` 诱导的 move 不同），因此实现时最好把 move generator 当作“有向”对象处理。

### 4.2 动态稀疏化：每个顶点一个 γᵢ（vertex-wise）

不是所有顶点都需要同样多候选。FILO 维护：

- `γᵢ ∈ [0,1]`（每个顶点）
- 实际候选数 `kᵢ = round(γᵢ · ngs)`
- 动态 GN：只保留 `N_{kᵢ}(i)` 这些候选

直觉：  
“近期被打乱/长期无改进”的区域 → 增大 γᵢ（允许更多候选）  
“已经足够好/不需要深搜”的区域 → γᵢ 保持较小（省时间）

论文给的默认值：

- `γ_base = 0.25`
- `λ = 2`（放宽时倍增）
- `δ = 0.5`（控制多久没改进才放宽）

并且：**只要本次 local search 找到了新的全局最好解 S\*，就 reset 相关顶点的 γᵢ 回到 γ_base**（避免“候选膨胀后一直不收敛”）。

#### 4.2.1 论文给出的 γ 更新触发条件（Section 2.3.2，Algorithm 6 的 UpdateSparsificationFactors）

论文不是“每次没改进就涨 γ”，而是给了一个 **按总迭代预算归一化的阈值**：

- 设 `Δ_CO` 为 CoreOpt 迭代次数；
- 设 `avg_cache = average(|V̄_S|)` 为历史 local search 后缓存顶点数的平均值；
- 设 `|V|` 为顶点总数（含 depot）。

当“涉及顶点 i 的 non-improving iteration 次数”达到：

- `N_i ≥ (δ · Δ_CO · avg_cache) / |V|`

就执行一次：

- `γ_i ← min(γ_i · λ, 1)`

并且：

- 一旦得到一个改进全局最好 `S*` 的解，且在该次 HRVND 后 `i ∈ V̄_S`（也就是 i 参与了这次优化的 kernel），则 `γ_i ← γ_base`。

实现建议（贴近论文意图）：

- 维护 `cnt_i`：顶点 i 参与的“non-improving HRVND 结束次数”计数（只对 `i ∈ V̄_S` 增加）。
- 当 `cnt_i` 达阈值就放宽一次 `γ_i`，并把 `cnt_i` 置零或减去阈值（两种都合理，只要行为稳定）。
- 注意论文提醒：若缓存上限 `C < |V|`，有些顶点即使参与了局部变化也可能被 LRU 淘汰而不再出现在 `V̄_S`，从而不会触发 γ 更新；但作者实证认为这不会阻碍找到好解，反而会提速（“heuristic filtering”）。

### 4.3 SMD（Static Move Descriptor）+ heap（Section 2.2.3）

核心思想：把“遍历邻域找改进 move”改成：

1. 初始化阶段：对当前允许的 move generators 计算 δ-tag（目标值变化）
2. 把“可能改进”的 SMD 放进二叉堆（heap）按 δ-tag 组织
3. 搜索阶段：在 heap 里找一个可行（容量不超）且改进的 move  
   - 论文实现不是严格 best-improvement：它在 heap 的数组里线性扫，找第一个可行项（rough-best-improvement）
4. 执行该 move
5. 只更新“受影响的一小撮 SMD”的 δ-tag（增量更新），继续

你可以理解为：**用数据结构把“改进 move”维持成一个动态集合**，避免每次都 O(n·ngs) 甚至 O(n²) 扫。

#### 4.3.1 SMD 四阶段（初始化/搜索/执行/更新）的实现骨架（按论文描述）

论文强调：SMD 本质是 “move + 其 δ-tag（目标变化）”。每个算子都需要一套：

1. **Initialization**：枚举当前允许的 move generators（来自 `T_γ`，并受 SVC 限制），计算 δ-tag；只把“改进 δ-tag<0”的 SMD 入 heap。
2. **Search**：寻找一个“可行且改进”的 SMD。
   - 论文实现不是严格 best-improvement：它不做“弹出 infeasible 再回填”，而是 **线性扫描 heap 的底层数组** 找第一个可行项（仍然大概率接近最优，因为 heap 大致按 δ-tag 排序）。
3. **Execution**：应用该 move，更新解的 route 结构与代价，并更新缓存集合 `V̄_S`（受影响顶点入缓存）。
4. **Update**：只更新“受影响的 SMD”（operator-specific 的局部集合），并对 heap 做相应的 decrease-key / increase-key（或先删后插）。

为了让“Update 只动一小撮”真正成立，你实现时需要准备两类局部索引：

- **route 级局部性**：知道某个 move 修改了哪些 route、哪些位置（前驱/后继）。
- **GN 级局部性**：知道某个顶点 i 的 active move generators 只来自 `N_{k_i}(i)`，因此受影响区域通常围绕少量 i。

## 5. SVC（Selective Vertex Caching）：把优化“锁定”到被扰动区域（Section 2.2.4）

### 5.1 缓存的内容是什么？

每个解 S 维护一个缓存顶点集合 `V̄_S`（最近被修改的区域）：

- 一个 move 会“直接影响”若干顶点（例如 relocate 会影响 i 本身、i 的前驱/后继、插入点 j 及其前驱等）
- 这些顶点会被标记为 cached
- 缓存上限 `|V̄_S| ≤ C`，采用 LRU（least recently used）淘汰
- 默认 `C=50`

### 5.2 缓存如何限制 local search？

关键不是“只在缓存里做 move”（那会太死），而是：

- **SMD 初始化阶段只考虑与缓存顶点相邻的 move generators**（至少一个端点在缓存里）
- 后续增量更新可能把一些“间接受影响的顶点”的 SMD 也带进来（所以是软限制）

论文把它解释为：从一个很小的“kernel（被打乱区域）”出发，局部搜索可能扩张，但扩张是受控的；整体表现为一种 **隐式动态分解**。

更形式化一点（按论文对 `\bar T_γ(S)` 的定义）：

- 有效 move generators `T_γ = ⋃_{i∈V} {(i,j),(j,i) ∈ E : j ∈ N_{k_i}(V \\ {i})}`，其中 `k_i = round(γ_i · ngs)`
- SVC 让初始化阶段只用：
  - `\bar T_γ(S) = ⋃_{i∈V̄_S} {(i,j),(j,i) ∈ E : j ∈ N_{k_i}(V \\ {i})}`
  - 也就是：至少一个端点属于缓存集合 `V̄_S`

但论文同时提醒：**后续 Update 阶段可能会把一些“非缓存端点”的 move generator 带进来**，原因是执行 move 后，某些 SMD 的 δ-tag 变得过期，需要更新才能保证搜索正确性；因此 SVC 是“强约束初始化、弱约束扩张”的策略，而不是硬剪枝。

## 6. Route Minimization（Algorithm 4）：目标是“压车数”，不是直接降距离

### 6.1 什么时候触发？

- 先估计理想 route 数 `k`：对需求做 greedy first-fit bin packing（下界）
- 若构造初解的 route 数 `|S| > k`，才运行 RouteMin

### 6.2 每次迭代做什么？

每轮（最多 `Δ_RM=1000` 次）：

1. 清空缓存 `V̄_S ← ∅`（让接下来优化聚焦在本轮破坏区域）
2. 选两条路线 `(r_i, r_j)`：
   - 先随机选一个 seed 客户 i，取其所在路线为 r_i
   - 再按 i 的最近邻客户 j（按距离递增）找一位属于其它路线的客户，得到 r_j
3. 把 r_i、r_j 的客户全部移除，放到待插入列表 L
4. 对 L：50% 随机打乱、50% 按需求降序（让大需求先插）
5. 逐个客户 i ∈ L：在现有路线里找“最小插入代价且保持可行”的位置插入  
   - 若所有路线都插不进去（必须开新车）：
     - 如果当前 |S| < k：直接开一条单客户路线
     - 否则按概率阈值 P 决定是否开单客户路线；不开则把该客户丢进下一轮的 `L̄` 继续尝试  
       P 从 1 按指数下降到 `P_f=0.01`（让算法“先尽量不开车，实在不行再开”）
6. 对当前解做一次 **受限 HRVND**：只用 Tier1，但把 `γᵢ=1`（RouteMin 不想在这里再搞复杂的动态稀疏管理）

这段的设计直觉非常清楚：RouteMin 是“容量结构优化器”，先把路线数压到合理水平，再交给 CoreOpt 去压距离。

## 7. Core Optimization（Algorithm 5–6）：FILO 真正的“迭代器主体”

### 7.1 一次 CoreOpt 迭代（Algorithm 6）

每轮（标准 `Δ_CO=10^5`，长跑 `10^6`）：

1. 清空缓存 `V̄_S ← ∅`
2. `Ŝ ← Shake(S)`：ruin&recreate（Algorithm 5）  
   - Shake 会把“被移除/重插入的客户集合”写进缓存（作为后续局部搜索 kernel）
3. `S' ← HRVND(Ŝ)`：用 HRVND 把局部最优推深（此时的 GN/SMD/SVC/γ 都生效）
4. 若 `S'` 优于历史最好 `S*`：更新 `S*`，并 reset 相关顶点的 `γᵢ`
5. 否则：按规则更新 `γᵢ`（长期无改进的区域放宽候选）
6. 更新 shaking 强度参数 `ωᵢ`（见 7.3）
7. 用 SA 接受准则决定 `S ← S'` 与否
8. 温度 T 乘法降温（从 `T0` 到 `Tf`，总共 Δ_CO 步）

#### 7.1.1 SA 接受准则与降温公式（纸面可直接实现）

论文给的 SA 接受条件写得很“可编程”（把 `exp` 写进 `ln U`）：

- 接受 `S'` 当且仅当：`Cost(S') < Cost(S) + T · ln U(0,1)`

这等价于：

- 若 `Δ = Cost(S')-Cost(S) < 0`：必接受
- 否则以概率 `exp(-Δ/T)` 接受

温度更新：

- `c = (T_f / T_0)^(1/Δ_CO)`
- 每轮：`T ← c · T`

### 7.2 Shake（Algorithm 5）：随机游走式 ruin + 贪心 recreate

**ruin（破坏）**：

- 随机选 seed 客户 `i'`
- 做长度为 `ω_{i'}` 的随机游走，每访问一个客户就把它从解里移除  
  游走的扩展方式两大类：
  - 沿同一条 route 前进/后退（局部破坏）
  - “跳到邻居路线”（跨 route 扩散破坏）：通过当前客户的近邻客户 j 找到其所属路线 r_j
- 过程中记录被访问客户集合 C、涉及路线集合 R（用于后续定义重插入顺序）

**recreate（重建）**：

- 先决定一个插入顺序（四选一：随机、按需求降序、按离 depot 距离升序/降序）
- 对每个被移除客户：贪心插入到“最小插入代价且可行”的位置；插不进则开单客户路线

直觉：这个 shake 不是“完全随机删一堆点”，而是倾向于删一段在结构上相关的区域（同路线或邻路线），让重插入后的解结构变化更可控。

#### 7.2.1 按论文伪代码复现 Shake（Algorithm 5）

论文的 Shake 有两个“随机布尔开关”控制随机游走的形态（我用更直白的变量名复述）：

1. 选 seed `i'`（从某个 route 集合 R 中随机选客户；CoreOpt 中 R 是全集）。
2. `i = i'`；`step = 0`；`C = []`（移除序列）；`VisitedRoutes = ∅`（避免重复跳入的 route）。
3. 采样一个布尔 `B_jump`（决定“更倾向沿 route 走”还是“更倾向跨 route 跳”）；采样一个布尔 `B_dir`（决定 next/prev 方向）。
4. repeat 直到 `step == ω_{i'}` 或提前 abort：
   - 记录当前客户：`C.append(i)`；`VisitedRoutes.add(route(i))`
   - 决定下一个客户 `j`：
     - 若当前 route 仍有多个客户且 `B_jump` 为真：沿 route 前进（`next(i)` 或 `prev(i)`，由 `B_dir` 决定）
     - 否则：跨 route 跳到“最近的已服务客户”：
       - 若允许跳到已访问 route：`j = NearestServedCustomer(S, i)`
       - 若要求跳到未访问 route：`j = NearestServedCustomer(S, i) such that route(j) ∉ VisitedRoutes`；若不存在则 abort
   - 从解里移除 i：`S = RemoveCustomer(S, i)`
   - `i = j`; `step += 1`
5. `C = DefineOrder(C, VisitedRoutes)`：决定重插入顺序（随机 / 按需求 / 按 depot 距离等）。
6. 对 `C` 中每个客户依次：在现有路线里找最小插入增量位置插入；若无可行位置则开单客户路线。

这段伪代码的意义是：你实现 Shake 时不需要“全局随机删点”，而是实现一个能复现论文随机游走策略的局部破坏器。

### 7.3 Shaking 强度 ω 的自适应（Section 2.3.2.1）

FILO 的 shaking 强度不是一个全局常数，而是 **每个客户 i 一个 ωᵢ**（当 i 被选为 seed 时，破坏步数=ωᵢ）。

反馈信号：一次迭代后，比较 `Cost(S') - Cost(S)`：

- 若 `0 ≤ Cost(S') - Cost(S) < Ω_LB`：破坏太弱（LS 基本把它“补回来了”）→ 倾向 **增大 ω**
- 若 `Cost(S') - Cost(S) > Ω_UB`：破坏太强（LS 补不回来）→ 倾向 **减小 ω**
- 若 `Ω_LB ≤ diff ≤ Ω_UB` 或出现改进：认为强度“差不多” → 做小幅随机抖动，避免 ω 停死在边界

阈值如何设：

- 先算当前解的平均边代价：`c̄_S = Cost(S) / (N + 2|S|)`（N 客户数，|S| 车数）
- `Ω_LB = c̄_S · I_LB`，`Ω_UB = c̄_S · I_UB`
- 默认 `I_LB=0.375`，`I_UB=0.85`

同时，为防止一个点被邻域更新“推到极端”，论文还加了“向 seed 的 ω_{i'} 靠拢但不超过 ±1”的限制（你实现时只要把这层 bound 想清楚即可）。

#### 7.3.1 论文给出的 ω 更新规则（可直接照抄实现）

论文定义：

- `Ω_LB = c̄_S · I_LB`，`Ω_UB = c̄_S · I_UB`
- `c̄_S = Cost(S)/(N + 2|S|)`
- `ω̃ = ω_{i'}`（本轮 seed 的 ω 值）
- `S_cache = V̄_Ŝ \\ {0}`：Shake 执行后，被缓存的顶点集合（不含 depot；注意它既包含 ruin 涉及的客户，也包含 recreate 插入影响到的客户）

对每个 `i ∈ S_cache`，更新 `ω_i`（论文是分段函数）：

- 若 `0 ≤ Cost(S')-Cost(S) < Ω_LB` 且 `ω_i < ω̃ + 1`：`ω_i ← ω_i + 1`（破坏偏弱 → 增强）
- 若 `Cost(S')-Cost(S) > Ω_UB` 且 `ω_i > ω̃ - 1`：`ω_i ← ω_i - 1`（破坏偏强 → 减弱）
- 否则（包括：`S'` 改进了 `S`，或 diff 落在合适区间）：在 `+1/-1` 两种微调里随机选一个（同时遵守 `ω_i ∈ [ω̃-1, ω̃+1]` 的“跟随上限”约束）

这个“相对 seed 的 ±1 跟随”是论文里很关键的工程细节：它防止某个点因为邻近区域反复被更新而把 ω 推到极端。

> 你做两阶段选择时，这个 ωᵢ 机制是非常像“在线算子强度选择”的：它让同一个 shake 变成一族动作（强度可变），而且强度和实例几何结构相关（dense 区域倾向较小 ω）。

## 8. 参数表（Table 1 的关键默认值，便于你复现/对齐）

初始化：

- `n_cw = 50`：savings 里考虑的邻居数
- `Δ_RM = 10^3`：RouteMin 迭代次数上限

GN / 动态稀疏化：

- `ngs = 25`：每个顶点的近邻候选数
- `γ_base = 0.25`：稀疏基线
- `δ = 0.5`：多久无改进才放宽（控制阈值）
- `λ = 2`：放宽倍数

CoreOpt：

- `Δ_CO = 10^5`（标准）/ `10^6`（long）
- `C = 50`：缓存顶点数上限
- `ω_base = ceil(ln |V|)`：ω 初值
- `n_EC = 25`：ejection chain 的探索规模上限
- `I_LB = 0.375, I_UB = 0.85`：shaking 强度的上下界因子
- `T0, Tf`：SA 初温/终温（论文表里给符号，具体值随调参而定）

## 8.1 “按论文能跑起来”的实现蓝图（数据结构与模块拆分）

如果你要复现到“能在 500–1000 规模上跑得动”的程度，最关键的是让每个组件都具备增量更新能力：

1. **Solution/Route 表示**
   - route：双向链表或数组 + `pred/succ`（O(1) 邻接查询）；并维护 `route_id[v]`、`pos[v]`（或 iterator）
   - route 缓存：`load(route)`、`cost(route)`、以及对某些邻域（如 2-opt*、tail/split）可用的 prefix/suffix 代价缓存
2. **距离与近邻**
   - `c(i,j)`：小规模可预存 distance matrix；大规模可按需算（但要缓存近邻距离）
   - `N_k(i)`：预计算近邻列表（同时服务于 Construction 与 GN；`n_cw` 与 `ngs` 用不同截断）
3. **SVC（LRU cache）**
   - 用 `deque + hash` 或 “时间戳 + 固定数组” 实现 LRU（上限 C=50）；每次 move 把受影响顶点 push 到 cache（更新 recency）
4. **SMD/heap**
   - 每个 operator 的一个 heap（或共享 heap 但带 operator/type 字段）
   - heap 元素：`(delta_tag, move_descriptor)`；descriptor 至少包含：参与 route、切点位置、涉及的客户序列端点等
   - Search 阶段按论文：线性扫 heap 的底层数组找第一个可行的 improving move（避免 pop/push 的维护成本）
5. **γ 管理**
   - `gamma[i]` + `cnt[i]`（non-improving 次数计数）；阈值用 `(δ·Δ_CO·avg_cache)/|V|`
6. **ω 管理**
   - `omega[i]` 初始化为 `ceil(ln |V|)`；每轮按 §7.3.1 对 `S_cache` 批量更新
7. **ejch（ejection chain）**
   - 用“节点=部分 relocation 序列”的 best-first tree（优先扩展当前累计改进最大的 partial sequence）
   - 控制复杂度的唯一硬阈值是 `nEC`（最多探索节点数），因此实现时要用优先队列 + 去重（避免无意义循环）

这部分蓝图之所以重要：FILO 的核心不是某个 move，而是把“局部搜索 + 自适应扰动 + 预算控制”做成一个在大规模仍可运行的工程系统。

## 9. 这篇里的“Adaptive/算法选择”到底发生在哪？

把 FILO 看成一个“自适应系统”，它主要做了三类在线决策：

1. **局部搜索预算分配（哪里值得深搜）**：SVC 选出被扰动区域作为 kernel，并通过缓存大小 C 控制扩张。
2. **邻域动作空间大小（γᵢ）**：长期无改进区域放宽候选集，提高跳坑概率；改进后收回。
3. **扰动强度（ωᵢ）**：把“ruin”从固定强度变成可自适应的强度族，按“破坏效果”闭环调节。

## 10. 和你的“两阶段组合/选择”如何对齐（非常具体）

### 10.1 能否和别的初始化器组合？

原理上可以：CoreOpt 只要求输入是一个可行解（或至少能被它的 recreate+feasibility 机制修复成可行）。  
因此你可以做：

- `Init = 神经初始化器 / 其它启发式`  
  `Iter = FILO CoreOpt`

**风险点**：FILO 的 shake/recreate 与 HRVND 假设“路线结构可局部重排且代价可快速增量评估”。如果你的 init 输出解表示不同（如 giant tour、带额外属性），需要对齐表示与增量计算。

### 10.2 能否只取它的一部分当组件（给两层 gate）？

可以拆出多个可选组件（动作粒度不同）：

- Gate1（初始化层）：`Construction` vs 你的其它 init；以及 `RouteMin` 是否启用
- Gate2（迭代层）：
  - 选 “是否触发更强 shake”（ω 调度）
  - 选 “是否放宽某些区域的 γᵢ”（候选集预算）
  - 选 “是否启用 ejch tier”（贵但可能跳坑）

这些全都是经典“算子选择/预算控制”范式，且每一步回报（Δcost/Δtime）都很好记录，适合做 online bandit / RL baseline。
