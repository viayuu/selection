# 07) Hybrid genetic search for the CVRP: Open‑Source implementation and SWAP* neighborhood（Vidal, 2021/2022）

- 预印本：arXiv `2012.10384`（PDF v2: Oct 2021）
- 本地 PDF：`adaptive/pdfs/07_hgs_cvrp_open_source_swapstar_2022.pdf`
- 公开预印本：`https://arxiv.org/pdf/2012.10384`
- 发表信息（来自开源仓库 README）：Computers & Operations Research 140 (2022) 105643，DOI `10.1016/j.cor.2021.105643`
- 官方开源仓库：`https://github.com/vidalt/HGS-CVRP`（本地：`adaptive/repos/07_hgs_cvrp_vidalt/`，commit `1a927955cd2861a29d978f0d359d6e647db9319c`）
- 本项目中的镜像副本（便于 EasyNCO 调用）：`EasyNCO/neural_solvers/methods/deepaco/HGS-CVRP-main/`

## 0.5 代码（开源链接 + 本地下载）与论文-代码对照

- 论文 Section 4 给出开源仓库：`https://github.com/vidalt/HGS-CVRP`。
- 本地已下载官方仓库：`adaptive/repos/07_hgs_cvrp_vidalt/`（上面列出 commit）。
- 关键文件（都在 `adaptive/repos/07_hgs_cvrp_vidalt/Program/`）：
  - `main.cpp`：解析参数 → 读 CVRPLib → `Genetic solver(params); solver.run();` → 导出解与 progress
  - `AlgorithmParameters.*` + `commandline.h`：参数默认值与 CLI（`-t/-it/-seed/-veh/-round/-nbGranular/-mu/-lambda/...`）
  - `Genetic.*`：GA 主循环（tournament → OX → Split → LocalSearch → Population）
  - `Population.*`：两子种群、biased fitness（Eq.1）、生存选择、penalty adaptation
  - `LocalSearch.*`：RI moves (1–9) + `swapStar()`（含 `ThreeBestInsert` 缓存与扇区剪枝）
  - `Split.*`：giant tour → routes（含 Vidal 2016 的 O(n) Split 无时长约束版本）
  - `Params.*`：`correlatedVertices`（granular restriction）、初始 penalty 标定、fleet size 默认值
  - `CircleSector.h`：SWAP* 的 route‑pair 几何剪枝（扇区重叠）

### 0.5.1 关键代码片段：惩罚自适应 + SWAP* 的“代价=移除+最优插入+惩罚”

**(1) penalty 自适应（目标可行比例控制）**：`Population::managePenalties()` 按近期可行比例调节 `penaltyCapacity/penaltyDuration`（带安全上下界），对应论文“controlled infeasibility + penalty adaptation”的工程落地。

```cpp
// Population.cpp
if (fractionFeasibleLoad < params.ap.targetFeasible - 0.05) params.penaltyCapacity *= params.ap.penaltyIncrease;
else if (fractionFeasibleLoad > params.ap.targetFeasible + 0.05) params.penaltyCapacity *= params.ap.penaltyDecrease;
```

**(2) SWAP\* 的核心代价分解**：`LocalSearch::swapStar()` 先预处理每个客户的 `deltaRemoval`，再用 “最优插入位置（缓存 top-3）+ 约束惩罚” 评估成对交换（以及两类 relocate 作为特殊情形）。

```cpp
// LocalSearch.cpp (inside swapStar evaluation loop)
double deltaPenRouteU = penaltyExcessLoad(routeU->load + params.cli[nodeV->cour].demand - params.cli[nodeU->cour].demand) - routeU->penalty;
double deltaPenRouteV = penaltyExcessLoad(routeV->load + params.cli[nodeU->cour].demand - params.cli[nodeV->cour].demand) - routeV->penalty;
mySwapStar.moveCost = deltaPenRouteU + nodeU->deltaRemoval + extraU
                    + deltaPenRouteV + nodeV->deltaRemoval + extraV;
```

## 0. 一句话总结（你读完应当留下的印象）

这篇短文的目标很明确：**把“十年打磨的 HGS-CVRP”用开源、可复现、尽量简洁的方式讲清楚**，并强调一个额外强邻域 **SWAP\***：

- SWAP\* 的 move 空间是 Θ(n⁴)（两条路线、两个客户、两侧插入位置），但作者给出一个 **O(n²) 级别的高效搜索法**（Theorem 1 + Algorithm 2）并进一步用几何剪枝让它在实践中不成为瓶颈。

> 在你的两阶段视角里：HGS 是一个“群体式迭代器”，其强项来自 **crossover 的多样化 + 强局部搜索的 intensification + 自适应惩罚 + diversity-aware 的生存选择** 的组合。

## 1. HGS-CVRP 的整体 pipeline（Figure 1 + Algorithm 1）

一个迭代的标准流程是：

1. **Parents selection**：二元锦标赛（binary tournament），按“质量 + 多样性”的 fitness 选父代
2. **Recombination**：OX crossover 在“无 depot 的客户排列（giant tour）”上做重组
3. **Split**：用 DP 把 giant tour 切成可行 CVRP routes（作者用 Vidal 2016 的线性 Split）
4. **Local search（Education）**：对 offspring 做一轮强局部搜索（包括 SWAP*）
5. **Insertion + population management**：把 offspring 放入（可行/不可行）子种群；触发淘汰与惩罚自适应
6. 终止：达到 `N_it` 次无改进或时间 `T_max`

Algorithm 1 在论文里就是以上 6 行逻辑的伪代码版。

## 2. Fitness = 质量排名 + 多样性排名（Equation 1）

HGS 的关键不是“用 GA”，而是“GA 的选择压力如何设计”。作者用 **rank-based + diversity-aware** 的 fitness：

1. 在种群 P 中，给每个个体 S：
   - `f^φ_P(S)`：按目标值（含惩罚）排序的排名（rank of solution quality）
   - `f^div_P(S)`：按多样性贡献排序的排名（rank of diversity contribution）
2. 多样性贡献的度量：broken-pairs distance（边结构差异）到 **nClosest 个最相似解** 的平均值
3. fitness：

`f_P(S) = f^φ_P(S) + (1 - nElite/|P|) · f^div_P(S)`

直觉：

- 质量排名是主导项（保证好解不会被多样性“误杀”）
- 多样性排名是次要项（让过于相似的解更难活下来）
- 系数 `(1 - nElite/|P|)` 让最好的 `nElite` 个体更稳定地保留下来

> 这就是 HGS 的 “advanced diversity control” 的核心落点：不是加随机扰动，而是把“多样性”写进选择与淘汰规则里。

### 2.1 开源实现如何落地 Equation (1)：biasedFitness（对照 `Population.cpp`）

论文写的是“两个 rank 的加权和”。开源代码里基本一字不差实现了这个思想（只是把 rank 归一化到 `[0,1]`）：

- 子种群 `pop` 始终按 `penalizedCost` 升序保持有序（`addIndividual` 插入时就插在正确位置），因此：
  - 位置 index 就是“质量排名”：`fitRank = idx / (|pop|-1)`
- 多样性贡献用 `averageBrokenPairsDistanceClosest(indiv, nClosest)`（离最近的 nClosest 个体的平均 broken-pairs distance）：
  - 代码把 diversity 值按“越大越好”排序，并用排序位置 `i/(|pop|-1)` 当作 `divRank`
- 最终（`updateBiasedFitnesses`）：
  - 若 `|pop| ≤ nElite`：`biasedFitness = fitRank`（不加 diversity term，避免 elite 数超过种群）
  - 否则：`biasedFitness = fitRank + (1 - nElite/|pop|) * divRank`

这和论文 Eq.(1) 的语义一致：质量为主，多样性为辅，而且 elite 的权重更偏向质量。

### 2.2 broken-pairs distance 的一个“可复现定义”（论文概念 + 代码实现）

论文里 broken-pairs distance 的直觉是“边结构差异”。开源实现给了一个非常具体的计算法（`Population.cpp:brokenPairsDistance`）：

- 对每个客户 `j`，检查它的后继是否与另一个解一致（允许反向边视作一致）：
  - 若 `succ1[j]` 既不等于 `succ2[j]`，也不等于 `pred2[j]` → 记一次差异
  - 并额外处理“与 depot 相连的断点”（route 起点）：若 `pred1[j]==0` 但另一个解里 `pred2[j] != 0` 且 `succ2[j] != 0` → 再记一次差异
- 最终归一化：`distance = differences / nbClients`

因此它不是抽象的“集合距离”，而是一个可以直接写成代码、并且对 route 边界（depot）有细节处理的距离度量。

## 3. 表示方式与 crossover：为什么用 giant tour + Split（Section 2）

### 3.1 为什么 crossover 在“无 depot 序列”上做？

如果在“带 depot 分段的 routes”上做 crossover，会很难保证容量可行、也很难定义一致的重组操作。

作者选择：

- 用客户排列（giant tour）作为遗传表示
- OX 只关心“客户相对顺序片段的继承”
- 之后用 Split 重新切分出 routes（容量可行）

这把“可行性处理”从 crossover 中剥离出去，让 crossover 变简单且稳定。

### 3.2 OX（ordered crossover）要点

- 随机选父代 1 的一个片段，原位继承到子代
- 按父代 2 的顺序循环填充剩余客户（跳过已出现的）

这就是经典 OX（Oliver et al. 1987），论文 Figure 2 给了示意。

### 3.3 Split（Vidal 2016 线性算法）

Split 的目标：给定 giant tour 的客户顺序，找到最优的 depot 分割点，使得：

- 每段 route 不超容量
- 总距离最小

作者强调：使用 Vidal 2016 的 O(n) Split（比 O(n²) 的 DP 更适合频繁调用）。

## 4. Local search（Education）：邻域、granularity 与 move 扫描策略

### 4.1 标准 RI（route improvement）邻域

论文复述了 HGS 传统 RI 的邻域组合：

- Relocate / Swap（并推广到长度为 2 的连续节点序列）
- 2-opt（intra-route）
- 2-opt*（inter-route）

### 4.2 Granular restriction：Γ 最近邻

为了把邻域规模从 O(n²) 压到 O(Γn)，只考虑节点对 `(i,j)`，其中 `j` 属于 `i` 的 Γ 个最近客户集合。

### 4.3 Move 扫描策略：random order + first improvement

作者强调一种简单但有效的实现方式：

- 随机遍历 (i,j) 的顺序
- 一旦发现改进 move 就立刻应用（first improvement）
- 直到达到局部最优

这是一种经典“便宜、稳定、并且不会被严格 best-improvement 拖慢”的局部搜索实现。

## 5. 受控探索不可行解 + penalty adaptation（Section 2, Population management）

HGS-CVRP 明确维护两类子种群：

- feasible subpopulation
- infeasible subpopulation

不可行性通过线性惩罚并入目标值（例如超载量 × penalty）。  
惩罚系数不是固定的，而是自适应调节以达到一个目标可行比例 `ξ_ref`：

- 观察一段时间内 local search 产出的“自然可行比例”
- 若可行解太少 → 增大 penalty（更强推回可行域）
- 若可行解太多 → 减小 penalty（允许穿越不可行域探索）

这就是一个典型的反馈控制：把“可行/不可行的探索平衡”稳定在目标区间。

另外，如果一个 offspring 做完 local search 仍 infeasible，论文说：

- 以 50% 概率做一次 Repair：把 penalty 提高 10× 再跑一遍 local search，争取修回可行

### 5.1 penalty adaptation 的“代码级可复现版本”（对照 `Population.cpp:managePenalties`）

论文只给了“反馈控制”的概念描述，但开源实现把所有阈值都写死了（因此非常适合你做 baseline）：

- 更新频率：每 `nbIterPenaltyManagement` 次迭代更新一次（默认 100；`AlgorithmParameters.cpp` / README）
- 维护两条长度为 100 的布尔队列：
  - `listFeasibilityLoad`：每个新加入的个体是否 `capacityExcess < eps`
  - `listFeasibilityDuration`：每个新加入的个体是否 `durationExcess < eps`
- 计算两种可行比例：
  - `fractionFeasibleLoad = mean(listFeasibilityLoad)`
  - `fractionFeasibleDuration = mean(listFeasibilityDuration)`
- 目标比例：`targetFeasible = 0.2`（和论文 Table 1 的 `ξ_ref` 对齐）
- 容忍带：`±0.05`（代码里是 `targetFeasible ± 0.05`）
- 乘法更新（分别对 capacity penalty 与 duration penalty 做）：
  - 若 `fraction < target-0.05`：`penalty *= penaltyIncrease`（默认 1.2）
  - 若 `fraction > target+0.05`：`penalty *= penaltyDecrease`（默认 0.85）
  - 并 clip 到安全范围 `[0.1, 100000]`

修复（Repair）也完全按论文：如果个体 infeasible，50% 概率用 `10× penalty` 再跑一次 local search（`Genetic.cpp` / `Population.cpp:generatePopulation`）。

## 6. Population management：µ/λ + clone removal + survivor selection

每个子种群大小控制在 `[µ, µ+λ]`：

- 初始化生成 `4µ` 个随机解（每个都先 local search），按可行性放入子种群
- 当某个子种群达到 `µ+λ`：
  1. 先删 clone（与其它个体完全相同）
  2. 若无 clone，则反复删除 fitness 最差的个体，删掉 λ 个（回到 µ）

这是一种非常典型且“概念简单”的 steady-state GA/MA 管理方式：  
小种群 + 持续插入 + 周期性清理，避免种群无限膨胀。

### 6.1 开源实现里的几个关键工程细节（对照 `Population.cpp` / `Genetic.cpp`）

1) **子种群始终保持按成本有序**

- `addIndividual` 会把新个体插入到 `subpop` 的正确位置（按 `eval.penalizedCost` 升序），因此 `subpop[0]` 永远是当前子种群最好的解。

2) **多样性数据结构是“边插入边维护”的**

- 每次插入新个体时，会对同一子种群里的所有个体计算一次 `brokenPairsDistance`，并把距离写入双方的 `indivsPerProximity` multiset（按距离从小到大排序）。
- 这样 `nClosest` 的平均距离可以 O(nClosest) 直接取 multiset 的前几个元素，不需要每次重算全距离矩阵。

3) **淘汰策略：clone 优先，否则删 worst biasedFitness**

- `removeWorstBiasedFitness` 会先检测 clone：`averageBrokenPairsDistanceClosest(indiv,1) < eps` 视作“存在距离 0 的最近邻”。
- 若存在 clone，优先删 clone；否则删 `biasedFitness` 最大的个体。
- 循环执行直到子种群从 `µ+λ` 缩回 `µ`。

4) **时间限制下的 restart**

- 若设置了 `timeLimit`，并且“无改进迭代数”达到 `nbIter`，代码会 `population.restart()` 重新生成种群继续跑（见 `Genetic.cpp`）。

这些细节解释了为什么 HGS 能在保持概念简洁的同时做到很强：它把“维护多样性所需的数据结构”做成了增量更新，不把代价留到选择/淘汰时再一次性爆炸。

## 7. SWAP*：为什么强、以及如何做到“足够快”（Section 3）

### 7.1 SWAP vs SWAP*

- Swap：跨两条 route 交换两个客户，位置固定（in place）
- **Swap\***：跨两条 route 交换两个客户，但 **插入位置可重新选择**（v 插到 r' 的任意位置，v' 插到 r 的任意位置）

因此 SWAP* 覆盖了更大的结构调整空间，能发现很多普通 Swap/Relocate/2-opt* 发现不了的改进。

### 7.2 Theorem 1：把“最佳插入位置”压缩到 Top‑3

关键结论（Theorem 1）：

对 routes r 与 r'，交换客户 v 与 v' 时：

- v 在 r' 的最佳插入位置，要么是 **v' 原来的位置（in place）**，
- 要么在“移除 v' 之前”对 v 在 r' 的 **前三个最佳插入位置**之中。  
v' 在 r 同理。

证明的核心就是插入增量代价：

- `Δ(v,i,j) = c_{iv} + c_{vj} - c_{ij}`

更“按论文可实现”的写法是先定义插入位置集合：

- 对 route `r=(r1,…,r|r|)`，插入位置集合 `P(r) = {(0,r1),(r1,r2),…,(r|r|-1,r|r|),(r|r|,0)}`
- `Δ_min(v,r) = min_{(i,j)∈P(r)} Δ(v,i,j)`

Theorem 1 的关键等式（论文 Eq.(2) 的含义）是：当你把 `v'` 从 `r'` 移除后，`v` 在新 route `r'^` 的最佳插入代价只可能是：

- 插在 `v'` 的原位置（in place）：`Δ(v, v'_p, v'_s)`
- 或者是“移除 v' 之前，v 在 r' 的 top‑3 插入位置之一”（因为移除 v' 只会删掉 `P(r')` 里的 2 个位置）

移除 v' 只会影响与 v' 相邻的两个插入位置，因此全局最优只可能落在“in place”或原 top‑3。

### 7.3 Algorithm 2：O(n²) 级别的 SWAP* 搜索

对每对 routes (r, r')：

1. **预处理阶段**：对每个 `v ∈ r`，计算 v 插入 r' 的 top‑3 位置；对 `v' ∈ r'` 同理  
   - `FindTop3Locations(v, r')` 代价 O(|r'|)
2. **搜索阶段**：枚举 `(v, v')` 对：
   - 需要处理一个小细节：top‑3 位置里可能“用到了 v'（或 v）本身”，而 swap* 时 v'/v 会先被移除，所以论文用
     - `k = min{κ : i^v_κ != v' 且 j^v_κ != v'}`（找第一个不依赖 v' 的 top‑3 位置）
     - `k' = min{κ : i^{v'}_κ != v 且 j^{v'}_κ != v}`（对称处理）
   - 然后用论文 Algorithm 2 的增量写法计算一次 swap* 的总代价变化：
     - `Δ_{v→r'} = min( Δ(v, v'_p, v'_s), Δ(v, i^v_k, j^v_k) ) - Δ(v, v_p, v_s)`
     - `Δ_{v'→r} = min( Δ(v', v_p, v_s), Δ(v', i^{v'}_{k'}, j^{v'}_{k'}) ) - Δ(v', v'_p, v'_s)`
     - 总改进：`Δ = Δ_{v→r'} + Δ_{v'→r}`
   - 取使 `Δ` 最小（最负）的 `(v,v')` 作为该 route 对的候选 swap*
3. **接受策略**：每个 route 对只应用一次“最好的 SWAP* move”（best per route pair）

论文给出复杂度推导，结论是整体 `O(n²)`。

### 7.4 进一步剪枝：极坐标扇区相交（polar sectors）

为了避免 O(n²) 在大实例上变瓶颈，作者还加了一个非常工程化的几何剪枝：

- 只对“从 depot 看过去扇区有交集”的 route 对 (r, r') 才尝试 SWAP*

直觉：两个 route 在空间上几乎不相交/不靠近时，做交换大概率没意义。

作者报告：加了这个剪枝后，SWAP* 的开销可下降到与其它标准邻域同量级。

## 8. 开源实现细节（论文 Section 4 + 仓库内代码映射）

论文 Section 4 说开源实现包含主要类：

- Individual：同时存 giant tour 与带 depot 的完整解；crossover 后立即 Split 重算 delimiters
- Population：维护可行/不可行子种群，并缓存用于 diversity 的解间距离
- Genetic：主循环与 crossover
- Split：线性 Split
- LocalSearch：所有邻域 + SWAP*（最关键的性能热点）
- CircleSector：极坐标扇区交集（给 SWAP* 剪枝）

仓库内 `EasyNCO/neural_solvers/methods/deepaco/HGS-CVRP-main/README.md` 也给了同样的结构说明，并列出了更完整的命令行参数（含 penalty 更新频率与系数）。

### 8.1 一个非常值得学的工程技巧：time stamps 替代 move descriptors

论文提到：LocalSearch 里为了避免重复评估 moves：

- 不用需要频繁 reinit 的“二值 move descriptor”
- 用 **整数时间戳**记录：
  - route 上次被修改的时间
  - 某个客户相关 moves 上次被评估的时间
- 只要 O(1) 比较时间戳，就能判断 move 是否需要重算

这类技巧在大规模本地搜索里非常关键（你做算子选择实验时也很容易变成瓶颈）。

## 9. 参数（Table 1）

论文给的 6 个核心参数与默认值：

- `µ = 25`：population size（最小）
- `λ = 40`：generation size（达到 µ+λ 触发淘汰）
- `nElite = 4`：elite 数（论文说为了抵消 SWAP* 带来的额外收敛，把它调小以增强多样性）
- `nClosest = 5`：多样性度量中最相似邻居数量
- `Γ = 20`：granular search 参数（近邻限制）
- `ξ_ref = 0.2`：目标可行比例（用于 penalty adaptation）

终止条件：

- `N_it`（默认 20,000 次无改进）或 `T_max`（时间限制）
- 若用时间限制，代码会多次“重启”并持续记录 best

## 10. 和你的“两阶段选择（init→iter）”如何对齐

### 10.1 HGS 的 init/iter 拆法

HGS 不像 ILS 那样“init 一个解 → iter 改进”，它更像：

- init：初始化一个 **种群分布**（4µ 个随机解 + local search 教育）
- iter：不断生成 offspring（crossover）并用 local search 强化，然后用 diversity-aware 的生存选择维持种群

因此如果你要做两阶段 gate，可以把 gate 定义得更贴近组件：

- Gate1（初始化选择）：用什么构造法/什么 neural initializer 来产生初始 individuals（甚至可以混合多种来源）
- Gate2（迭代选择）：用 HGS 的 local search 组件（含 SWAP*）还是用别的迭代器（FILO / AILS-II / SISRs / POPMUSIC 等）

### 10.2 “能否组合”的原理结论

从原理上 HGS 的组件非常可组合：

- 任何能输出 CVRP 解（或 giant tour + Split 能还原）的初始化器，都能用来 seed HGS 的种群
- HGS 的 local search（尤其 SWAP*）也可以作为其它框架（如 ILS/ALNS/RL）的一个算子

你要证明的“组合有效”在这里有很强的 OR 侧支撑：  
HGS 的强大本身就是“多组件互补 + 自适应控制”的结果，不是单一技巧。把不同方法的初始化与迭代阶段解耦组合，在概念上完全合理。
