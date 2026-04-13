# 01) Knowledge-guided local search for the Vehicle Routing Problem（KGLS, Arnold & Sörensen, 2019）

- 论文：Florian Arnold, Kenneth Sörensen, *Knowledge-guided local search for the vehicle routing problem*, Computers & Operations Research 105 (2019) 32–46
- DOI：`10.1016/j.cor.2019.01.002`
- 本地 PDF：`adaptive/pdfs/01_knowledge_guided_local_search_2019.pdf`

## 0.5 代码与数据可获得性（论文给出的链接）

- 论文在引言处写明：heuristic + benchmark instances 可在 `http://antor.uantwerpen.be/routingsolver/` 获取（目前该链接返回 404）。
- 虽然原始下载站点已失效，但**作者后来把 KGLS 的实现维护在 GitHub 并公开源码**：
  - 官方仓库（Java）：`https://github.com/ArnoldF/LocalSearchVRPXXL`（README 明确说明包含 KGLS 与 very-large-instance 扩展）
  - 本地已下载：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/`（commit `c7651080c5062b52ec73fc5bd71eae14fca17c6d`）
- 因此本笔记后续“实现细节”会以论文为主，并用上述源码对照关键步骤（惩罚/坏边轮换、扰动、CE/RC/LK 等）。

**论文 → 代码**（最关键文件映射）：
- 入口与主循环：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/KGLS.java`
- 构造初解（CW）：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/construction/ClarkeWright.java`
- 惩罚/坏边函数（width/length/rotation）与 penalized cost：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/datastructures/CostEvaluator.java`
- 扰动（惩罚驱动）与改进（LS+LK）：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/localsearch/LocalSearch.java`
- 主要邻域（inter-route）：  
  - segment relocation：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/localsearch/SegmentMoveOperator.java`
  - cross-exchange：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/localsearch/CrossExchangeOperator.java`
  - relocation chain：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/localsearch/RelocationChain.java`
- route 内强化（intra-route）：`adaptive/repos/01_kgls_arnoldf_LocalSearchVRPXXL/src/localsearch/LinKernighan.java`

### 0.5.1 关键代码片段：坏边轮换 + 惩罚驱动扰动（对应论文 Algorithm 2/3 的精神）

1) **坏边评分 `b(i,j)` + rotation**（`CostEvaluator.determineEdgeBadness`）：对当前解中每条边算 value，并除以 `(1+penalty)`；每次调用后按 `width → length → width_length → ...` 轮换。

```java
// CostEvaluator.determineEdgeBadness(...)
int penalty = edgePenalties.getOrDefault(edge, 0);
edge.setValue((int)(value / (1 + penalty)));
...
if (penalizationCriterium.equals("width")) penalizationCriterium = "length";
else if (penalizationCriterium.equals("length")) penalizationCriterium = "width_length";
else penalizationCriterium = "width";
```

2) **扰动 = “惩罚最坏边，然后在惩罚代价下做局部搜索”**（`LocalSearch.perturbateSolution`）：先启用 penalized distance，构建 edgeRanking；循环挑最坏边加惩罚，并从边端点出发搜改进 move。

```java
// LocalSearch.perturbateSolution(...)
costEvaluator.enablePenalization();
costEvaluator.determineEdgeBadness(solution.getRoutes());
while (appliedChanges < numPerturbations) {
    Edge worstEdge = costEvaluator.getAndPenalizeWorstEdge();
    Pair<Integer, Set<Route>> result = localSearch(..., /*intraRouteOpt=*/false, runParameters);
    appliedChanges += result.getFirst();
}
costEvaluator.disablePenalization();
```

3) **惩罚后的代价整形（λ）在代码里是硬编码 0.1**：`CostEvaluator.getDistance` 在 penalizationEnabled=true 时返回 `base + 0.1 * baselineCost * penalty(edge)`（对应论文 `c_g = c + λ p L` 的一个实现版本）。

### 0.5.2 论文伪代码 vs 该仓库默认参数（避免你复现时“看起来不像”）

开源实现把很多论文里“可调节/实验确定”的项暴露成参数（或写死为默认值）。最常见的差异点是：论文的 “P=30 次 perturbation” vs 代码默认 `num_perturbations=3`（但可从命令行覆写）。

`KGLS.java` 的默认参数如下（你可以用 `key=value` 覆写）：

- `depth_lin_kernighan=5`（LK 深度）
- `depth_relocation_chain=3`（RC 深度）
- `num_perturbations=3`（每轮扰动“惩罚-搜索”次数）
- `neighborhood_size=20`（近邻表规模）
- `moves=[segment_move,cross_exchange,relocation_chain]`（inter-route 邻域集合；intra-route 的 LK 在 `improveRoute` 中单独调用）

## 0. 一句话总结（你读完应当留下的印象）

这篇论文想证明一件事：**“只靠本地搜索 + 合理剪枝 + 合理扰动 + 少量 VRP 结构知识”就能做出很强、而且很快的 CVRP 启发式**。作者把它包装成一个确定性的 GLS（Guided Local Search）框架：用“惩罚坏边”来持续改变搜索地形，从而反复跳出局部最优。

## 1. 论文贡献点（按“你复现/移植时真正有用的点”整理）

1. **一个非常干净的 GLS 框架（deterministic）**：没有随机初始解、没有随机邻域顺序、没有复杂的 population；靠“惩罚驱动的扰动”来做多样性。
2. **三类互补邻域的组合（LS3）**：  
   - **LK（Lin–Kernighan）**：做 *intra-route* 的 TSP 强化（每条 route 里把顺序尽量压到局部最优）。  
   - **CE（CROSS-exchange family）**：做 *inter-route* 的大邻域交换（可退化成 relocate/swap/crossover 等）。  
   - **RC（Relocation Chain）**：作为补充邻域（作者实测：RC 很耗时但值得，特别是 routes 较短时效果明显）。
3. **“知识引导”的核心不是 ML，而是“坏边判别”**：提出并验证了两个很有效的坏边指标：
   - **cost**（边长）  
   - **width**（边的“横向宽度”，反映 route 的“扁/窄/紧凑”）
4. **惩罚策略的关键发现**：单一指标（只惩罚长边/只惩罚宽边）都不错，但 **轮换（rotation）更好**：`bw → bc → bw,c`。作者的解释是：不同实例偏好不同指标；以及多样性本身重要。
5. **计算层面的关键**：剪枝（pruning）+ 局部更新（只在“被扰动过的 route 区域”做优化）让它在大实例上跑得非常快。
6. **工程层面的可扩展性展示**：同一套思路轻改即可扩展到 MDVRP、MTVRP（并且给了非常直接的做法）。

## 2. 问题设定与符号（CVRP）

- 节点：depot + customers  
- 成本：欧氏距离（对称）  
- 约束：每条 route 载重 ≤ Q  
- 目标：总距离最小

作者强调：**CVRP 的两类子任务**天然对应两类邻域：

- inter-route：分配客户到哪个 route  
- intra-route：每条 route 内部顺序优化（近似一个小 TSP）

KGLS 的设计就是“先把 intra-route 做得很强（LK），再用 inter-route 邻域持续改变分配，再对被影响的 routes 立刻做 LK 复原”。

## 3. KGLS 的整体算法结构（Algorithm 3 的可复现版本）

### 3.1 总体框架：Construction → Initial optimization → 반복 (Perturbation → Optimization)

KGLS 可以视为“局部搜索为主、GLS 惩罚为辅”的确定性迭代框架：

1. **Construction（构造初解）**：
   - 用 parallel Clarke–Wright (CW) 生成可行解。
   - 如果 CW 用车数太多，做一次“route 最小化导向”的 CW（见 4.1）。
   - 对每条 route 做一次 LK（先把 route 内部顺序压好）。
2. **Initial optimization（初始局部优化）**：
   - 用 inter-route 操作 **CE + RC** 做一次 steepest descent（每次选最优改进 move）。
   - 每当某条 route 被改变，就立刻对这条 route 做 LK（intra-route 强化）。
   - 初始化边坏度函数 `b(·) ← bw(·)`（先用 width 指标）。
3. **主循环直到时间上限**：
   - **Perturbation（扰动阶段）**：重复 P 次（作者最终选 P=30）：
     1) 找到当前解中“最坏”的边 `(i,j)`（最大 `b(i,j)`）  
     2) 给这条边加惩罚 `p(i,j) ← p(i,j)+1`  
     3) 在“必须先移除这条边”的条件下，尝试 CE / RC 的 move，并用 **惩罚后的代价 `c_g`** 来评估（不是原始 `c`）。
   - **Optimization（再优化阶段）**：
     - 先对被影响的 routes 做 LK
     - 然后对 perturbation 影响到的 routes 反复 CE / RC（此时评估用原始 `c`），每次改动 route 就做 LK
     - 扰动阶段结束后 **轮换坏边指标 b(·)**（rotation，见 5.2）

这套结构很像你现在做的“init→iter”：  
KGLS 本质是一个 **iteration 模块**，输入任意初解，输出改进解；它内部的自适应状态是 `p(i,j)` 与 `b(·)` 的轮换。

### 3.2 KGLS 的“确定性”设计（为什么作者强调这点）

作者刻意把系统做成 deterministic：

- 扰动不是随机踢，而是“惩罚最坏边”
- 邻域搜索不是随机遍历，而是 steepest descent
- 轮换策略是固定顺序

作者的观点：很多 VRP metaheuristic 依赖随机性，但很少解释“随机到底贡献在哪里”。KGLS 试图用一个可解释的机制（惩罚）替代随机。

## 4. 初始化（Construction）细节：CW + “用车数最小化导向”的改造

### 4.1 为什么要关心 routes 数量？

即使 CVRP 的目标只有距离，实际中“routes 越多往往越差”（多出 depot 边、分配更碎）。  
KGLS 的 CE/RC 能偶尔消除 route，但不擅长“主动造更少 routes 的初解”，所以作者在构造阶段就做一次 route 最小化导向。

### 4.2 判定初解是否“用车数过多”

先算车辆数下界：

- `M_min = ceil(D / Q)`，`D` 为总需求

再用 parallel CW 得到 `M_CW`。当 `M_CW > M_min + 1` 时，认为“装箱很难/初解用车偏多”，触发改造版 CW。

### 4.3 改造版 savings：把“高需求客户先聚”编码进 savings

作者把经典 CW savings `s(i,j)` 与需求融合，构造 weighted savings（归一化后相加）：

- 直觉：让高需求客户优先彼此相连，减少“高需求分散导致装不下”的情况  
- 论文里报告：在 Uchoa et al. (2017) 的 100 个实例中，有 33 个实例初解 routes 数被成功减少，有些实例解质提升显著。

更贴近原文的写法是：把 savings 与 “d(i)+d(j)” 都做一次归一化后相加，例如：

- `sw(i,j) = s(i,j) / max_{k,l} s(k,l) + (d(i)+d(j)) / max_{k,l} (d(k)+d(l))`

（公式的符号与排版在 PDF 抽取中会有轻微歧义，但实现思路是明确的：**两个分量都缩放到 [0,1] 附近，再相加排序**。）

## 5. “Knowledge-guided”到底是什么：坏边定义 + 惩罚轮换

### 5.1 GLS 的基本形式（KGLS 使用边惩罚）

维护每条边 `(i,j)` 的惩罚计数 `p(i,j)`；在扰动阶段用“惩罚后的代价”指导 move 选择。

论文给了一个非常具体、可直接复现的 GLS 代价整形（Equation 1）：

- `c_g(i,j) = c(i,j) + λ · p(i,j) · L`

其中：

- `c(i,j)`：原始边长
- `p(i,j)`：边 `(i,j)` 被惩罚的次数（只对“当前解里出现的边”会不断累积）
- `L`：平均边长的尺度因子（论文用“起始解总成本 / 客户数”作为 proxy）
- `λ`：惩罚强度（论文在预实验里选 `λ=0.1`，并在正文中固定使用）

**实现层面要注意**：在 perturbation 阶段，接受的是 “`c_g` 改进” 的 move，因此：

- 一个 move 可以在 `c_g` 下是改进，但在原始 `c` 下可能是变差  
  这就是 GLS 能“跳出局部最优”的根本机制。

KGLS 把“扰动”做成了一个确定流程（Algorithm 2/3 的精神）：  
反复“惩罚最坏边 → 在该边的局部搜索空间里试图删掉它（用 `c_g` 评估）”。

关键问题变成：**“哪个边最该惩罚？”**  
作者把这个称为 **badness/utility**：`b(i,j)` 越大，越“坏”，越先惩罚。

### 5.2 两个核心坏边特征：cost vs width

1) **cost**：`c(i,j)`（边长）

2) **width**：`w(i,j)`  
来自作者的 data mining 观察：高质量解往往 routes 更“窄、更紧凑”。  
width 的几何含义（更贴近原文定义）：

- 对一条 route r，先计算 **重心** `g(r)`：route 上所有客户坐标的均值（不含 depot）。
- 取 **主轴**：从 depot 指向 `g(r)` 的方向向量 `u = normalize(g(r) - depot)`。
- 取 **垂直轴**：`u⊥`（2D 中可直接用 `(−u_y, u_x)`）。
- route 的 “width” 在原文里是：客户在 `u⊥` 方向投影的跨度（max−min）。

论文最终在 GLS 里用的是“edge width”，即 Figure 5 所示：  
对边 `(i,j)`，`w(i,j)` 是端点在 `u⊥` 方向上的投影差值（绝对值）：

- `proj⊥(x) = (x - depot) · u⊥`
- `w(i,j) = |proj⊥(x_i) - proj⊥(x_j)|`

（实现上你只要能在 route 改变后更新 `g(r)` 和 `u⊥`，即可计算该 route 中每条边的 width。）

论文还提到另一个与“好解”相关的 route-level 指标 **compactness**（紧凑度）：  
每个客户到“depot→g(r)”这条直线的平均距离（越小越紧凑）。KGLS 最终没有直接用 compactness 作为 badness，而是用 width（更直接地可归因到边）。

作者把 width 与 cost 变成坏边函数（都除以 `1+p(i,j)` 防止一直惩罚同一条边）：

- `bw(i,j) = w(i,j) / (1 + p(i,j))`
- `bc(i,j) = c(i,j) / (1 + p(i,j))`
- `bw,c(i,j) = (w(i,j) + c(i,j)) / (1 + p(i,j))`

### 5.3 最有效的策略：rotation（轮换惩罚指标）

作者对比了：

- 只用 `bw`、只用 `bc`、只用 `bw,c`
- 随机惩罚（作为对照）
- **rotation**：每个 perturbation phase 后按固定顺序切换坏边函数

结论（在他们测试集上）：

- `bw` 略优于 `bc`  
- `bw,c` 接近  
- **rotation 最好**，顺序是 `bw → bc → bw,c`（论文也测试了其他轮换，发现这三者轮换是“甜点”）

作者解释：

- 不同实例可能偏好不同“坏边定义”（instance-dependent）  
- 以及“只盯一种指标”会导致搜索过窄，轮换引入必要多样性

## 6. 局部搜索算子（LS3）到底做了什么

### 6.1 LK（intra-route）

- 每当 inter-route move 改动了某条 route，就立刻对这条 route 跑 LK
- 这相当于把“route 内部顺序”始终保持在一个很强的局部最优附近，让 inter-route move 的贡献更纯粹地体现在“分配/结构”上

### 6.2 CE（CROSS-exchange family, inter-route）

CE 是一个“超级家族”：包含 swap/relocate/Or-exchange/crossover/CROSS-exchange 等。  
论文里把它当作统一实现：交换两个 routes 的子串（子串可逆），形成新 routes。

**关键工程点**：CE 的 neighborhood 极大，所以论文给的是一个“按顺序搜索 + 强剪枝”的实现思路（Algorithm 1），核心叫 sequential search：

1. **先枚举 starts（只看局部 1-step 交换）**，并且只保留 “不会立刻变差” 的 starts（`c1 ≤ 0` 或 `c1* ≤ 0`）  
   - 对每条被考虑的边 `(I_k, I_{k+1})`，最多只评估 `4C` 个 starts（来自 `I_k` 与 `I_{k+1}` 的 C 近邻，各自有两种连接方式）。
2. 对每个保留的 start，再 **延长子串** 并计算第二个 cross 的代价 `c2`（或 `c2*`），当 `c1+c2 ≤ 0` 且满足容量约束时，记录为候选 move。
3. 最后用 **steepest descent**（选最优改进）执行 1 个候选 move。

更贴近原文的代价定义（对应 Figure 3 的 swap 情况）：

- start 代价（示例）：  
  `c1 = c(I_k, J_l) + c(I_{k+1}, J_{l-1}) - c(I_k, I_{k+1}) - c(J_l, J_{l-1})`
- 第二个 cross 代价（示例）：  
  `c2 = c(I_{k+1}, J_{l+1}) + c(J_l, I_{k+2}) - c(I_{k+1}, I_{k+2}) - c(J_l, J_{l+1})`

然后延长子串（例如把 `J_{l+1}` 继续并入交换子串）会改变 `c2`，重复评估直到遇到 depot 或容量约束阻止继续延长。

**实现建议（按论文思路落地）**

- 预处理每个客户的 C 近邻列表（C=30）。
- route 用双向链 + 数组索引混合结构（或至少能 O(1) 得到前驱/后继），方便快速计算 `c1/c2`。
- start 过滤（`c1 ≤ 0`）是 CE 能快的关键：在好解附近，绝大多数 start 会被立刻剪掉。

### 6.3 RC（Relocation Chain, inter-route）

RC 可以理解为“多步 relocate 链”：把一个客户从 route A 移走会导致 A 变松；为了填补/可行，继续把另一个客户挪到 A；链式传播，直到回到可行并产生改进。

论文给的实现思路比“概念描述”要具体得多（Section 2.3）：

1. RC 从一个 relocate 开始：把节点 `I_k` 从 route r_i 移到 route r_j 的某个位置（可能导致 r_j 超载）。
2. 这个 relocate 的代价用 “插入增量 + 原位置移除带来的 detour 变化” 计算：
   - 插入增量（只考虑插在 `J_l` 的前/后两种）：
     - `c_I = min_{J*_l ∈ {J_{l-1}, J_{l+1}}} ( c(I_k,J_l) + c(I_k,J*_l) - c(J_l,J*_l) )`
   - 原 route 的 detour 变化（移除 I_k）：
     - `c_D = c(I_{k-1}, I_{k+1}) - c(I_k, I_{k+1}) - c(I_k, I_{k-1})`
   - 若 `c_I + c_D ≤ 0`，认为该 relocate 有“partial gain”，可作为 RC 的起点（对应 LK 的 partial gains 思想）。
3. 若第一步 relocate 后 r_j 仍可行，则直接把这一步当作候选 move；否则尝试继续链式 relocate：
   - 下一步 relocate 必须 **恢复上一条 route 的可行性**，且累计代价仍非正；
   - 不断继续直到达到最大链长。

论文还给了多个实现剪枝（非常关键）：

- **链长上限取 3**：作者实测链长 4 的计算成本可达 10×，因此固定限制为 3 次 relocate。
- **只考虑 C 近邻插入**：与 CE 类似，relocate 只允许把节点插到其 C 个近邻附近（减少候选位置）。
- **每个目标 route 只保留一个插入位置**：如果近邻里有多个点都在同一条目标 route 上，只取插入代价最小的那个位置（减少分支）。
- **RC 内禁止“插回到已被 eject 的位置”**：避免重新计算与回路（原文说这样能减少额外 cost 计算）。

论文还提到一个很工程的加速：  
对每个 RC 起点，生成一串候选 moves 后，不只执行一个，而是：

1. 先执行最优改进（steepest descent）
2. 把所有与之“干扰”的候选 move 从列表删除（干扰定义：共享被搬移节点，或搬移节点的新旧邻接关系冲突）
3. 再从剩余候选中继续执行下一个最优改进

（这个技巧的意义是：你已经花了代价生成了很多候选 move，就尽量多利用；同时用干扰规则避免代价重算。）

## 6.4 复现时的“代码级蓝图”（把论文实现思路落成工程）

如果你要实现一个“尽量贴近论文”的 KGLS，通常会把状态拆成三类缓存：

1. **route-level 缓存**（每条 route 一份）：
   - 当前载重、长度（cost）
   - 客户坐标和的累积（用于快速更新重心 `g(r)`）
   - 主轴/垂直轴向量（用于 width）
2. **node-level 预处理**：
   - C 近邻列表（按欧氏距离排序）
3. **GLS 记忆状态**：
   - `p(i,j)`：对当前解中出现过的边累计惩罚（可用哈希表按无向边存，避免 O(n²)）

然后按 Algorithm 3 的流程驱动：

- perturbation 阶段：只在被惩罚的那条边 `(i,j)` 的“局部搜索空间”里找 `c_g` 改进（CE/RC 的 start 必须涉及 `(i,j)` 或其端点）。
- optimisation 阶段：只对 perturbation 改动到的 routes 做 LK + CE/RC 复原（避免全局反复扫）。

## 7. 剪枝（pruning）与可扩展性：为什么 KGLS 会这么快

### 7.1 “C 近邻剪枝”：只考虑近邻连接

对 inter-route operators，只考虑新边在 “每个节点的 C 个最近邻”中出现：

- 默认 C=30  
- 对比 C=20/30/40：
  - 短时间内 C=20 更快  
  - 长时间 C=30 更好  
  - C=40 太松导致邻域过大，性价比下降

作者也尝试了 α-nearness（类似 Helsgaun 对 TSP 的 α-nearness），以及一种“宽度增强的距离度量”，但最终发现 **简单的欧氏距离近邻** 已足够好。

### 7.2 “只优化被扰动的 routes”：局部更新

扰动阶段只对受影响的 routes 做 LK + CE/RC，没被改动的 routes 不重复扫描。  
这其实是一个很重要的“局部性原则”：每次 perturbation 的改动面积小，优化就聚焦小区域 → 可扩展。

## 8. 参数与复现要点（你真的要复现时最关键的清单）

**核心参数（论文结论）**

- perturbation moves：`P = 30`
- pruning neighbors：`C = 30`（长时间更好）
- penalization：rotation `bw → bc → bw,c`
- time limit：`3 * N / 100` minutes（每 100 客户 3 分钟）

**实现要点（作者报告）**

- Java 实现，单线程
- 节点对象预存：
  - `d(·)`、`d_g(·)`、`p(i,j)` 等矩阵信息  
  - C 近邻列表与插入代价等预处理
- routes 用可随机访问的动态数组结构（论文里用 ArrayList），使 move 评估快；执行 move 的插入删除虽然是线性，但频率相对小。

## 9. 实验结果概览（你写 related work / baseline 对比时可直接引用的结论）

### 9.1 Uchoa et al. (2017) 100 个 CVRP 实例（N=100..1000）

作者用同一时间预算（3min/100 customers），报告：

- KGLS 平均 gap（对 BKS）约 `0.44%`，平均用时 `~5.3` minutes  
- 相比 ILS、HGSADC：
  - 解质大致可比 ILS  
  - 在大实例上明显更快（“时间-质量” trade-off 非常好）

### 9.2 MDVRP（Cordeau et al. 实例）

在 MDVRP 上做了很轻量的改造（给每 depot 加一个 dummy empty route 来允许把客户迁入新 depot），结果 gap 也非常小，且耗时更短。

### 9.3 MTVRP（多 trip + time horizon）

把 MTVRP 化成：

1) 先用 KGLS 解对应的 VRP（忽略 time horizon）  
2) 每次得到一个 VRP 解，就用 3 个 bin packing heuristic（First Fit / First Fit Decreasing / Best Fit）尝试把 routes 分配给 M vehicles，检查是否满足 time horizon  
3) 找到可行就记录，最终取 best feasible

关键观察：KGLS 生成大量 VRP 解很快，所以“多次尝试 + bin packing 检查”的策略在 MTVRP 上很自然。

## 10. 对你的“两阶段 init+iter 组合”研究的直接启发

### 10.1 KGLS 作为“迭代器”组件是否可组合？

非常可组合，理由：

- 输入只需要一个可行解（route 划分 + 顺序），不依赖“初解来自哪个初始化器”
- 自适应状态 `p(i,j)` 只与“当前解结构 + 搜索历史”有关

组合风险主要在：

- 如果初始化器产生的 routes 数非常大/非常碎，KGLS 需要先把 routes 压下去（它能做但不一定最快）  
  → 这提示你 Stage-1（init 选择）最好也考虑“路线数/紧凑度”的偏好。

### 10.2 把 KGLS 的“惩罚轮换”当作一个可学习的 policy

KGLS 里唯一“知识注入”的点是：坏边函数 `b(·)` 的选择与轮换。  
你完全可以在你的 two-stage 设想里把它变成更一般的决策：

- Stage-2 不只选 iter=KGLS，而是选 `(iter=KGLS, mode=bw/bc/bw,c/rotation, C, P)`  
- 输入特征：`ψ(i, sol0)`（例如 routes 的宽度/紧凑度统计，edge 长尾程度）

这会把 KGLS 从“固定启发式”升级为“可组合、可调度、可选择”的迭代模块。
