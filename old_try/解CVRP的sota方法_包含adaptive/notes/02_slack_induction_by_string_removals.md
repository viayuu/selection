# 02) Slack Induction by String Removals for Vehicle Routing Problems（SISRs, Christiaens & Vanden Berghe, 2020）

- 论文：Jan Christiaens, Greet Vanden Berghe, *Slack Induction by String Removals for Vehicle Routing Problems*, Transportation Science (2020)
- DOI：`10.1287/trsc.2019.0914`
- 本地 PDF：`adaptive/pdfs/02_slack_induction_by_string_removals_2020.pdf`

## 0.5 代码可获得性（论文是否给出开源链接）

- 在论文全文（`adaptive/fulltext/02_slack_induction_by_string_removals_2020.txt`）中未检索到官方开源仓库/代码下载链接（GitHub/GitLab/Bitbucket 等）。
- 为了把“论文伪代码 → 可运行程序”的一些实现细节对齐（特别是 `string/split-string removal` 与 `blink insertion` 的细节），我额外下载了一份**第三方开源实现（非原作者）**作为参考对照：
  - 仓库：`https://github.com/hankarudova/open-source-sisr-routing`（branch `cvrp`）
  - 本地：`adaptive/repos/02_sisrs_rudova_open-source-sisr-routing/`（commit `fe1a33a6b5bf0dfaba87fb65565dcb4db2d9a89c`）
- 免责声明：该仓库不是论文官方实现，参数命名/初始化策略/边界处理可能与原作者版本不同；本笔记在引用代码时会明确标注“third-party”，并以论文描述为准。

**论文 →（third-party）代码**（关键文件映射）：
- 主循环（SA 接受 + ruin&recreate）：`adaptive/repos/02_sisrs_rudova_open-source-sisr-routing/src/Sisrs.cpp`
- 实例读入与邻接矩阵（adjacency list/matrix）：`adaptive/repos/02_sisrs_rudova_open-source-sisr-routing/src/ProblemInstance.cpp`
- 关键超参（`alpha/beta/gamma/max_cardinality/average_removed` 等）：`adaptive/repos/02_sisrs_rudova_open-source-sisr-routing/config.json`

### 0.5.1 关键代码片段：Adjacent String Removal + Split String Removal + Blink 插入（帮助你对齐实现细节）

**(1) Ruin：决定一次“删多少/删几条 route”**（third-party `Sisrs::ruin`，代码注释直接标了 Eq.5–7）：

```cpp
float cardinality = std::min((float)max_cardinality,
    (problem.customer_count - sol_neighbor->absent_customers.size()) / (float)sol_neighbor->routes.size());
float max_strings = (4.0f * average_removed) / (1.0f + cardinality);
std::uniform_real_distribution<float> dist(1.0f, max_strings);
int strings_to_remove = std::min((size_t)std::floor(dist(gen)), sol_neighbor->routes.size());
```

**(2) String vs Split-string removal**：以概率 `alpha` 选 split-string；split-string 里用 `beta` 决定“保留片段”长度 `m`（这正对应论文的两段式 removal）。

```cpp
std::uniform_real_distribution dist_alpha(0.0f, 1.0f);
if (dist_alpha(gen) > alpha || to_remove_cardinality == 1 || route_total_length == to_remove_cardinality) {
    to_remove_customers = randomSubstringIncluding(customer_vector, to_remove_cardinality, customer_index); // string
} else {
    int m = 1;
    std::uniform_real_distribution<float> dist_beta(0.0f, 1.0f);
    while (m < route_total_length - to_remove_cardinality && dist_beta(gen) > beta) ++m;
    // 先抽一个 (l+m) 的连续串，再“掐掉中间 m 个”，形成 split-string
}
```

**(3) Recreate：blink insertion 的落地方式**：遍历所有插入位置，但以 `gamma` 概率“眨眼跳过”，从而避免每次都做完整枚举（对应论文的 blinks）。

```cpp
if (insert_here_cost < insert_best_cost && (dist_blink(gen) > gamma * 100)) {
    insert_best_cost = insert_here_cost;
    insert_best_position = j;
    insert_best_route_index = i;
}
```

## 0. 一句话总结

SISRs（读作 *scissors*）是一个“极简但很强”的 Ruin&Recreate：**只用 1 个 ruin（相邻字符串删除）+ 1 个 recreate（带 blinks 的 greedy insertion）**，并用“Slack Induction”的解释体系告诉你：为什么这种破坏方式会特别有效，尤其在大规模 CVRP/VRP 变体上。

## 1. 这篇论文的核心贡献（按可复现价值排序）

1. **一个新的 ruin 机制：Adjacent String Removal**  
   - 不是随机删点、也不是半径删点，而是删“连续子串（string）”，并且删的 strings **彼此相邻（围绕同一个 seed 的近邻区域）**。  
   - 核心目的是在一个局部区域制造“大块 slack”（capacity slack + spatial slack），让重建时更容易形成更好的 routes 分区。
2. **一个新的 recreate：Greedy insertion with blinks**  
   - 插入时并不总是评估所有位置：以 blink rate `β` 跳过一部分位置，形成一种隐式的“按排名的随机化”选择（但不用先给位置排序）。
3. **当“最小化车辆数”是主目标时的 Fleet Minimization**  
   - 引入 absences-based acceptance：用“缺席次数”这种记忆，避免算法反复遗漏同一批客户，并更系统地把 routes 数压到下界附近。
4. **一个很有解释力的新概念：spatial slack**  
   - 把“删掉一段路径”理解为“释放了可达区域”，从而解释为何“删 strings + 聚集删除”比“删散点”更容易重组出好解。

## 2. 放回经典大框架：Ruin&Recreate + SA（但只保留最必要的东西）

作者明确对比了“越来越复杂的 ALNS（越来越多的 R−/R+）”路线，提出反向观点：**减少算子数量也能做很强**。

SISRs 的总体结构：

- 用 Simulated Annealing（SA）做全局 acceptance（不是每步只接受改进）
- 每轮用同一个 R−R+ 生成邻居：  
  `s* = recreate( ruin(s) )`

作者用一个很实用的解表示：

- `s = {T, A}`  
  - `T`：当前 routes 集合  
  - `A`：absent customers（暂时未被服务的客户集合）  
  - 注意：这让算法可以自然表示“部分解/不可行解”，便于做 fleet minimization。

## 3. Slack Induction：作者到底想“诱导”出什么 slack？

作者把 slack 分成两类：

### 3.1 capacity slack

删掉客户后，车辆载重空间被释放 → 后续插入更容易可行。

### 3.2 spatial slack（这篇的标志性概念）

作者用一个几何直觉：把车辆的行驶距离看成资源，则一条边 `(u,v)` 的“可达区域”类似以 `u,v` 为焦点的一簇椭圆（直观上：你能在不增加太多距离的情况下绕到哪里）。  
删掉一段 detour 后，即使总距离不一定减少，也可能扩大“可达区域”，让路径重组空间变大，这就是 spatial slack。

这套解释的重要作用是：它把“ruin 应该删什么”从经验变成更可讨论的设计目标。

## 4. SISRs 的 3 条设计前提（Premises）：为什么它要“删得多、删得近、删成串”

作者在 5.1 给了三条非常工程化的失败案例（这对你设计新 ruin 很有用）：

1. **删得不够多**：释放不了足够 capacity slack → 无法把高 demand 客户塞进更好的 routes。
2. **删得不够集中（不相邻）**：slack 分散到很多 routes 上，重建时像做了很多次“小 ruin”，效果弱、收敛慢。
3. **只按邻近删点（比如 radial removal）但不删成串**：可能没删掉完整 detour，很多“坏结构”仍残留。

因此他们的结论是：

> **删多个、彼此相邻的 strings**，可以在少数 routes 上制造强 slack，并且 slack 区域高度重叠，更利于重建阶段做结构性重组。

## 5. Ruin：Adjacent String Removal 的具体算法（Algorithm 2）

### 5.1 你需要的预处理：adjacency list

对每个客户 `c`，预先构建邻接列表 `adj(c)`：按距离从近到远排序的客户列表（包含自己作为第一个元素）。

这一步是 SISRs 的“相邻”定义来源，也是可扩展到任意 VRP 变体的关键接口（只要你能定义距离）。

### 5.2 “删多少”：先确定 strings 数 ks，再确定每个 string 长度 lt

作者用一套很干净的采样规则（公式 5–9）：

1) 先确定全局最大 string 长度（跟当前解平均 route 长度相关）：

- `lmax_s = min(Lmax, avg_tour_cardinality)`  

2) 再把 “平均删除客户数 c̄” 转成可选 strings 数的上界：

- `kmax_s = 4*c̄/(1 + lmax_s) - 1`  

3) 实际删除 strings 数：

- `ks = floor( U(1, kmax_s + 1) )`

4) 对每个被选中的 route `t`，其 string 长度上界：

- `lmax_t = min(|t|, lmax_s)`

5) 实际删除长度：

- `lt = floor( U(1, lmax_t + 1) )`

**直觉**：

- 你给一个 c̄（想平均删多少客户），算法会在“多删短串”与“少删长串”之间自适应选择。  
- `Lmax` 约束了过长串（作者最后用 `Lmax=10`，见参数表）。

**论文原文对这套采样的一个关键解释（来自参数表的文字）**

- 当当前解的 routes 足够长（平均/典型 tour cardinality ≥ `Lmax`）时：`lmax_s≈Lmax=10`，于是 `ks` 的取值通常落在 `1..3`（也就是说：**多数时候只删 1–3 条 tours，但每条删一个较短的 string**）。
- 当搜索刚开始（初解每条 tour 只有 1 个客户）时：`lmax_s=1`，于是 `ks` 可能达到 `1..19`（很多 tours 各删 1 个客户），这基本对应“从 trivial 初解快速合并 routes”的构造过程。

这段解释非常重要：它说明 SISRs 的 R− 在不同阶段会自动表现出“粗构造（多 tour）→ 局部改进（少 tour）”的形态。

### 5.3 “删哪些 routes”：围绕 seed 的“相邻 tours”

ruin 从一个 seed customer 开始：

1. 选 seed：`c_seed ← randomCustomer(s)`  
2. 遍历 `adj(c_seed)`：对每个客户 `c`，找到它所在 tour `t`  
3. 如果 `t` 还没被 ruinned（每个 tour 最多 ruinned 一次），则在 `t` 上删一个 string  
4. 重复直到删了 `ks` 个 tours（也就是 `ks` 条 strings）

这里的关键是：你并不是“随机选 ks 条 routes”，而是通过 `adj(c_seed)` 去选 **离 seed 区域最近的 ks 条 routes**。这就是“Adjacent string removals”产生集中 slack 的核心机制。

### 5.4 两种 string 删除模式：string vs split-string（各 50%）

**(A) string removal**  
从 tour `t` 中随机选一个长度为 `lt` 的连续子串，但要求子串包含 `c*_t`（其中 `c*_t` 是该 tour 中距离 seed 最近的客户）。  
效果：删掉一段空间上相邻的访问块，很容易删掉 detour。

更贴近“可直接写代码”的细节：

- 设 tour `t` 的客户序列（不含 depot）为 `v[0..|t|-1]`，`c*_t` 位于位置 `pos`。
- 所有“长度为 lt 且包含 pos”的子串起点 `start` 满足：  
  `start ∈ [pos-lt+1, pos] ∩ [0, |t|-lt]`
- 论文做法是：在这些可行 start 中 **均匀随机选 1 个**，然后删除 `v[start : start+lt]`。

**(B) split-string removal**  
这是一个很“聪明的多样性注入”：

- 先选一个长度为 `l + m` 的连续段（包含 `c*_t`）  
- 然后“保留”其中连续的 `m` 个客户（不删），其余 `l` 个删掉  
- `m` 的生成过程由 `α` 控制（几何式增长）：从 `m=1` 开始，若 `U(0,1) < α` 则停止，否则 `m++`，直到 `m=mmax`

因为 `α` 很小（作者用 `α=0.01`），所以：

- “小 m”发生概率很低（少保留、多删集中在 `c*_t` 附近）
- “m=mmax”发生概率很高（更像删 route 的开头/结尾，常发生在靠 depot 的位置）

这相当于在“强局部 slack”与“更结构化的 route 端点扰动”之间做随机混合，增强搜索的覆盖。

更贴近原文的概率描述（Table 3 的文字）：

- `P(m=1)=α`
- `P(m=2)=(1-α)α`
- `P(m=3)=(1-α)^2 α`
- …  
- `P(m=mmax)=(1-α)^(mmax-1)`（截断几何分布的尾部）

实现 split-string removal 时，你可以按论文的“循环自增”方式生成 m（无需显式采样几何分布），然后：

1. 随机选一个长度为 `lt + m` 且包含 `c*_t` 的连续段（方式与 string removal 相同，只是长度变成 `lt+m`）
2. 在这个段里再随机选一个长度为 `m` 的连续子串作为“保留块”
3. 删除剩余 `lt` 个客户

## 5.5 Ruin 的代码级要点（Algorithm 2 的三个“容易写错”的地方）

1. **`adj(c)` 要包含自己作为第一个元素**：这样 `c_seed` 所在 tour 会最先被 ruinned（原文明确假设）。
2. **每个 tour 每轮最多 ruinned 一次**：用集合 `R` 记录 ruinned tours（Algorithm 2 的 Line 4 与 Line 10）。
3. **`c*_t` 的定义不是“tour 中随机一个点”**：它是“tour t 中距离 seed 最近的客户”。Algorithm 2 通过扫描 `adj(c_seed)` 来隐式实现：第一次遇到属于 tour t 的客户 c，就是 `c*_t`（因为 adjacency list 已按距离排序）。

## 6. Recreate：Greedy insertion with blinks（Algorithm 3）

### 6.1 先排序 absent customers（多种 order 混合）

重建时先对 `A` 排序。作者提供 4 种 order，并用固定权重随机选择：

- Random（权重 4）
- Demand（权重 4，需求大者先插）
- Far（权重 2，离 depot 远者先插）
- Close（权重 1，离 depot 近者先插）

这一步已经是一种轻量的“启发式选择”：不同实例/不同局部状态可能偏好不同插入顺序。

### 6.2 插入位置搜索：blink rate β 的含义

对每个 customer `c`：

- 依次遍历 tours（随机顺序）
- 对每条可插的 tour，遍历所有插入位置 `P_t`
- 但每个位置以概率 `β` 被“眨眼跳过”（不评估），以概率 `1-β` 才评估
- 在被评估的位置里选最小插入代价的那个
- 如果所有 tours 都插不进：新开一条 route

**插入代价的实现**：论文默认用经典增量代价（插到边 `(u,v)` 之间）：

- `Δ_ins(c; u,v) = d(u,c) + d(c,v) - d(u,v)`

在实现时通常会把每条 tour 表示成 “depot + customers + depot” 的序列，这样 `(u,v)` 就是相邻节点对。

### 6.3 blink 机制的一个很漂亮的性质：隐式 rank-based sampling

论文给出一个非常清晰的结论（式 10）：

如果把所有可能插入位置按代价从好到坏排序，那么 blink 相当于以几何分布选择 rank：

- `p(r) = (1-β) · β^(r-1)`，`r=1,2,...`

但算法不需要显式排序位置，只要顺序遍历+随机 blink 就能得到这种分布。

作者实测：**β=0（总选最优位置）反而是 sub-optimal**；小 β（例如 1%）更好，因为它给重建注入了结构多样性（类似 ALNS 中的噪声插入/随机扰动，但更“可控”）。

## 7. Metaheuristic：SA（Algorithm 1）如何驱动 SISRs

SA 在这里的角色比较纯粹：

- neighbor：永远是 `ruin+recreate`  
- acceptance：按温度和 `Δdist` 接受一定比例的劣解（逃逸）
- cooling：指数冷却

重要实现细节：

- 初始解：每个客户单独一条 route（因此一开始 routes 数非常大，但保证 trivially feasible）

更贴近原文的 SA 更新公式（Equation 1–4）：

- 冷却：`T_{k+1} = c·T_k`
- `c = (Tf/T0)^(1/f)`（f 为总迭代次数）
- 接受准则（等价写法）：接受 `s*` 若  
  `dist(s*) < dist(s) - T·ln(U(0,1))`

实现建议：

- 直接用上面这条“不含 exp 的不等式”会更数值稳定（避免 `exp(-Δ/T)` 下溢）。
- 对标准 CVRP（非 fleet minimization 阶段），recreate 会在插不进去时新建 tour，所以最终 `A=∅`，`dist(s)` 就是总路长即可。

参数（论文给出一组推荐值）：

- `T0=100`, `Tf=1`
- 迭代次数 it(v) 按 problem size 线性插值，并给了基准：  
  - `it(100)=3×10^7`  
  - `it(1000)=3×10^8`

这解释了为何 SISRs 很强调“每次 neighbor 生成要很快”：迭代数巨大，单次必须极轻量。

## 8. Fleet Minimization（Algorithm 4）：当“车辆数最少”是第一目标

很多 VRP 变体/业务 KPI 是 hierarchical objective：先最少车，再最短路。作者给了一个非常工程的两阶段：

### 8.1 核心思想：absences-based acceptance

对每个客户维护一个 absence counter `abs_c`：它在搜索过程中“有多少次处于 absent 集合 A 中”。

fleet minimization 阶段的 acceptance 规则是：

接受邻居 `s*` 如果：

- `|A*| < |A|`（未服务客户更少），或
- `sumAbs(A*) < sumAbs(A)`（未服务客户的“历史缺席总次数”更小）

这条规则的直觉非常清晰：

- 你不仅想“尽快可行”，还想避免反复遗漏同一批难客户

### 8.2 强制压车：remove “最不重要的” tour

每次找到一个全服务解（`A* = ∅`）就记录为候选最优，然后删除当前解里一个“最弱”的 tour：

- 删除 `sumAbs(t)` 最小的 tour（其客户在历史上很少缺席 → 相对容易被其他 routes 重新覆盖）

然后继续 SISRs 的 ruin&recreate 尝试在更少 routes 下把客户补回去。

### 8.3 参数

fleet minimization 的迭代预算作者建议取总迭代的 10%（经验上几分钟即可压到最小车数）。

## 8.4 直接可抄的参数表（Table 3）

论文给出的“默认最优”参数（对复现 baseline 很关键）：

- Ruin：
  - `c̄ = 10`（平均移除客户数；作者强调太小不够跳坑，太大难重建）
  - `Lmax = 10`（字符串最大长度）
  - `α = 0.01`（split-string 的“停止概率”，很小 → m 往往取到 mmax）
- Recreate：
  - `β = 0.01`（blink rate）
  - absent customer 排序策略权重：Random 4、Demand 4、Far 2、Close 1
- Fleet minimization：
  - 预算：总迭代的 10%
- SA：
  - `T0 = 100`, `Tf = 1`
  - 迭代数 `it(v)`：按 v=100..1000 线性插值，且设置 `it(100)=3×10^7`, `it(1000)=3×10^8`

## 9. 结果与作者结论（你写 related work/对比时有用的点）

论文在多个 benchmark（含 Uchoa 等）上报告：

- 在 large-scale（500–1000）上非常强、非常鲁棒
- 相比一些更复杂的方法（例如 UHGS、ILS-SP），SISRs 在大实例上能赢很多，并且波动更小（robustness）
- 同时强调：其结构简单，便于复现/移植到 VRP 变体

## 10. 对你“两阶段 init + iter 可组合”的直接启发

### 10.1 SISRs 在 pipeline 里是“迭代器”，且天然可组合

- 它只需要当前解（哪怕是很差的初解）  
- 不依赖特定初始化来源  
→ 你完全可以做 “LEHD init + SISRs iter” 或 “GLOP init + SISRs iter” 的组合实验。

### 10.2 你可以把 SISRs 拆成“init/iter 两段选择”的最自然示例

SISRs 本身就是两段结构：

- ruin：决定“删什么”（强结构决策，动作空间大但反馈密集）
- recreate：决定“怎么补”（插入顺序 + blink rate 等）

因此对你的两层 gate 来说，SISRs 提供了一种很典型的“可学习动作空间”：

- Gate1：选择 ruin 强度（`c̄, Lmax, ks 分布`）与模式（string vs split-string）  
- Gate2：选择 recreate 顺序策略与 β（以及是否附加局部搜索）

### 10.3 把 SISRs 融合到你在写的“算子选择 baseline”里

如果你把每个 iterator 当作 operator：SISRs 的特点是能提供非常清晰的 credit：

- `credit = Δcost / Δtime`  

而且它“非平稳”很强（早期需要大破坏，后期需要小破坏）。这非常适合你在 `adaptive/instance_based_algorithm_selection.md` 第 12 节里提到的 dynamic bandit / adaptive pursuit / ALNS 权重更新等 baseline 做 Stage-2 调度对比。
