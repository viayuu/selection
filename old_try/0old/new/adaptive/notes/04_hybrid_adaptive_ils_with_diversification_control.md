# 04) A hybrid adaptive Iterated Local Search with diversification control to the CVRP（AILS-PR, Máximo & Nascimento, 2020/2021）

- 预印本：arXiv `2012.11021`（PDF 标注 “Preprint submitted to EJOR, Dec 2020”）
- 本地 PDF：`adaptive/pdfs/04_hybrid_adaptive_ils_diversification_control_2021.pdf`
- 公开预印本：`https://arxiv.org/pdf/2012.11021`
- 备注：你旧笔记里写了 `10.1016/j.ejor.2021.02.024`，但该 DOI 不在本 PDF 文本中；若你确实用发表版引用，请用官方页面核对后再补。

## 0.5 代码可获得性（论文是否给出开源链接）

- 在这份 arXiv/EJOR 版本 PDF 的正文中未检索到作者提供的开源仓库/代码下载链接（GitHub/GitLab/Bitbucket 等）。
- 但同一作者后续工作 AILS‑II 的代码在 INFORMSJoC 提供了可下载仓库（见本项目 `adaptive/notes/06_ails_ii_adaptive_iterated_local_search.md`），并且**把 AILS‑PR 中最关键的两条自适应闭环（ω 与 η）以非常直接的类实现出来**，可用于验证“论文公式如何落到工程代码”：
  - 本地 AILS‑II 源码：`adaptive/repos/06_ails_ii_informsjoc_2023_0106/`
  - `DiversityControl/OmegaAdjustment.java`：对应 AILS‑PR Algorithm 2 的比例控制更新（`ω ← ω * dβ / d̄`）
  - `DiversityControl/AcceptanceCriterion.java`：对应 AILS‑PR 的阈值接受（`b̄ = f + η( f̄ − f )`）与 η 的收敛调节

### 0.5.1 用 AILS‑II 开源代码“验证/澄清” AILS‑PR 的两个核心公式（注意：这是 AILS‑II 的实现，但机制同源）

**(1) ω 的比例控制更新（对应 AILS‑PR Algorithm 2）**：AILS‑II 里写成了一个非常直白的“把均值距离拉回目标距离”的更新。

```java
// DiversityControl/OmegaAdjustment.java
obtainedDist = meanLSDist.getDynamicAverage();
omega += ((omega/obtainedDist*idealDist.idealDist) - omega); // = omega * ideal / obtained
omega = Math.min(omegaMax, Math.max(omega, omegaMin));
```

**(2) 阈值接受与 η 的调节（对应 AILS‑PR Section 4.1.2）**：阈值是 `upperLimit + eta*(avg-upperLimit)`；并且代码里把阈值 cast 成 `int`（如果你的目标值是整数，这个细节会影响“刚好卡在阈值附近”的接受行为）。

```java
// DiversityControl/AcceptanceCriterion.java
eta *= alpha;
eta = Math.max(eta, etaMin);
thresholdOF = (int)(upperLimit + (eta*(averageLSfunction.getDynamicAverage()-upperLimit)));
return solution.f <= thresholdOF;
```

## 0. 一句话总结（你读完应当留下的印象）

AILS-PR 把 ILS 里最难调、也最影响“跳坑/收敛”的两件事做成**反馈闭环**：

1. **扰动强度 ω**：用结构距离 `d(s, sr)` 把不同扰动算子的“平均跳跃幅度”对齐到目标 `dβ`  
2. **接受阈值 η**：用目标接受流量 `κ` 动态调节接受阈值，控制搜索的“松紧程度”

然后再用 **elite set + Path-Relinking（PR）** 做重型 intensification。

> 在你的两阶段视角里：这是一整套“迭代器（iteration）”的经典手工自适应 baseline（无训练、无离线标注），非常适合拿来对比你后续 learning-based 的 gate。

## 1. 论文贡献点（按“你复现/移植时最有用的点”整理）

1. **Perturbation degree 的自适应（Algorithm 2）**：每个 removal heuristic Hr_k 独立维护 `ω^{Hr_k}`，用比例控制逼近目标距离 `dβ`。
2. **Acceptance criterion 的自适应（Section 4.1.2）**：阈值 `b̄ = f + η( f̄ − f )`，并用“接受流量控制”更新 η（而不是 SA 那种温度）。
3. **算子选择非常克制**：作者提到试过像 ALNS 那样按历史改进更新算子概率，但效果不佳；最终 **每轮随机选 removal heuristic**，把主要 adaptiveness 放在“强度 + 接受阈值”上。
4. **PR 的一个重要简化**：PR 的“相似性”只看客户属于哪条 route（集合交集），**不管 route 内顺序**；动作变成“把客户搬回对应 route”，更稳、更便宜。
5. **可复现性好**：CVRP 的构造初解、扰动算子、可行化与局部搜索都写成了非常具体的 appendix 算法（你实现 baseline 时很省力）。

## 2. AILS 的通用骨架（Algorithm 1）

AILS 结构和标准 ILS 一样，但多了两步自适应更新：

1. 构造初解 `s`
2. `sr, s* ← LocalSearch(s)`（sr=reference solution，s*=best found）
3. repeat:
   - `s ← Perturbation(sr, Hr_k)`（每次随机选一个 removal heuristic Hr_k）
   - `s ← LocalSearch(s)`
   - **更新 `ω^{Hr_k}`**（用距离反馈，见 §4）
   - `sr ← Apply acceptation criterion to s`（用阈值 b̄ 决定是否替换 sr）
   - **更新 η**（用接受流量反馈，见 §5）
   - 若更好则更新 `s*`

## 3. “距离”到底是什么：d(s, sr)（Equation 1）

对 CVRP，他们用边集合的对称差当作结构距离：

- `E_s`：解 s 的边集合
- `d(s, sr) = |E_s △ E_sr|`

这个 d 一箭双雕：

- 用来调 perturbation 强度（让每次跳跃“到位”）
- 用来管 elite set 的多样性（保持解之间别太像）

## 4. 自适应扰动强度：更新 ω（Algorithm 2）

### 4.1 ω 的意义

每个 removal heuristic `Hr_k` 有一个强度参数 `ω^{Hr_k}`，决定一次 perturbation 要删掉多少客户（或删多长字符串等）。

### 4.2 更新法：让“平均距离”逼近 dβ

为每个 Hr_k 维护统计量：

- `it^{Hr_k}`：该 Hr_k 被使用的次数
- `d^{Hr_k}`：累计距离和（累计 `d(s, sr)`）

每次用 Hr_k 产生 s 并 local search 后：

- `it^{Hr_k} += 1`
- `d^{Hr_k} += d(s, sr)`

当 `it^{Hr_k} == γ`（γ 是更新周期）时更新：

- `ω^{Hr_k} ← ω^{Hr_k} · dβ / d^{Hr_k}`
- 并 clip：`ω^{Hr_k} ← min{size, max{1, ω^{Hr_k}}}`  
  对 CVRP：`size = n`（客户数）
- 然后把 `it^{Hr_k}, d^{Hr_k}` 清零

直觉：

- 若 Hr_k 这段时间平均跳得太近（d 小）→ ω 变大  
- 若跳得太远（d 大）→ ω 变小

这就是一个非常经典的“比例控制（P-control）”式自适应，优点是可解释且实现简单。

## 5. 自适应接受准则：阈值 b̄ + 调 η（Section 4.1.2）

### 5.1 阈值形式：b̄ = f + η( f̄ − f )

定义：

- `f(s)`：当前解目标值
- `f`：最近 `min(it, γ)` 次迭代里出现的最好目标值（窗口 best）
- `f̄`：local search 产出的解质量的运行均值（weighted average）

阈值：

- `b̄ = f + η( f̄ − f )`，其中 `η ∈ [0,1]`
- 所以 `b̄ ∈ [f, f̄]`
  - η 大 → 阈值接近 f̄ → 更松 → 接受更多
  - η 小 → 阈值接近 f → 更严 → 接受更少

### 5.2 f̄ 的更新（Equation 2）

论文使用一个“冷启动 + 后续滑动平均”的分段更新：

- 若 `it > γ`：`f̄ ← f̄(1−1/γ) + f(s)/γ`
- 若 `it ≤ γ`：`f̄ ← (f̄(it−1) + f(s))/it`

### 5.3 用目标接受流量 κ 来调 η

设定目标接受比例 `κ ∈ [0,1]`（论文称 flow）。

- 统计自上次更新以来的实际接受比例 `κ_r`
- 每当累积到一批（论文表述：每 γ 个被接受解）就更新一次：
  - `η ← max{ ε, κ·η / κ_r }`
  - ε 是很小的正数，防止 η 降到 0

直觉：

- 若实际接受太少（κ_r < κ）→ η 上升 → 阈值更松
- 若实际接受太多（κ_r > κ）→ η 下降 → 阈值更严

## 6. CVRP 具体实现（Online Appendix A）：初始化、扰动、可行化、局部搜索

这一部分是你复现 baseline 时最直接能用的“配方”。

### 6.1 初解（Appendix A.1, Algorithm 1）

作者认为初解质量对最终影响不大，因此用一个很简单且可扩展的构造：

1. 车辆数下界：`m = ceil( (1/Q) · Σ_i q_i )`
2. 初始化 m 条路线，每条路线随机放 1 个客户
3. 剩余客户按随机顺序插入：
   - 先用插入启发式 `Ha1` 选“插到哪条 route”
   - 再在该 route 内选“最低插入增量代价”的位置插入

此时可能 infeasible（route 超载），后面用 feasibility heuristic 修复。

按 Appendix A.1 的伪代码更细一点（Algorithm 1）：

- `Va ← V_c`（未分配客户集合）
- for `j=1..m`：
  - 随机取 `v ∈ Va`，令 `R_j ← (0, v, 0)`，并 `Va ← Va \\ {v}`
- while `Va ≠ ∅`：
  - 随机取 `v ∈ Va`，`Va ← Va \\ {v}`
  - 用 `Ha1` 选择 `v` 要插入的 route `R̂`（见 §6.2）
  - 在 `R̂` 内选择最低插入增量位置插入 `v`

作者在 appendix 里明确提醒：这个构造 **不保证可行**（可能超载），因此会立刻进入 feasibility procedure（§6.3）。

### 6.2 扰动算子（Appendix A.2）

扰动由 “removal + insertion” 两段组成；作者用 3 个 removal + 2 个 insertion。

#### Removal Hr1：Concentric removal

- 随机选中心客户 `v_r`
- 删除 `v_r` 以及其 `ω^{H1}-1` 个最近邻客户（一个几何团块）

#### Removal Hr2：Proximity removal（用相对近邻排名定义 proximity）

关键是一个“相对 proximity index”（尺度无关）：

- `Π_R(v)`：route R 中其它客户相对 v 的“全局近邻排名”集合
- `ρ`：每轮随机取 `ρ ∈ [1, floor(n/m)]`
- `minset(a,S)`：集合 S 中 a 个最小元素之和

`prox(v,ρ,R) = minset(min(ρ,|R|-2), Π_R(v)) / min(ρ,|R|-2)`

其中（按 Appendix A.2 原文解释）：

- `Π_R(v)` 不是“距离值”，而是 **全局近邻排名**：把 route `R\\{0,v}` 中每个客户 u，在“全图 V 上距 v 的近邻排序”里的名次取出来，形成一个集合。
  - 直觉：如果 v 的 route-mates 在全局排序里都很靠前（名次小），说明 v 与当前 route 很“贴合”。
- `minset(a,S)` 是集合 S 中 a 个最小元素的和；因此 `prox` 取的是“前 ρ 个最贴近 route-mates 的平均全局名次”，具有尺度无关性。

移除策略：

- 把未删除客户集合按 prox 降序排序（prox 越大越“贴合当前 route”，越优先删）
- 按一个强烈偏向排序前部的概率分布采样并删除 `ω^{H2}` 个（Appendix 给了显式分布）：
  - 记 `S` 为“尚未被删除的客户集合”，`o_v ∈ {0,…,|S|-1}` 为 v 在上述排序中的位置（0 表示排序第一）
  - 论文使用一个指数型权重来采样，等价实现可以写为：`P(o_v) ∝ 2^{-|o_v|}`（越靠前概率越大）
  - 你实现时更建议直接按论文给的权重做 `alias sampling` 或累计分布采样（|S| 可能很大）

#### Removal Hr3：Removal of vertex sequences

从 route 中随机截取若干段连续序列（长度随机），直到累计删除数达到 `ω^{H3}`：

- 每次截取前，随机采样一个序列长度 `ℓ ∈ [1, floor(n/m)]`（论文按当前 m 归一化）
- 随机选择一条 route 与一个起点位置，取连续 ℓ 个顶点形成“环状序列”
- 序列允许跨越 depot（例如 `(..., 8, 11, 0, 2, ...)`），但 depot 不会被删除（只删除客户 8/11/2）

#### Insertion Ha1：Insertion by Proximity

- 先选 route：`R̂ = argmin_R prox(v, ρ, R)`
- 再选插入位置：最小化插入增量  
  `d(v_i,v)+d(v,v_{i+1})−d(v_i,v_{i+1})`

这对应 Appendix 的 Eq.(A.3)：`î = argmin_i Δ_ins(i)`。

#### Insertion Ha2：Insertion by Cost

直接全局找“插入增量代价最小”的 route+位置。

#### Perturbation Procedure（Algorithm 2）的额外点：允许改变 route 数

每次 perturbation，有概率 `1/γ` 把 route 数 m 改变 ±1（且不低于下界 m）：

- m−1：删除一条随机 route
- m+1：加入一条空 route

这是一个“结构层”的扰动：不仅重排客户，也允许 route 数变化。

按 Appendix A.2 的伪代码（Algorithm 2）更完整地写，是：

1. `s' ← s`（copy）
2. 以概率 `1/γ` 把 route 数 `m` 增减 1（同时保持 `m ≥ m_lower`，m_lower 由 Eq.(A.1) 给出）：
   - `m ← m-1`：随机删除一条 route（其客户会进入“待插入集合”）
   - `m ← m+1`：加入一条空 route
3. 采样 `ρ ∈ [1, floor(n/m)]`
4. 选一个 removal `Hr_k`，删除 `ω^{Hr_k}` 个客户（得到 removed list）
5. 在 `{Ha1,Ha2}` 中随机选一个 insertion，把 removed list 全部插回解中

### 6.3 可行化与局部搜索：共用邻域、不同准则（Appendix A.3–A.4）

#### Inter-route 邻域（λ=1）

`N = {Shift, Swap, 2-opt*}`，并给了 O(1) 的增量代价公式（Δ）。

#### Intra-route 邻域

`N^- = {Shift-, Swap-, 2-opt-}`，intra-route LS 用 best improvement，只接受 Δ<0。

#### Neighborhood Search（Algorithm 3–4）：候选表 LM + 近邻集合 δ(v)

他们维护候选 move 集合 `LM`，候选来自每个客户 v 的近邻集合 `δ(v)`：

- `|δ(v)| = ϕ`（参数，控制复杂度）
- appendix 的表述是“按 proximity/近邻次序取最近的 ϕ 个点”；工程实现里通常就是按欧氏距离预计算 v 的 ϕ-NN 列表（与 Hr2 的全局近邻排名可复用同一套排序/索引）
- 复杂度约 `O(n·ϕ)`

Algorithm 3 的关键循环：

1. `LM ← ∅`
2. 对每条 route 用 `UpdateLM` 填充候选
3. while `LM` 非空：
   - 选一个最优 move Θ（依据 mode 不同）
   - 应用 move；删除涉及被修改 routes 的旧候选；对受影响 routes 做 intra-route LS
   - 再对受影响 routes 调用 `UpdateLM`
4. 若 `LM` 为空但解仍 infeasible：新增一条 route（给 feasibility 一个“容器”），直到可行

Feasibility mode 与 Local search mode 的核心差异：

- **Feasibility mode**：优先减少超载（引入 feasibility gain Ω），必要时允许成本上升
- **Local search mode**：只做成本下降（同时不允许可行性变差）

#### 6.3.1 Feasibility gain Ω 与排序指标 Λ（Appendix A.3.1 的可复现公式）

论文把“容量可行性”写得非常工程化（你实现时可以几乎直接照搬）：

- 定义 route 的 slack（Eq.(A.7)）：
  - `slack(R) = Q - Σ_{i∈R} q_i`
  - `slack(R) ≥ 0` 即该 route 可行（不超载）
- 设一次 inter-route move 只影响两条 route：origin `R_i` 与 destination `R_j`（move 后变成 `R_i' , R_j'`）。
- 定义 feasibility gain（Eq.(A.8)）为“超载量（负 slack）的减少”：
  - `Ω(s,s') = min(0, slack(R_i')) + min(0, slack(R_j')) - min(0, slack(R_i)) - min(0, slack(R_j))`
  - 因此 `Ω>0` 表示总超载量减少（更接近可行）
- 定义 move 的成本变化 `Δ(s,s')`（论文对 Shift/Swap/2-opt* 都给了 O(1) 的增量公式）
- 定义 feasibility procedure 的排序指标（Eq.(A.9)）：
  - 若 `Δ ≤ 0`：`Λ = Δ`（成本不升就直接优先）
  - 若 `Δ > 0`：`Λ = Δ / Ω`（单位可行性增益的代价）

并且 feasibility procedure 在 UpdateLM 阶段会加一个“route 对过滤条件”（Appendix A.3.1 总结）：

- 只考虑 `(slack(R_i) ≥ 0) XOR (slack(R_j) ≥ 0)` 的 route 对（一个可行、一个不可行），因为它的目标是把超载 route 修回可行。

Local search mode 则更严格（Appendix A.4）：

- 不用 `Λ`，直接用 `Δ` 排序（best improvement）
- move 需要满足：`Ω ≥ 0` 且 `Δ < 0`（既不让可行性变差，又要真改进成本）

这解释了为什么作者能保证整体复杂度 `O(n·ϕ)`：他们把 inter-route 搜索的候选限制为 `v_j ∈ δ(v_i)`（每个点只看 ϕ 个候选），再在每个候选对上枚举一个很小的 move 集合 `N={Shift,Swap,2-opt*}`（λ=1）。

## 7. PR（Path-Relinking）：elite set + route pairing + priority（Section 5）

### 7.1 Elite set 家族 E（Algorithm 3）

维护 `E = {E_m, E_{m+1}, ..., E_n}`（按 route 数分桶）：

- 每个 `E_m` 容量上限 `σ`
- 同一 `E_m` 内任意两解距离 > `dβ`

更新逻辑是典型的 “quality + diversity”：

- 未满：只要与所有已有解距离足够大就加入
- 满了：如果不比最差解差，且不会太靠近更优的解，则允许进入，并踢掉距离≤dβ 的较差解（或替换一个“最接近且更差”的解）

### 7.2 PR 主过程（Algorithm 4）

对当前 local optimum 解 s（m 条 route）：

1. 从 `E_m` 随机挑一个 elite 解 `s_e`
2. 在 {s, s_e} 中随机指定：初始解 `s_i` 与引导解 `s_g`
3. 计算 route 匹配 `φ`（Algorithm 5）
4. 定义 `NF`：初始解中“不在匹配 route 交集里”的客户集合  
   `NF = ⋃_k R_i^k \\ (R_i^k ∩ R_g^{φ(k)})`
5. 随机选一套 priority 规则 `C`（10 个之一）
6. while `NF` 非空：
   - 计算每个 `v ∈ NF` 的 priority `p_v`（Appendix B）
   - 取最高 priority 的 `v̂`（tie 用 SHIFT 代价更小的）
   - 设 `v̂` 在 guide 解中属于 route `R_g^l`  
     → 把 `v̂` 搬到初始解里对应的 `R_i^{φ^{-1}(l)}`，并以最小插入增量代价的位置插入
   - 若当前解可行且更好：更新 PR best `s_b`
7. 对 `s_b` 再做一次 local search（把 PR 得到的半熟解“煮熟”）
8. 用 `s_b` 更新 elite set

关键点：PR 的相似性只看“客户属于哪条 route”（集合），不管顺序；因此 PR 的终点未必等于 guide 解，但能更稳定地产生高质量中间解。

### 7.3 Route pairing（Algorithm 5）

不用 Hungarian（太贵），用贪心：

- 重复 m 次：选尚未配对的 route 对 `(R_i^k, R_g^l)` 使交集最大 `|R_i^k ∩ R_g^l|`  
  然后设 `φ(k)=l` 并移除这两条 route

### 7.4 Vertex priority（Appendix B）

Appendix B 把“先搬哪个客户”写成一个可复现的离散规则系统：对每个候选客户 `v ∈ NF`，先判断它在“origin/destination 两条 route 的可行性状态”，再用一套规则 `C` 把这些状态映射成优先级 `p_v`。

**(1) 定义状态（Table B.1）**

记 `R_o(v+) / R_o(v-)` 为 v 所在 origin route 在“移除 v 之前/之后”的可行性；`R_d(v-) / R_d(v+)` 为 destination route 在“插入 v 之前/之后”的可行性。则：

- Origin 状态 `o ∈ {1,2,3}`（附带 priority factor）：
  - `o=1`：`R_o(v+)` infeasible → `R_o(v-)` feasible，factor `+1`
  - `o=2`：`R_o(v+)` infeasible → `R_o(v-)` infeasible，factor `+1`
  - `o=3`：`R_o(v+)` feasible → `R_o(v-)` feasible，factor `-1`
- Destination 状态 `d ∈ {4,5,6}`（附带 priority factor）：
  - `d=4`：`R_d(v-)` feasible → `R_d(v+)` infeasible，factor `-1`
  - `d=5`：`R_d(v-)` infeasible → `R_d(v+)` infeasible，factor `-1`
  - `d=6`：`R_d(v-)` feasible → `R_d(v+)` feasible，factor `+1`

作者的直觉也写得很明确：状态 1/2/6 对“走向可行解”更有利（加分），状态 3/4/5 更不利（减分）。

**(2) 定义 10 套规则 C1..C10（Table B.2）**

作者没有用 64 种全组合，而是挑了 10 套代表性规则。每个规则 `C` 指定“哪些状态是 active 的”：只有 active 的状态才会贡献它的 `±1` factor。

因此优先级的计算方式非常简单：

- 初始化 `p_v = 0`
- 若 origin 状态 `o` 在规则 `C` 中是 active：`p_v += factor(o)`
- 若 destination 状态 `d` 在规则 `C` 中是 active：`p_v += factor(d)`
- 最终 `p_v ∈ {-2,-1,0,1,2}`

论文还提供了一个“查表版”（Table B.3）：对每个 `(o,d)` 与每个 `C_k` 直接给出最终 `p_v`。你实现时最省事的方式就是把 Table B.3 预抄成 `priority[C][o][d]` 的 lookup。

**(3) tie-breaking**

Algorithm 4 还规定：若有多个 v 的 `p_v` 相同，选其中 SHIFT 代价（把 v 挪到目标 route）最小的那个。

## 8. irace 调参结果（Table 1）

论文用 irace 得到 5 个关键参数：

- `γ = 20`：更新周期（ω 更新、route 数变化概率 1/γ、acceptance 统计窗口等）
- `κ = 0.35`：目标接受流量
- `dβ = 24`：目标结构距离（同时也是 elite 的最小距离）
- `σ = 63`：每个 `E_m` 的最大容量
- `ϕ = 60`：近邻集合大小（控制 `O(nϕ)`）

## 8.1 复现时的实现蓝图（把 Appendix A 变成可运行代码）

如果你要把 AILS-PR 当作“可对照的 online baseline”，最关键的是把 Appendix A 的数据结构写对：

1. **基础表示**
   - route 用数组/链表都可以，但一定要维护：
     - `load(route)`（便于计算 slack）
     - `route_id[v]` 与 `pos[v]`（便于 O(1) 定位 v 在哪条 route 哪个位置）
2. **近邻集合 δ(v)**
   - 预计算每个客户 v 的 ϕ 个近邻（通常按欧氏距离排序取前 ϕ）
   - 这会直接把 inter-route 的候选对从 O(n²) 降到 O(nϕ)
3. **候选表 LM**
   - 按 Algorithm 4：对每个候选对 `(v_i, v_j)`，你只需要在 `N={Shift,Swap,2-opt*}` 里保留“对当前准则最好的那一个 move”
   - 工程实现可以用字典/哈希：`LM[(v_i, v_j)] = best_move`，并维护一个可快速取 argmin 的结构（例如把 move 放到 list 后每轮线性扫；ϕ=60 时可接受）
4. **Feasibility 相关增量**
   - `slack(route) = Q - load(route)`（Eq.(A.7)）
   - `Ω(s,s')` 只依赖两条 route 的 slack 变化（Eq.(A.8)），因此不需要全局扫描
   - feasibility 模式用 `Λ` 作为比较键（Eq.(A.9)），local search 模式用 `Δ`
5. **扰动与强度/阈值的闭环**
   - `ω^{Hr_k}` 的统计量：`it^{Hr_k}`, `d^{Hr_k}`（Algorithm 2）
   - η 的统计量：记录最近一段窗口的接受率 `κ_r`（用于 `η ← max{ε, κη/κ_r}`）

你会发现 AILS-PR 的“自适应”主要是 **少量标量参数的反馈控制**，因此非常适合作为你后续 learning-based gate 的强 baseline：实现简单、解释清晰、跑起来稳定。

## 9. 对你“两阶段选择 / baseline 设计”的直接启发（很具体）

1. **这是一个非常“好上手”的 online baseline**：无训练、无离线枚举组合，用反馈闭环自动调参。
2. **它提示你：算子选择不一定要先做“算子概率学习”**：作者尝试过概率更新但不如随机；但“强度/阈值”这类连续参数的闭环非常有效。
3. **你可以把 AILS-PR 当作 Gate2 的候选迭代器**，或把它内部的 Hr1/Hr2/Hr3 当作 Gate2 动作集；再用你的实例特征去预测：
   - `dβ`（需要多大跳跃）  
   - `ϕ`（邻域预算）  
   - 哪个 Hr 更适合（几何团块 vs 字符串 vs proximity）
