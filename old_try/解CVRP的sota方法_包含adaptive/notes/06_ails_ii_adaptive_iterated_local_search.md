# 06) AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-scale CVRP（Máximo, Cordeau, Nascimento, 2024）

- 论文：Vinícius R. Máximo, Jean‑François Cordeau, Mariá C. V. Nascimento, *AILS‑II: An Adaptive Iterated Local Search Heuristic for the Large‑scale Capacitated Vehicle Routing Problem*, INFORMS Journal on Computing (2024)
- DOI：`10.1287/ijoc.2023.0106`
- 本地 PDF：`adaptive/pdfs/06_ails_ii_adaptive_ils_large_scale_cvrp_2024.pdf`
- 预印本：`http://arxiv.org/pdf/2205.12082`

## 0.5 代码（开源链接 + 本地下载）与论文-代码对照

- 论文 Section 4 明确给出源码地址：`https://github.com/INFORMSJoC/2023.0106`（MIT License）。
- 本地已下载：`adaptive/repos/06_ails_ii_informsjoc_2023_0106/`（commit `4ff361153681635149e30305057251e9035ec6b7`）。
- README 同时指出：持续维护版本在 `https://github.com/vinymax10/AILS-CVRP`。
- 代码组织与论文流程（Figure 1 / Algorithm 1）对应（核心文件）：
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/SearchMethod/AILSII.java`：主循环（construct → feasible → LS → perturb → feasible/LS → adapt → acceptance）
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/SearchMethod/InputParameters.java`：命令行参数（`-file/-rounded/-limit/-stoppingCriterion/-dMax/-dMin/-gamma/-varphi`）
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/SearchMethod/Config.java`：默认超参（`etaMin/etaMax/gamma/dMin/dMax/varphi`、扰动算子池与插入启发式）
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/DiversityControl/*`：收敛式多样性/接受控制（`DistAdjustment/OmegaAdjustment/AcceptanceCriterion`）
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/Perturbation/*`：`Concentric/Sequential` removal + insertion（`InsertionHeuristic`）
  - `adaptive/repos/06_ails_ii_informsjoc_2023_0106/src/Improvement/*`：feasibility repair + local search

### 0.5.1 从源码读到的“关键实现细节”（帮助你做 baseline 对齐）

- **停止条件（Time/Iteration 两套）**：`SearchMethod/StoppingCriterionType.java`，以及 `AILSII.stoppingCriterion()`：
  - 既可以按 `-stoppingCriterion Time -limit seconds`，也可以按 `Iteration -limit iters`
  - 另外支持 `bestF <= optimal` 提前停止（如果你传了 `-best`）。
- **扰动算子池是“写死的 2 个”**：`Config.perturbation = [Sequential, Concentric]`；`AILSII` 用反射 `Class.forName(\"Perturbation.\" + ...)` 实例化（`AILSII.java` 构造函数）。
- **目标距离 `d_i` 的收敛实现**：`DiversityControl/DistAdjustment.java` 维护一个共享的 `IdealDist`：
  - 每轮按指数更新 `idealDist.idealDist *= alpha`，并 clamp 到 `[dMin, dMax]`
  - `alpha` 在 Iteration 模式下是 `pow(dMin/dMax, 1/limit)`；Time 模式下通过“已用时间比例”估计总迭代数再计算同样的指数因子。
- **每个扰动算子自己的 `ω_k` 自适应**：`DiversityControl/OmegaAdjustment.java`：
  - 用 `Mean(config.getGamma())` 维护 `obtainedDist = meanLSDist.getDynamicAverage()`
  - 周期性（每 `gamma` 次）做一个比例调整：`omega += ((omega/obtainedDist*idealDist.idealDist) - omega)`，并 clamp 到 `[1, size-2]`
  - `AILSII.search()` 里把一次 local search 后的结构距离写回：`selectedPerturbation.getChosenOmega().setDistance(distanceLS)`。
- **接受准则 η 的收敛实现**：`DiversityControl/AcceptanceCriterion.java`：
  - 维护 `upperLimit`（近期最好值）与 `Mean(config.getGamma())` 的均值
  - 阈值 `thresholdOF = upperLimit + eta*(avg-upperLimit)`（注意：代码里 `thresholdOF` 被 cast 成 `int`）
  - `eta` 通过 `eta *= alpha` 逐步衰减到 `[etaMin,etaMax]`，其中 `alpha` 会根据 Time/Iteration 模式与剩余预算估计动态调整（实现了“越到后期越严格”的效果）。
- **插入启发式与 `φ/varphi` 的实际含义**：`Perturbation/Perturbation.java`：
  - `InsertionHeuristic.Distance` 实际上只看 very few（代码里 limit=1）的候选位置
  - `InsertionHeuristic.Cost` 使用 `limitAdj = varphi`（默认 40）扩大候选位置集合
  - 候选位置来自节点的 KNN 列表与 depot 特判（`getBestKNNNo2`）。

### 0.5.2 关键代码片段（把论文公式“钉”在实现上）

**(1) 目标距离 `d_i` 的收敛（convergent diversity control）**：`DistAdjustment.distAdjustment()` 每步把 `idealDist` 乘一个 `alpha` 并 clamp 到 `[dMin,dMax]`。

```java
// DiversityControl/DistAdjustment.java
idealDist.idealDist *= alpha;
idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
```

**(2) ω 的比例控制更新（对应论文/流程图里“update perturbation degree”）**：

```java
// DiversityControl/OmegaAdjustment.java
obtainedDist = meanLSDist.getDynamicAverage();
omega += ((omega/obtainedDist*idealDist.idealDist) - omega); // = omega * ideal / obtained
omega = Math.min(omegaMax, Math.max(omega, omegaMin));
```

**(3) 阈值接受（threshold acceptance）**：阈值在代码里 cast 成 `int`（如果你的目标值是整数，这会影响边界接受行为）。

```java
// DiversityControl/AcceptanceCriterion.java
thresholdOF = (int)(upperLimit + eta*(averageLSfunction.getDynamicAverage()-upperLimit));
return solution.f <= thresholdOF;
```

## 0. 一句话总结（你读完应当留下的印象）

AILS‑II 是把 AILS‑PR（上一代 AILS + PR）**改造成能跑到 3,000–30,000 节点**的版本：  
核心不是“更复杂”，而是：

1. **去掉 PR（太慢）**  
2. **把多样性控制做成“收敛式（convergent）”**：早期更探索、后期更严格  
3. **把局部搜索改成更适合大规模的邻域组合 + 预算裁剪**（只在被扰动 routes 上做搜索、用 SWAP* 但强限制邻域）

因此它是一个非常典型的“**大规模工程化自适应迭代器**”，也非常贴合你要研究的“迭代阶段的算子/策略选择”。

## 1. 这篇相对 AILS‑PR 的关键变化（Section 3.3）

论文明确列了 AILS‑II 为了大规模做的改动（你复现/做 baseline 最该关注）：

1. **Convergent diversity control**：
   - perturbation 目标距离从 `dMax → dMin` 逐步减小（越到后期越精修）
   - acceptance 参数 η 从 1 衰减到 ε=0.01（阈值越来越严）
2. **局部搜索邻域更换**：
   - intra-route：SHIFT, SWAP, 2-opt
   - inter-route：SHIFT, 2-opt*，以及 **SWAP\***（来自 Vidal 2022）
3. **SWAP\* 的邻域裁剪**：不在整条 route 上枚举，而是只考虑每个客户的 `φ` 个近邻（减少与 route 长度的耦合）
4. **Feasibility algorithm 改写**：不再对 “Δ≤0/Δ>0” 分两套排名，而是统一用“成本变化 / 可行性增益”的比值排名 move（更稳定/更省分支）
5. **Local search 与 feasibility 只在 modified routes 上做**：
   - 只对 perturbation 改动到的 routes 做邻域搜索（巨幅降成本）
   - inter-route move 必须涉及至少一条被扰动的 route
6. **扰动算子池简化**：保留 concentric + sequence removal；插入用 cost + distance（不再搞更多 removal 变体）

## 2. 总体框架（Algorithm 1）

Algorithm 1 给的 AILS‑II 骨架非常标准：

1. `s ← Construct an initial solution`
2. `sr, s* ← LocalSearch(s)`（sr 是 reference）
3. repeat:
   - `s ← Perturbation(sr)`
   - `s ← LocalSearch(s)`
   - **Update diversity control parameters**（perturbation 目标距离 & ω 更新）
   - `sr ← Apply acceptance criterion to s`
   - **Update acceptance criterion**（η 衰减）
   - 若更好则更新 `s*`

注意：AILS‑II 仍然允许 perturbation 后先 infeasible，再用 feasibility strategy 修回可行（与 AILS‑PR 相同的总思路）。

## 3. Perturbation degree：收敛式目标距离 dᵢ（Section 3.1）

### 3.1 目标距离 dᵢ 的意义

AILS 系列的核心思想是：给定 reference `sr`，希望 perturbation+local search 得到的解 `s` 与 `sr` 的结构距离落在一个“合适范围”。

AILS‑PR 用的是固定目标距离；AILS‑II 引入了收敛式：

- `dᵢ` 从 `dMax` 逐步减小到 `dMin`
- 直觉：早期允许更大跳跃（探索），后期更小跳跃（精修）

### 3.2 dᵢ 的更新公式（论文原文给法）

论文写法是每次迭代做一次乘法更新，使得 `dᵢ` 从 `dMax` 平滑收敛到 `dMin`：

- `dᵢ ← dᵢ · (dMin/dMax)^(1/it)`

其中 `it` 在正文里写作“预计最大迭代数”，而 Figure 1 的流程图标注为“预计剩余迭代数”。两种写法的目标一致：让 d 在搜索后期变小。

一个不容易跑偏的实现方式是：

- 若 stoppingCriterion=Iteration：`it = (max_iter - cur_iter + 1)`（剩余迭代数）
- 若 stoppingCriterion=Time：`it ≈ time_left / avg_time_per_iter`（用滑动平均估计剩余迭代数，至少 clamp 到 1）

这样每次更新用的是“剩余步数”，会自动适配不同 time budget。

### 3.3 ω 的在线更新（来自 Figure 1 的流程图，文本里是“Update of perturbation degree”）

AILS‑II 只有两个 removal heuristic：

- `R1`：Concentric removal
- `R2`：Sequence removal

每个 removal `R_k` 都带一个强度参数 `ω_k`（一次 batch 移除的客户数），并维护一个自适应距离估计 `d_k`。Figure 1 给出了几乎“照抄就能写”的更新过程（关键细节比正文更全）：

**初始化（Figure 1）**

- `η ← 1`
- `ω1 = ω2 ← 30`（默认初值）
- `d_i ← dMax`（理想距离）
- `d_k` 从 0 开始，但为了避免除零，作者建议在重置时设为 `1e-5`

**每次执行某个 removal `R_k` 后（得到本轮 local search 输出 s）**

1. 计算结构距离 `d(s,sr)`（distance 定义继承 AILS 系列；CVRP 常用 `|E_s △ E_sr|`）
2. 更新 `d_k`（Figure 1）：
   - `d_k ← α·d_k + (1−α)·d(s,sr)`
3. 其中 `α` 不是常数，而是一个随“该算子近期被调用次数”变化的系数（Figure 1）：
   - `it_k`：removal `R_k` 被执行的次数
   - `t_k = it_k mod γ`
   - `α = (t_k + 1) / (t_k + 2)`（从 1/2, 2/3, 3/4… 逐步逼近 1）
   - 并且每过 `γ` 次（进入新周期）重置：`α ← 0.5` 且 `d_k ← 1e-5`
4. 用 `d_i / d_k` 直接更新 `ω_k`（Figure 1）：
   - 若 `d_i > d_k`：`ω_k = min(d_i/d_k, n)`
   - 否则：`ω_k = max(d_i/d_k, 1)`
5. 因为 `ω_k` 必须是整数（移除多少个客户），实现里通常做：`ω_k ← clamp(round(ω_k), 1, n)`。

## 4. Acceptance criterion：收敛式阈值（Section 3.2）

AILS‑II 的接受准则是一个“阈值接受（Threshold Acceptance）”风格：

- 阈值：`θ = f + η( f̄ − f )`
- 其中：
  - `f`：最近 γ 次迭代的最好解值（窗口 best）
  - `f̄`：local search 产出的解质量平均（running average）
  - `η ∈ (0,1]` 控制阈值松紧（η 越大越松）

与 AILS‑PR 不同的是：AILS‑II 不再用“目标接受流量 κ”去调 η，而是用一个 **收敛式衰减**：

- `η` 从 1 逐步衰减到 `ε = 0.01`
- 论文写法：每步 `η ← η · ε^(1/it)`（同样是“乘法收敛”思想，Figure 1 标注 it 为“预计剩余迭代数”）

直觉：早期允许接受更差的解来探索；后期更保守，更像 intensification。

补一个实现层面的关键点（Figure 1）：

- 接受规则是阈值接受：若 `f(s) < θ` 则 `sr ← s`（否则 sr 不变）
- `it` 的取值建议与 §3.2 保持一致（Iteration 用剩余迭代数；Time 用“剩余时间/平均迭代耗时”估计）

## 5. Perturbation：算子池与插入策略（Section 3.3 + Table 1）

AILS‑II 的 perturbation 采用 **batch** 风格：

1. 先用 removal heuristic R_k 删除 `ω_k` 个客户
2. 再用 addition heuristic 把这些客户一次性插回去（允许回到原位，属于设计取舍）

Figure 1 的流程图还给了一个很明确的调度：`R_k` 与 insertion heuristic 都是 **random choice**（每轮随机选一种），并把 “选择哪个算子” 与 “调多大强度 ω_k” 解耦（强度是自适应的，算子是随机的）。

论文总结（Table 1）：

- removal：Concentric removal、Sequence removal
- insertion：Insertion by cost、Insertion by distance

与 AILS‑PR 相比，少掉了 proximity removal 等更复杂算子（为了速度与可控性）。

## 6. Local search / Feasibility：邻域组合 + 大规模剪枝（Section 3.3）

### 6.1 邻域组合（“为什么用 SWAP*”）

论文给的组合是：

- intra-route：SHIFT, SWAP, 2-opt
- inter-route：SHIFT, 2-opt*，SWAP*

这里最大的变化是 inter-route 的 **SWAP → SWAP\***：

- SWAP\* 是 Vidal 2022 强调的“跨两条路线交换两个客户，但插入位置可重新最优选择”的强邻域  
  对大规模而言更有改进能力，但也更贵，所以必须做剪枝。

### 6.2 SWAP\* 的邻域限制

论文说：为每个顶点只考虑其 `φ` 个最近邻（varphi/phi 参数），避免在长 route 上枚举过多候选。

### 6.3 只在 modified routes 上做搜索（最关键的复杂度控制）

这是 AILS‑II 能上 30k 的决定性工程点之一：

- local search / feasibility 只对 perturbation 改动到的 routes 做
- inter-route move 至少涉及一条“被扰动”的 route

直觉：如果你一次 perturbation 只改动了局部结构，那么在全局上扫描邻域是浪费；把搜索预算锁定在“刚被打乱的区域”，在大规模时收益/成本比最高。

### 6.4 Feasibility 的 move 排名（改成单一准则）

AILS‑PR 的 feasibility 在 `Δ≤0` 与 `Δ>0` 时用两套策略；AILS‑II 改为：

- 用 “成本变化 / feasibility gain” 的比值统一排名 moves

这更像一个“单位修复代价”指标，工程上通常更稳定，也更省逻辑分支。

按论文文字的实现意图，可以把它写成一个统一的打分函数（便于你复现 baseline）：

- `Ω(s,s')`：move 带来的 infeasibility reduction（例如超载量下降；若同时有多类约束，就把各类 violation 做加权和）
- `Δ(s,s') = cost(s') - cost(s)`
- score：`score = Δ / max(Ω, ε)`（ε 是很小的正数，防止除零）
- feasibility 阶段选 `score` 最小的 move（因为它“单位可行性修复带来的成本代价”最低；若 Δ<0 则天然优先）

## 7. 代码与参数（Section 4–5：对复现很友好）

### 7.1 代码来源

论文给的开源地址（Java）：`https://github.com/INFORMSJoC/2023.0106`

并在论文里直接解释了代码入口与包结构（对复现很友好）：

- 入口（Source Code 1 的含义）：读取命令行参数 → 构造 Instance → 构造 `AILSII` 对象 → 调 `search()` 开始搜索。
- 包结构（论文列出的摘要）：
  - `Auxiliary`：解间距离计算、diversity control 用的动态平均
  - `Evaluators`：move 的 cost/feasibility 增量评估、move 执行
  - `DiversityControl`：acceptance criterion、perturbation degree 的自适应控制
  - `Data`：读入实例
  - `Improvement`：local search 与 feasibility（含 SWAP* 的裁剪版本）
  - `SearchMethod`：AILS-II 主流程
  - `Perturbation`：addition/removal heuristics
  - `Solution`：solution / route / vertex 的数据结构

### 7.2 命令行参数（Section 5）

论文列出关键参数（我只摘你做对比最需要的）：

- `-varphi`：近邻集合大小（默认 40）
- `-gamma`：调整 ω 的周期（默认 30）
- `-dMax`：初始目标距离（默认 30）
- `-dMin`：最终目标距离（默认 15）
- `-stoppingCriterion`：Time 或 Iteration
- `-limit`：时间秒数或迭代次数

示例：

- `java -jar AILSII.jar -file Instances/X-n214-k11.vrp -rounded true -best 10856 -limit 100 -stoppingCriterion Time`

### 7.3 终止条件（实验设定）

实验中作者使用 `3n` 秒（n=客户数）作为 time limit，并在小/中/大规模数据上做对比。

## 7.4 复现要点（把 Figure 1 的“可跑”细节落到代码里）

如果你的目标是“先复现一个能跑的 baseline”，AILS‑II 最关键的实现点其实不在邻域本身，而在 **预算裁剪与收敛式控制**：

1. **只在 modified routes 上做搜索**
   - perturbation 之后，记录被影响到的 routes（remove 与 insert 都算）
   - feasibility 与 local search 都只扫描这些 routes
   - inter-route move 必须涉及至少一条 modified route（否则禁止）
2. **SWAP\* 的 φ 裁剪**
   - 预计算每个客户的 φ-NN 列表（默认 40）
   - SWAP\* 的候选只来自这些近邻，避免 route 很长时枚举爆炸
3. **收敛式参数更新**
   - `d_i ← d_i · (dMin/dMax)^(1/it)` 与 `η ← η · ε^(1/it)` 的核心是 `it`（剩余迭代数）的估计：Iteration 最简单；Time 则用滑动平均估计剩余迭代次数
4. **扰动强度 ω 的闭环**
   - 对每个 removal `R_k` 维护：`it_k`、`d_k`、`ω_k`
   - 每个 γ 周期重置 `d_k` 到 `1e-5`，并用 `α=(t_k+1)/(t_k+2)` 让 `d_k` 的更新“越到后期越平滑”

做到以上 4 点，你的 AILS‑II 复现就已经抓住了论文里“能扩展到 30k”的核心工程原因。

## 8. 和你的“两阶段选择（init→iter）”如何对齐

### 8.1 AILS‑II 更像哪个阶段？

几乎完全是 **iteration**（迭代改进器）：

- 初解只是起点（作者并不强调构造质量）
- 核心性能来自 perturbation + local search + diversity control

### 8.2 你可以用它做哪些“经典 baseline”？

最容易落地的 baseline 有三类（全都不需要训练）：

1. **固定参数 AILS‑II**：直接用论文默认 dMax/dMin/varphi/gamma（作为最强手工迭代器基线）
2. **两阶段 gate 的 Gate2 baseline**：Gate2 只决定 “用 AILS‑II vs 用别的 iter（如 FILO/HGS/SISR）”
3. **更细粒度的算子选择 baseline**：把 AILS‑II 的内部选择显式化：
   - 选 removal（concentric vs sequence）
   - 选 insertion（cost vs distance）
   - 选 varphi（邻域预算）与 dMax/dMin（跳跃尺度）

这些动作都能用实例特征做 conditional policy（learning-based）或用 bandit 在线调度（heuristic-based），非常贴合你要做的“两阶段选择 + 实例特征”方向。
