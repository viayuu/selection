# 根据问题性质选择算法（Instance-Based Algorithm Selection）调研报告（跨领域，偏可实现）

> 你关心的“算子选择”这里指更外层的版本：**给定一个问题实例（problem instance），根据它的性质/特征来选择最合适的算法/求解器/配置**。  
> 这在文献里更常被叫做：
>
> - **Algorithm Selection / Per-instance Algorithm Selection**（按实例选算法）
> - **Algorithm Portfolio / Portfolio-based Selection**（算法组合与选择/调度）
> - **Instance-Specific Algorithm Configuration (ISAC)**（按实例选配置/集群配置）
> - 在机器学习领域也属于 **Meta-learning / AutoML** 的核心组成部分（先选算法再调参）

本报告侧重“你要能真的实现出来”的角度：定义、数据管线、特征、模型、系统设计、评估与常见坑；并给出一套你可以直接迁移到“初始化/迭代组件选择”的映射。

---

## 1. 为什么“根据问题性质选算法”是合理的（动机与理论背景）

### 1.1 No Free Lunch（为什么你不该期待一个算法通吃）

经典结论：对足够广的任务分布，没有任何单一优化算法在平均意义上优于其他所有算法。  
代表性论文：Wolpert & Macready (1997) “No free lunch theorems for optimization” DOI `10.1109/4235.585893`。

现实意义：

- 你观察到的现象（不同实例上不同方法 win）不是“实现错了”，而是**应当存在的结构**。
- 这为“算法选择/组合”提供了合理性：**实例异质性 + 算法偏好性** ⇒ 存在可学习的映射。

### 1.2 Rice 框架：算法选择问题的标准形式

经典起点：Rice (1976) “The Algorithm Selection Problem” DOI `10.1016/S0065-2458(08)60520-3`。

把问题形式化：

- 实例空间：`I`（每个实例 `i ∈ I`）
- 算法集合：`A = {a1, ..., ak}`
- 性能度量：`m(i, a)`（时间、解质量、PAR10、gap、成功率…）
- 特征提取：`φ(i) -> x ∈ R^d`
- 选择器：`S(x) -> a`（或输出一个 schedule / 分配时间）

目标：对未知新实例 `i`，在给定预算下选出使 `m(i,a)` 最优的 `a`（或 schedule）。

你现在在做的“solver zoo 选择”就是 Rice 框架的直接实例化；把“算法”换成“初始化/迭代组件组合”也同样成立。

---

## 2. 你真正要实现的是哪一种“选择”？先把任务类型分清

算法选择在工程里至少分 3 类（强烈建议你在实现前明确是哪一类）：

### 2.1 选择单一算法（Single-algorithm selection）

输出：`a*`。  
适用：你只能/只想跑一个算法（或只跑一次 pipeline）。

优点：系统简单、部署成本低。  
缺点：风险高（预测错=彻底错）。

### 2.2 选择一个**算法调度/组合**（Algorithm scheduling / Portfolio schedule）

输出：一个分配时间的 schedule，例如：

`[(a2, 5s), (a5, 10s), (a1, rest)]`

适用：

- 你允许并行或串行跑多个方法
- 你想降低“选错”的尾部风险（robustness）

这在 SAT/CSP/规划里非常常见（有的实例“谁都难”，但总有一个能在短时间解出来；schedule 能显著提高 solved fraction）。

### 2.3 选择“算法配置”（Per-instance algorithm configuration）

输出：`(algorithm, hyperparameters)`，甚至对每个实例给不同配置。  
代表体系：ISAC / SMAC / ParamILS / irace。

意义：很多时候“算法 A vs B”不如“算法 A 的不同配置”差异大；配置空间更细但也更难。

### 2.4 多阶段/分层选择（Two-stage / hierarchical selection）——你现在的“两阶段选择”属于这一类

当你的求解 pipeline 天然分成多个阶段（最典型就是 **初始化 init** + **迭代改进 iter**），算法选择也可以做成分层决策：

- Stage-1：只看实例 `i` 的性质，选择初始化器 `init ∈ I`，得到中间产物 `sol0`
- Stage-2：在 `sol0` 的基础上再选择迭代器 `iter ∈ T`（可视为条件选择），得到最终解 `sol1`

输出不再是“一个算法”，而是“一个**组合**（或一个多阶段 schedule）”：

`select(i) = (init(i), iter(i, sol0))`

这一类方法的关键优势是：

1. **避免组合爆炸的最硬方式**：不用把每个 `(init,iter)` 当作独立“算法”暴力建模（虽然也可以做 baseline）
2. **允许使用条件信息**：Stage-2 可以用 `sol0` 的结构特征（这在实践中通常是性能提升的关键）
3. **更贴近真实系统**：很多 OR/启发式算法内部其实就是“多阶段自适应调度”，把它外显成 two-stage selection 很自然

---

## 3. 典型系统管线（强烈推荐按这个 checklist 实现）

下面是“按实例选算法”的最通用工程管线，你可以直接照做。

### 3.1 数据：你需要什么训练数据？

对一批训练实例 `i=1..N`，对每个算法 `a∈A` 跑出：

- `y_{i,a}`：性能（时间/质量/二者组合）
- `status_{i,a}`：成功、超时（censored）、无解…
- `cost_feat(i)`：特征计算耗时（**非常重要**，不要忽略）

你最终要学的是：`x_i -> 选择 a`（或预测 `y_{i,a}`）。

**注意：超时是“右删失数据”**。如果你直接把超时当成固定大常数，会引入系统性偏差（下面会讲处理方法）。

### 3.2 特征：哪些算“问题性质”？

特征大体分两类：

1. **静态特征（static features）**：只看输入本身，不跑算法  
   - 规模 `n`、密度、统计量、图结构特征、几何分布特征等  
   - 优点：稳定、可解释、成本可控  
2. **动态/探测特征（probing / dynamic features）**：在实例上跑很短的探测过程再抽特征  
   - SATzilla 的 probing 就是经典例子  
   - 在数值线性代数里也很常见：先跑几步 Krylov，看残差下降速率再决定用哪个 solver/preconditioner  
   - 优点：信息量更大、更贴近“可解性/难度”  
   - 缺点：会引入额外开销；探测本身也可能失败或噪声大

工程经验：  
**先做 static 特征把系统跑通，再加 probing 特征做性能上限。**

### 3.3 学习目标：你到底在预测什么？

常见三种建模目标（选择哪种会决定你后面模型怎么写）：

#### A) 直接分类：预测“最优算法是谁”

- label：`a*_i = argmin_a y_{i,a}`
- 模型：多分类

缺点：  
“把第 1 名预测成第 2 名”与“预测成最差”在损失里同权，但真实代价差很多 ⇒ 需要 cost-sensitive 改造。

#### B) 回归：预测每个算法的性能，再选最小

- 模型：`ŷ_{i,a} = f_a(x_i)`（每算法一个回归器）  
  然后 `select = argmin_a ŷ_{i,a}`。

优点：能自然体现“第二名也很接近”的情况；也便于做 schedule（预测运行时间分布）。

难点：删失数据与 heavy-tail（运行时间常是长尾）。

#### C) 排序/学习排序：预测算法的相对顺序（更鲁棒）

在你只关心“选谁更好”而不关心绝对值时，排序/对比学习往往更稳：

- 训练 pairwise：学习 `P(a better than b | x)`
- 或直接学习一个 score：`s_a(x)`，然后按 score 排序

机器学习领域的 meta-learning（“算法排名预测”）大量使用这一类（例如排名方法对分类器选择：Brazdil & Soares 2000）。

---

## 4. 经典系统与代表方法（你可以直接拿来当模板）

这一节选的是“真的经典 + 真的可复现 + 真的经常被后续引用”的几条线。

### 4.1 Algorithm Portfolios：组合与选择思想的源头之一

- Gomes & Selman (2001) “Algorithm portfolios” DOI `10.1016/S0004-3702(00)00081-3`

核心观点：

- 与其赌一个算法，不如构造一个 portfolio（算法集合）
- 通过选择/调度，让系统接近“虚拟最优求解器（VBS）”

它奠定了后续 SATzilla、SUNNY、AutoFolio 等的基本叙事。

### 4.2 SATzilla：最经典的“特征 + 性能预测 + 组合调度”系统

Xu, Hutter, Hoos, Leyton-Brown (2008)  
“SATzilla: Portfolio-based Algorithm Selection for SAT” DOI `10.1613/jair.2490`

为什么 SATzilla 特别值得你学？

1. **特征工程很系统**：既有静态特征，也有 probing 特征  
2. **预测目标很工程化**：不是只预测 best solver，还要考虑 timeouts、PAR10 等评价  
3. **解决“删失数据”问题**：SAT 里超时很多，这是现实系统绕不开的
4. **可扩展到 schedule**：不是只能选一个 solver，可以做组合/调度

把 SATzilla 抽象成你可以照搬的模板：

- 输入：`x_i`（实例特征）
- 输出：对每个 solver 的预测运行时间/成功概率
- 决策：选最小预测值，或生成 schedule（先跑预测最快的若干个）

### 4.3 SUNNY：kNN 的“懒惰”组合选择（非常实用的 baseline）

Amadini, Gabbrielli, Mauro (2014)  
“SUNNY: a Lazy Portfolio Approach for Constraint Solving” DOI `10.1017/S1471068414000179`

核心思想非常朴素，但强到能当长期 baseline：

1. 用特征空间里的 **相似实例**（kNN）作为“经验库”
2. 在相似实例上表现好的算法，在新实例上也更可能好
3. 输出可以是：
   - 单一算法（投票/最优）
   - 或一个 schedule（按相似实例统计最有效的算法排序，并按 solved fraction 分配时间）

优点：
- 你不需要训练复杂模型，几乎零调参
- 适合“特征足够表达相似性”的场景
- 对非线性、长尾、删失往往比你想象得稳健

缺点：
- 特征空间度量不好时会崩
- 需要足够多的历史实例覆盖

### 4.4 ISAC：先聚类实例，再对每个簇选配置/算法（很贴合“问题族分群”）

代表性工作：
- Kadioglu et al. (2010) “ISAC – Instance-Specific Algorithm Configuration” DOI `10.3233/978-1-60750-606-5-751`
- Malitsky (2015) “Instance-specific algorithm configuration” DOI `10.1007/s10601-015-9210-1`

典型流程：

1. 用实例特征对训练集聚类成若干簇 `C1..Cm`
2. 对每个簇单独做：
   - 选一个最适合的算法，或
   - 用配置器（ParamILS/SMAC/irace）找到簇内最优配置
3. 新实例先分到某个簇，再用该簇的配置/算法

它的优势：
- 解释性强：你能说“这类实例用这套方法”
- 对异质性很强的问题族很有效（你现在 TSPLIB 的 win-bucket 现象其实就暗示了分群结构）

### 4.5 ASlib：算法选择的基准库（做研究/验证必备）

Bischl et al. (2016) “ASlib: A benchmark library for algorithm selection” DOI `10.1016/j.artint.2016.04.003`

ASlib 提供了标准化的：

- 实例特征
- 多算法的性能矩阵（含超时）
- 标准评价指标与 split

你后续如果要把“init+iter 组合选择”写成论文，ASlib 的标准范式非常值得对齐。

### 4.6 AutoFolio：用“算法配置”自动搭建算法选择器（AutoML for algorithm selection）

Lindauer, Hoos, Hutter, Schaub (2015)  
“AutoFolio: An Automatically Configured Algorithm Selector” DOI `10.1613/jair.4726`

AutoFolio 的视角特别适合你：

- 选择器本身也有很多 design choice：用分类还是回归？用什么模型？怎么归一化？怎么处理缺失？怎么做 schedule？
- AutoFolio 用 SMAC 之类的配置器在一个“选择器设计空间”里自动搜索

这对你做“组合（init×iter）”非常关键：  
当动作空间扩大后，你很容易陷入“手调选择器很痛苦”的问题；AutoFolio 思想提供了一个系统化解法。

### 4.7 Hydra：自动构造 portfolio（不仅选，还自动“造组合”）

Xu, Hoos, Leyton-Brown (2010)  
“Hydra: Automatically Configuring Algorithms for Portfolio-Based Selection” DOI `10.1609/aaai.v24i1.7565`

思想：

1. 不先给定一个固定 solver 集合
2. 用配置器反复寻找“能补齐当前 portfolio 弱点”的新配置/新算法变体
3. 逐步构造一个更强的 portfolio

把它映射到你现在的研究设想：

- 你不一定要固定“初始化器集合/迭代器集合”不变  
  你也可以用类似 Hydra 的思想去“自动长出”更互补的组件集合（例如不同 revision 长度、不同 destroy 强度）。

### 4.8 <claspfolio>2：Answer Set Programming (ASP) 领域的经典 portfolio 选择系统（“强可复现的领域模板”）

Hoos, Lindauer, Schaub (2014)  
“claspfolio2: Advances in Algorithm Selection for Answer Set Programming” DOI `10.1017/S1471068414000210`

为什么它代表性强？

- ASP/SAT/CSP 这类领域的求解器往往高度工程化、参数/随机性丰富、实例异质性极强；因此 portfolio 选择几乎是“标配”。
- claspfolio2 体现了一套非常通用的实践：  
  **实例特征 + 运行时预测/排名 +（可选）调度/并行**，并把它打磨成可用于比赛/基准的系统。

你可以把它当成“领域无关模板”来学：

- 如何定义 solver portfolio
- 如何做 feature pipeline（含 static/probing）
- 如何把预测转成单一选择或 schedule
- 如何处理超时/删失与长尾运行时

### 4.9 Empirical Performance Models (EPM) 与 Runtime Prediction：算法选择系统背后的“发动机”

很多成熟的算法选择系统（SATzilla、claspfolio、AutoFolio 等）核心都离不开：  
**用机器学习模型预测某算法在某实例上的性能**（尤其是 runtime / success probability）。

两篇非常“代表性+被广泛引用”的综述/体系化工作：

- Hutter, Xu, Hoos, Leyton-Brown (2014) “Algorithm runtime prediction: Methods & evaluation” DOI `10.1016/j.artint.2013.10.003`
- Hutter, Hamadi, Hoos, Leyton-Brown (2006) “Performance Prediction and Automated Tuning of Randomized and Parametric Algorithms” DOI `10.1007/11889205_17`

它们的价值在于：把“选择器”背后的技术栈讲清楚（特征、删失、长尾、模型选择、评估协议等），你实现时会少踩很多坑。

### 4.10 Benchmark 与 Challenge：为什么算法选择领域这么强调“公开基准与竞赛”

算法选择是一个“系统工程问题”（特征、模型、预算、调度、评估协议都影响结果），因此该领域非常强调公开基准与挑战赛来避免“只在自家数据上好看”：

- Kotthoff, Hurley, O'Sullivan (2017) “The ICON Challenge on Algorithm Selection” DOI `10.1609/aimag.v38i2.2722`
- ASlib 基准库（你上面已经看到）是竞赛与复现实验常用的数据底座：DOI `10.1016/j.artint.2016.04.003`

对你写论文/做可复现研究的直接启发：

- 尽量把你的 “solver/pipeline selection” 数据组织成接近 ASlib 的格式（features + performance matrix + meta）
- 报告 SBS/VBS、PAR10/mean gap、以及跨分布/跨规模的泛化（你已经在做 win-bucket，这非常契合挑战赛范式）

### 4.11 经典综述与跨领域线索（帮你“快速建立全景”，也给你更多可引用来源）

如果你想快速把“算法选择”这条线从 SAT/CSP 扩展到更广领域（数值优化/机器学习/工业求解等），下面这些是非常常被引用、也非常适合当“论文 related work 背景段落”的材料：

**(1) Combinatorial search 的算法选择综述（强相关、强代表性）**

- Kotthoff (2014) “Algorithm Selection for Combinatorial Search Problems: A Survey” DOI `10.1609/aimag.v35i3.2460`  
  重点价值：把组合搜索（SAT/CSP/ASP 等）的“特征—预测—选择/调度—评估”全流程系统化整理。
- Kotthoff (2015) “On Algorithm Selection, with an application to combinatorial search problems” DOI `10.1007/s10601-015-9214-x`  
  重点价值：更偏“方法论/形式化”，并讨论了选择器构建时的一些关键工程与评价问题。

**(2) 连续黑箱优化（BBO）里的算法选择：实例特征(Ela) + 成本敏感学习**

这条线特别适合拿来“类比”你要做的事：  
它把“实例性质”做成一套可复用的 **Exploratory Landscape Analysis (ELA)** 特征，然后用这些特征去选择/排名优化器。

- Kerschke et al. (2015) “Algorithm selection for black-box continuous optimization problems: A survey on methods and challenges” DOI `10.1016/j.ins.2015.05.010`  
  重点价值：明确讲了连续优化里算法选择的难点（采样成本、特征噪声、泛化）与常用套路。
- Mersmann et al. (2011) “Exploratory landscape analysis” DOI `10.1145/2001576.2001690`  
  重点价值：ELA 特征的原点之一（从“函数地形”抽可解释特征）。
- Mersmann et al. (2012) “Algorithm selection based on exploratory landscape analysis and cost-sensitive learning” DOI `10.1145/2330163.2330209`  
  重点价值：把“选错代价不对称”显式建模（这与你的 gap/time/timeout 场景很像）。

**(3) 工具与生态：把论文里的思路快速落地**

- `ASlib` 本身就是“算法选择的标准数据格式+基准库”（4.5）  
- `llama`（R 包）提供了很多算法选择的 baseline 组件（特征、学习器、选择策略等），适合拿来对照实现：DOI `10.32614/cran.package.llama`

---

## 5. 模型与方法：从“能跑”到“能打”的几条经典路线

这一节给你“选型指南”：你到底用分类、回归、排序、还是 kNN/聚类？

### 5.1 最推荐的起步路线（强实用）

如果你要快速做出一个可靠系统，我建议优先顺序：

1. **SBS/VBS 基线**（必须做，见第 8 节评估）
2. **SUNNY/kNN** 做一个强 baseline（几乎零训练成本）
3. **回归预测每算法性能**（RF/XGBoost/MLP 都可以）然后 argmin
4. 需要更稳时做 **schedule**（降低尾部风险）
5. 最后才上 RL/复杂 gating（否则很难 debug）

### 5.2 分类 vs 回归 vs 排序：什么时候选哪个？

#### 分类（预测 best 算法）

适合：
- 算法数不多
- best 和 second 差距大（错一次代价大）

风险：
- label 噪声：当多个算法很接近时，“谁是 best”不稳定
- 代价不均：错成第二名还好，错成最差很惨 ⇒ 需要 cost-sensitive

经典解决：
- 直接学 `m(i,a)` 的回归，而不是学 `argmin`

#### 回归（预测性能矩阵）

适合：
- 你希望进一步做 schedule / 风险控制
- 你要支持 multi-objective（例如 quality+time）

关键点：
- 删失数据：超时不等于“真实时间=timeout”，而是“真实时间>=timeout”
- heavy-tail：很多 runtime 统计在对数空间更稳定

#### 排序/学习排序（预测算法排名）

适合：
- 你主要关心“选谁更好”，不在意绝对值
- 多算法差距小但有稳定偏好

经典参考（机器学习算法选择中的排名方法）：
- Brazdil & Soares (2000) “A Comparison of Ranking Methods for Classification Algorithm Selection” DOI `10.1007/3-540-45164-1_8`

### 5.3 处理删失（timeouts）的经典方式

删失是算法选择系统的常态：SAT/CSP/MIP 都大量存在超时。

常见处理（从简单到强）：

1. **PAR10 / capped loss**（评估层面）  
   - 超时记为 `10×timeout`（SAT 社区常用），用于系统评估
2. **建模层面做 survival / censored regression**  
   - 目标不是预测一个点，而是预测 `P(T <= t | x)` 或分位数
3. **两阶段模型**：先预测“能否在预算内解出”（classification），再预测“若能解出需要多久”（regression）

SATzilla 类系统就是围绕这一点做了大量工程化设计（你如果要做 schedule，这是必须面对的问题）。

### 5.4 EPM / Runtime Prediction：从“预测 best”升级到“预测性能曲线/分布”

如果你希望选择器不仅能“选对”，还能：

- 输出 top-k（给后续 schedule/并行）
- 在不确定时保守（降低选错尾部风险）
- 支持多指标（time+quality）

那么一个非常成熟的路线是：**做 Empirical Performance Model (EPM)** ——对每个算法建模 `y = m(i,a)` 的预测器。

代表性参考（体系化、强可复用）：

- Hutter, Xu, Hoos, Leyton-Brown (2014) “Algorithm runtime prediction: Methods & evaluation” DOI `10.1016/j.artint.2013.10.003`
- Hutter, Hamadi, Hoos, Leyton-Brown (2006) “Performance Prediction and Automated Tuning of Randomized and Parametric Algorithms” DOI `10.1007/11889205_17`

一个最常见、工业级稳健的 EPM 配方（你实现时非常建议从这里开始）：

1. **特征**：`x = φ(i)`（可选再拼算法配置 `θ`）
2. **目标**：运行时间建议在 log 空间建模：`z = log( runtime + 1 )`（长尾更稳定）
3. **模型**：Random Forest / GBDT（对非线性与离群值很稳、对特征尺度不敏感）
4. **删失处理**：
   - 最简单：先做“是否超时/是否成功”的分类器 `p_solve(x)`，再对成功样本回归 `runtime`
   - 更强：censored regression / survival（直接学 `P(T <= t | x)` 或分位数）
5. **决策**：
   - 单选：`argmin_a E[T_a | x]` 或 `argmin_a E[gap_a | x]`
   - schedule：按预测的成功概率/期望时间构造时间分配（见下一节）

> 你在 CO 里的 `gap` 选择与 SAT/CSP 的 `runtime` 选择在形式上完全一致：  
> 只要把 `m(i,a)` 换成你关心的指标（如 `aug_gap` 或 `gap/time`），EPM 配方直接可复用。

### 5.5 从“选一个算法”到“构造一个 schedule”：经典 portfolio 调度思想

当你担心“选错一次就爆炸”（尾部风险很重），或很多实例“很难预测”，schedule 常常比单选更稳。

两类最常见 schedule：

1. **串行 schedule（sequential）**：在一个 CPU 上按顺序给若干算法分配时间片  
   例如：`[(a2, 2s), (a5, 5s), (a1, rest)]`
2. **并行 schedule（parallel）**：多核/多机同时跑多个算法（或不同随机种子/不同配置）  
   这在 SAT/ASP/CSP 里非常常见（portfolio parallelism）。

如何从预测模型构造 schedule（常用启发式）：

- **Top-k + 均分**：选预测最好的 k 个算法，每个给 `budget/k`
- **按 solved-prob 分配**：若你能预测 `p_solve(a|x)`，可按概率比例分配时间片
- **基于 ERT 的贪婪构造**：反复加入“带来最大期望收益”的算法（尤其当你能近似得到 runtime CDF）

在你当前项目的可落地方向：

- 你已经实现了 top-k/rejection 的离线模拟评估；这几乎就是 schedule 的雏形  
  （“先跑 top-1，没把握再跑 top-k”）。

进一步的代表性思路：

- Lindauer et al. (2018) “Selection and Configuration of Parallel Portfolios” DOI `10.1007/978-3-319-63516-3_15`  
  系统性讨论了并行 portfolio 的选择与配置问题（如果你未来上集群很有参考价值）。

### 5.6 置信度/风险感知的选择（Confidence-aware / Risk-aware selection）

现实中最痛的不是“平均差 0.1%”，而是“偶尔选错导致灾难性退化”。经典做法是把“不确定性”显式纳入决策：

1. **top-k + fallback**：当 `best` 与 `second` 预测差距很小（或模型方差很大）时，直接输出 top-k，并用 schedule/并行降低风险
2. **rejection option**：当置信度低时拒绝自动决策，改用更稳健但更贵的 baseline（你现在的 rejection-based 选择就是这个思想）
3. **预测分位数而不是均值**：用 conservative 的 `P90 runtime` 或 `P90 gap` 做决策（避免均值被幸运样本误导）

工程上，Random Forest/GBDT 往往可以很便宜地提供“近似不确定性”（树间方差、分位数回归等）。

### 5.7 多目标选择（time + quality + feasibility）：把 trade-off 显式化

很多领域并不是单一指标：

- SAT/CSP：成功率 + runtime
- CO：gap + runtime（甚至还有 memory）
- 工程求解：误差/精度 + runtime

常见处理方式（从最简单到更稳）：

1. **加权和标量化**：`score = gap + λ·time`（或 `gap/time`），用 λ 表示你愿意用多少时间换 1% gap
2. **字典序（lexicographic）**：先满足可行/超时，再比质量（很多工业场景更符合 KPI）
3. **Pareto + 决策规则**：先预测每算法的 (gap,time) 分布，再选满足约束的 Pareto 最优集合（再在集合内挑风险最小/均值最好）

> 对你现在的 “Augmented Gap” 比较：  
> 如果未来要把时间也纳入论证，建议尽早把 “gap 与 time 的标量化规则”固定下来，否则结论会非常依赖指标定义。

---

## 6. 特征工程：跨领域最有“迁移价值”的套路

你要实现“根据问题性质选算法”，特征是第一生产力。下面给跨领域最常用的可迁移套路。

### 6.1 静态结构特征（便宜、稳定、可解释）

- **规模类**：n、m、约束数、变量数、图密度…
- **统计类**：系数分布、度分布、需求分布、距离分布…
- **图结构类**：聚类系数、连通性、中心性、谱特征（可选）
- **几何/空间类**（TSP/CVRP 特别常见）：凸包占比、近邻距离统计、分布均匀性/聚团性…

你在 TSPLIB 分析里已经构建了一批“几何可观测特征”，这正是典型的“可部署特征”（推理时也能算）。

### 6.2 动态 probing 特征（信息量大但有开销）

跨领域的共同范式是：

> **在实例上跑一个很短的探测过程，观察早期收敛/可行化/改进趋势，把趋势当特征。**

例子（帮助你迁移思路）：

- SAT：跑一些基本传播/启发式几步，统计约束传播强度、冲突率等（SATzilla 风格）
- MIP：跑 LP relaxation、观察 root node bound 改进速度
- 数值线性代数：跑几步 GMRES/CG，看残差下降率，选择更合适的预条件器/迭代法
- 组合优化神经求解：跑少量 rollout/局部改进 steps，看初期改进曲线

实践建议：
- probing 的成本必须纳入总预算（否则选出来的“最好算法”整体反而更慢）
- probing 特征经常是“实例难度”的强 proxy，能显著提高选择器上限

### 6.3 特征成本建模（经常被忽略但很关键）

你的系统最终要最小化的是：

`total_cost = cost_feature(i) + cost_algorithm(i, selected)`

很多论文/实现只优化 `cost_algorithm`，导致：
- 特征提取过慢，抵消了选择收益
- 或 probing 太重，得不偿失

工程上常见解法：

- 分层特征：先算 cheap features，如果置信度足够就直接选；不够再算贵特征（类似“rejection-based”）
- 多阶段选择：cheap 模型做粗筛（top-k），贵模型在 top-k 上精细预测

这和你当前项目里的 top-k / rejection 策略是同构的，只是对象从 solver 变成“算法”。

### 6.4 Meta-features（元特征）与 Landmarking：机器学习/AutoML 里最常用的“实例性质表示法”

在机器学习的算法选择（选择分类器/回归器/聚类算法）领域，“实例”是一个数据集，`φ(i)` 被称为 **meta-features**。  
这一套思想非常成熟，而且很多概念可以直接迁移到 CO（把“数据集”换成“问题实例集合/单实例”，把“学习器”换成“求解器/组件”）。

代表性综述/概念框架：

- Smith-Miles (2009) “Cross-disciplinary perspectives on meta-learning for algorithm selection” DOI `10.1145/1456650.1456656`
- Brazdil & Giraud-Carrier (2018) “Metalearning and Algorithm Selection: progress, state of the art...” DOI `10.1007/s10994-017-5692-y`

常见 meta-features 的分类（非常经典，且基本跨领域通用）：

1. **Simple/Size**：规模与形状  
   - `n_samples`, `n_features`, `n_classes`, 稀疏度、缺失率、类别不平衡度…
2. **Statistical**：统计分布  
   - 每维的 mean/std/skewness/kurtosis，相关系数分布，主成分解释率…
3. **Information-theoretic**：信息论  
   - entropy、mutual information、noise ratio…
4. **Model-based**：用一个简单模型拟合后抽“模型结构”  
   - 小决策树深度、叶子数、训练误差、margin 分布…
5. **Landmarking / Probing（非常重要）**：跑一组“廉价基线算法”得到它们的表现作为特征  
   - 例如在 ML 里：kNN/朴素贝叶斯/线性模型的快速交叉验证分数  
   - 在 CO 里对应：Nearest Neighbor/Greedy insertion/少步局部搜索的 early objective、early improvement slope

为什么 landmarking 很关键？

- 它通常比纯静态统计特征更贴近“可解性/难度”，因此对算法选择的上限贡献很大
- 代价是额外计算开销，所以需要和预算一起设计（见 6.3）

### 6.5 特征工程的“工程坑”：泄漏、尺度、缺失与稳健性

这类系统最容易栽在工程细节：

1. **信息泄漏（leakage）**：  
   - 例如用 test 集整体统计量做归一化；或用“最优解相关信息”当特征（部署时不可得）  
   - 你在 TSPLIB 特征里区分 observable vs opt-dependent 是非常正确的做法
2. **尺度问题**：  
   - kNN/聚类对尺度极敏感；一定要标准化/白化，否则距离无意义  
3. **缺失特征**：  
   - probing 特征可能在某些实例失败/超时；需要明确缺失值策略（impute + missing indicator 是常见稳健做法）
4. **跨分布泛化**：  
   - 特征分布 shift 时选择器容易失效；建议定期做“OOD 检测/置信度下降时 fallback”

---

## 7. 评估：如何证明“根据问题性质选算法”真的有效？

### 7.1 三个必须报告的基线（不然别人不会信）

1. **SBS (Single Best Solver)**：在训练集上整体最好的单一算法  
2. **VBS (Virtual Best Solver / Oracle)**：每个实例都选真实最优算法（理论上限）  
3. **Random/Uniform**：随机选算法（sanity check）

你的选择器要做到：

- 明显优于 SBS（否则选择没意义）
- 尽量逼近 VBS（越接近说明“实例异质性可被特征捕捉”）

### 7.2 指标选择：不要只看平均值

常见指标（跨领域可用）：

- **平均性能**：mean gap / mean runtime / mean PAR10
- **win/tie/loss**：逐实例胜率（你已经在做）
- **tail 指标**：95th percentile runtime / worst-case gap（选择的价值经常体现在尾部风险降低）
- **anytime/AUC**：如果你用 schedule 或迭代型算法，曲线面积比最终值更合理

补充几个“算法选择领域非常标准、非常常用”的定义（建议你在报告里明确写出来，避免口水战）：

1. **PAR10（SAT/CSP 竞赛与 ASlib 常用）**  
   - 若实例在 timeout 内 solved：`par10 = runtime`  
   - 若超时：`par10 = 10 * timeout`  
   - 系统得分取平均：`mean(PAR10)`  
   直觉：强烈惩罚超时（比 plain mean runtime 更能体现“选错的灾难性”）。

2. **Solved fraction / success rate**  
   - `solved_rate = (# solved within timeout) / N`  
   对 schedule/portfolio 特别重要：有时你宁愿牺牲一点平均时间，也要提高 solved rate。

3. **Total wall-clock cost（把特征成本算进去）**  
   - `total_time = feature_time + algorithm_time`  
   这是部署时真正关心的 KPI；很多选择器在论文里忽略 feature_time，落地会吃亏。

4. **Quality 指标的“可比性”**  
   - 你在 CO 里常用 `gap(%)`，但要确保 reference（best known / optimal / baseline）一致  
   - 如果出现“负 gap / 极端 gap”，往往是 evaluation 定义不一致或 reference 错配（你之前遇到的 gap 异常就是典型症状）

5. **Schedule/anytime 评价**  
   - 报告 `best_so_far(t)` 曲线或 AUC，会比“最终一次结果”更公平  
   - 对固定预算的 portfolio，`time-to-target`（达到某个阈值所需时间）也很常用

### 7.3 统计检验（推荐）

如果你要写论文/报告：

- paired sign test / Wilcoxon（同一实例配对）
- 报告置信区间（bootstrap）

---

## 8. 给你一份“可落地到你项目”的实现蓝图（把算法选择变成代码）

你现在的目标是：根据实例性质，在候选方法（或 init×iter 组合）之间选择。

### 8.1 把“算法集合”定义清楚：你选的对象是什么？

你可以有三种粒度（由易到难）：

1. **整 solver**（你原来做的 solver selection）
2. **两段式 pipeline**：`init` 与 `iter` 分开选（你现在要证明组合有效）
3. **更细粒度算子**：在 iter 内选 destroy/repair/neighborhood（对应你前一份算子选择报告）

这份报告关注 1/2（按实例选“算法/组合”），但它们完全可以嵌套到 3。

### 8.2 特征设计建议（结合你已有的 TSPLIB 特征体系）

推荐你按“可部署/可观测”优先：

- 规模：`n`
- 几何/分布：你在 `benchmarks/analyze_tsplib70_logs.ipynb` 已经有一批（grid_entropy、nnd_over_expected、hull_fill 等）
- CVRP 额外：需求统计（mean/std/max）、容量紧张度 proxy、depot-客户距离统计、聚团性
- （可选）初始化探测特征：用某个 cheap initializer 做几步，记录 early improvement 曲线（当作 probing）

### 8.3 建模路线（推荐你从最稳的开始）

**Baseline 0：SBS/VBS**

先把四个组合（或更多组合）的性能矩阵整理出来，得到：
- SBS：全局最强单方法
- VBS：实例级别 oracle 上限

**Baseline 1：kNN（SUNNY 风格）**

做一个最简单的 selector：

- 距离：标准化后的欧氏距离或 cosine
- 预测：看 k 个近邻里哪个方法平均最好
- 输出：单方法或 top-k

**Baseline 2：回归预测性能**

- 对每个方法训练一个回归器预测 `gap`（或 `gap/time`）
- 选预测最小者
- 可加不确定性：预测方差大时用 top-k + 备用方法（风险控制）

**Baseline 3：排序学习**

如果你的“谁是 best”噪声很大（多个方法差距极小），排序往往更稳。

### 8.4 “组合爆炸”怎么办（init×iter 很多时）

当组合数 `|Init|×|Iter|` 变大，你会遇到：

- 监督标签更稀疏（很多组合没测）
- 训练数据成本爆炸

经典应对有两条线：

1. **结构化因子模型**：把组合性能分解成  
   `score(init, iter | x) ≈ s_init(init|x) + s_iter(iter|x, init_out)`  
   这与两层 gate 非常接近（先选 init，再在条件上选 iter）。
2. **portfolio + schedule**：不强行选一个组合，而是在 top-k 上做少量预算分配（降低选错风险）。

### 8.5 一个“你可以直接照着做”的落地实现清单（强建议按这个顺序）

**Step 0：定义评价与预算**（先定规则再做模型）

- 指标：只看 `aug_gap`？还是 `gap+time`？有没有 timeout？  
- 预算：每实例允许的总推理时间（含特征提取）是多少？  
- 选择输出：单一方法？top-k？还是 schedule？

**Step 1：把结果组织成“ASlib 风格”的三张表**

1. `instances.csv`：实例 ID、元信息（n、来源、类别/分布标签…）
2. `features.csv`：实例特征 `φ(i)`（区分 observable vs opt-dependent）
3. `performance.csv`：每个 `(instance, method)` 的指标（gap/time/status）

你现在已经有 `method -> {instance -> aug_gap}` 的结构；把它标准化成表格后，后续所有实验会变得非常顺。

**Step 2：先做 4 个 baseline（不然模型无从谈起）**

- Random（均匀随机选方法）
- SBS（训练集整体最强单方法）
- VBS（oracle 上限）
- kNN/SUNNY（几乎零训练成本的强基线）

**Step 3：做一个最稳的回归 EPM（每方法一个回归器）**

- 目标：`aug_gap`（或 `gap/time`）  
- 模型：RandomForestRegressor / XGBoost  
- 决策：`argmin`  

**Step 4：加风险控制（你会立刻看到尾部改善）**

- top-k（k=2/3）+ fallback
- rejection：置信度低时改用 SBS 或更稳健迭代器

**Step 5：做“分桶画像”分析（你已经很擅长这个）**

- 按 `n` 分桶：看不同规模上 selector 的 win ratio
- 按你已有的几何特征分桶：比如 cluster-like vs uniform-like

这一步不仅是报告漂亮，更关键是：它会反向指导你“特征是不是抓住了偏好性”。

**Step 6：再考虑更复杂的结构（两层 gate / 因子分解 / schedule）**

- 如果单选已经明显优于 SBS，说明特征→偏好的信号存在  
  此时再把动作空间扩大到 init×iter 或 schedule，收益更可控。

### 8.6 你的“两阶段选择”（init→iter）如何放进算法选择经典框架里？

把你想做的事情写成一个标准的 **hierarchical algorithm selection**：

- 实例：`i`（例如一个 TSPLIB/CVRPLIB 实例）
- Stage-1（初始化选择）：`init ∈ I`
- Stage-2（迭代选择）：`iter ∈ T`
- 初始化输出：`sol0 = Init(init, i)`（以及时间 `t_init`、初始质量 `q0`）
- 迭代输出：`sol1 = Iter(iter, i, sol0)`（以及时间 `t_iter`、最终质量 `q1`）

决策策略（两层 gate）可写成：

```text
x  = φ(i)                 # 实例特征（推理时可获得）
init ~ π1(init | x)

sol0, log0 = run_init(init, i)
z  = ψ(i, sol0, log0)     # 条件特征：实例 + 初解结构/质量/耗时等
iter ~ π2(iter | z)

sol1, log1 = run_iter(iter, i, sol0)
```

这与 Rice 框架的关系：

- Rice 的 `S(φ(i)) -> a` 是 “单步选择一个算法”
- 你的两阶段选择是 “先选 `init`，再在 `init` 产物条件下选 `iter`”

如果你把每个 `(init,iter)` 组合当成一个“复合算法”，它仍然是 Rice 框架；  
但 two-stage 的价值在于：**它显式利用了“sol0 结构”这一强信息源**，并在数据上更节省（更容易做因子化/分层建模）。

### 8.7 两阶段选择最关键的点：特征要分成“实例特征 φ”与“条件特征 ψ”

你之前日志分析已经说明：不同方法对不同 `n`/分布特征有偏好。两阶段选择要进一步回答：

1. **Stage-1（选 init）应该看什么？**  
2. **Stage-2（选 iter）除了实例，还应该看 `sol0` 的什么结构？**

#### 8.7.1 Stage-1：实例特征 φ(i)（只用 observable）

建议优先使用“推理时可得、成本可控”的 observable 特征（你在 TSPLIB notebook 的 observable feature 体系就是很好的起点）。

对 TSP 类实例（例子）：

- 规模：`n`
- 分布/几何：`grid_entropy`, `nnd_over_expected`, `pc_anisotropy`, `hull_fill`, `pairdist_cv`, `r_max_over_mean` …
- （可选）cheap landmarking：跑 NN/greedy insertion + 少步 2-opt，看 early improvement slope

对 CVRP 类实例（例子）：

- 规模：客户数 `n`、车辆容量 `Q`
- 需求统计：mean/std/max、重尾程度、最大需求占比
- 空间分布：同样的几何特征（对 depot 的径向统计、聚团性 proxy）
- （可选）可行性紧张度 proxy：`sum(demand)/Q` 与车辆数下界、近似 route count 下界等

> 你可以把 φ(i) 做成两种形态：  
> - 手工特征向量（更可解释、少依赖）  
> - 学习到的 embedding（例如用你已有的 Encoder_h/Naive_Encoder 输出一个实例表示，再拼手工特征）

#### 8.7.2 Stage-2：条件特征 ψ(i, sol0, log0)（这是两阶段选择能赢的关键）

Stage-2 的选择，本质上是回答：“在这个实例上，给定这个初解结构，哪个改进器/迭代策略更可能带来收益？”

因此 ψ 至少应包含：

1. **实例特征 φ(i)**（同上）
2. **初解质量与代价**：`q0`（如 tour length / CVRP cost）、`t_init`
3. **初解结构特征（solution features）**：这部分是 two-stage 的增益来源

TSP 常用的 `sol0` 特征（可实现、与邻域偏好强相关）：

- `length(sol0)` 与 `length(sol0)/n`
- 边长统计：mean/std/分位数、长边比例（top 10% edge length fraction）
- 交叉/几何异常：交叉边数量（线段相交计数的近似）、锐角/钝角比例
- “局部可改进潜力” proxy：随机采样若干对边，估计 2-opt 正收益的比例/期望收益

CVRP 常用的 `sol0` 特征（与 destroy/repair/route-local-search 偏好强相关）：

- 路线数 `#routes`、平均路线长度、长度方差
- 载重利用率分布：mean/std、near-capacity route 比例
- slack 分布（如果有时间窗/时长约束则更关键）
- depot 径向结构：每条路线最大半径/半径方差（反映“分区” vs “混杂”）
- 交叉/重叠 proxy：路线之间的几何交叉数量、客户-路线簇混杂程度

log0 还能贡献一些“运行时信号”（强烈建议记录）：

- 初始化是否稳定（多次随机种子下方差）
- early improvement curve（如果 init 本身有内部迭代）

> 你在 `my/` 的两层 gate 设想里已经写到 “Gate2 输入 data + sol0”；  
> 这里的 ψ(i,sol0) 就是把这个想法具体化成可落地特征。

#### 8.7.3 与你的“两阶段选择”代码对齐：φ/ψ 在仓库里如何落地（实现提示）

你已经有一套很接近最终形态的工程骨架（两层 gate + `data + sol0` 条件输入）。为了把它变成“可复现的算法选择实验”，建议你把 **特征与性能矩阵** 明确落盘：

1. **φ(i)：实例特征（stage-1 输入）**  
   - 你在 `benchmarks/analyze_tsplib70_logs.ipynb` 已经实现了 TSPLIB 的一批 observable 特征  
   - 建议导出成 `features.csv`（每行：instance_id + feature columns），这会让后续 kNN/RF/EPM 训练非常顺

2. **q0/sol0：初始化输出（stage-2 条件）**  
   - 在 EasyNCO 环境里，TSP/CVRP 的 cost 计算都可以通过 env 的 cost 函数完成（你现在的 reward 也依赖它）
   - 建议把每个 `(instance, init)` 的 `sol0`（路径/路线表示）与 `q0,t_init` 保存下来（否则 stage-2 数据难复现）

3. **ψ(i,sol0)：条件特征（stage-2 输入）**  
   - 你目前在 `my/` 原型里已经把 “sol0 长度 + 一些简单统计”拼进 Gate2（这是正确方向）  
   - 下一步可以逐步增加上面列出的结构特征（边长分布、route slack 分布、2-opt potential proxy 等），并始终记录计算耗时

4. **学习到的 embedding（可选）**  
   - 如果你用论文同款 encoder 作为特征提取器（例如你已有的 `Encoder_h/Naive_Encoder` 体系），建议明确区分：  
     - φ(i)：只用实例编码（推理时可得）  
     - ψ(i,sol0)：实例编码 + “解结构编码/统计”（避免把不可得信息混进 stage-1）

为了让两阶段选择的数据“可复现、可复查、可复用”，我非常建议你把数据按 **(instance, init, iter)** 三个主键拆成几张表（ASlib 的思路，但扩展到 two-stage）：

- `instances.csv`：`instance_id, problem, n, source, ...`（元信息）
- `features_phi.csv`：`instance_id, f1, f2, ...`（φ(i)，只放 observable）
- `init_runs.csv`：`instance_id, init_id, seed, q0, t_init, sol0_path, ...`  
  - `sol0_path` 指向一个二进制文件（`.pt/.npz/.pkl`）存 tour/routes，别塞进 CSV
- `features_psi.csv`：`instance_id, init_id, g1, g2, ...`（ψ 的“解结构统计”部分；也可以把 φ 拼进来）
- `iter_runs.csv`：`instance_id, init_id, iter_id, seed, q1, t_iter, status, ...`（最终表现）

这样你后续想做的任何研究变体（监督/排序/RL、top-k/rejection、只换特征、不换算法集合等）都能在同一套数据底座上快速迭代。

这样做的好处是：你可以同时跑 “手工特征选择器” 与 “embedding 选择器”，并在同一套评估协议（SBS/VBS/win-bucket）下公平对比。

#### 8.7.4 一个非常实用的建模分解：把 Stage-2 学成“改进量 Δ”，而不是直接学最终 q1

在 two-stage 场景里，Stage-2（选 iter）最自然的学习目标其实是“给定初解能带来多少改进”，而不是“最终成本是多少”。原因：

- `q0`（初解质量）在 Stage-2 决策时是已知的；你没必要让模型重复学习它
- 不同 init 产生的 `q0` 尺度/分布可能不同；直接回归 `q1` 会让模型更难泛化

一个简单但常常非常有效的分解是：

`q1(i, init, iter) = q0(i, init) + Δ(i, init, iter)`

其中 `Δ` 是 iter 相对 init 的“改进量”（对最小化问题通常 `Δ ≤ 0` 越小越好）。你可以：

- 训练 `f_iter(ψ(i,sol0)) ≈ Δ`（每个 iter 一个回归器，或一个多输出模型）
- 选择：`iter* = argmin_iter ( q0 + ŷΔ_iter )` 等价于 `argmin_iter ŷΔ_iter`

这会带来两个工程好处：

1. **更贴合“迭代器的职责”**：迭代器本质就是把现有解往下推，你直接学它的增益更自然
2. **更利于组合扩展**：当 `|Init|×|Iter|` 变大时，你至少把一半复杂度“压到 Stage-2 的条件特征”里，而不是把所有组合当独立算法

如果你未来把评价从 `q` 换成 `gap/time`，也可以做同样的分解（例如学 `Δgap`、或学 `Δq` 再换算到 gap）。

#### 8.7.5 进一步的“自适应”：Stage-2 可以做成 probe-based 的动态选择（很像经典 portfolio schedule）

如果你允许 Stage-2 多花一点点时间做试探（这在很多系统里非常常见），可以用一个非常通用的思路降低选错风险：

1. 对每个候选 iter 先跑 **很短的 probe**（固定 1~k 步，或固定很小的时间片）
2. 观察 early improvement（例如 `Δq/k` 的斜率、或者是否快速消除明显坏结构）
3. 把 probe 结果作为 `log0/log_probe` 的一部分，再决定把剩余预算给谁

它等价于：把 Stage-2 从“静态一次性选择”升级为“动态调度/短试探 + 分配预算”，在算法选择文献里属于非常标准的 portfolio/schedule 体系（尤其能显著改善尾部风险）。

### 8.8 训练与数据：两阶段选择怎么“标注/学习”才划算？

两阶段选择的核心难点是组合空间 `|I|×|T|`。经典做法有三类（你可以按成本从低到高逐步走）：

#### 8.8.1 Baseline：把每个组合当成独立算法（最简单、但成本最高）

直接学：

- `x=φ(i)` → 选 `(init,iter)`  
或预测每个组合的 `m(i, init, iter)` 再 argmin。

它的优势是实现简单、能给你一个“上限对照”；劣势是组合一多数据就爆炸。

#### 8.8.2 分层监督学习（推荐你首先实现的“可解释版本”）

1. 先训练 Stage-2：  
   - 数据：对每个实例 i、每个初始化器 init 跑出 `sol0`，再把所有 iter 的结果跑一遍  
   - 学习：`ψ(i,sol0) -> best iter`（或预测每个 iter 的收益/时间）
2. 再训练 Stage-1：  
   - label 可以定义为：在 oracle Stage-2 下该 init 的最终最好结果  
   - 即 `label_init(i) = argmin_init min_iter m(i,init,iter)`

这等价于“先把 Stage-2 学好，再用 Stage-2 的 oracle 定义 Stage-1 的目标”。  
你可以把它看成一种分解：先学局部决策（iter），再学全局入口（init）。

#### 8.8.3 在线/弱标注：contextual bandit / RL（当全组合标注太贵时）

当你不想为每个 i 跑全组合时，可以直接把两阶段选择当作两步决策：

- action1：选 init
- action2：选 iter
- reward：最终 `-gap` 或 `-(gap+λ·time)`（按你的 KPI）

用 REINFORCE/Actor-Critic 更新两层 policy：  
这与你在 `my/` 原型里写的两层 gate + baseline 的思路一致。

优点：不需要全组合标签；缺点：训练更不稳定，需要精心设计 reward、baseline、探索策略。

### 8.9 评价与论证：如何证明“两阶段选择”真的比“单阶段/单方法”强？

建议报告至少这几条（对应算法选择领域的标准基线）：

1. **SBS（单一 best 组合）**：固定一个 `(init,iter)` 全部实例用它  
2. **Two-stage SBS**：固定 init 与固定 iter（不随实例变）  
3. **Two-stage selector**：你的 `π1, π2` 随实例变  
4. **VBS（oracle）**：
   - 组合 oracle：`min_{init,iter} m(i,init,iter)`（上限）
   - 两阶段 oracle：`min_init min_iter m(i,init,iter)`（与组合 oracle 等价，但写法更贴近两阶段）

以及你已经很擅长的两类分析：

- 逐实例 win/tie/loss（只看 `aug_gap` 或 `gap/time`）
- 分桶 win（按 `n`、按分布特征、按 slack 紧张度…）

两阶段选择真正“能证明有效”的形态通常是：

- selector 明显优于 SBS  
- 且在某些子群（例如某个 n 段、某类分布）里 win ratio 明显更高  
  这就直接支持“不同阶段/组件对不同实例有偏好性”的论点。

---

## 9. 推荐阅读清单（围绕“按实例选算法”）

下面按“用途”分组列一份更工程化的必读清单（均为代表性/高引用/社区广泛使用的工作）：

**A. 基础框架（先建立共同语言）**

- Rice 1976, Algorithm Selection Problem, DOI `10.1016/S0065-2458(08)60520-3`
- Wolpert & Macready 1997, No Free Lunch, DOI `10.1109/4235.585893`
- Gomes & Selman 2001, Algorithm portfolios, DOI `10.1016/S0004-3702(00)00081-3`

**B. 经典系统（你实现时最像“参考实现”的）**

- SATzilla (SAT), DOI `10.1613/jair.2490`
- SATzilla-07（早期设计分析也很值得看）, DOI `10.1007/978-3-540-74970-7_50`
- SUNNY (CSP/CP), DOI `10.1017/S1471068414000179`
- claspfolio2 (ASP), DOI `10.1017/S1471068414000210`
- Hydra（自动构造 portfolio）, DOI `10.1609/aaai.v24i1.7565`

**C. Benchmark/竞赛范式（怎么做“像领域里一样”的评估）**

- ASlib, DOI `10.1016/j.artint.2016.04.003`
- ICON Challenge on Algorithm Selection, DOI `10.1609/aimag.v38i2.2722`

**C2. 综述（快速建立全景，写 related work 很好用）**

- Kotthoff 2014, Algorithm Selection for Combinatorial Search Problems: A Survey, DOI `10.1609/aimag.v35i3.2460`
- Kotthoff 2015, On Algorithm Selection (Constraints), DOI `10.1007/s10601-015-9214-x`

**D. 运行时/性能预测（EPM 是选择器背后的发动机）**

- Algorithm runtime prediction (methods & evaluation), DOI `10.1016/j.artint.2013.10.003`
- Performance prediction & automated tuning, DOI `10.1007/11889205_17`
- （并行 portfolio 方向）Selection and Configuration of Parallel Portfolios, DOI `10.1007/978-3-319-63516-3_15`

**D2. 算法配置（与“选择”强耦合：构造 portfolio / 自动搭建选择器）**

- ParamILS, DOI `10.1613/jair.2861`
- SMAC（Sequential Model-Based Optimization for Algorithm Configuration）, DOI `10.1007/978-3-642-25566-3_40`
- irace package, DOI `10.1016/j.orp.2016.09.002`

**E. 自动化构建选择器（AutoML for algorithm selection）**

- AutoFolio, DOI `10.1613/jair.4726`

**F. Meta-learning（把“实例性质”系统化）**

- Smith-Miles 2009, Meta-learning for algorithm selection, DOI `10.1145/1456650.1456656`
- Brazdil & Giraud-Carrier 2018, Metalearning & Algorithm Selection, DOI `10.1007/s10994-017-5692-y`

**F2. 连续优化里的“实例特征 → 选优化器”（给你很多特征/评估启发）**

- Kerschke et al. 2015, Algorithm selection for black-box continuous optimization (survey), DOI `10.1016/j.ins.2015.05.010`
- Mersmann et al. 2011, Exploratory landscape analysis, DOI `10.1145/2001576.2001690`
- Mersmann et al. 2012, ELA + cost-sensitive learning for algorithm selection, DOI `10.1145/2330163.2330209`

**G. （可选）AutoML：算法选择+超参优化**

- Auto-WEKA, DOI `10.1145/2487575.2487629`

**H. （可选）现代方向：深度学习/表示学习做选择（更偏研究前沿）**

- Loreggia et al. 2016, Deep Learning for Algorithm Portfolios, DOI `10.1609/aaai.v30i1.10170`

---

## 10. 你可以直接把这份报告接到现有仓库工作流的方式（建议）

你已经在做的日志分析（逐实例 win、n 分桶偏好）本质上就是算法选择的 evaluation 报告格式。下一步你可以：

1. 把 `method -> {instance -> aug_gap}` 变成一个训练表  
2. 把你已有的 instance features 拼上去  
3. 用 kNN / RF 回归先做一个 selector baseline  
4. 在 TSPLIB/CVRPLIB 上做 cross-validation，看 selector 是否显著优于 SBS  
5. 如果有效，再把选择对象细化到 init×iter 或 iter 内算子池

这条路径基本就是：**从“证明组合有效” → “把组合变成可学习的选择器”**。

---

## 11. 面向你“两阶段选择（init→iter）”的更具体落地：实例特征怎么提？怎么接经典算子选择 baseline？

你现在的系统本质上同时包含两类“选择”：

1. **外层（instance-based）**：按实例性质选择 *初始化器 init*（以及可能的 *迭代器 iter*）。  
2. **内层（operator selection / hyper-heuristic）**：在迭代阶段（或更细粒度）按反馈自适应选择“算子/邻域/模块”。  

两者并不冲突，反而可以非常自然地融合：**外层用实例特征做“先验/粗分流”，内层用在线反馈做“后验/微调”**。

算子选择（AOS/超启发式/bandit/RL）的更系统综述与公式/伪代码细节见：`adaptive/operator_selection_classics.md`。

下面给你一套可以直接照做的“特征提取配方 + baseline 组合方式”。

### 11.1 先把可用信息分三层：φ / ψ / log（你提特征时不容易混）

**(A) φ(i)：实例特征（Stage-1 输入）**  
只依赖输入数据（TSP 坐标；CVRP: depot+customers+demands+capacity），推理时可得。  
目标：捕捉“算法偏好性”的大方向（规模、分布、紧张度、结构性）。

**(B) ψ(i, sol0, log0)：条件特征（Stage-2 输入）**  
只在你已经做完初始化、拿到初解 `sol0` 后才可得。  
目标：捕捉“这个初解更适合用哪类改进器”的信息（局部可改进潜力、结构缺陷、路线负载形态等）。

**(C) online log / probe（可选）：短试探的动态信号**  
例如对多个 iter 各跑 1~k 步，看 early improvement slope；或者记录迭代阶段最近 W 步的改进速率。  
目标：在“模型不确定/实例 OOD”时降低选错风险（经典 portfolio schedule 思路）。

> 工程建议：  
> - **先把 φ + ψ 做好**（离线可复现），再上 probe（否则难 debug）。  
> - 特征必须记录提取耗时；很多选择器“理论上变好，整体却变慢”就是 feature_time 没算。

### 11.2 φ(i) 怎么提：两条路线 + 一个强 baseline

你可以并行维护两套 φ(i)（论文里也常这么做）：

1. **手工特征（可解释、稳定、便宜）**：便于做统计分析与写论文 ablation。  
2. **学习到的 embedding（表达力强）**：例如用你现成的 encoder（你仓库里已经有）。  

强烈建议保留一个 **kNN/SUNNY baseline** 只用手工 φ：它几乎零训练成本，但非常难被“水模型”打败，是长期可靠对照组。

### 11.3 TSP 的 φ(i)（只依赖坐标）的“可实现特征清单”

以下特征都不依赖最优解，可部署；并且你在 `benchmarks/analyze_tsplib70_logs.ipynb` 已经实现/接近实现了其中很多（可以直接复用）。

**规模与尺度**

- `n`（节点数）
- `area_bbox`：bounding box 面积（先做平移/缩放到 [0,1] 可省去绝对尺度）
- `bbox_aspect = (x_span / y_span)`（形状拉伸）

**分布形状（PCA/凸包）**

- PCA：`pc1_var_ratio`、`pc_anisotropy = λ1/λ2`
- `hull_ratio = area(convex_hull) / area_bbox`
- `hull_fill = n / area(convex_hull)`（密度 proxy；注意归一化）

**近邻结构（clusteriness / uniformity）**

- 最近邻距离 `nnd` 的 mean/std/CV
- `nnd_over_expected`：与均匀分布期望最近邻的比值（越小越聚团）
- `grid_entropy`：把坐标落到 G×G 网格计数后算熵（越大越均匀）

**距离分布（全局几何的低维摘要）**

- 采样若干对点距离，计算 mean/CV/分位数（避免 O(n^2)）

**Landmarking（强烈推荐，信息量大）**

- 跑一个极便宜构造法（NN / greedy insertion）得到 `len_nn`
- 再跑少量 2-opt（比如 20~50 次随机交换）得到 `Δlen_2opt_probe`、`improve_rate`

Landmarking 在算法选择里非常经典：它相当于“让实例自己暴露可解性”。对你这种“不同方法偏好不同实例”的问题通常非常有效。

### 11.4 CVRP 的 φ(i)（输入=坐标+需求+容量）的“可实现特征清单”

CVRP 除了几何分布，还多了“可行性/紧张度”和“需求结构”，这些往往直接决定算子偏好（例如 removal/repair 的有效性）。

**规模与紧张度（非常关键）**

- `n`（customers 数）
- `Q`（capacity）
- `sum_demand / Q`（车辆数下界 proxy）
- `max_demand / Q`、`p95_demand / Q`（大需求客户比例）

**需求分布**

- mean/std/CV、skewness（偏态）、kurtosis（重尾）
- `gini` 或 top-k 需求占比（是否存在“少数大客户”）

**几何分布（客户+depot）**

- 客户坐标做 TSP 那套特征（PCA/网格熵/近邻统计/凸包）
- 额外加“到 depot 的径向统计”：`r_mean/r_std/r_cv/r_max_over_mean`

**Landmarking（CVRP 同样适用）**

- 构造一个非常便宜的可行解（例如 sweep / NN insertion）得到 `cost_greedy`
- 跑少量 relocate/swap（或你已有迭代器的 very-few-step probe）得到 early improvement slope

### 11.5 ψ(i, sol0)：TSP 的“解结构特征”（决定 Stage-2 能否赢）

你现有实现里已经用 `length0 + segment stats`（见 `my/features.py`），这是正确的最小起点。下面是一些“信息量更大但仍可实现”的 ψ 特征（建议按成本逐步加）：

**(1) 质量/归一化**

- `q0 = length(sol0)`
- `q0/n`（每节点平均边长）
- `q0 / sqrt(n * area_bbox)`（弱尺度不变的归一化；不需要最优解）

**(2) 边长分布（局部搜索偏好很敏感）**

- `edge_mean/std/CV`
- `edge_p90/p99`
- `long_edge_frac = #(edge > p90) / n`

**(3) 交叉/几何“坏结构” proxy（2-opt/3-opt 特别吃这个）**

- 交叉边数量（可用随机采样边对近似，避免 O(n^2)）
- 角度异常比例（连续 3 点夹角过小/过大）

**(4) 2-opt 改进潜力（强 feature，且可用随机采样估计）**

随机采样 K 对边 `(a,b),(c,d)`，估计

`gain = dist(a,b)+dist(c,d) - dist(a,c)-dist(b,d)`

统计：

- `p_gain_pos = P(gain>0)`  
- `gain_mean_pos = E[gain | gain>0]`  

直觉：如果 `p_gain_pos` 很低，说明 tour 已经“2-opt 很饱和”，可能需要更强/更全局的迭代器。

### 11.6 ψ(i, sol0)：CVRP 的“路线结构特征”（决定 destroy/repair 类算子效果）

把 `sol0` 看成 routes 列表后，建议最少做这些统计（全部都可 O(n) 计算）：

**路线规模与长度形态**

- `num_routes`
- `route_len_mean/std/CV`（路线长度分布）
- `route_size_mean/std`（每条路线客户数分布）

**载重与 slack（destroy/repair 偏好高度相关）**

- `load_util_mean/std`（载重利用率）
- `near_capacity_frac = #(util > 0.9)/num_routes`
- `slack_mean/std/min`

**几何结构（分区 vs 混杂）**

- 每条路线的“最大径向”与“径向方差”（到 depot 的半径统计）
- 路线之间的覆盖重叠 proxy（例如路线 centroid 的距离分布）

这些 ψ 特征能直接解释：  
为什么某些迭代器/破坏修复策略在“路线很满/很紧”时更有效，而在“路线很松/几何更均匀”时反而浪费时间。

### 11.7 怎么把“经典算子选择（AOS）baseline”接到你的 Stage-2（不需要你重写 solver）

你现在 Stage-2 的动作空间是：从若干 *迭代器* 里选一个（LEHD-RRC / 2-opt / DACT / …）。  
经典 AOS 文献的动作空间通常更细（邻域算子/破坏算子/修复算子），但你可以先用一个非常实用的映射：

> **把每个“迭代器 iter”当作一个高层 operator**，每次只运行一个很小的 chunk（固定步数或固定时间片），然后根据改进量给它记分/更新权重。

这相当于把 Stage-2 从“一次性选择 iter”升级为“iter 的在线调度/自适应分配预算”，属于非常标准的 portfolio schedule / operator scheduling baseline。

下面给你几类强 baseline（你可以直接拿来做对照）：

**Baseline A：ALNS 风格权重自适应（工程上最常用）**

- 维护每个 iter 的权重 `w_j`（初始均匀或由 φ/ψ warm-start）
- 每个 chunk 选择 `j ~ roulette(w)`  
- 观察本 chunk 的 credit（见 11.8），用指数平滑更新：

`w_j ← (1-ρ) w_j + ρ · score(credit_j)`

其中 `score` 可以用分档（新 best / 有改进 / 无改进）或直接用 `credit` 的截断值。

**Baseline B：多臂老虎机（MAB）：UCB / Thompson / EXP3**

把 iter 看成 arms：

- reward/credit：用“单位时间改进量”或“是否改进”的二值奖励
- UCB1：适合相对平稳；Thompson：实现简单且探索自然；EXP3：更抗非平稳/对抗性

优势：更“理论化/标准化”，写论文时很好解释；作为 baseline 很有说服力。

**Baseline C：Choice Function（超启发式里最经典的启发式选择器）**

`CF(j) = α·(近期改进) + β·(距离上次使用时间) + γ·(与上一个 iter 的组合效应)`

它比 bandit 多了一个“结构化探索（recency）”与“二阶协同（sequence effect）”，在某些实例上非常稳健。

**Baseline D：Probe-then-commit（非常像 SATzilla/SUNNY 的 schedule 思想）**

- 对每个 iter 先跑一个 very-short probe（例如 1~k 步 / 0.1 秒）
- 按 probe 的单位时间改进率排序，剩余预算给 top-1 或 top-k

这往往能显著降低“模型/策略选错导致灾难性退化”的尾部风险。

### 11.8 credit（奖励/记分）怎么定义：建议你统一用 “Δq / time” 的视角

不管你是做 RL gate，还是做 AOS/bandit baseline，你的“反馈信号”尽量统一，否则对比不公平。

对最小化问题（TSP/CVRP cost 越小越好），一轮 chunk 的自然 credit 是：

- `Δq = q_before - q_after`（正数表示有改进）
- `credit = Δq / max(time_chunk, ε)`（强调性价比）

常见稳健化：

- `credit = clip(Δq/time, 0, c)`（避免极端值）
- 或分档：`new_global_best -> 5, improved -> 3, accepted -> 1, else -> 0`（ALNS 常用）

这样你才能回答你关心的研究问题：  
“某些实例上，**不同迭代模块的‘单位预算收益’**显著不同，因此选择/调度是有价值的。”

### 11.9 把“外层实例特征”与“内层算子选择”融合：推荐的 3 种组合方式（由易到难）

**融合方式 1：外层做 warm-start，内层做在线自适应（强实用、最推荐先做）**

- 用 φ/ψ 预测一个 `prior p(iter|ψ)`（哪怕只是一个简单回归/softmax）
- 初始化 `w_j ← p_j` 或初始化 bandit 的先验（Thompson 特别方便）
- 运行过程中再用 ALNS/bandit 更新 `w`

直觉：实例特征给“先验偏好”，在线反馈纠正“预测误差/非平稳”。

**融合方式 2：外层选一个“算子选择策略/超参数”（meta-selection）**

对不同类型实例，你可能需要不同的 AOS 风格：

- clustered + 初解很差：更强探索（大扰动/更 aggressive 的 removal）
- uniform + 初解已很强：更保守精修（2-opt/轻量邻域）

你可以让 Stage-2 先选 “AOS 模式”（比如 ρ、温度、探索率、tabu 长度），再在该模式下做 operator selection。

**融合方式 3：Contextual bandit / RL（把 ψ 当 context，每步更新）**

如果你愿意把 Stage-2 细化到多步决策（chunk-by-chunk），那它自然就是 contextual bandit / RL：

- context：`ψ(i, current_sol, progress)`  
- action：选 iter 或选低层 operator  
- reward：`Δq/time`

这条路上限最高，但也最难调；建议以 11.7 的 baseline 作为对照逐步上去。

### 11.10 与你仓库当前实现的直接对齐点（你可以马上改哪几处）

你现在已经有一个非常好的“最小两阶段特征实现”：

- `my/features.py`：
  - `tsp_instance_features(...)` 基本对应 φ(i)（当前默认是 paper encoder embedding）
  - `tsp_gate2_features(...)` 对应 ψ(i,sol0)（当前是 `length0 + tour edge stats`）

如果你要更系统地做特征 ablation，建议按下面顺序扩展（每一步都能独立跑实验）：

1. **给 φ 增加一套“手工特征 extractor”**（对齐你 notebook 里的几何特征：grid_entropy、nnd_*、PCA/hull…）  
2. **给 ψ 增加 2-opt potential / crossing proxy**（只做随机采样近似即可）  
3. **把 Stage-2 从“一次性选 iter”改成 “chunk 调度 iter”**，接入 11.7 的 AOS baseline（ALNS / bandit / choice function）  
4. 在 TSPLIB/CVRPLIB 上做：  
   - φ-only vs φ+ψ  
   - one-shot Stage-2 vs AOS Stage-2 schedule  
   - 逐实例 win + n 分桶 win（你现在已经有成熟分析模板）

这样你就能非常清楚地回答：  
“性能提升到底来自哪：实例分流？初解条件信息？还是迭代阶段的在线自适应？”

---

## 12. 把经典“算子选择 baseline”真正落到你的两阶段系统里（online & offline，偏好上手可复现）

这一节更“工程+复现导向”：我把经典算子选择/超启发式/AOS 文献里最常用、最好上手的 baseline 按 **online / offline** 分组，全部映射到你当前的 Stage-2（以及可选的 Stage-1→Stage-2 融合），并明确标注它们属于：

- **启发式算子选择（heuristic / rule-based）**：主要由手工规则/打分函数/权重更新构成，不需要训练数据（或只需要极少统计）。
- **学习式算子选择（learning-based）**：明确做“学习/估计”（bandit/RL/监督学习/离线学习等），通常需要更多日志或训练数据，但可系统化提升。

> 重要澄清：很多经典 AOS 方法（如 ALNS 权重更新、Adaptive Pursuit）虽然“像学习”，但它们通常被当作 **启发式自适应** 的 baseline（参数少、易复现、很稳）。  
> 我在这里会按“是否显式建模为学习问题（bandit/RL/监督模型）”来分。

### 12.1 统一接口：把 Stage-2 写成“chunked operator scheduling”（几乎所有 baseline 都能套）

你要把 Stage-2 的“选 iter”从一次性决策，改造成下面这种标准接口（不要求你重写 solver，只需要把迭代器拆成可重复调用的小段）：

- 你有算子集合 `O = {o1,...,ok}`  
  - 最容易上手的做法：`O` 就是你的 iterators（LEHD-RRC / 2-opt / DACT / …）。  
  - 更细粒度（以后再做）：把一个迭代器内部的不同 move/removal/repair 当作低层算子。
- 总预算 `B`：可以是总迭代步数、总 wall-clock 秒数、或总 “solver calls” 次数。
- chunk 长度 `c`：每次选一个算子，只运行 `c` 步（或 `τ` 秒），得到一次可观察反馈。
- 状态/上下文：  
  - `φ(i)`：实例特征（Stage-1 用）  
  - `ψ(i, sol_t, log_t)`：条件特征（Stage-2 用；可用你现有 `sol0` 的特征，并可加入 progress）
- credit（奖励）：建议统一用 `Δq / Δt`（见 11.8），或其分档版本。

把它写成通用循环就是：

1. `sol ← sol0`，`best ← sol0`  
2. for `t=1..T`（直到耗尽预算 `B`）：  
   - 计算 `context_t = ψ(i, best, progress)`（或只用 `best` 的少量统计）  
   - 选择算子 `o_t`（用下面任一 baseline）  
   - 运行 `o_t` 一个 chunk：得到 `sol'`、耗时 `Δt`、质量 `q'`  
   - 更新 `best`（若更好）  
   - 计算 `credit_t` 并更新选择器内部状态（在线方法才需要）  
3. 返回 `best`

这样，你就把 Stage-2 变成了一个“标准算子选择问题”，下面所有 baseline 都可以直接套进去。

#### 12.1.1 chunk 公平性：按“步数”还是按“时间”？（这是很多 AOS 实验不公平的根源）

当你把“不同迭代器/算子”放到同一个选择器里，最容易踩的坑是：**每个算子的单位成本不同**。

- 如果你按“固定步数/固定 move 次数”定义 chunk：  
  某些算子一步很贵（例如复杂 repair 或大 neighborhood），会吃掉更多时间；对比会偏向“便宜但改进小”的算子。
- 如果你按“固定 wall-clock 时间片”定义 chunk：  
  每个算子得到相同时间预算，更接近公平；但实现上需要你能中断/限定迭代器运行时长，或至少能控制其内部 steps 上限。

**强烈建议（上手且公平）**：

1. 以 **时间片** 作为 chunk（例如 `τ=0.1s/0.2s/0.5s`），并用 `Δq/Δt` 做 credit；  
2. 如果你暂时做不到时间片中断，就用固定 steps，但务必用 `Δq/Δt` 归一化，并在报告里说明“chunk cost 不同，credit 已按时间归一化”。

#### 12.1.2 incumbent / acceptance：输出 best，不代表你必须“只接受改进”（但先从最简单开始）

算子选择实验里通常同时维护两个对象：

- `best`：best-so-far（最终输出建议用它）  
- `cur`：当前解（用于继续搜索/允许短期劣化换长期更优）

**好上手版本（建议你先用）**：

- `cur` 永远等于 `best`（也就是只接受改进），这样系统最稳定、最易 debug

**更接近经典 HH 的版本（等你跑通再加）**：

- 引入一个 move acceptance（例如 Great Deluge / Late Acceptance / SA），允许 `cur` 暂时变差  
- credit 仍然建议按 `best` 的改进来记（更稳），或者同时记录 `cur` 的接受率作为额外信号

### 12.2 一张“上手优先”的 baseline 地图（你应该先复现哪些）

如果你的目标是尽快做出可写进论文的、可信的对照实验，建议按顺序复现：

**最推荐优先级（又强又简单）**

1. **Fixed schedule / VNS 风格轮转**（启发式、online 但不学习）：强 sanity baseline  
2. **ALNS-style 权重自适应**（启发式、online）：工程上最常用，参数少  
3. **Adaptive Pursuit**（启发式、online）：经典、稳定、极易实现  
4. **Bandit-UCB / Thompson**（学习式、online）：理论清晰、写论文好解释  
5. **Probe-then-commit / top-k schedule**（启发式+轻度学习、online）：强抗选错风险

**等你跑通后再加（提升研究深度）**

6. **Dynamic MAB（非平稳 bandit）**：Fialho 等的 “dynamic multi-armed bandit for AOS” 系列  
7. **Choice Function / Tabu HH**：超启发式传统 baseline（解释性强）  
8. **Offline 学习（监督/离线 HH）**：用于展示“实例特征+条件特征”确实能学到偏好  

#### 12.2.1 Baseline 对照表（online/offline × heuristic/learning）

下面这张表是“做实验时的 checklist”：你只要把每个 baseline 按同一套 `预算 B + chunk + credit` 接口跑一遍，就能快速证明“选择/调度是否有效”。

| Baseline | Online/Offline | Heuristic / Learning | 主要输入 | 需要日志反馈？ | 适合作为… |
|---|---|---|---|---|---|
| Fixed schedule / Round-robin | online | heuristic | 无（或 very cheap progress） | 否（可选） | 强 sanity 对照 |
| ALNS weight update | online | heuristic | credit（可分档） | 是 | 工程强 baseline |
| Adaptive Pursuit | online | heuristic | credit（估计 `Q`） | 是 | 经典概率更新 baseline |
| Choice Function | online | heuristic | credit + recency（可选 synergy） | 是 | 解释性强 baseline |
| Tabu HH selection | online | heuristic | 最近使用历史（可选 perf） | 是（可选） | 低成本多样性机制 |
| UCB1 / Thompson | online | learning | credit（可二值化） | 是 | 理论清晰 baseline |
| EXP3 | online | learning | credit（可二值/截断） | 是 | 抗非平稳/对抗 baseline |
| Dynamic MAB (discount / window) | online | learning | credit + 时间衰减 | 是 | 非平稳更贴合搜索 |
| Probe-then-commit / top-k schedule | online | heuristic (+light learning) | probe credit（短试探） | 是 | 强抗选错风险 |
| Offline fixed schedule/weights | offline | heuristic | 训练集统计 | 否（部署时） | “不靠特征”的强对照 |
| Offline supervised `ψ→argmax credit` | offline | learning | `ψ(i,sol,progress)` | 否（部署时） | “特征能学到偏好”的证据 |
| Offline sequence mining / HH offline learning | offline | learning | 轨迹序列 | 否（部署时） | “序列效应”的证据 |

### 12.3 Online 启发式 baseline（不需要训练集，也最容易复现）

#### 12.3.1 Fixed schedule / Round-robin / VNS-style（启发式，online）

**类型**：启发式算子调度（不学习）。  
**核心**：按固定顺序轮流给每个算子一个 chunk（可加“失败就跳过/缩短”的小规则）。  

为什么它重要：  
- 它是“完全不靠学习”的强对照；如果你的 AOS/学习策略都打不过它，说明 credit/接口设计可能有问题。

复现方案（最简单版）：

- `order = [o1,o2,...,ok]` 循环
- 每次取下一个 `o`，跑 chunk
- 若 `Δq<=0` 连续多次，可把该算子暂时跳过 L 轮（变成弱 tabu）

#### 12.3.2 ALNS-style 权重更新（启发式，online，最常用）

**代表性论文（ALNS 的经典范式）**：
- Ropke & Pisinger (2006) ALNS for PDPTW, DOI `10.1287/trsc.1050.0135`
- Pisinger & Ropke (2007) “A general heuristic for vehicle routing problems”, DOI `10.1016/j.cor.2005.09.012`

**类型**：启发式算子选择（score/weight adaptation），online。  

**为什么它适合你的 Stage-2**：  
- 不需要任何训练数据；只要你能定义“本 chunk 是否改进”  
- 机制简单但很强，尤其适合 VRP/CVRP 这类“破坏-修复-局部搜索”风格

最常见的 ALNS 选择器由三部分组成：

1. **score 规则（credit → 分档奖励）**  
   典型 4 档（你也可以改）：  
   - `s=σ1`：产生新全局最优（best improved）  
   - `s=σ2`：产生当前解改进（improved）  
   - `s=σ3`：产生可接受但不改进（accepted）  
   - `s=σ4`：被拒绝/无效（rejected）  
2. **指数平滑更新权重**  
   `w_j ← (1-ρ) w_j + ρ·s_j`
3. **roulette wheel 采样**  
   `P(j) = w_j / sum(w)`

复现建议（强上手默认值）：

- `ρ`：0.1（更稳）或 0.2（更快反应）  
- `σ`：`(5, 3, 1, 0)`（非常常见）  
- chunk：固定步数（例如 10~50 次 move）或固定时间片（例如 0.1~0.5s）

把它映射到你的 two-stage：

- `o_j` 可以是“迭代器模块”（LEHD-RRC / 2-opt / DACT / …）  
- 或者更细粒度时：`o_j` 是某个 destroy 或 repair 算子；ALNS 天然支持 “removal×insertion” 的二层选择（以后再做）。

#### 12.3.3 Adaptive Pursuit（启发式，online，经典概率更新 baseline）

**代表性论文**：
- Thierens (2005) “An adaptive pursuit strategy for allocating operator probabilities” DOI `10.1145/1068009.1068251`

**类型**：启发式算子选择（probability adaptation），online。  

**核心直觉**：  
- 始终把概率质量“追逐”到当前估计最好的算子上，但不会完全饿死其他算子。  
- 比 “直接 greedy 选最好” 更稳；比 ALNS 更像“明确的概率更新公式”。

典型实现要维护两类量：

- `p_j`：选算子 j 的概率（和为 1）  
- `Q_j`：算子价值估计（可用滑动平均 credit）

每轮：

1. 按 `p` 采样一个算子 `j`  
2. 得到 credit，更新 `Q_j`（例如指数平均）  
3. 找到当前最好的算子 `j* = argmax Q_j`  
4. 用追逐式更新概率：

- `p_{j*} ← p_{j*} + β · (p_max - p_{j*})`
- `p_{j}  ← p_{j}  + β · (p_min - p_{j})`  for `j != j*`

其中 `p_min` 保证探索（常设为 `0.01~0.05`），`p_max = 1 - (k-1)p_min`。

复现建议（强上手默认值）：

- `β=0.2`（反应快）或 `0.1`（更稳）  
- `p_min=0.02`（k<=10 时很合适）  
- `Q` 的更新：`Q_j ← (1-α)Q_j + α·credit`，`α=0.1`

你会发现：Adaptive Pursuit 在你的 Stage-2 场景里是一个非常“干净”的 baseline——参数少、机制明确、很好写进论文对照。

#### 12.3.4 Choice Function（启发式，online，超启发式经典）

**代表性线索**：
- Cowling et al. (2002) Hyper-heuristics 源头之一，DOI `10.1007/3-540-46004-7_1`
- Ochoa et al. (2012) HyFlex benchmark（cross-domain HH），DOI `10.1007/978-3-642-29124-1_12`
- Özcan et al. (2012) 改进 choice function（cross-domain），DOI `10.1007/978-3-642-32964-7_31`

**类型**：启发式算子选择（score function），online。  

Choice Function 的价值是：它把 exploration/exploitation 做成可解释的加和结构：

`CF(j) = α·Perf(j) + β·Recency(j) + γ·Synergy(last,j)`

在你的 Stage-2 映射里：

- `Perf(j)`：算子 j 最近 W 次的平均 `credit`（或分档 score）
- `Recency(j)`：距离上次使用 j 的轮数（越久越鼓励探索）
- `Synergy(last,j)`：上一次选的算子与当前算子组合的历史收益（捕捉“序列效应”）

复现建议：

- 先做不含 `Synergy` 的简化版（只用 Perf+Recency），已经很强且实现简单  
- 窗口 `W=20~50`  
- `α=1, β=0.1` 起步；`γ` 先设 0

#### 12.3.5 Tabu-based heuristic selection HH（启发式，online，超好上手）

**代表性论文**：
- Kendall & Hussin (2005) tabu-based HH for exam timetabling，DOI `10.1007/0-387-27744-7_15`

**类型**：启发式算子选择（tabu），online。  

复现非常简单：

- 维护 tabu list（长度 L）记录最近使用过的算子  
- 每轮从非 tabu 的算子里选“当前最看好的”（可以是随机、也可以用简单 Perf 排序）  
- 选中后加入 tabu

它的价值是：  
- 在你的 Stage-2 里能很好地避免“反复卡死在同一种迭代器上”，是非常便宜的多样性机制。

### 12.4 Online 学习式 baseline（bandit / RL）：更标准、更好写论文

#### 12.4.1 Stationary Multi-Armed Bandit（学习式，online）

如果你把每个算子看成一个 arm，reward=credit，那么 Stage-2 立刻变成 MAB。  
常用三件套（都很经典、很好复现）：

（直接面向“超启发式/算子选择”的 bandit 应用例子：  
- “A Multi-Armed Bandit selection strategy for Hyper-heuristics” (2017) DOI `10.1109/cec.2017.7969356`）

**UCB1（Upper Confidence Bound）**  
- Auer et al. (2002) “Finite-time Analysis of the Multiarmed Bandit Problem” DOI `10.1023/a:1013689704352`
- 选择规则：`argmax_j ( mean_reward_j + c * sqrt(log t / n_j) )`

**Thompson Sampling（后验采样）**  
- Thompson (1933) 原始论文，DOI `10.2307/2332286`（或 Biometrika 版本 `10.1093/biomet/25.3-4.285`）
- 实现上非常简单（尤其 reward 取二值“是否改进”时，用 Beta-Bernoulli 即可）

**EXP3（adversarial bandit，抗非平稳/对抗）**  
- Auer et al. (2002) “The Nonstochastic Multiarmed Bandit Problem” DOI `10.1137/s0097539701398375`
- 维护权重 `w_j`，按 `p_j ∝ w_j` 采样，用重要性加权更新

为什么这些对你“好上手”：

- 你只需要能定义 credit（11.8），不需要任何额外特征  
- bandit 自带 exploration，能作为 ALNS/Choice 的“学习式对照组”  
- 写论文时也更容易说清楚“学习在做什么”

#### 12.4.2 Dynamic / Non-stationary Bandit for AOS（学习式，online，强相关经典线）

你做组合优化的迭代改进时，算子收益往往强烈非平稳（早期与后期完全不同）。  
这正是 AOS 文献里 “dynamic multi-armed bandit” 的动机。

**代表性经典论文（强建议你重点读）**：

- Fialho et al. (2008) “Adaptive operator selection with dynamic multi-armed bandits” DOI `10.1145/1389095.1389272`
- Fialho et al. (2009) “Dynamic Multi-Armed Bandits and Extreme Value-Based Rewards for Adaptive Operator Selection…” DOI `10.1007/978-3-642-11169-3_13`
- Fialho et al. (2009) “Extreme compass and Dynamic Multi-Armed Bandits…” DOI `10.1109/cec.2009.4982970`
- （更偏综述/方法论的书章节）Maturana et al. (2011) “Adaptive Operator Selection and Management in Evolutionary Algorithms” DOI `10.1007/978-3-642-21434-9_7`

它们对你最有用的点：

- 明确讨论了 **reward 的设计**（例如 extreme value-based：更重视“偶尔的大改进”）  
- 明确讨论了 **非平稳**：如何遗忘旧信息、如何让策略跟随阶段变化  
- 很适合解释“为什么 Stage-2 的在线调度比一次性选择更稳健”

好上手复现路径（不需要复现论文所有细节）：

1. 先实现一个 **discounted UCB** 或 **sliding-window UCB**（遗忘旧 reward）  
2. 再加一个 “extreme value credit” 选项：  
   - 例如把 reward 定义为 `max(0, Δq)` 的分位数/截断，强调罕见的大改进  
3. 与 12.4.1 的 stationary bandit 对比，看是否更稳、更抗阶段变化

#### 12.4.3 Contextual Bandit（学习式，online，和你的 ψ 特征最契合）

当你把 `ψ(i, sol_t, progress)` 作为上下文（context），每一步选择算子，就变成 contextual bandit：

- 输入：`x_t = ψ(...)`
- 动作：选择算子 `o_t`
- 回报：`credit_t`

它比纯 bandit 更强：能学到“在不同解结构/不同阶段应该选不同算子”。  
但也更难：你需要更多数据、并要处理 exploration 与泛化。

建议你把它作为“有把握后再做”的提升项；但在论文叙事上非常自然（因为你已经在做 `data+sol0` 的 gate2）。

#### 12.4.4 Reinforcement Learning HH（学习式，online，作为高阶 baseline/对照）

在 HH 文献里 RL 通常指：

- 状态：搜索阶段/当前解结构/最近改进情况/算子历史
- 动作：选算子
- 奖励：改进幅度、是否接受、单位时间收益

RL baseline 可以很强，但也最容易“调参地狱”。  
如果你要上 RL，建议把它放在已有 bandit/ALNS 强 baseline 的基础上，作为“学习式上限探索”。

补充：两类“可引用且比较经典”的 RL HH/AOS 线索（你写 related work/做对照很有用）：

1. **直接把 RL 用于算子选择/AOS 的讨论**  
   - “Adaptive operator selection with reinforcement learning” (2021) DOI `10.1016/j.ins.2021.10.025`
2. **RL + move acceptance（超启发式里很常见的组合）**  
   - “A Reinforcement Learning - Great-Deluge Hyper-Heuristic for Examination Timetabling” (2010) DOI `10.4018/jamc.2010102603`  
   - “Reinforcement learning with EGD based hyper heuristic system for exam timetabling problem” (2011) DOI `10.1109/ccis.2011.6045110`  
   - （cross-domain 里也常把 selection 与 acceptance 打包设计）“Late acceptance-based selection hyper-heuristics for cross-domain heuristic search” (2013) DOI `10.1109/ukci.2013.6651310`

### 12.5 Offline 算子选择：你如何用训练集把 Stage-2 的选择“学出来”

这里的 “offline” 我按两种含义拆开（很重要）：

1. **offline training（离线训练，在线执行）**：在训练集上学一个策略/模型，测试时不再更新（或只做很轻的在线校正）。  
2. **offline per-run（离线、不依赖特征）**：只在训练集统计出一个固定 schedule/固定参数，部署时直接用（不学习/不更新）。

下面给你一组“非常好上手、也很适合写论文对照”的 offline baseline。

#### 12.5.1 Offline 固定 schedule / 固定权重（启发式，offline）

最朴素但很重要：  

- 在训练集上穷举/搜索一个 schedule（例如 `[(o2,10 chunks),(o5,20 chunks),...]`）  
- 或者搜一个固定权重向量 `w`（roulette 采样）  
- 测试时完全不变

这是“完全不使用实例特征/条件特征”的强对照：  

- 如果你的 φ/ψ 模型打不过它，说明特征或训练协议有问题  
- 如果能明显超过它，说明“按实例/按初解结构选择”确实带来信息增益

#### 12.5.2 Offline 监督学习：`ψ -> 选算子`（学习式，offline，最适合你）

这是把 Stage-2 变成一个“按状态选动作”的标准监督学习问题，也是你目前两层 gate 继续深化最自然的一步。

数据怎么构造（关键！）：

- 你需要日志样本：`(x_t=ψ(i,sol_t,progress), action=o, outcome=credit)`  
- 这些样本可以来自：
  - 你离线跑过的多算子轨迹（比如随机探索策略、或多种 baseline 生成轨迹）
  - 或者对同一 `sol_t` 尝试多个算子各跑一个 chunk（更贵但更干净）

一个非常现实但经常被忽略的问题：**离线数据的分布会“锁死”你能学到的策略**。

- 如果你的日志只来自某一个强策略（例如永远偏向某个算子），那么模型几乎看不到其他算子在其他状态下的表现 → 学出来的策略也会偏置（典型的 coverage / covariate shift 问题）。

好上手且实用的数据收集策略（建议你至少做第 1 个）：

1. **混合探索策略收集**：  
   - 例如 50% round-robin + 50% 随机，或者 70% ALNS + 30% 随机  
   - 目的是保证每个算子都在不同类型状态上被调用过，模型才有机会学到“什么时候该用谁”
2. **固定状态多算子试验（更干净，但贵）**：  
   - 对同一个 `sol_t`，让每个算子都跑一个 chunk，得到一个“局部对照表”  
   - 这会大幅降低 label 噪声，尤其适合你要做论文里的干净 ablation
3. **迭代式收集（DAgger 思想）**：  
   - 先用一个粗糙策略收集数据训练 `π`，再用 `π` 跑一遍收集新数据继续训练  
   - 好处是减少 train-test 状态分布漂移；坏处是实验复杂度上升

两种最上手的建模方式：

1. **回归式（推荐）**：对每个算子训练 `f_o(x) ≈ E[credit | x, o]`  
   - 决策：`argmax_o f_o(x)`  
   - 好处：天然支持 top-k / schedule / 风险控制  
2. **分类式**：label 取 “在这个 x 上跑哪个算子最好”  
   - 好处：实现简单  
   - 坏处：label 噪声大（算子差距很小时）

复现建议（非常工程化）：

- 输入特征：从简开始（`q`, `Δq_recent`, `p_gain_pos`, `route_slack_stats`, …）  
- 模型：RandomForest / XGBoost（对噪声与非线性很友好，几乎不用调）  
- 输出：top-1 或 top-k schedule（用 12.3.1 的 schedule 接口）

#### 12.5.3 Offline “序列学习/子序列挖掘”：学一个强 schedule（学习式，offline）

很多 HH 文献发现：**算子不是独立的，序列组合很关键**（例如 A 后接 B 特别有效）。  
因此 offline 学习经常直接学“算子序列/子序列”。

一些相关工作（用于你写 related work/方法对照）：

- “Analysing heuristic subsequences for offline hyper-heuristic learning” DOI `10.1007/s10732-018-09404-7`（也有会议版 `10.1145/3319619.3326760`）  
- “Clustering of hyper-heuristic selections … for offline learning” DOI `10.1145/3067695.3076025`

如何把它落到你的 Stage-2（好上手版本）：

1. 用训练集跑出若干条“较好的轨迹”（来自 ALNS/bandit/人工配置）  
2. 从轨迹中统计频繁子序列/高收益子序列（长度 2~4 就够用）  
3. 用这些子序列拼成一个 schedule（或对不同实例簇拼不同 schedule）  

这会给你一个非常好的对照：  
“不用任何在线学习，只用离线学出来的 schedule”能到什么水平？

#### 12.5.4 Offline 学习 Selection Hyper-Heuristic（学习式，offline）

这类工作更接近“真正的 offline 学策略”，通常会用神经网络/序列模型来学习何时选哪个启发式。  
代表例子（给你读法，不一定要立刻复现）：

- “Offline Learning with a Selection Hyper-Heuristic: An Application to Water Distribution Network Optimisation” (2021) DOI `10.1162/evco_a_00277`
- “Offline Learning for Selection Hyper-heuristics with Elman Networks” DOI `10.1007/978-3-319-78133-4_16`
- （cross-domain 里常见的“学习式选择 HH”范式之一）“A tensor-based selection hyper-heuristic for cross-domain heuristic search” (2015) DOI `10.1016/j.ins.2014.12.020`

把它映射到你这里：

- state/context：`ψ(i, sol_t, progress)`（你完全可以用手工特征 + encoder embedding 混合）
- action：选迭代器/算子
- training：从离线轨迹做 imitation / sequence prediction /（可选）离线 RL

### 12.6 你最值得做的“融合 baseline”：offline prior + online adaptation（强实用，论文也好讲）

你现在已经有 Stage-1 的实例分流（init 选择），又有 Stage-2 的条件特征（sol0）。  
最自然、也最实用的一种组合是：

1. 用 offline 模型（监督回归/分类）给出 `prior p(o|ψ)` 或 `prior score(o|ψ)`  
2. 在线用 ALNS/Adaptive Pursuit/bandit 做更新（纠错 + 适应非平稳）  

这在算法选择领域对应 “prediction + schedule”，在 HH 领域对应 “learned prior + AOS adaptation”。  
它往往能做到：

- 平均更好（因为 prior 有信息）  
- 尾部更稳（因为 online adaptation 能纠错）

### 12.7 复现实验你必须先固定的 6 件事（否则 baseline 很难公平比较）

为了让你的 Stage-2 AOS baseline “可复现、可解释、可对比”，建议你在实验方案里把下面 6 条写成硬规则：

1. **预算定义**：`B` 是总步数、总时间、还是总 chunk 数？  
2. **chunk 定义**：每次调用算子到底做多少工作？（固定步数/固定时间片）  
3. **credit 定义**：`Δq`、`Δq/time`、还是分档 score？（建议同时报告）  
4. **接受准则**：只接受改进？还是允许劣化以跳出局部最优？（影响巨大）  
5. **best-of-run 还是 last solution**：输出用 best（强烈建议）还是最后一步？  
6. **随机性与重复次数**：每实例跑几次 seed？报告 mean/median/best？（HH/AOS 波动很常见）

当你把这些固定下来，你就能非常干净地回答你核心论点：  
“在同等预算下，Stage-2 通过算子选择/调度，确实能在某些实例上显著超过单一迭代器；且不同实例的赢家不同，因此‘选择’是必要且可学的。”
