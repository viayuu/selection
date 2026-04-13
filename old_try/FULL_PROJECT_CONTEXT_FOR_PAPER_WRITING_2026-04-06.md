# 全项目论文写作上下文交接文档

本文档不是某一条单独代码线的说明书，而是写给“完全没有上下文的另一个 AI 模型”的全局交接材料。

目标是把这整个研究工作区里，和当前论文写作相关的内容尽可能完整地交代清楚，包括：

- 最早的 motivation 验证阶段
- 中间做过的各类方法与原型
- 为什么研究问题一再收缩与转向
- 当前最可信的主线是什么
- 代码、实验、日志、分析、结果解释时最容易踩的坑是什么
- 用户在这整个过程中反复强调过的偏好、口径和关注点是什么

如果另一个 AI 之后要继续：

- 写论文
- 写 related work / method / implementation / experiments
- 补实验分析
- 解释代码和结果
- 继续推进当前主线

那么它应该先读这份文档，再按文中的路线去看更细的子文档。

---

## 0. 最重要的一句话

这个工作区不是一个单一项目，而是一个持续演化的研究工作区。

从最早的 “验证 initialization / iteration 是否能拆开并组合”，到后来的：

- 双 gate RL
- 每步 operator selection
- 单实例 fully-online RL
- contextual bandit / Neural-LinUCB
- supervised learning-to-rank

这些都是真实发生过的探索，不是无关噪声。

如果要写论文或继续研究，不能只盯着 `2cmab2/`，而要理解：

> 当前主线之所以会是 `2cmab2`，是因为前面很多尝试暴露了 RL 侧的难点、evaluation 侧的难点、reward 与最终 mean cost 不一致的问题、以及“初始化选择”这个问题本身更容易被做清楚。

---

## 1. 先认清这个工作区

当前工作区根目录：

- `/public/home/zhoucl/shiys`

它同时包含多条研究线与参考材料。最重要的目录如下：

- `9motivation/`
  - 很早期的 motivation 验证与平台命令实验
- `EasyNCO/`
  - 底层 NCO 平台
  - 很多初始化器 / 迭代器 / env / data loader 都来自这里
- `统计.md`
  - 平台里“哪个问题支持哪些方法”的总统计
- `实验设计.md`
  - 早期统一 benchmark 设想
- `methods源代码/初始化迭代组合分析.md`
  - LEHD / GLOP 的 init / iter 可拆分性分析
- `1two_gate/`
  - 双 gate RL 原型：先选 initializer，再选 iterator
- `1step/`
  - 每一步选 operator 的迭代式选择原型
- `1online/`
  - 单个 TSPLIB 实例上的 fully-online on-policy RL 主线
- `2cmab/`
  - 早期 Neural-LinUCB / contextual bandit 版本
- `2cmab2/`
  - 当前最主要的 contextual bandit initialization selector 主线
- `2l2r/`
  - 与 `2cmab2` 并行的 supervised learning-to-rank 分支
- `9nss论文/`
  - NSS 原论文、源码、数据集、参考分析

这几个目录的关系不是并列无关，而是有明显的时间演化和方法论继承关系。

---

## 2. 当前最应该怎么理解整个研究问题

如果只看最终留下来的最稳定表述，当前问题可以概括为：

> 对于一个组合优化实例，尤其是 `TSP` 或 `CVRP`，只在 initialization 阶段做一次方法选择，目标是让被选方法在该实例上尽量表现更好。

这里有几个关键限制，是在长期讨论后逐步收敛出来的：

### 2.1 先只做 initialization

原因不是 iteration 不重要，而是：

- 迭代阶段 RL 太复杂，credit assignment 更难
- iteration policy 很容易陷入超参、reward 设计和 collapse 分析
- 想先把“实例特征 -> 方法选择”这个研究问题单独讲清楚

### 2.2 先只做 TSP + CVRP

不是因为这两个问题一定最重要，而是因为：

- 当前平台里这两个问题的可选方法最多
- 最适合做“method selection”
- 其它很多问题可行动作太少，不适合先讲 selection 论文

### 2.3 不是在做 solver training，而是在做 selector

核心训练对象不是 LEHD、GLOP、ICAM 这类 solver 本身，而是一个 selector：

- 输入实例
- 选择方法
- 方法本身通常视作固定 black-box / fixed arm

### 2.4 不是完整 MDP，更接近 contextual bandit

尤其在 `2cmab`、`2cmab2` 主线中，问题被重新整理为：

- 上下文：实例特征
- 动作：选哪个 initialization method
- 奖励：该方法在这个实例上的离线表现映射后的 reward

这比早期 step-level RL 更容易形成清晰的 method section 和 experiment section。

---

## 3. 研究问题是如何演化过来的

下面这条时间线非常重要，因为它解释了为什么后面很多设计看起来“像折中”，其实是被之前的尝试逼出来的。

### 3.1 最早的 motivation：先验证“初始化-迭代”是不是能拆开

最早不是直接做 `2cmab2`，而是在问一个更基础的问题：

> 一个求解 pipeline 里的 initialization 和 iteration 能不能分离？如果能，是否可以交叉组合不同方法？

对应的关键材料：

- `methods源代码/初始化迭代组合分析.md`
- `9motivation/experiment.md`

这个阶段做的事情主要有两类：

1. 理论 / 代码层面阅读 GLOP、LEHD 等方法
2. 在 EasyNCO 平台里尝试用尽量少修改的方法去跑：
   - LEHD init + LEHD iter
   - GLOP init + GLOP iter
   - 以及更进一步的跨方法组合

这个阶段的关键认知是：

- 很多方法本质上确实可以抽象成 `init(x) -> s0` 与 `iter(x, s0) -> s*`
- 这使得“先选 initializer，再选 iterator”的 pipeline decomposition 变得合理
- 但 EasyNCO 原始 `eval.py` 架构默认是一套 policy 同时服务 init 和 iter，跨方法组合工程上不直接成立

这也是后面 `1two_gate/` 这条线的重要前史。

### 3.2 早期平台级 benchmark 设想：尽可能覆盖很多问题

`实验设计.md` 和 `统计.md` 对应的是另一个很早期、但非常重要的阶段：

- 想做统一 benchmark
- 尽量覆盖 EasyNCO 中所有已有问题与方法
- 统一用 `NoIteration`，只看 initialization

从这个阶段沉淀出来的非常重要的事实是：

- 平台里方法-问题支持矩阵非常不均匀
- 真正适合先做 selector 的，其实主要就是 `TSP` 和 `CVRP`
- 这直接影响了后来的主线收缩

可以说：

> `统计.md` 是后面“为什么先只做 TSP+CVRP”的关键证据之一。

### 3.3 第一代 RL 原型：`1two_gate/`

当确认 init / iter 可以拆分后，最自然的下一步就是：

> 不再把 solver 当一个整体选，而是分两次选。

对应目录：

- `1two_gate/`

核心思路：

- Gate1：选择 initializer
- Gate2：基于实例和 `sol0` 再选择 iterator

这是一个明显受 NSS “实例级 selector” 启发、但又比 NSS 更细粒度的方案。

它回答的是：

> 如果一个实例适合的初始化方法和适合的迭代方法并不一致，能不能通过两次决策把它们拆开？

这一阶段的核心特点：

- TSP 原型
- 训练时在线生成随机 TSP
- 强化学习范式，训练 gate，不训练 solver
- reward 通常是最终长度的负值

这条线非常重要，因为它是第一次把“选择器”从 solver-level 拉到了 pipeline-level。

### 3.4 第二代 RL 原型：`1step/`

之后发现“双 gate 各选一次”仍然太粗。

隐含问题是：

- 一个实例在整个 improvement 过程中，未必始终适合同一种 iterator
- 优化过程可能是阶段性的

于是演化为：

> 初始化只选一次，但迭代阶段的 operator 每一步都重新选。

对应目录：

- `1step/operator_policy/`
- 以及相关分析文档 `1step/operator_selection_iteration_report.md`

这时问题已经变成典型的 solution-improvement MDP / operator scheduler：

- 状态：当前完整解
- 动作：下一步选哪个 operator
- 转移：operator 作用后得到下一解

这个方向更接近 hyper-heuristic / learned search control。

### 3.5 第三代 RL 主线：`1online/`

再往后，工作进一步转向：

> 单个 TSPLIB 实例、完全 online、on-policy 的训练。

对应目录：

- `1online/`

它的设定更“纯”：

- 一个 TSPLIB 实例就是一个 run
- 在线几何增强构成 batch
- 训练时采样，评估时 greedy
- init once + per-step operator selection
- reward = `-final_length`

这条线的重要意义不是它最终成为论文主线，而是它逼出了很多后来在 `2cmab2` 里仍然会反复讨论的问题：

- online / offline 到底怎么界定
- reward 到底应该直接优化什么
- 日志要细到什么程度才方便复盘
- collapse 是训练阶段还是 greedy eval 阶段发生的

### 3.6 第一代 contextual bandit：`2cmab/`

在经历了多条 RL 原型之后，研究重点开始回收：

> 不再做长 horizon 的 operator selection，转而做 initialization-only 的 solver selection。

于是出现了 `2cmab/`：

- 任务：TSP / CVRP 上的 online Neural-LinUCB 求解器选择
- 特征：实例编码后得到 `phi(x)`
- 每个 arm 自己维护 LinUCB 统计量

这一代已经很接近当前主题了，但它仍有明显缺点：

- `phi` 只编码实例，不编码动作
- 需要 per-arm 独立统计量
- 训练实现更慢
- GPU 利用不理想

### 3.7 当前主线：`2cmab2/`

`2cmab2/` 是前面所有经验收缩后的当前主线。

它保留了几个最重要的元素：

- initialization-only
- TSP + CVRP
- selector 而非 solver
- contextual bandit 视角
- 引入 NSS 风格图编码与手工特征

同时又做了几个关键升级：

- 采用 action-conditioned 表征 `phi(x, a)`
- Neural-LinUCB 与 NeuralUCB 都做成共享表示 + bandit 头
- 数据协议更清晰
- 训练、验证、checkpoint、详细日志、后处理分析更系统

---

## 4. 各条研究线分别做了什么

这一节按目录逐个总结。

### 4.1 `9motivation/`：最早的动机验证

关键文件：

- `9motivation/experiment.md`

这个阶段最重要的事情是：

- 尝试用 EasyNCO 原生 `eval.py` 跑 LEHD / GLOP 以及跨方法组合
- 验证 “init + iter” 这个 decomposition 是否合理
- 搞清楚平台原生架构为什么不方便直接做 cross-policy composition

这个阶段的论文意义：

- 它可以作为方法动机的来源
- 可以说明为什么后续会把 solver-level selection 拆成更细的 pipeline-level 选择

这个阶段的工程意义：

- 暴露出 EasyNCO 的单-policy 假设
- 暗示后续很多实验最好不要直接改平台大逻辑，而是外围包装

### 4.2 `methods源代码/初始化迭代组合分析.md`：init / iter 可分离性的理论与源码证据

这是非常高价值的一份“研究动机与方法拆解”文档。

它围绕 LEHD 和 GLOP 做了很细的分析，结论大致是：

- LEHD 和 GLOP 都可以抽象成 init + iter
- 它们的 iter 部分在某些意义上只依赖 `(instance, current_solution)`，因此具备组合可能性
- 组合是否有效取决于分布偏移和解表示兼容性

如果之后论文需要一个更强的 motivation section，这份文档可以直接转化为：

- 为什么把 solver 当成“由 initialization 与 iteration 两部分组成”是合理的
- 为什么值得研究 initialization-only selector

### 4.3 `1two_gate/`：双 gate RL 原型

关键词：

- TSP 原型
- Gate1 选 initialization method
- Gate2 选 iteration method
- RL 训练 gate，不训练 solver

关键文件：

- `1two_gate/train_two_gate_tsp.py`
- `1two_gate/solver_zoo.py`
- `1two_gate/features.py`
- `1two_gate/paper_encoder.py`
- `1two_gate/实现说明.md`
- `1two_gate/实现验证分析.md`
- `1two_gate/原论文和双gate对比.md`

它的研究意义：

- 第一次把选择粒度从“选完整 solver”细化到“分组件选”
- 说明 pipeline decomposition 是可操作的

它的局限：

- 仍然假设整个迭代阶段只需要选一次方法
- 训练是 RL，variance 和超参问题依然存在

### 4.4 `1step/`：每步 operator 选择

关键词：

- 初始化只选一次
- 之后每一步迭代都重新选 operator
- 更接近 operator scheduler / hyper-heuristic

关键材料：

- `1step/operator_selection_iteration_report.md`
- `1step/operator_policy/` 下的实现

这条线的核心贡献不是最后实验最好，而是：

- 它明确指出“双 gate 一次性选 iterator”过粗
- 让整个问题更贴近 solution-improvement policy

但它同时也带来了更多复杂性：

- 长 horizon
- reward 设计更麻烦
- 评估与日志更难

### 4.5 `1online/`：单实例 fully-online RL 主线

关键词：

- 单个 TSPLIB 实例
- fully online
- on-policy REINFORCE
- 一实例一 checkpoint
- augmentation batch

关键文件：

- `1online/README.md`
- `1online/IMPLEMENTATION.md`
- `1online/train_online_instance_policy.py`
- `1online/trainer.py`

这条线的关键价值：

- 把 “online” 这个词落到了最严格的语义上
- 训练与评估日志极细
- 可以分析 collapse、探索、greedy gap、operator 随 step 的变化

它对后面 `2cmab2` 的影响不在方法形式，而在研究习惯上：

- 用户非常强调详细日志
- 希望每步都能知道选了哪个动作、各动作 score 是多少、真实结果是什么
- 评测不能只给 summary，必须给足够细的 trace 方便复盘

### 4.6 `2cmab/`：第一代 bandit selector

关键词：

- initialization-only
- Neural-LinUCB
- `phi(x)` 而不是 `phi(x, a)`
- per-arm 统计量

关键文件：

- `2cmab/README.md`
- `2cmab/COMPARISON.md`
- `2cmab/IMPLEMENTATION.md`

它是从 RL 向 contextual bandit 转向的第一代稳定实现。

它留下了两类重要遗产：

1. 研究层面：
   - 明确当前主题可以讲成 Neural-LinUCB / solver selection
2. 工程层面：
   - 对比出 `2cmab2` 的很多改动到底改善了什么

### 4.7 `2cmab2/`：当前主线

这是当前最重要的目录。

它的详细交接文档已经单独写好：

- `2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`

这份文档非常详细，重点覆盖：

- 数据协议
- 模型结构
- Neural-LinUCB / NeuralUCB 的实现
- encoder、manual features、arm embedding
- 训练、eval、日志、checkpoint、analysis
- synthetic 与 benchmark 结果

本总文档不重复那 2000 多行细节，只强调它在全局演化中的位置。

### 4.8 `2l2r/`：与 bandit 并行的 supervised ranking 分支

关键词：

- supervised learning-to-rank
- 同样的实例表示
- 同样的 arm 空间
- 同样的 TSP + CVRP
- 但训练目标不再是 bandit 式更新，而是 ranking loss

关键文件：

- `2l2r/IMPLEMENTATION.md`
- `2l2r/运行说明.md`

它存在的研究意义是：

- 作为对照，回答“bandit 是否真的有必要”
- 或者至少给出一个强的 supervised baseline

它说明：

- 当前并不是只剩一条思路
- 用户也在比较 bandit selector 与 supervised selector 哪条更适合这个问题

---

## 5. 为什么当前会收敛到 `2cmab2`

这是整个项目最关键的“路线选择解释”。

### 5.1 因为 RL 原型太重

从 `1two_gate` 到 `1step` 再到 `1online`，问题越来越细，也越来越强，但同时：

- 训练方差更大
- 代码与实验更复杂
- 很难很快形成清晰论文

### 5.2 因为 initialization-only 更容易定义清楚

对于论文写作来说，initialization-only 有几个天然优势：

- action 空间容易定义
- evaluation 比较清楚
- 同一个实例上各方法结果可离线预先跑好
- 更适合做 selector benchmarking

### 5.3 因为 TSP + CVRP 真正拥有足够多的候选方法

`统计.md` 的支持矩阵基本决定了：

- 大规模统一多问题 selector 作为第一篇论文不够扎实
- TSP + CVRP 最适合作为当前落地范围

### 5.4 因为 NSS 提供了很好的参考坐标系

NSS 的意义不是直接照搬其 supervised training，而是提供了：

- 数据集
- encoder 思路
- 手工特征思路
- benchmark 场景

这让 `2cmab2` 很容易放在一个熟悉的研究语境里：

- 不是凭空提出 selector
- 而是在已有 solver selection 研究之上，转向 contextual bandit / Neural-LinUCB

---

## 6. `2cmab2` 为什么重要，以及它到底在做什么

这部分只做全局摘要。更细的细节请看单独文档。

### 6.1 一句话

`2cmab2` 当前做的是：

> 在 `TSP + CVRP` 的 initialization-only 场景下，基于 NSS 数据和 benchmark，用 contextual bandit 方法学习“实例应该选哪个初始化方法”。

### 6.2 它不是 NSS 监督学习

虽然用了 NSS 的：

- 数据
- encoder 结构思路
- 手工特征风格

但它不是 NSS 式监督学习分类器。

外层范式是：

- contextual bandit
- 逐样本选择 arm
- 根据选中的 arm 更新 bandit 统计量

### 6.3 它在语义上更接近 online，但数据上又是离线模拟

这是用户反复追问、反复澄清过的问题。

准确的表述应该是：

- 训练流程语义：online contextual bandit
- 数据来源：离线已经知道所有 arm 结果的数据集

因此它是：

> 用离线完备结果数据，去模拟 online contextual bandit 交互。

这一点在论文里要表述得非常谨慎，不能简单写成“offline supervised”，也不能直接夸成“真实 online deployment”。

---

## 7. `2cmab2` 中最关键的设计认知

下面这些点，几乎都是用户在实现、调试、分析过程中反复追问出来的，也是以后写 method / implementation section 时最容易丢失的上下文。

### 7.1 当前的主对比对象是谁

在 `2cmab2` 当前主线上，比较对象至少包括：

- Random
- SingleBest
- SingleBestPerProblem
- Oracle
- LinUCB
- Neural-LinUCB
- NeuralUCB Diag
- 以及后来平行的 `2l2r`

而且用户非常在意：

- 不只是 top1
- 也不只是 reward
- 还要看 mean cost
- 还要看是否超过每一个单一方法

### 7.2 reward 与 mean cost 不是一回事

这是一个极其重要的经验教训。

当前 `2cmab2` 中 reward 通常来自“基于 rank 或相对位置映射到 `[0,1]`”之类的设计，直觉上是：

- 排名越好，reward 越大

这在 bandit 语义上是合理的，因为：

- 它把“效果更好”转成了更高 reward

但这带来一个后续分析上的核心矛盾：

- 训练 loss 下降，并不保证 val/test 的 mean cost 明显下降
- top1 accuracy 提高，也不一定等于 mean cost 变好

用户后来多次观察到：

- loss 已经接近 0
- 但 val mean cost 没明显改善

这不是偶发现象，而是由目标函数与最终 metric 之间的不完全一致导致的。

### 7.3 top1 与 mean cost 也不是一回事

这同样非常重要。

对 selector 来说：

- top1 命中率看的是“是否恰好选到 oracle 最优 arm”
- mean cost 看的是“选到的 arm 的实际平均代价有多低”

如果几个 arm 的 cost 非常接近，那么：

- top1 不高
- 但 mean cost 可能仍然不错

反过来，如果总是选到次优但明显差的 arm：

- top1 低
- mean cost 会更差

因此论文里不能只报 top1，也不能只报 reward。

### 7.4 Neural-LinUCB 中的表示层训练，不等于直接优化最终 mean cost

当前实现里，Neural-LinUCB 的表示层训练目标大致是：

- 用缓存下来的 `theta_at_pull`
- 与当前 `phi(x,a)` 做内积
- 去拟合当时观测到的 reward

也就是一个“让表示空间更适配 bandit 线性头”的目标。

这和直接最小化最终 mean cost 不是一回事。

因此后面出现：

- loss 很漂亮
- val mean cost 却不动

从方法上并不意外。

### 7.5 `phi(x, a)` 是 `2cmab2` 相比 `2cmab` 的关键升级

这一点在 `2cmab/COMPARISON.md` 里解释得很清楚。

`2cmab`：

- `phi(x)`

`2cmab2`：

- `phi(x, a)`

这意味着：

- arm 信息不再只靠“每个 arm 各自一组统计量”表达
- 而是直接进入表示层

这让共享线性头和共享 bandit 统计量成为可能，也更贴近“实例-方法匹配度”的建模直觉。

### 7.6 TSP 与 CVRP 虽然共用任务框架，但图编码器不是完全共享的

这一点用户问得很多。

NSS 原始 encoder 的处理方式是：

- TSP 输入节点特征维度是 2
- CVRP 输入节点特征维度是 3

所以它们不能用完全相同的首层线性映射。

在当前实现中，通常理解为：

- TSP 与 CVRP 各有自己的输入投影 / encoder 分支
- 但在更高层的 selector 框架、动作空间和线性 bandit 结构上是统一的

这也是为什么后来会有：

- problem one-hot
- problem-specific head
- problem-aware split

等设计讨论。

### 7.7 problem-specific head 的动机

用户多次追问：

> 同一个 arm 在 TSP 和 CVRP 上的表现可能完全不同，如果共享一个线性参数，会不会互相污染？

这推动了对以下问题的认真思考：

- 只靠 problem one-hot 是否足够
- 是否应该让线性头对问题类型做区分
- TSP / CVRP 是否应共享还是部分共享

当前主张大体是：

- 统一动作空间仍然有价值
- 但问题类型差异需要被显式建模
- 共享表示 + 问题特定线性头，是更合理的折中

### 7.8 表示层 buffer 与线性头 rebuild buffer 需要分开

这也是后期很重要的一次设计调整。

用户最开始对“最近窗口训练”有疑问：

- 如果不是从头训练，为什么会忘掉早期信息？

随后梳理出的更合理方案是：

- 表示层训练可以用一个最近窗口 buffer
- 线性头 rebuild 可以用更长的历史或不同的窗口

原因：

- 表示层训练过大的 history 会越来越慢
- 线性头 rebuild 相对便宜，可以保留更多 bandit 统计信息

这个设计体现了一个非常重要的工程判断：

> 表示学习和 bandit 统计更新，不必共享同一个 history 视图。

### 7.9 用户非常在意“可复盘”

这导致后来很多日志、trace、analysis 功能都被做得很细。

用户不满足于：

- 一个总的 `summary.json`

而是希望有：

- 每次选臂时选了哪个 arm
- 每个 arm 的 score 是多少
- 最终选中 arm 对应的真实详细结果
- train / val / test 上的选择分布
- top1/top2/top3
- 每个 arm 被选后真实效果如何
- 和 single_best / oracle / 每个单一方法相比如何

这对论文写作非常有帮助，因为它逼着实验部分不再只报一两个 aggregate 指标。

---

## 8. 关于论文与源码阅读，这个项目里发生过什么

这个工作区不是“拍脑袋实现”，而是反复对照过论文原文和论文代码。

其中最重要的阅读对象包括：

- NSS 论文与源码
- Neural-LinUCB 论文与 openreview 代码
- NeuralUCB 论文与源码
- LEHD 论文与源码
- GLOP 论文与源码

而且用户不止一次要求：

- 重新认真阅读论文原文
- 对照论文实现检查当前计划有没有知识性错误
- 尽量遵循“论文原来怎么做，你就怎么做”

这意味着以后另一个 AI 写论文时，必须尊重以下事实：

1. 当前实现里有些地方是“尽量贴论文”
2. 也有些地方是出于工程需求做的改造
3. 文中要明确区分“严格论文设定”和“为当前 CO 场景做的合理改写”

### 8.1 对 Neural-LinUCB 的理解演化

用户针对 Neural-LinUCB 问过很多非常细的问题，例如：

- encoder 在这个过程中会不会训练
- 当前实现是 online 还是 offline
- 每次选臂到底基于什么选
- 这样能否迁移到真正 online 场景
- 为什么不每次都更新 encoder，而是隔一段时间更新
- 原论文是不是每次都从固定初始化点开始训隐藏层
- 如果当前实现不是从头训，会不会偏离“深度表征 + 浅层探索”的理论

这些问题最终逼出了两层重要认知：

1. 论文理论版 Neural-LinUCB 与工程实现版 Neural-LinUCB 之间有差异
2. 当前实现更偏工程可用，而不完全等同于论文理论证明设置

### 8.2 对 encoder 的理解也经过了很多轮澄清

用户后来几乎把 encoder 相关的每一个细节都追问了一遍，例如：

- manual feature 在哪里拼
- scale 在哪里拼
- TSP 与 CVRP 的输入维度差异如何处理
- encoder 到底是 transformer 还是 GNN
- 第一层之后是否就可以看作已经“编码成统一表示”
- 为什么要加 arm embedding
- 一个实例对应多个候选 arm 时，`phi(x,a_1), phi(x,a_2), ...` 是否一样
- `arm_embedding` 是一个 arm 的还是所有 arm 的

这意味着：

- 未来写 method 或 implementation section 时，必须把维度变化写清楚
- 不然另一个人很容易误解当前实现

---

## 9. 用户在整个过程中反复强调过的研究偏好

这一节非常重要，因为它不只是“工程偏好”，而是会影响论文写作和实验叙事。

### 9.1 用户更喜欢“讲清楚”，而不是“只给结论”

体现为：

- 注释要求很详细
- 文档要求说明网络结构、维度变化、训练过程
- 不满足于抽象描述，希望有代码对照、公式对照、例子对照

### 9.2 用户非常重视论文与实现是否一致

多次明确提出：

- 先再次仔细审查计划
- 重点检查有没有知识性错误
- 对照原论文和原实现
- 不要想当然

所以以后写论文时：

- 不能把实现说得比论文更“原教旨”
- 也不能忽略我们做过的工程改写

### 9.3 用户更关心实际 mean cost / 真实效果，而不是好看的 surrogate

这体现在很多对话中：

- 对 summary.json 里的字段不满足
- 希望看到每个 arm 的真实记录
- 对 loss 下降但 val mean cost 不动非常敏感
- 希望把 selector 和 single_best / oracle / 每个单方法逐个比较

### 9.4 用户高度重视复盘与解释

希望日志里有：

- 每步选择明细
- score
- chosen arm 的真实 cost/reward/rank
- 评测进度
- 周期性 val
- 每段训练的 loss 变化

这意味着：

- 论文里的图表和分析应该尽量延续这种“可解释、可复盘”的风格

### 9.5 用户最终是要写论文给别人看

因此另一个 AI 后面写论文时，最好优先满足：

- 叙事清晰
- 变量定义清楚
- 不混淆不同研究阶段
- 明确区分哪些是历史尝试，哪些是当前主线

---

## 10. 当前主线之外，还有哪些重要但容易被忽略的上下文

### 10.1 `2cmab2` 不是唯一“最终可写”的方向

虽然现在最重要的是 `2cmab2`，但：

- `2l2r` 是一个真实存在的平行分支
- 它可能成为重要 baseline，甚至可能在某些指标上更强

如果写论文，一定要想清楚：

- 是否只写 bandit
- 还是要把 ranking 版也纳入实验

### 10.2 `1online` 等 RL 主线不一定会写进当前论文，但它们提供了重要背景

这些工作可以在论文中扮演：

- motivation 历程
- why not RL / why init-only first
- future work

的角色。

### 10.3 早期 benchmark 设想并没有完全废掉

`实验设计.md` 和 `统计.md` 说明：

- “多问题大统一 benchmark” 不是错方向
- 只是对第一阶段论文来说太大了

因此这部分很适合作为：

- future work
- extensibility
- platform-level implication

的来源。

---

## 11. 如果要写论文，当前最合理的叙事主线是什么

这是给另一个 AI 的建议，不是唯一答案。

### 11.1 一个可行的论文主线

可以把论文叙事组织成：

1. 组合优化初始化方法存在实例依赖性
2. TSP / CVRP 上已有大量 initialization methods，但 single-best 不总是最优
3. 现有 NSS 这类方法主要是 supervised selector
4. 我们改用 contextual bandit / Neural-LinUCB 视角做 one-shot initialization selection
5. 使用 NSS 风格实例表示，但训练范式是 bandit，而非 supervised
6. 在 synthetic splits 与 TSPLIB/CVRPLIB 上评估
7. 同时分析 reward/top-k/mean cost/arm distribution/baseline comparison

### 11.2 哪些内容适合作为 motivation

可以引用的背景包括：

- `统计.md` 中 TSP/CVRP 可选方法很多
- `methods源代码/初始化迭代组合分析.md` 中 pipeline 可分解
- 早期 RL 原型说明“更复杂的 pipeline-level selection 是可能的，但 initialization-only 是更稳妥的第一步”

### 11.3 哪些内容不要混在主方法里

不建议在当前主论文里把下面几条主线都混成“我们的方法”：

- `1two_gate`
- `1step`
- `1online`
- `2cmab2`
- `2l2r`

更合理的处理方式是：

- 以 `2cmab2` 为主方法
- 其它路线作为研究背景 / 失败探索 / 后续方向

---

## 12. 如果另一个 AI 继续接手，应该优先读哪些文件

建议按以下顺序：

### 第 1 组：全局认知

- `统计.md`
- `实验设计.md`
- `methods源代码/初始化迭代组合分析.md`

### 第 2 组：当前主线总览

- `2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`

### 第 3 组：当前主线代码与分析

- `2cmab2/run_offline.py`
- `2cmab2/neural_linucb.py`
- `2cmab2/neural_ucb_diag.py`
- `2cmab2/encoders.py`
- `2cmab2/analysis.py`
- `2cmab2/analysis.ipynb`

### 第 4 组：平行方法与对照

- `2l2r/IMPLEMENTATION.md`
- `2l2r/run_l2r.py`
- `2cmab/COMPARISON.md`

### 第 5 组：研究演化背景

- `1two_gate/实现说明.md`
- `1two_gate/原论文和双gate对比.md`
- `1step/operator_selection_iteration_report.md`
- `1online/README.md`
- `1online/IMPLEMENTATION.md`

### 第 6 组：NSS 原始参考

- `9nss论文/论文原文.md`
- `9nss论文/neural-solver-selection/`
- `9nss论文/NSS论文实现讲解_模型设计与训练流程.md`

---

## 13. 当前环境与运行口径

这是操作层面的交接。

### 13.1 当前环境

- 当前已经是在服务器环境
- 当前工作目录就是服务器上的研究工作区
- 可以直接跑代码
- 推荐环境：`conda` 的 `easynco_zhoucl`
- 用户偏好：尽量用 `tmux` 跑长任务，避免断线

### 13.2 当前一个重要变化

用户后来明确更新过环境变化：

- 现在代码已经直接放到服务器上
- 以后默认可以直接在服务器环境跑

所以不要再按“本地机 / 只能写计划不能执行”的旧语境理解当前状态。

---

## 14. 本项目里反复出现过的几个关键概念口径

这一节适合另一个 AI 在写作时直接套用，避免口径飘移。

### 14.1 “online” 的口径

有两种口径，必须区分：

1. 严格 online
   - 如 `1online`
   - 每轮真的是基于当前策略交互、采样、更新
2. 离线数据模拟 online contextual bandit
   - 如 `2cmab2`
   - 数据是静态的，但训练流程语义是 online bandit

### 14.2 “selector” 的口径

这里的 selector 不是：

- 训练 solver 本身
- 训练一个全流程求解器

而是：

- 给定实例
- 选择现有 solver / init method / arm

### 14.3 “arm” 的口径

在当前 bandit 系列里，arm 通常就是：

- 一个候选 initialization method

### 14.4 “reward” 的口径

reward 不一定等于最终优化目标本身。

在当前 `2cmab2` 中，经常是：

- 根据 cost / rank 做映射
- 越好的方法得到越大的 bandit reward

### 14.5 “val 好不好”的口径

不能只看：

- loss
- reward
- top1

还要看：

- mean cost
- 与 single_best 的比较
- 与每个单一方法的比较

---

## 15. 当前最值得保留的经验结论

下面这些是跨多轮实现、调试、分析之后得到的经验，不一定全都要写进论文正文，但对后续研究非常重要。

### 15.1 initialization-only 是一个真实且清晰的问题

它不是“为了简化才凑出来的子问题”，而是在平台支持矩阵、方法可组合性和研究可写性约束下自然浮现出的第一阶段核心问题。

### 15.2 reward 设计会深刻影响 bandit 学到什么

特别是当 reward 来自 rank 映射时，loss 很好并不代表 mean cost 很好。

### 15.3 top1 不是唯一目标

对 selector 来说，mean cost 往往更重要，也更贴近实际使用价值。

### 15.4 日志一定要足够细

没有细日志，很难解释：

- 为什么某次 val 没提升
- 为什么 loss 降了但 cost 不降
- selector 是否在某个问题上偏向少数 arm

### 15.5 问题类型差异不能被忽略

TSP 与 CVRP 虽然可以统一进一个 selector 框架，但它们的输入形式、方法生态和 arm 表现分布确实不同。

### 15.6 研究历史本身就是论文论证的一部分

也就是说：

- `1two_gate`
- `1step`
- `1online`
- `2cmab`
- `2cmab2`
- `2l2r`

不是“杂乱无章的历史遗留”，而是逐步收缩问题、发现更合适建模方式的过程证据。

---

## 16. 当前最直接的写论文建议

如果另一个 AI 现在就要开始写论文，我建议：

1. 先把主论文只围绕 `2cmab2` 组织
2. 把 `2l2r` 视为强 baseline 或平行实验，而不是主方法
3. 用 `统计.md` 和 `methods源代码/初始化迭代组合分析.md` 支撑 motivation
4. 在 method section 里明确说明：
   - 这是 initialization-only selector
   - 训练范式是 contextual bandit
   - 数据上是离线已知结果，流程上模拟 online bandit
5. 在实验 section 里一定不要只报 loss / reward / top1
   - 要突出 mean cost
   - 要比较 single_best / oracle / 各单方法
6. 如果要写 future work，可以自然引向：
   - step-level operator selection
   - fully online deployment
   - 更大问题族

---

## 17. 最后给另一个 AI 的一句提醒

这个项目最容易犯的错误，不是代码没看懂，而是：

> 把当前主线 `2cmab2` 当成一个凭空出现的最终方案，忽略它前面的研究动机、失败尝试、问题收缩、以及用户对“真实效果、可解释性、复盘能力”的长期强调。

只要抓住这条主线：

- 先有 “init / iter 是否可分”的 motivation
- 再有 RL 原型
- 再收缩到 initialization-only
- 再转向 contextual bandit
- 再与 supervised ranking 平行比较

那么整个工作区就会变得非常清楚。

---

## 18. 在对话里被反复澄清过的高频问题

这一节不是代码目录说明，而是把很多“如果只看代码和文档，很容易漏掉”的对话上下文写下来。

### 18.1 关于 plan 审查这件事

在真正开写代码之前，用户对 plan 的审查非常严格，典型要求包括：

- 比较不同智能体给出的方案
- 分析对方方案和当前方案各自的不足
- 参考论文原文与论文代码逐条核对
- 重点检查有没有知识性错误
- 不满足于“概念上差不多”，而是要看是否真的符合论文做法

因此这个项目从一开始就不是“先写再说”，而是经历过多轮 method-level 审查。

### 18.2 关于两类问题 TSP / CVRP 的统一方式

用户多次追问：

- 计划里是如何区分两类问题的
- one-hot 是否真的足够
- 当前说法是否合理
- NSS 原始 encoder 是如何处理两类问题的

最终形成的稳定认知是：

- 不能把 TSP 和 CVRP 完全当成同一种输入
- 它们至少在节点输入维度上不同
- 统一任务框架可以做，但编码与 head 设计要考虑 problem type

### 18.3 关于 NSS checkpoint

用户后来明确说过：

> 不要管 NSS 的 checkpoint。它做的是监督学习，和我现在要做的东西不一样。

这句话非常重要，因为它决定了：

- 可以复用 NSS 的数据、encoder、特征风格
- 但不应该把当前 bandit / ranking selector 说成“基于 NSS checkpoint 微调”

### 18.4 关于 `pt` 数据与 json 的关系

用户曾经一度说“不需要接触 pt 数据，只用跑出来的 json”，后来又纠正：

> 还是需要接触 pt，因为实例特征是从 `tsp100` 数据集中提取的，而 json 是每个实例对应的结果。

因此以后描述数据构造时，不能偷懒写成“只有结果 json”，而要说明：

- 实例内容来自数据集文件
- 方法结果来自离线结果文件

### 18.5 关于“当前实现到底算 online 还是 offline”

这是整个项目中最常被问到的问题之一。

用户反复问过：

- 我的实现是 online 的还是 offline 的
- 这样能迁移到 online 上吗
- 最大阻碍是不是 reward 需要知道排名

稳定答案是：

- `2cmab2` 的训练流程语义是 online contextual bandit
- 但 reward 来自预先离线知道的结果，因此是“离线结果驱动的 online 交互模拟”
- 若要迁移到真正 online，最大的改动之一就是 reward 获取方式不能依赖完整离线 rank

### 18.6 关于 Neural-LinUCB 里 encoder 是否训练

用户多次问：

- encoder 在这个过程中会训练吗
- 现在的实现会不会更新 encoder
- 为什么不每次都更新而是每隔若干 step 才更新

这里沉淀出的关键口径是：

- bandit 主循环每步都更新线性统计量
- 深度表示层通常不是每步更新，而是周期性更新
- 这是“深度表征 + 浅层探索”设计的一部分

### 18.7 关于“原论文是不是每次从头训隐藏层”

这是一个非常深的理论问题，用户也追问得很细。

核心争议点是：

- 原论文理论证明常假设每个 epoch 的隐藏层训练都从固定初始化点开始
- 当前实现为了工程可用性，通常不会严格这么做

这意味着：

- 当前实现更偏工程版 Neural-LinUCB
- 理论保证与工程做法之间存在差距
- 论文里如果要声称“完全遵循原理论设定”就会有风险

### 18.8 关于 train_every / representation_steps / batch_size

用户对这些参数做过很多轮追问：

- 它们分别是什么意思
- 为什么 bandit 里没有传统 RL 意义下的 batch size
- 是否可以加 batch size
- 加了之后和普通 RL 里的 batch size 有什么异同

最后形成的稳妥解释是：

- `train_every`：bandit 交互多少步后触发一次表示层训练
- `representation_steps`：一次表示层训练内部做多少轮优化
- `representation_batch_size`：表示层训练时的 mini-batch 大小
- bandit 主循环本身仍是一条样本一步一步更新，不是拿一整批 transition 一起做 Q-learning / PPO 那种更新

### 18.9 关于 checkpoint 保存频率

用户明确要求过：

- 不要只在 epoch 结束后存 checkpoint
- 要支持每隔若干 step 自动保存

这和普通 offline epoch-based 训练不同，更接近 online / streaming 训练语义。

### 18.10 关于 eval 频率

用户也明确要求过：

- 不要只在 epoch 结束后 val
- 应该每隔若干 step eval 一次

这是因为当前场景本质上被用户看作 online 过程，而 epoch=1 只是遍历一次数据，不该成为唯一的验证边界。

### 18.11 关于日志必须非常详细

用户希望在 train 和 eval 里都看到：

- 每次选臂的 arm id / arm name
- 每个 arm 的 score
- 当前选中 arm 对应的真实 cost/reward/rank
- 周期性训练 loss
- eval 进度条与分段统计

这不是附加要求，而是整个项目后期的核心工程要求之一。

### 18.12 关于 analysis.ipynb

用户很强调 notebook 分析，经历过多次改写：

- 去掉 overall、macro_average
- 让 top1 / top2 / top3 放到一个表里
- 重新生成 analysis.ipynb
- 增加各单方法 mean cost 对比

这说明：

- 论文图表风格更偏“可直接拿来汇报 / 写作”的结构化分析
- 不是仅仅把 json dump 出来就结束

---

## 19. 在 `2cmab2` 线上已经发生过的重要工程改动

下面这些不一定会写进论文正文，但它们是当前实现为什么长成今天这样的重要原因。

### 19.1 从只输出 summary，到输出详细 per-step / per-instance trace

最开始的输出不够细，后面逐步加上了：

- 每次选臂详细日志
- `score`
- 当前 arm 的真实结果记录
- periodic val
- analysis notebook

这使得 `2cmab2` 后来变得更像一个“研究平台”，而不只是一个脚本。

### 19.2 从 epoch 级 checkpoint / eval，到 step 级 checkpoint / eval

因为用户把这条线理解成 online 语义，所以：

- checkpoint 保存从 epoch 级改成 step 级更合理
- eval 也从 epoch 末才做，改为 step 周期性做

### 19.3 表示层训练加入 mini-batch

用户很关心：

- 为什么训练慢
- 为什么 GPU 显存低但 Util 高
- 能不能像普通训练那样 batch 化

因此后来表示层训练明确做了 mini-batch 化。

### 19.4 表示层 buffer 与线性头 buffer 分离

这是一个比较后期、但很关键的改动：

- 表示层训练只看最近窗口，避免越来越慢
- 线性头重建保留更长历史，避免 bandit 统计过度遗忘

### 19.5 NeuralUCB 的实现也被重新检查并修改

用户后期明确要求：

- 检查 NeuralUCB 的实现
- 参考论文和源码
- 修改实现
- 让日志风格和 Neural-LinUCB 一致

因此当前工作区并不是只有 Neural-LinUCB 被认真打磨过，NeuralUCB 也被单独审查和修正过。

### 19.6 analysis 工具不断增强

用户要求过的分析项包括：

- loss 曲线
- train 集 arm 分布
- val 集 arm 分布
- test 集各方法 mean cost
- top1 / top2 / top3
- 与 single_best / oracle / 每个单一方法的比较
- benchmark 数据集上的同类图表

这些需求推动生成了：

- `analysis.py`
- `analysis.ipynb`
- `run_output_analysis.ipynb`
- 相关绘图和 json 汇总工具

---

## 20. 已经出现过的重要现象、异常和经验解释

这些内容未必最终写进论文，但对理解实验结果非常关键。

### 20.1 NeuralUCB 训练过程中出现过“卡在某一步”的情况

用户曾报告：

- 程序在某个 step 例如 `9710/14000` 长时间停住
- 终端看起来不动
- GPU 上可能仍有活动，或后来确认进程已经没有了

这个问题后来用户选择暂时不再追究，但它说明：

- 这条工程线在实际服务器运行中确实出现过不稳定问题
- 以后如果继续大规模实验，需要更稳的运行监控与日志

### 20.2 loss 明显下降，但 val mean cost 基本不变

这是 `2cmab2` 后期非常重要的观察。

用户明确指出：

- `neural_linucb.log` 中 loss 不断下降，甚至接近 0
- `periodic_val.jsonl` 里 val mean cost 没明显改善

这是后面结果解释中必须正面处理的问题之一。

可能的结构性原因包括：

- reward 是 rank-based 或 surrogate
- 表示层训练目标不是直接最小化 mean cost
- top1 与 mean cost 不一致
- 数据中多个方法 cost 很接近

### 20.3 GPU 利用率高、显存却不高

用户关心过：

- 为什么显存占用低，但 GPU-Util 很高
- 是否有优化空间

这通常意味着：

- 不是在吃大 batch 的显存密集计算
- 更像是很多小 batch / 小 kernel / 频繁前向反向

这与 bandit + 周期性表示层训练的实现方式高度一致。

### 20.4 top1 长时间不提升，可能和数据集本身有关

用户后来甚至提出一个判断：

> Neural-LinUCB 方案中 top1 一直没什么变化，原因可能是数据集的问题。

这也推动了从 EasyNCO 结果数据转向 NSS synthetic split 数据的决定。

### 20.5 benchmark 图表里 selector 与某单方法完全一样，可能是分析 bug

用户后来还发现过一种异常：

- 某张图里 selector 和某个单一方法（比如 `bq`）数值完全一样

这提醒以后继续分析时必须警惕：

- 绘图代码或聚合代码可能有 bug
- 不能因为图画出来了就默认它是对的

---

## 21. 写论文时最容易犯的具体错误

下面这些都是“未来另一个 AI 很可能会犯”的错误。

### 21.1 把所有分支写成一个统一方法

错误写法：

- 好像 `1two_gate`、`1step`、`1online`、`2cmab2` 都是同一篇论文的不同实验

更合理的写法：

- 当前论文主线聚焦 `2cmab2`
- 其它是研究演化背景 / 前期探索 / future direction

### 21.2 把 `2cmab2` 写成纯 offline supervised learning

错误原因：

- 数据是离线的，但训练流程不是普通 supervised classification

### 21.3 把 `2cmab2` 写成严格理论版 Neural-LinUCB

错误原因：

- 当前实现里有不少面向 CO 场景与工程可用性的改写

### 21.4 只写 top1，不写 mean cost

错误原因：

- 用户最在意的是真实表现
- top1 只能反映一部分 selector 质量

### 21.5 忽略 TSP / CVRP 的差异

错误原因：

- 两类问题共享框架，不代表完全同质
- 编码、方法可用性、性能分布都不同

### 21.6 忽略日志与分析需求

这个项目的一个鲜明特点就是：

- 实现和分析是一起被设计出来的

如果论文只讲模型，不讲：

- 详细日志
- 结果复盘
- per-arm 分析

会丢掉这个项目很有特色的一部分。

---

## 22. 相关文件索引

### 顶层

- `统计.md`
- `实验设计.md`
- `methods源代码/初始化迭代组合分析.md`

### 研究线

- `9motivation/experiment.md`
- `1two_gate/实现说明.md`
- `1two_gate/实现验证分析.md`
- `1two_gate/原论文和双gate对比.md`
- `1step/operator_selection_iteration_report.md`
- `1online/README.md`
- `1online/IMPLEMENTATION.md`
- `2cmab/README.md`
- `2cmab/COMPARISON.md`
- `2cmab/IMPLEMENTATION.md`
- `2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`
- `2l2r/IMPLEMENTATION.md`
- `2l2r/运行说明.md`

### NSS 相关

- `9nss论文/论文原文.md`
- `9nss论文/NSS论文实现讲解_模型设计与训练流程.md`
- `9nss论文/neural-solver-selection/`

---

## 23. 这份文档和 `2cmab2` 子文档的关系

这份文档的作用是：

- 交接整个研究历程
- 让另一个 AI 理解为什么当前是这样

而下面这份文档的作用是：

- 深入交接当前主线 `2cmab2` 的实现与实验细节

对应文件：

- `2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`

最合理的使用方式是：

1. 先读本文件
2. 再读 `2cmab2` 子文档
3. 再看具体代码和输出
