# 全项目交接文档（给下一位 AI 模型）

本文档的目标，是把当前工作区里与毕业论文、实验分析、代码实现和 slides 相关的上下文尽可能完整地交代清楚，让一个**完全没有上下文**的新模型也能快速接手工作。

这不是论文正文，也不是某一条代码线的 README，而是一份“总项目 handoff 文档”。

适用场景：

- 继续写论文正文
- 继续改 slides
- 继续补实验 / 跑评测 / 做分析图
- 解释 `2cmab2`、`2l2r` 或旧分支代码
- 回答“为什么最后选这个方案、之前那些方案为什么没继续”的问题

建议把它当作当前项目的**第一入口文件**来读。

---

## 0. 一句话总述

当前论文主线是：

> 在 `TSP + CVRP` 上做 **initialization-only** 的实例级方法选择，把问题建模为 **单步 contextual bandit**，主方法是 **Neural-LinUCB**；研究动机来自“不同神经求解器在不同实例上的表现具有互补性”，而不是继续追求一个统一占优的单一求解器。

但要理解为什么会走到这一步，必须知道这个工作区之前已经真实尝试过：

- motivation 验证
- 初始化/迭代可拆分性分析
- 双 gate RL
- 每步 operator 选择
- 单实例 fully-online RL
- 早期 contextual bandit 原型
- supervised learning-to-rank 对照分支

这些都不是噪声，而是当前论文问题定义收缩和转向的原因。

---

## 1. 当前工作区与真实活跃路径

### 1.1 工作区根目录

当前工作区根目录是：

- `/public/home/zhoucl/shiys`

它**不是一个单一项目**，而是一个研究工作区。

### 1.2 当前真正活跃的项目目录

本次论文与实验最相关的实际代码和文档，主要在：

- `/public/home/zhoucl/shiys/old_try/2cmab2`
- `/public/home/zhoucl/shiys/old_try/2l2r`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9`
- `/public/home/zhoucl/shiys/old_try/1two_gate`
- `/public/home/zhoucl/shiys/old_try/1step`
- `/public/home/zhoucl/shiys/old_try/1online`
- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection`

注意：

- 根目录下也有 `EasyNCO/`、`9nss论文/`、`平台方法统计统计.md` 等资料
- 但论文和当前 selector 主线代码，实际主要是在 `old_try/` 下面
- 工作区根目录**不是 git repo**

### 1.3 建议优先阅读顺序

如果是一个新模型刚接手，建议按下面顺序读：

1. 本文档
2. `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/introduction.tex`
3. `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/method.tex`
4. `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/results.tex`
5. `/public/home/zhoucl/shiys/old_try/2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`
6. `/public/home/zhoucl/shiys/old_try/analysis-output/2026-04-06-full-project-results/analysis-report.md`
7. `/public/home/zhoucl/shiys/old_try/2cmab2/neural_linucb.py`
8. `/public/home/zhoucl/shiys/old_try/2cmab2/build_dataset.py`
9. `/public/home/zhoucl/shiys/old_try/2cmab2/encoders.py`

---

## 2. 当前论文在讲什么

### 2.1 当前论文主问题

当前论文聚焦的问题是：

> 对于一个 `TSP` 或 `CVRP` 实例，只在初始化阶段做一次方法选择，应该选哪个 initialization method，才能让这个实例最终表现更好？

关键词：

- initialization-only
- one-shot selection
- instance-level method selection
- contextual bandit
- Neural-LinUCB

### 2.2 当前论文明确**不**在讲什么

当前主论文**不是**：

- 每一步迭代都选 operator 的 RL 论文
- 完整 MDP / policy gradient 主线论文
- 训练 solver 本体的论文
- 纯 NSS 风格的 supervised selector 论文

更准确地说：

- solver 是固定候选方法池
- 训练对象是 selector
- 主方法是 bandit，不是 solver training

### 2.3 当前论文核心动机

核心动机来自一个非常朴素但重要的观察：

> 现有神经组合优化方法在不同实例上的表现具有互补性，没有一个方法在所有实例、所有分布上都占优。

因此研究问题从：

- “能不能设计一个统一最强的方法”

转成：

- “给定实例后，应该选哪个方法”

---

## 3. 研究问题是怎么一步步演化到这里的

这一部分很重要，因为后面很多“为什么这样设计”的答案，都来自前面尝试过的失败路线。

### 3.1 motivation 验证阶段

最早的问题不是直接做 bandit，而是先验证：

1. 方法之间是否真的存在实例级互补性
2. initialization 和 iteration 是否可以拆开理解
3. 若方法有互补性，选择问题是否值得单独做

对应的关键材料：

- `/public/home/zhoucl/shiys/平台方法统计统计.md`
- `/public/home/zhoucl/shiys/old_try/FULL_PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/appendix.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/slides.tex` 中的动机页

这个阶段形成的关键结论：

- TSP 和 CVRP 上都不存在稳定支配其他方法的单一 solver
- 方法偏好不能只靠单一规模特征解释
- initialization method selection 是一个值得单独研究的问题

### 3.2 init / iter 可拆分性分析

早期很重要的一条思路是：

> 一个求解 pipeline 能否拆成 `initialization` 和 `iteration` 两部分，再分别做选择？

这对应了后面双 gate 和 step-level 方案的前史。

### 3.3 第一代 RL 原型：`1two_gate`

目录：

- `/public/home/zhoucl/shiys/old_try/1two_gate`

核心设定：

- Gate1：选 initializer
- Gate2：选 iterator
- 一个 episode 里各决策一次

它回答的问题是：

> 适合的初始化方法和适合的迭代方法，能不能拆开两次选？

结果：

- 训练很容易坍缩到单一组合
- slides 和论文里已经把它作为负结果写进去
- 当前主要价值是“失败案例”和“研究路径演化证据”

### 3.4 第二代 RL 原型：`1step`

目录：

- `/public/home/zhoucl/shiys/old_try/1step`
- `/public/home/zhoucl/shiys/old_try/1step/operator_selection_iteration_report.md`

核心设定：

- 初始化选一次
- 迭代阶段每一步都选一个 operator

它比 `1two_gate` 更细粒度，但代价是：

- 状态空间和 credit assignment 更复杂
- collapse 与调参问题更明显
- 不适合作为当前毕业论文的第一主线

### 3.5 第三代 RL 主线：`1online`

目录：

- `/public/home/zhoucl/shiys/old_try/1online`
- `/public/home/zhoucl/shiys/old_try/1online/README.md`

核心设定：

- 单个 TSPLIB 实例
- 完全 online、on-policy
- init once + per-step operator selection
- REINFORCE
- 训练时在线 augmentation，评估时 greedy

这条线很纯，但目前留下来的结果大多只是 smoke / prototype 级别。

主要结论：

- 代码通路打通了
- 但没有形成稳定、可论文主讲的实验结果

### 3.6 第一代 contextual bandit：`2cmab`

目录：

- `/public/home/zhoucl/shiys/old_try/2cmab`
- `/public/home/zhoucl/shiys/old_try/2cmab/README.md`

核心变化：

- 明确把问题重新整理成 contextual bandit
- Neural-LinUCB 成为主方法
- 开始摆脱多步 RL 的复杂性

但这一版还比较早：

- 数据协议主要来自 EasyNCO 结果
- 代码和输出组织不如 `2cmab2` 稳定
- 后来基本被 `2cmab2` 取代

### 3.7 第二代 contextual bandit：`2cmab2`

目录：

- `/public/home/zhoucl/shiys/old_try/2cmab2`

这是当前论文的**主线实现**。

关键词：

- TSP + CVRP
- initialization-only
- shared action space + feasible mask
- HybridContextEncoder
- Neural-LinUCB
- problem-specific linear heads
- NSS synthetic train/val/test
- TSPLIB / CVRPLIB benchmark

### 3.8 并行监督学习分支：`2l2r`

目录：

- `/public/home/zhoucl/shiys/old_try/2l2r`

这个分支的作用不是替代 `2cmab2`，而是做一个并行对照：

- `2cmab2`：bandit 风格
- `2l2r`：supervised learning-to-rank

这个对照很重要，因为它帮助回答：

> 如果不走 bandit，而是直接用监督排序，效果会怎样？

---

## 4. 当前论文与 slides 的状态

### 4.1 论文目录

论文模板目录：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9`

关键文件：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/main.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/slides.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/abstract.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/introduction.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/method.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/experiments.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/results.tex`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/sections/thesis/conclusion.tex`

### 4.2 当前论文主叙述

当前论文正文的主叙述已经比较稳定：

- 动机：实例级方法互补性
- 任务：TSP/CVRP 上的 initialization-only selection
- 建模：单步 contextual bandit
- 方法：Neural-LinUCB
- 结果：合成混合数据集优于 SingleBest，LIB benchmark 上泛化有限
- 负结果：双 gate RL / step-level operator 选择容易坍缩

### 4.3 当前 slides 状态

slides 文件：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/slides.tex`

已做过大量微调，当前大致状态：

- 研究背景、方法、实验、结果、总结结构完整
- 结果页是按 `TSP` 与 `CVRP` 分页
- 方法页分成：
  - 实例编码
  - 动作条件表示
  - 动作评分
  - 线性头在线更新
  - 表示层周期更新
- 负结果与 collapse 已加入 slides
- 总结页已经改成上下结构，不再是左右双栏

### 4.4 LaTeX / VSCode 相关现状

已知设置：

- LaTeX Workshop 的 `autoBuild` 已被关掉，不再“保存即编译”
- VSCode 自动保存可以保留
- 当前更推荐手动编译

当前相关设置文件：

- `/public/home/zhoucl/shiys/.vscode/settings.json`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/.vscode/settings.json`

---

## 5. 当前主线：`2cmab2` 的任务定义

### 5.1 当前最准确的任务定义

`2cmab2` 当前做的是：

> 给定一个 `TSP` 或 `CVRP` 实例，从当前问题可用的方法池里选一个 initialization method，并根据该方法在该实例上的 reward 在线更新 selector。

这就是单步 contextual bandit。

### 5.2 不是完整 online deployment，但语义是 online bandit

这里有一个很重要的边界：

- 数据本身来自离线全信息结果
- 但训练协议模拟 online partial feedback

也就是：

1. 离线阶段，所有方法在每个实例上的真实 cost 已经可得
2. 训练时，selector 只“观察自己选中的 arm 的 reward”
3. 再按 bandit 方式更新统计量和表示层

因此更准确的表述是：

> 用离线全信息结果数据，模拟在线 contextual bandit 交互。

---

## 6. `2cmab2` 的代码结构

### 6.1 训练与评测主入口

- `/public/home/zhoucl/shiys/old_try/2cmab2/run_offline.py`
  - 当前最核心入口
  - 负责训练、周期性验证、保存 checkpoint、最后 test

- `/public/home/zhoucl/shiys/old_try/2cmab2/evaluate_nss_benchmarks.py`
  - 用训练好的模型在 TSPLIB / CVRPLIB benchmark 上测

### 6.2 核心实现文件

- `/public/home/zhoucl/shiys/old_try/2cmab2/build_dataset.py`
- `/public/home/zhoucl/shiys/old_try/2cmab2/features.py`
- `/public/home/zhoucl/shiys/old_try/2cmab2/encoders.py`
- `/public/home/zhoucl/shiys/old_try/2cmab2/neural_linucb.py`
- `/public/home/zhoucl/shiys/old_try/2cmab2/default_settings.py`

### 6.3 分析文件

- `/public/home/zhoucl/shiys/old_try/2cmab2/analysis.py`
- `/public/home/zhoucl/shiys/old_try/2cmab2/analysis.ipynb`
- `/public/home/zhoucl/shiys/old_try/2cmab2/analysis/STAGE2_NEURAL_LINUCB_FORMAL_ANALYSIS_2026-04-02.md`

---

## 7. `2cmab2` 的数据协议

### 7.1 它其实支持两套数据源

`2cmab2` 代码支持两种数据来源：

1. `easynco_results`
2. `nss`

### 7.2 当前论文主线使用哪一套

当前默认配置见：

- `/public/home/zhoucl/shiys/old_try/2cmab2/default_settings.py`

当前论文主线用的是：

- `DATA_SOURCE = "nss"`

也就是说当前主线实验不是直接从 EasyNCO 的结果 JSON 构造，而是直接用：

- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/datasets`

### 7.3 当前数据命名口径

论文当前使用的两种数据集名称：

- **合成混合数据集**
  - NSS synthetic train/val/test
  - TSP 10000 / 1000 / 1000
  - CVRP 10000 / 1000 / 1000
  - 规模范围 50--499
  - 均匀分布 + 高斯混合分布

- **LIB 基准数据集**
  - TSPLIB + CVRPLIB
  - 主要用于分布外泛化评测

注意：

- 论文主文里已经删掉了 `Uniform-100` 相关内容
- 主文里也不再反复使用 “NSS 数据集” 这个叫法

---

## 8. 动作空间与问题约束

### 8.1 当前论文主线动作空间：9 个 arm

当前主线统一动作空间是 9 个方法：

- 0: `bq`
- 1: `elg`
- 2: `lehd`
- 3: `t2t`
- 4: `t2t500`
- 5: `difusco`
- 6: `difusco500`
- 7: `omni`
- 8: `mvmoe`

### 8.2 TSP 与 CVRP 的可行动作不同

- TSP 可用：
  - `bq / elg / lehd / t2t / t2t500 / difusco / difusco500`
- CVRP 可用：
  - `bq / elg / lehd / omni / mvmoe`

### 8.3 为什么要统一动作空间

统一动作空间 + feasible mask 的好处：

- 不同问题共用一套全局 arm 编号
- 共享方法（如 `bq / elg / lehd`）可以共享经验
- 具体问题再通过 mask 过滤不可选动作

---

## 9. `2cmab2` 当前实现的样本结构

核心样本结构在：

- `/public/home/zhoucl/shiys/old_try/2cmab2/build_dataset.py`

每个 `InstanceSample` 一条样本对应一个实例，里面已经聚合了：

- `uid`
- `problem`
- `global_index`
- `nodes`
- `problem_onehot`
- `manual_raw`
- `manual_with_scale`
- `linucb_context`
- `feasible_mask`
- `costs`
- `rewards`
- `ranks`
- `best_arm`
- `result_records`

也就是说，一条样本本身就包含了：

- 实例上下文
- 可行动作集合
- 所有动作的离线真实表现

---

## 10. 当前主方法：Neural-LinUCB

### 10.1 总体结构

当前主方法可以写成：

- 实例编码：`c(x) ∈ R^270`
- 方法嵌入：`e(a) ∈ R^16`
- 动作条件表示：`phi(x,a) ∈ R^64`
- 线性头：problem-specific `(A^{-1}, b, theta)`

### 10.2 编码器结构

实现文件：

- `/public/home/zhoucl/shiys/old_try/2cmab2/encoders.py`

它的设计不是 Transformer 或 GNN 的通用新实现，而是：

- 前半部分直接复制 NSS 风格的层次化编码器
- 后半部分新增当前任务需要的包装层

具体是：

1. `Encoder_h`
   - NSS 层次化图编码器本体
2. `DualNSSEncoder`
   - TSP 和 CVRP 各保留一个 encoder 实例
3. `HybridContextEncoder`
   - 图编码 + 手工特征 + problem one-hot 融合成上下文

### 10.3 为什么 TSP 与 CVRP 需要两个首层 encoder

因为两者输入维度不同：

- TSP：节点特征是 `[x, y]`
- CVRP：节点特征是 `[x, y, demand]`

代码上非常明确：

- TSP 首层：`Linear(2, embedding_dim)`
- CVRP 首层：`Linear(3, embedding_dim)`

对应位置：

- `/public/home/zhoucl/shiys/old_try/2cmab2/encoders.py`

### 10.4 图编码维度

当前默认维度链路：

- TSP 输入：`[B, n, 2]`
- CVRP 输入：`[B, n+1, 3]`
- 图编码输出：`256`
- 再拼一个 `scale`：`257`
- 再拼 `manual_raw(11)` 和 `problem_onehot(2)`：`270`

即：

- `c(x) ∈ R^270`

### 10.5 手工特征 `manual_raw`

实现文件：

- `/public/home/zhoucl/shiys/old_try/2cmab2/features.py`

当前 `manual_raw` 维度是 11。

组成如下：

- 前 9 维：几何统计特征
  - 所有点对距离标准差
  - 质心 x
  - 质心 y
  - 平均半径
  - 距离离散值统计
  - 最近邻距离标准差
  - 聚类比例
  - 离群点比例
  - 簇半径
- 对 CVRP：
  - 再加 2 维 demand 统计
  - demand mean
  - demand std
- 对 TSP：
  - demand 那两位补 0

### 10.6 CVRP 的容量限制到底在哪里

这是一个很容易被问到的问题，必须讲清楚。

当前实现中：

- CVRP 节点张量只有 3 列：`x / y / demand`
- **capacity 没有单独作为第四维输入**

原因是：

- 在 NSS 数据生成阶段，`demand` 已经被 `capacity` 归一化了
- 也就是存进数据集的不是绝对需求，而是 `demand / capacity`

对应原始代码：

- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/datasets/data_utils.py`

关键行：

```python
'demand': (demand / capacities[:, None]).float()
```

因此当前模型看到的第三维：

- 不是原始 demand
- 而是“占容量比例”

所以：

- 容量约束是**隐式进入模型**的
- 它被吸收进 `demand` 这一维
- 这就是为什么维度上没有单独体现 `capacity`

### 10.7 动作条件表示与 arm embedding

当前设计：

- `c(x) ∈ R^270`
- `e(a) ∈ R^16`
- 拼接 `[c(x), e(a)] ∈ R^286`
- 再过两层 MLP 得到 `phi(x,a) ∈ R^64`

这里的 `e(a)` 是：

- 每个 arm 自己的 embedding
- 不是所有 arm 的一个共享向量

在一次评估一个实例的多个候选 arm 时：

- 上下文 `base` 会复制成 `M` 份
- arm embedding 会查出 `M` 个不同的向量
- 拼接后得到 `M` 个不同的 `(x,a)` 联合表示

### 10.8 线性头与 `theta`

当前实现中，Neural-LinUCB 的线性头是按问题拆开的：

- TSP 一套 `(A_inv, b, theta)`
- CVRP 一套 `(A_inv, b, theta)`

也就是说：

- 深度表示层共享
- 线性 UCB 头 problem-specific

`theta` 在代码里不是 `torch.Parameter`，而是在线维护的线性统计量结果：

- `theta = A_inv @ b`

它在当前实现里通常是：

- `numpy.ndarray`
- 形状 `[64]`

### 10.9 `theta_at_pull` 是什么

表示层训练时会缓存一个：

- `theta_at_pull`

它表示：

- 某条样本在“被选中那一刻”的线性头参数

这个量会和当前的 `phi(x_i,a_i)` 做内积，用来回归真实 reward。

### 10.10 当前 reward 设计

当前 reward 不是直接用 cost，而是实例内排名 reward：

\[
r(x, a)=1-\frac{\operatorname{rank}(x,a)-1}{|\mathcal{A}(x)|-1}
\]

解释：

- 最优方法 reward = 1
- 最差方法 reward = 0

优点：

- 跨问题数值更稳定

缺点：

- 丢失绝对 cost gap 信息
- 这也是后面出现“loss 降了但 mean length 不怎么动”的原因之一

### 10.11 当前表示层训练目标

当前表示层训练不直接用 policy gradient，而是一个回归式目标：

\[
\mathcal{L}(\psi)=\frac{1}{B}\sum_{i=1}^B
\left(
{\theta_i^{\mathrm{pull}}}^{\top}\phi(x_i,a_i)-r_i
\right)^2
\]

代码位置：

- `/public/home/zhoucl/shiys/old_try/2cmab2/neural_linucb.py`

关键含义：

- `phi(x_i,a_i)` 是当前表示层输出
- `theta_i^{pull}` 是历史缓存下来的线性头
- `r_i` 是真实 reward

也就是说：

- reward 是作为 target 进入表示层训练的
- 不是作为输入特征

### 10.12 为什么 loss 会降但 mean length 不一定降

这是当前实验分析中的一个核心现象：

- 训练 loss 很稳地下降
- 甚至接近 0
- 但 val/test 上的 mean length 不一定同步改善

当前最可信的解释是：

1. 表示层训练目标是拟合**排名 reward**
2. 排名 reward 只保留顺序，不保留绝对差距
3. top-1 / reward 变好不一定等价于 mean length 真正变好

这也是论文中已经写进去的一个局限。

---

## 11. `2cmab2` 的训练、日志与输出

### 11.1 默认运行配置

当前默认参数集中定义在：

- `/public/home/zhoucl/shiys/old_try/2cmab2/default_settings.py`

其中比较关键的默认值包括：

- `METHOD = "neural_linucb"`
- `DATA_SOURCE = "nss"`
- `EPOCHS = 1`
- `CHECKPOINT_EVERY_STEPS = 1000`
- `EVAL_EVERY_STEPS = 1000`
- `ALPHA = 1.0`
- `HIDDEN_DIM = 64`
- `ARM_EMBED_DIM = 16`
- `TRAIN_EVERY = 100`
- `REPRESENTATION_STEPS = 10`
- `PROBLEM_SPECIFIC_HEADS = True`

### 11.2 当前日志

当前一条典型运行会产出：

- `neural_linucb.log`
- `periodic_val.jsonl`
- `summary.json`
- `checkpoints/`
- `traces/`

其中：

- `neural_linucb.log`
  - 训练日志
  - 含 loss、step 级进度、评测摘要
- `periodic_val.jsonl`
  - 每次 step-based val 的结构化记录
- `traces/selector_*.jsonl`
  - 逐实例选择详情

### 11.3 一个重要改动

当前训练和评测都已经从“只按 epoch 保存/评测”改成了：

- 每隔若干 step 保存 checkpoint
- 每隔若干 step 做 val

这点在当前代码和实验习惯里已经是默认认知。

---

## 12. 并行分支：`2l2r`

### 12.1 它是什么

`2l2r` 是与 `2cmab2` 并行的监督排序分支。

核心差异：

- `2cmab2`：bandit，只用选中的 arm 更新
- `2l2r`：supervised ranking，同一实例上的所有可行 arm 一起进 loss

### 12.2 保留它的意义

它不是当前论文主线，但它很有价值，因为它能回答：

> 如果改用监督排序，而不是 bandit，效果会不会更好？

### 12.3 它的实现特点

- 复用和 `2cmab2` 相同的数据协议
- 复用和 `2cmab2` 相同的 NSS 风格编码器
- 只是训练目标从 bandit 换成 ranking loss

关键文件：

- `/public/home/zhoucl/shiys/old_try/2l2r/IMPLEMENTATION.md`
- `/public/home/zhoucl/shiys/old_try/2l2r/run_l2r.py`
- `/public/home/zhoucl/shiys/old_try/2l2r/default_settings.py`

### 12.4 当前推荐 loss（在该分支内）

从 `default_settings.py` 看，当前 `2l2r` 内部更推荐的不是传统 pairwise/listwise，而是：

- `two_stage_default_gate`

也就是先默认留在 `single_best_per_problem`，再学什么时候 switch，以及 switch 到谁。

但要注意：

- 这只是 `2l2r` 分支内部的当前推荐
- **不是**当前毕业论文主线

---

## 13. 当前最可信的实验结论

### 13.1 motivation 层面的结论

从附录和 slides 的动机表里，可以稳定支持以下结论：

- TSPLIB 上不存在统一最优方法
- CVRPLIB 上也不存在统一最优方法
- 方法偏好会随着分布变化而变化
- 因此实例级方法选择是合理的研究问题

### 13.2 当前论文主结果（按论文写法）

当前论文主文强调的是：

- 合成混合数据集上，Neural-LinUCB 优于 SingleBest
- TSP benchmark 上有一定泛化
- CVRP benchmark 上泛化不足

当前主文表中的数值：

#### TSP

- 合成混合数据集：
  - Oracle: `7.8699`
  - Neural-LinUCB: `7.9420`
  - SingleBest: `7.9578`
- TSPLIB：
  - Oracle: `8.1422`
  - Neural-LinUCB: `8.2116`
  - SingleBest: `8.2334`

#### CVRP

- 合成混合数据集：
  - Oracle: `18.1781`
  - Neural-LinUCB: `18.2785`
  - SingleBest: `18.4167`
- CVRPLIB：
  - Oracle: `68.3098`
  - Neural-LinUCB: `73.7296`
  - SingleBest: `68.7926`

### 13.3 更严格的全项目分析结论

更严格的跨项目分析在：

- `/public/home/zhoucl/shiys/old_try/analysis-output/2026-04-06-full-project-results/analysis-report.md`

里面有几个非常重要的结论：

1. 在 legacy 3000-instance 协议上，没有学习型 selector 真正打败 `SingleBestPerProblem`
2. 在当前 NSS 2000-instance 协议上，`Neural-LinUCB` 明显优于 `NeuralUCB-Diag`
3. `top1` 波动并不必然带来 `mean_cost` 改善
4. TSPLIB/CVRPLIB benchmark 上存在明显的 `top1 / mean_cost` 失配

这几条结论非常适合用于：

- 写 discussion
- 写 limitation
- 写答辩时的口头解释

### 13.4 关于 `top1` 和 `mean length`

这是整个项目里反复出现的一个关键主题：

- `top1` 高，不一定 mean length 低
- reward 优化得更好，不一定 mean length 更好
- 这不是 bug，而是 reward 定义带来的目标失配

这也是为什么论文里已经把 mean length 作为主指标，而不是只报 top-1。

### 13.5 当前最重要的输出目录与文件

如果要继续看现成结果，最关键的目录是：

- 主 NSS 训练输出：
  - `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server`
- 其中最重要的子目录：
  - `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/analysis_partial`
  - `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib`

尤其建议先看：

- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/analysis_partial/summary.md`
- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/analysis_partial/loss_avg_curve.png`
- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/summary.txt`
- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/tsplib_mean_cost_vs_single_methods.png`
- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server/benchmark_eval_tsplib_cvrplib/cvrplib_mean_cost_vs_single_methods.png`

历史但仍有参考价值的早期 `2cmab2` 训练输出：

- `/public/home/zhoucl/shiys/old_try/2cmab2/outputs/2026-03-25-18-53-19_neural_linucb`

当前论文和 slides 的 PDF 产物通常在：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/main.pdf`
- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/slides.pdf`

---

## 14. 已经很明确的负结果与失败经验

### 14.1 双 gate RL 容易 collapse

这一点在：

- 当前论文 results
- 当前 slides
- 旧日志与分析

里都已经很明确。

主要结论：

- gate 很容易收缩到单一方法
- 训练后期 mean length 基本平台化

### 14.2 更细粒度控制不一定更好

直觉上看，“每一步都选”应该更强，但当前项目中恰恰不是这样：

- 决策粒度更细
- 训练更难
- collapse 更严重
- 更难形成稳定、可写论文的结果

### 14.3 单实例 fully-online RL 目前更像 prototype

`1online` 的价值主要在于：

- 验证 fully-online 训练流程
- 暴露 reward / logging / collapse 问题

而不是作为当前论文主实验证据。

---

## 15. 当前论文写作上的术语与口径约束

下面这些是已经稳定下来的写法，下一模型最好不要再反复改口径。

### 15.1 当前主文常用词

推荐使用：

- `Neural-LinUCB`
- `SingleBest`
- `Oracle`
- `合成混合数据集`
- `LIB 基准数据集`
- `mean length`
- `top-1 accuracy`
- `初始化方法选择`
- `上下文老虎机`

### 15.2 当前主文尽量不要再用的表达

不推荐继续在主文中使用：

- `SingleBestPerProblem`
  - 当前主文里已经尽量简化成 `SingleBest`
- `NeuralUCB-Diag`
  - 主文里已经删掉它，不再作为主结果对象
- `Uniform-100`
  - 已从主文移除
- `NSS 数据集`
  - 只在首次介绍时提来源，后面不反复说
- 类似 `1two_gate`、`s2_04_full_linear_history` 这样的文件夹名式表达
  - 不应该直接写进论文正文

### 15.3 当前论文中已经明确写进去的局限

主文当前已接受的局限包括：

- reward 与最终 mean length 目标不完全一致
- 当前只做 initialization，不做完整两阶段控制
- 当前只覆盖 TSP 与 CVRP
- CVRP benchmark 泛化仍弱

---

## 16. 当前 slides 写作上的风格约束

slides 文件：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9/slides.tex`

当前已经形成的风格偏好：

- 更像答辩 PPT，不像论文段落
- 一页只讲一个核心点
- 框不能滥用
- 但完全无层次也不行
- 表格尽量保留完整，不随便删方法行
- 同种问题的“表 + 分析”尽量在同一页

已经做过的版式调整包括：

- 结果页表格与分析在同页
- 方法页拆成多页
- 结论页改为上下结构
- 实验设置页改成两个主框
- 若干方法页字体放大

---

## 17. 环境、运行与编译

### 17.1 Python / 训练环境

当前服务器上建议用：

- `conda` 环境：`easynco_zhoucl`

运行训练或评测时，优先使用：

- `tmux`

### 17.2 `2cmab2` 典型运行方式

典型主入口：

- `python /public/home/zhoucl/shiys/old_try/2cmab2/run_offline.py ...`

benchmark 评测入口：

- `python /public/home/zhoucl/shiys/old_try/2cmab2/evaluate_nss_benchmarks.py ...`

### 17.3 论文与 slides 编译

常见目录：

- `/public/home/zhoucl/shiys/old_try/sustechthesis-1.3.9`

主要文件：

- `main.tex`
- `slides.tex`

如果需要手动编译，优先用：

```bash
xelatex -interaction=nonstopmode -halt-on-error slides.tex
```

论文正文因为有参考文献，通常还需要：

- `biber`
- 再次 `xelatex`

### 17.4 PDF 打不开时的已知经验

曾经出现过：

- PDF 在生成过程中被打开
- 查看器缓存了半成品文件

因此如果出现“打不开 PDF”，先确认：

- 编译是否已完成
- 重新打开 PDF
- 必要时重启 VSCode 预览

---

## 18. 需要特别记住的实现细节（FAQ 式摘要）

### 18.1 `e(a)` 是什么

- `e(a)` 是 arm embedding
- 每个 arm 一个可学习向量
- 当前默认维度 16

### 18.2 为什么 `e(a) ∈ R^16`

- 本质是当前实现的超参数选择
- 不是论文强制规定
- 作用是给方法身份一个低维可学习表示

### 18.3 为什么要加 arm embedding

因为仅有实例上下文 `x` 不够，selector 还需要知道“当前在评分的是哪个方法”。

### 18.4 `phi_net` 是什么

- 它是把 `[c(x), e(a)]` 压到 `phi(x,a)` 的 MLP
- 当前输出 64 维

### 18.5 `theta` 是什么

- 最后一层线性 UCB 头的参数
- 当前实现里是在线维护的线性统计量结果，不是普通神经网络参数

### 18.6 `theta_at_pull` 是什么

- 某条历史样本在被选中时缓存的 `theta`
- 用于后续表示层训练

### 18.7 当前 reward 如何进入表示层训练

- reward 作为 target
- `pred = theta_at_pull^T phi(x,a)`
- 用 MSE 回归这个 target

### 18.8 为什么 CVRP 输入只有 3 维

- `[x, y, demand]`
- 容量约束已经被归一化进 demand
- 所以没有单独的 capacity 维

---

## 19. 当前最值得继续推进的问题

如果下一模型要继续真正“做事”，最值得推进的方向是：

1. 继续完善论文正文与 slides
2. 继续把 `2cmab2` 的主结果和 discussion 写得更稳
3. 明确 reward 失配和 benchmark 泛化不足的解释
4. 如果还要做方法改动，优先考虑：
   - reward redesign
   - 更强的跨问题表示
   - 更合理的 OOD 泛化策略
5. `2l2r` 可作为对照，但不建议再把论文主线切回 supervised

---

## 20. 给下一模型的直接建议

如果你是下一位接手这个项目的模型，建议你按这个顺序开始：

1. 先读本文件
2. 再读当前论文 `introduction / method / results / conclusion`
3. 再看 `2cmab2` 的 `build_dataset.py / encoders.py / neural_linucb.py`
4. 再看 `analysis-report.md`
5. 如果需要解释“为什么不继续做 RL”，再去看 `1two_gate / 1step / 1online`

最重要的是，不要误判当前主线。

当前主线不是：

- `1online`
- `1step`
- `1two_gate`
- `2l2r`

而是：

- **`2cmab2` + 当前 thesis/slides**

---

## 21. 本文档与旧文档的关系

旧文档：

- `/public/home/zhoucl/shiys/old_try/2cmab2/PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`
- `/public/home/zhoucl/shiys/old_try/FULL_PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`

它们仍然有价值，但这份文档更新了：

- 论文当前表述
- slides 当前状态
- `2l2r` 对照分支信息
- `2cmab2` 的实现细节解释
- 当前最可信的结果口径
- 最近反复确认过的术语与限制条件

如果新模型只读一份，优先读本文件。
