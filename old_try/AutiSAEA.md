# AutoSAEA 论文与代码详解

## 1. 我先给出一句话总结

`0AutoSAEA/` 里的工作，本质上是在做一件事：  
不要提前假设“哪一种 surrogate model + 哪一种 infill criterion 一定最好”，而是把它们当成一个分层的在线选择问题，在优化过程中边做边学，动态决定这一轮应该用哪种代理模型、配哪种样本填充策略。

论文把这个问题建模为一个 two-level multi-armed bandit（TL-MAB）问题；代码则给出了一个 MATLAB 版本的实现，其中核心主循环在 `0AutoSAEA/AutoSAEA/AutoSAEA.m`。

---

## 2. `0AutoSAEA/` 文件夹里有什么

### 2.1 核心内容

- `0AutoSAEA/Surrogate-Assisted_Evolutionary_Algorithm_With_Model_and_Infill_Criterion_Auto-Configuration (1).pdf`
  - 论文正文。
- `0AutoSAEA/AutoSAEA/`
  - 论文对应的 MATLAB 实现。

### 2.2 代码目录的功能划分

- `AutoSAEA.m`
  - 算法主循环，真正实现“模型+准则”的自动配置。
- `RUN_AutoSAEA.m`
  - 单个测试函数上的多次重复运行封装。
- `run_myexperiment.m`
  - 批量实验入口，跑 CEC2005 和 CEC2015。
- `GP_lcb_arm.m`、`GP_ei_arm.m`
  - GP 模型对应的两个 low-level arm。
- `RBF_pre_arm.m`、`RBF_ls_arm.m`
  - RBF 模型对应的两个 low-level arm。
- `PRS_pre_arm.m`、`PRS_ls_arm.m`
  - PRS 模型对应的两个 low-level arm。
- `KNN_eoi_arm.m`、`KNN_eor_arm.m`
  - KNN 模型对应的两个 low-level arm。
- `Low_level_r.m`
  - 低层 reward 的实现。
- `TL_UCB.m`
  - UCB 选择公式的实现。
- `DEoperator.m`
  - 用于生成候选 offspring 的 DE 变异+交叉。
- `DE_optimizer.m`
  - 在 surrogate 上做局部搜索时用的 DE 优化器。
- `my_rbfbuild.m`、`my_rbfpredict.m`
  - RBF 模型训练与预测。
- `GP_fit.m`、`predictor.m`
  - GP 建模与预测。
- `prs/`
  - PRS 相关工具函数。
- `funcname/`、`CEC05/`、`input_data/`、`cec15problems.*`
  - benchmark 函数、CEC 数据和 CEC2015 的 MEX 接口。

---

## 3. 论文到底在解决什么问题

### 3.1 问题背景

论文针对的是 expensive optimization problems（EOPs），也就是：

- 目标函数没有显式表达式；
- 每次真实评估都很贵；
- 如果直接用普通进化算法做大量真实评估，成本会非常高。

因此就会用 surrogate-assisted evolutionary algorithms（SAEAs）：

- 用便宜的 surrogate model 近似真实目标函数；
- 再用某种 infill criterion 决定“下一次应该真实评估哪个点”。

### 3.2 传统 SAEA 的核心痛点

论文强调：SAEA 的效果非常依赖两样东西。

- surrogate model 选什么
- infill criterion 选什么

而且这两者不是独立的，它们是耦合的。

例如：

- GP 可以输出均值和不确定性，因此天然适合 LCB、EI 这种 acquisition function；
- RBF、PRS 更像是回归拟合器，常搭配 prescreening 或 local search；
- KNN 更像分类器，适合做 level-based 的 exploitation / exploration。

不同问题上最适合的组合并不一样，所以论文的主张不是“找一个万能组合”，而是：

**在优化过程中在线地、自适应地选择 surrogate model 和 infill criterion。**

### 3.3 如果你不熟悉老虎机、多臂老虎机、双层老虎机，先看这里

这一小节我尽量不用论文口吻，而是用直觉来讲。

#### 3.3.1 “老虎机”这个词到底是什么意思

这里的“老虎机”，不是说 AutoSAEA 真的在玩赌场机器，而是借用了一个经典比喻。

你可以先想象一台最简单的老式拉杆老虎机：

- 你每拉一次，就要付出一次成本；
- 它会随机给你一个回报；
- 你不知道这台机器真实的“平均回报”是多少；
- 只能靠不断试玩来估计。

英文里这种机器常被叫做 `one-armed bandit`：

- `arm` 指的是机器侧面的拉杆；
- `bandit` 是一种调侃，说它像“会抢你钱的机器”。

所以后面说到 `bandit`，不要把它理解成“强盗算法”，而要理解成：

**一个“每次选一个动作，然后获得随机回报”的试错问题。**

#### 3.3.2 什么叫多臂老虎机（Multi-Armed Bandit, MAB）

如果现在不是只有 1 台老虎机，而是有很多台，每台平均回报都不同，就变成了多臂老虎机问题。

例如你面前有 4 台机器：

- A 机器可能长期平均回报最高；
- B 机器偶尔很高，但不稳定；
- C 机器看起来一般；
- D 机器你还没怎么试过。

你的目标是：

- 在有限尝试次数内，尽量拿到更多总回报；
- 同时逐渐学会“哪台机器更值得玩”。

这里每一台机器就叫一个 `arm`。

所以：

- `arm` = 一个可选项
- `pull an arm` = 选一次这个可选项
- `reward` = 这次选择得到的反馈

放到算法里，`arm` 不一定真的是机器，它可以是：

- 一个模型
- 一个策略
- 一个算子
- 一个候选方法

#### 3.3.3 多臂老虎机最核心的矛盾：exploration 和 exploitation

MAB 最核心的矛盾只有一个：

- `exploitation`：继续用目前看起来最好的那个选择
- `exploration`：去试试那些还不太确定、但也许更好的选择

比如：

- 你已经试过 A 机器很多次，感觉它挺赚钱；
- 但 D 机器你只试过 1 次，也许它其实更强，只是你还不知道。

如果你永远只选当前最好的：

- 容易过早锁死在一个“看起来不错但未必最优”的选择上。

如果你永远乱试：

- 又会浪费很多机会在差的选择上。

所以 bandit 问题的本质就是：

**如何在“利用已知好选择”和“探索未知可能性”之间做平衡。**

#### 3.3.4 UCB 是怎么帮你做这个平衡的

UCB 可以理解成一种“乐观估计分数”。

它通常长这样：

- 当前这个 arm 过去的平均表现
- 再加上一个“探索奖励”

直觉上就是：

- 如果一个 arm 历史表现很好，那它分数高；
- 如果一个 arm 还没被试过几次，那它也会因为“不确定”而获得额外加分。

所以 UCB 的风格不是：

- 只看历史平均值，完全贪心；

也不是：

- 完全随机乱试。

而是：

**对“看起来不错”且“还可能被低估”的 arm 保持乐观。**

你可以把它理解成：

- 老选手靠成绩拿分；
- 新选手靠潜力拿分。

#### 3.3.5 什么叫双层老虎机

普通 MAB 是：

- 一次直接从很多 arm 里选 1 个。

双层老虎机则是：

- 先做第一层选择；
- 再在第一层结果限定的范围内做第二层选择。

举个生活化例子：

你去点外卖，不是直接在“所有菜”里选，而是分两步：

1. 先选菜系
   - 川菜
   - 粤菜
   - 日料
   - 汉堡
2. 再在该菜系里选具体菜
   - 如果你选了川菜，再选水煮鱼还是宫保鸡丁
   - 如果你选了日料，再选寿司还是拉面

这就是一个典型的两层选择结构：

- 第一层是大类
- 第二层是大类下面的具体选项

#### 3.3.6 AutoSAEA 里的“双层”到底对应什么

在 AutoSAEA 里，这个两层结构非常明确。

第一层选的是 surrogate model：

- GP
- RBF
- PRS
- KNN

第二层选的是该模型能搭配的 infill criterion：

- GP 下面选 `LCB` 或 `EI`
- RBF 下面选 `prescreening` 或 `local search`
- PRS 下面选 `prescreening` 或 `local search`
- KNN 下面选 `L1-exploitation` 或 `L1-exploration`

所以它不是“8 个完全平级的方法随便挑 1 个”，而是：

1. 先决定“这轮更适合用哪种 surrogate 模型”
2. 再决定“在这个模型下该怎么挑新样本”

这就是论文里说的 `two-level multi-armed bandit`。

#### 3.3.7 为什么 AutoSAEA 不直接把 8 个组合全当平级 arm

当然也可以直接把下面 8 个组合平铺：

- GP-LCB
- GP-EI
- RBF-prescreening
- RBF-local-search
- PRS-prescreening
- PRS-local-search
- KNN-L1-exploitation
- KNN-L1-exploration

但论文认为这样会丢掉结构信息。

因为 AutoSAEA 想学的不只是：

- “8 个组合里哪个最好”

还想学：

- “哪一类 surrogate 模型整体更适合当前问题”
- “在这个 surrogate 下面，哪种 infill criterion 更合适”

也就是说，它想同时学会：

- 模型层面的偏好
- 模型内部策略层面的偏好

#### 3.3.8 reward 在这里是什么

在 bandit 里，选完 arm 之后必须有反馈，不然你没法学。

这个反馈就叫 `reward`。

AutoSAEA 里，reward 不是“我选了 GP，所以给 GP 一分”这么简单，而是：

- 先真正生成一个新点；
- 对这个新点做一次真实评估；
- 看这个新点相对于当前 population 表现如何；
- 再把这个结果转成 reward。

所以在这里：

- `arm` 是“这轮用哪种模型/哪种准则”
- `reward` 是“这轮这么选，最终带来了多大帮助”

#### 3.3.9 high-level arm 和 low-level arm 怎么理解

如果你看到论文里的这些词，可以这样翻译：

- `high-level arm`
  - 第一层选择项
  - 在这篇论文里就是 surrogate model
- `low-level arm`
  - 第二层选择项
  - 在这篇论文里就是和该模型匹配的 infill criterion

因此：

- 高层解决“用什么模型”
- 低层解决“模型确定后怎么生成要真实评估的新点”

#### 3.3.10 把这几个词串起来，AutoSAEA 就容易懂了

如果你已经接受下面这件事：

- AutoSAEA 每一轮都要在几种“方法组合”之间做选择；
- 每一种选择做完后，都能观察到结果好不好；
- 后面轮次应该参考前面轮次的历史表现；

那么 bandit 语言其实只是给这件事起了一个标准名字。

换成最朴素的话，AutoSAEA 做的就是：

- 这一轮先选一个 surrogate 大类；
- 再选这个 surrogate 下面的一种样本选择策略；
- 跑完以后看效果；
- 好用的以后多用一点；
- 不确定但可能有潜力的也偶尔继续试；
- 最终逐渐学会“当前问题更适合哪种模型+准则组合”。

如果你记住这段朴素版本，后面再看 `TL-UCB`、`TL-R`、`high-level arm`、`low-level arm`，基本就不会再觉得这些术语很突然了。

---

## 4. 论文方法的核心思想

### 4.1 高层和低层分别选什么

论文把选择过程拆成两层：

- 高层 arm：选 surrogate model
- 低层 arm：在该模型对应的合法 infill criterion 里再选一个

论文里固定了 4 个高层模型：

- GP
- RBF
- PRS
- KNN

对应的 8 个合法组合 arm 是：

- `(GP, LCB)`
- `(GP, EI)`
- `(RBF, prescreening)`
- `(RBF, local search)`
- `(PRS, prescreening)`
- `(PRS, local search)`
- `(KNN, L1-exploitation)`
- `(KNN, L1-exploration)`

所以它不是“任意两两组合”，而是“先选模型，再从与该模型兼容的策略里选准则”。

### 4.2 为什么要 two-level，而不是直接把 8 个组合当成 8 个平级臂

论文的逻辑是：

- 如果直接把 8 个组合全平铺成一个普通 MAB，可以做，但没有利用结构信息；
- two-level 结构能体现“模型是父层、准则是子层”的关系；
- 这样做可以减少无效搜索，并且更适合处理“同一个模型下有多个准则”的耦合问题。

也就是说，论文不是只想学“哪个组合最好”，还想学：

- 哪类模型整体上更适合当前问题；
- 该模型下哪种 infill criterion 更合适。

---

## 5. 论文里用到的 surrogate model 和 infill criterion

### 5.1 四个 surrogate model

#### 5.1.1 GP

Gaussian Process 的特点是：

- 能预测均值；
- 还能给出预测不确定性；
- 适合构造带 exploration/exploitation 平衡的 acquisition function。

所以它搭配：

- LCB
- EI

#### 5.1.2 RBF

RBF 用径向基函数来拟合目标面，特点是：

- 建模成本通常比 GP 更低；
- 对很多连续优化问题很常见；
- 适合用预测值直接排序，或者在 surrogate 上做局部优化。

所以它搭配：

- prescreening
- local search

#### 5.1.3 PRS

PRS 是 polynomial response surface，本质上就是多项式响应面：

- 拟合速度快；
- 在低维、较平滑问题上可能很好用；
- 但表达复杂非线性时能力有限。

它也搭配：

- prescreening
- local search

#### 5.1.4 KNN

KNN 在这篇论文里不是拿来做精确回归，而是做 level classification：

- 先按适应度把当前种群分成若干 level；
- 再预测新点属于哪个 level；
- 重点找“看起来像好 level”的点。

所以它搭配：

- L1-exploitation
- L1-exploration

### 5.2 各种 infill criterion 在做什么

#### 5.2.1 LCB

形式是：

- 预测均值减去若干倍标准差。

直觉上：

- 均值小表示“可能更优”；
- 方差大表示“值得探索”；
- 两者结合是在 exploitation 和 exploration 之间折中。

#### 5.2.2 EI

Expected Improvement 的核心是：

- 相对当前最好值，期望能改进多少。

它比单纯看均值更“主动”地考虑：

- 可能改进的幅度；
- 不确定性带来的机会。

#### 5.2.3 Prescreening

先用 DE 产生一批 offspring，然后：

- 用 surrogate 对这批候选打分；
- 挑 surrogate 预测最好的那个去做真实评估。

这是一种“先大致筛一遍”的策略。

#### 5.2.4 Local Search

不是只在一批离散 offspring 里选，而是：

- 直接在 surrogate 上做一个局部优化；
- 找 surrogate 预测的局部最优点；
- 再对这个点做真实评估。

#### 5.2.5 L1-exploitation

先找那些被 KNN 预测为最好 level 的候选点，然后：

- 选择离这些优秀样本最“贴近”的点；
- 更偏 exploitation。

#### 5.2.6 L1-exploration

同样先限制在“被预测为最好 level”的候选里，然后：

- 选离这些优秀样本更“分散”的点；
- 更偏 exploration。

---

## 6. 论文中的 TL-UCB 和 TL-R

### 6.1 TL-UCB

论文用 two-level UCB 来选 arm。

高层先选 surrogate model：

- 当前价值高的模型更容易被选中；
- 但很久没试过的模型也会因为 UCB exploration bonus 被重新尝试。

选定模型后，再在其对应的低层 arm 中做同样的 UCB 选择。

所以 TL-UCB 的精神非常清楚：

- 不是贪心地永远用当前最好；
- 而是在“已知表现”和“继续探索”之间平衡。

### 6.2 TL-R

论文最关键的一个设计是 reward 不是直接拿真实目标值，而是做了层次化设计。

#### 6.2.1 低层 reward

低层 reward 取决于新点 `x_t` 在当前 population 里的排名。

如果：

- 新点是最好的，则 reward 接近 1；
- 新点很差，则 reward 接近 0。

这样设计的好处是：

- reward 有界；
- 不稀疏；
- 不同阶段之间更容易比较；
- 不会直接受目标函数绝对尺度影响太大。

#### 6.2.2 高层 reward

高层 reward 不是直接由真实函数值给出，而是：

- 看本轮低层 arm 的价值提升了多少；
- 再把这部分提升回传给对应 surrogate model。

论文这样设计的目的是：

- 让高层模型的价值来自它“带出的低层策略表现”；
- 让模型价值不是孤立定义，而是体现模型与准则的耦合效果。

### 6.3 这套设计背后的真正思想

我觉得这篇论文最值得记住的不是公式本身，而是下面这个观点：

**AutoSAEA 并不是在学习一个 surrogate，而是在学习“对当前问题应该怎样使用 surrogate”。**

也就是：

- 哪种模型更适合这个问题；
- 在这个模型下，是该更激进 exploit，还是更谨慎 explore。

---

## 7. 论文算法流程，用自然语言重述一遍

### 7.1 初始化阶段

1. 用 LHS 在搜索空间里采样 `N` 个初始点。
2. 真实评估这些点。
3. 把它们放进数据库 `D`。
4. 初始化所有 arm 的统计量。
5. 先保证每个合法组合 arm 至少被试一次。

### 7.2 迭代阶段

每一轮做下面这些事：

1. 从数据库 `D` 中取最好的 `N` 个样本当当前 population。
2. 如果 8 个合法组合还没都试过，就按 warm-up 方式依次试。
3. 否则用 TL-UCB：
   - 先选 surrogate model；
   - 再选该模型下的 infill criterion。
4. 用选中的“模型+准则”协同生成一个新点。
5. 对这个新点做一次真实评估。
6. 把该点加入数据库。
7. 根据排名更新 low-level reward。
8. 再把 low-level 的变化传给 high-level。
9. 直到达到 FE 上限。

### 7.3 最终输出

输出数据库里最好那个点。

---

## 8. 论文实验结论怎么理解

### 8.1 benchmark

论文主要做了三类实验：

- CEC2005，维度 10 和 30
- CEC2015，维度 10 和 30
- 一个真实油藏生产优化问题

### 8.2 论文声称证明了什么

论文想证明 3 件事：

- AutoSAEA 比若干已有 SAEA 更强；
- 它的自适应选择行为确实会随问题变化而变化；
- 它不仅对 benchmark 有效，对真实工程问题也有效。

### 8.3 我对实验结论的理解

论文的实验重点并不只是“平均值赢了多少”，而是：

- 不同问题确实偏好不同 arm；
- 因此固定单模型、固定单准则的 SAEA 很难一直占优；
- 自动配置机制带来的泛化性，才是 AutoSAEA 的主卖点。

换句话说，这篇论文真正想卖的是“适应能力”，不是某个 surrogate 自身比别人强。

---

## 9. 代码入口与整体执行逻辑

### 9.1 批量实验入口：`run_myexperiment.m`

`0AutoSAEA/AutoSAEA/run_myexperiment.m` 是整个工程的批量实验入口。

它做的事情是：

- 定义 CEC2005 的 15 个函数名；
- 定义 CEC2015 的 15 个函数名；
- 设定维度 `10` 和 `30`；
- 每个问题跑 `20` 次；
- CEC2005 用 `variable_domain` 取上下界；
- CEC2015 用 `cec15problems('eval', x, idx)` 调 MEX 评估。

也就是说，它是一个“论文复现实验总控脚本”。

### 9.2 单问题多次运行：`RUN_AutoSAEA.m`

`RUN_AutoSAEA.m` 的职责是：

- 对某个给定函数做多次重复试验；
- 每次调用一次 `AutoSAEA(...)`；
- 记录每轮运行的 best-so-far 曲线；
- 最后保存结果到 `result/` 目录。

它更像是“实验包装器”，而不是算法本体。

### 9.3 算法本体：`AutoSAEA.m`

`AutoSAEA.m` 才是真正对应论文算法的实现主体。

这个文件可以分成 4 大块：

- 参数初始化
- 初始样本生成与真实评估
- 两层 arm 选择
- 候选点真实评估与 reward 更新

---

## 10. `AutoSAEA.m` 的逐段解释

### 10.1 初始化参数

主函数一开始固定了几个关键参数：

- `F = 0.5`
- `CR = 0.9`
- `apha = 2.5`
- `level = 5`
- `initial_sample_size = 100`
- `MaxFEs = 1000`

这和论文实验里的主要设定是一致的：

- 种群规模 100
- 总真实评估预算 1000
- KNN 分 5 个 level
- UCB 的控制参数 `alpha = 2.5`

### 10.2 初始种群的生成

代码用 LHS 生成 100 个样本：

- 每个样本都调用真实目标函数 `FUN` 评估；
- 评估结果记在 `fitness`；
- 全部历史样本放到 `hx`；
- 全部历史目标值放到 `hf`；
- `CE` 和 `gfs` 用来记录每次 FE 后的收敛轨迹。

也就是说，`hx/hf` 不是“当前种群”，而是**整个历史数据库**。

### 10.3 当前 population 的定义

在每轮迭代开始时，代码会：

- 对 `hf` 排序；
- 取历史数据库里最好的前 100 个样本作为 `ghx/ghf`。

这个实现和论文是一致的：

- 当前 population 不是单纯继承上一代；
- 而是始终由数据库中的 top-`N` 样本构成。

这点很重要，因为它说明 AutoSAEA 更像“database-driven search”，而不是传统意义上只维护一个代际种群。

### 10.4 warm-up：先把 8 个组合 arm 都试一遍

代码里 `num_arm = 8`。

在前 8 轮：

- 每轮固定执行一个组合 arm；
- 目的是让后续 UCB 不会遇到“某个 arm 从未被选过”的情况。

这是论文里“每个合法 arm 至少试一次”的实现版本。

### 10.5 正式进入 TL-UCB 选择

从第 9 轮开始，代码分两步做选择。

第一步，选高层 surrogate model：

- RBF
- GP
- PRS
- KNN

第二步，在被选中的模型下面选低层 arm：

- RBF 下在 `prescreening / local search` 中选
- GP 下在 `LCB / EI` 中选
- PRS 下在 `prescreening / local search` 中选
- KNN 下在 `L1-exploitation / L1-exploration` 中选

### 10.6 真实评估与数据库更新

无论选中了哪个 arm，流程最后都会走到同一件事：

- 生成一个 candidate
- 如果这个点不在历史数据库中，就做真实评估
- 把它追加进 `hx/hf`
- 更新 `CE` 和 `gfs`
- 计算低层 reward

这说明整套算法始终是“每轮新增 1 次真实评估”。

---

## 11. 8 个 arm 在代码里分别怎么实现

### 11.1 `RBF_pre_arm.m`

这部分对应论文里的 `(RBF, prescreening)`。

实现过程是：

1. 用当前 top-100 样本 `ghx/ghf` 拟合 RBF。
2. 对 `DEoperator` 生成的一批 offspring 做 surrogate 预测。
3. 选预测值最小的那个 offspring。
4. 如果这个点没出现过，就做真实评估。
5. 根据它在当前 population 中的排名计算 reward。

它的本质就是：

**RBF 负责打分，DE 负责给候选，prescreening 负责从候选中挑一个。**

### 11.2 `RBF_ls_arm.m`

对应 `(RBF, local search)`。

与 `RBF_pre_arm` 的区别在于：

- 它不是在一批给定 offspring 里挑最好的；
- 而是调用 `DE_optimizer.m`，直接在 RBF surrogate 上做局部优化。

局部搜索的搜索边界是：

- 当前 top-100 样本在每一维的最小值和最大值。

这与论文公式里“在当前局部子空间做 local search”是对应的。

### 11.3 `PRS_pre_arm.m`

对应 `(PRS, prescreening)`。

逻辑和 `RBF_pre_arm` 几乎完全平行：

1. 用当前样本拟合二次 PRS。
2. 对 DE 产生的 offspring 做预测。
3. 选预测值最小的那个。
4. 真实评估。
5. 计算 reward。

### 11.4 `PRS_ls_arm.m`

对应 `(PRS, local search)`。

逻辑和 `RBF_ls_arm` 平行，只是 surrogate 从 RBF 换成 PRS。

### 11.5 `GP_lcb_arm.m`

对应 `(GP, LCB)`。

实现过程是：

1. 先把 `ghx + ghf` 合并后去重，避免 GP 因重复点不稳定。
2. 拟合 GP。
3. 对每个 offspring 计算：
   - 预测均值
   - 预测 MSE
   - `LCB = mean - w * std`
4. 选 LCB 最小的点。
5. 真实评估并更新 reward。

这里代码中的 `w = 2`，和论文一致。

### 11.6 `GP_ei_arm.m`

对应 `(GP, EI)`。

做法是：

1. 拟合 GP。
2. 对每个 offspring 计算 EI。
3. 由于代码里把 EI 写成了一个带负号的量，所以最后是取最小值。
4. 再做真实评估。

这是一种常见写法：

- 如果目标是最小化问题，可以把“最大 EI”改写成“最小负 EI”。

### 11.7 `KNN_eoi_arm.m`

对应 `(KNN, L1-exploitation)`。

流程是：

1. 先按当前 `ghf` 的排序把样本分成 `level = 5` 个等级。
2. 训练 KNN 分类器，预测 offspring 属于哪个 level。
3. 只保留那些被预测为最好 level 的点。
4. 计算它们到优秀父代集合 `Parents_L1` 的距离。
5. 选择“离优秀样本整体最近”的点。

代码里具体实现成：

- 对每个候选点，算它到 `Parents_L1` 所有点的距离；
- 取这些距离的最大值；
- 再在候选之间选这个最大值最小的点。

这就是一种 min-max 式的“贴近好区域”的 exploitation。

### 11.8 `KNN_eor_arm.m`

对应 `(KNN, L1-exploration)`。

与 exploitation 的区别是：

- 仍然只在“被预测为最好 level”的候选里选；
- 但改成尽量离已有优秀样本远一些。

代码里具体是：

- 对每个候选点，先看它到 `Parents_L1` 最近的那个距离；
- 再选这个最近距离最大的候选。

也就是 max-min 形式，更偏向在 promising 区域内做发散探索。

---

## 12. 代码里的 reward 和 UCB 是怎么落地的

### 12.1 `Low_level_r.m` 几乎直接实现了论文公式

`Low_level_r.m` 做的事情非常直白：

1. 把当前 population 的适应度 `ghf` 和新点 `candidate_fit` 拼在一起；
2. 看新点在这个集合里的排序位置；
3. 用
   `reward = -(1/N) * rank + (N+1)/N`
   计算 reward。

这和论文给出的 low-level reward 公式是对应的。

所以这一部分论文和代码的一致性很高。

### 12.2 `TL_UCB.m` 也很直接

`TL_UCB.m` 只有一行核心公式：

- `q_value + sqrt(alpha * log(id) / count)`

这里：

- `q_value` 是当前 arm 的经验价值；
- `id` 是当前迭代轮数；
- `count = length(sum_reward)` 是该 arm 被选过多少次。

从思想上看，它就是标准 UCB。

---

## 13. 这份代码与论文最重要的一致处

### 13.1 一致处 1：问题建模一致

论文的核心是：

- 高层选 surrogate；
- 低层选 infill criterion；
- 组合后产生一个真实评估点。

代码完全保留了这个结构。

### 13.2 一致处 2：8 个合法组合一致

论文里使用的 8 个组合 arm，在代码中都有一一对应的函数实现。

### 13.3 一致处 3：候选点生成机制一致

论文中的核心候选生成机制包括：

- GP + acquisition
- RBF/PRS + prescreening
- RBF/PRS + local surrogate search
- KNN + level-based 筛选

这些在代码里都落地了。

### 13.4 一致处 4：低层 reward 基本一致

论文低层 reward 是基于排名的线性 reward，代码也是这么实现的。

---

## 14. 这份代码与论文最值得注意的差异

这一节很重要，因为如果只看论文，很容易以为代码逐行实现了论文伪代码；但实际上不是完全一模一样。

### 14.1 差异 1：论文里的 TL-R 高层回传，在代码里被简化了

论文中的高层 reward 逻辑是：

- 先更新 low-level arm 的价值；
- 再根据 low-level 价值的变化量，构造 high-level reward；
- 最终保证 high-level value 等于其关联 low-level value 的均值。

但是代码里没有显式维护：

- `Q_H`
- `Q_L`
- `T_H`
- `T_L`
- 高层 reward 的显式传播公式

代码实际做的是：

- 每个 low-level arm 维护一个历史 reward 列表，比如 `Save_rp`、`Save_gl`；
- 每个 high-level model 维护一个“该模型下所有低层 reward 的拼接列表”，比如 `rbf_model`、`gp_model`；
- low-level 的 value 用 `mean(Save_*)` 表示；
- high-level 的 value 用 `mean(model_reward_list)` 表示。

这意味着：

- 代码保留了“高层看模型、低层看具体策略”的思想；
- 但并没有严格按论文的公式 (26) 到 (30) 去做 reward propagation。

### 14.2 差异 2：代码里的 high-level value 是“按选择次数加权”的

论文理论上想要的是：

- high-level value = 该模型关联 low-level arm 当前 value 的简单平均。

而代码中：

- `rbf_model` 是所有 RBF 相关 reward 的历史拼接；
- `mean(rbf_model)` 实际上是一个**按每个 low-level arm 被选择次数加权后的平均 reward**。

这两者并不完全相同。

所以更准确地说：

- 代码实现的是一个“经验均值版的双层 UCB”；
- 不是论文里那个严格推导后的 TL-R 版本。

### 14.3 差异 3：warm-up 顺序和论文伪代码的展示顺序不同

论文里列出的合法组合顺序通常是：

- GP 两个
- RBF 两个
- PRS 两个
- KNN 两个

而代码前 8 步的顺序是：

- RBF-prescreening
- GP-LCB
- RBF-local-search
- GP-EI
- PRS-prescreening
- PRS-local-search
- KNN-L1-exploitation
- KNN-L1-exploration

这不是原则性错误，但说明实现层面对 warm-up 顺序做了自己的安排。

### 14.4 差异 4：代码把一些实验参数写死了

论文把 `N`、`MaxFEs`、`alpha` 等参数作为算法输入来描述。

但代码里直接固定成：

- `N = 100`
- `MaxFEs = 1000`
- `alpha = 2.5`
- `level = 5`

所以这份代码更像“论文实验版实现”，而不是一个高度参数化的通用框架。

### 14.5 差异 5：代码里有若干工程化/兼容性小问题

我在阅读代码时看到几个值得注意的点：

- `DEoperator.m` 文件名和主函数名 `DEoperating` 不一致。
- `KNN_eor_arm.m` 文件中的函数名写成了 `Knn_eor_arm`。
- `AutoSAEA.m` 的返回值里有 `s_l_s`，但函数体里并没有真正赋值。
- `AutoSAEA.m` 里有 `in1 = randperm(num_arm)`，但实际没有使用。
- GP 分支用了 `try/catch`，一旦建模失败会直接把 reward 置 0，这很实用，但也会掩盖潜在问题。

这些不一定影响论文主思想，但说明代码更偏实验原型，而不是精修过的工程版本。

---

## 15. 代码依赖与运行注意点

### 15.1 这份代码不是完全自包含的

从当前 `0AutoSAEA/AutoSAEA/` 文件夹来看，GP 分支依赖的一些函数并不在该目录中直接出现，比如：

- `regpoly0`
- `corrgauss`
- `Direct`
- `findtheta`

这说明代码大概率默认：

- 用户本地 MATLAB 环境里已经有相关工具箱/函数；
- 或作者运行时额外挂了其他路径。

所以如果只拿这个文件夹直接跑，GP 部分不一定能无缝运行。

### 15.2 MATLAB 工具箱依赖

从代码调用方式看，至少还依赖：

- `lhsdesign`
- `ClassificationKNN.fit`

这通常意味着需要 MATLAB 的统计/机器学习相关工具箱支持。

### 15.3 CEC2015 是 Windows MEX 版本

目录里给了：

- `cec15problems.mexw64`

这说明当前附带的是 **Windows 64 位 MATLAB MEX**。

如果换到 Linux 或 macOS：

- 这个 MEX 不能直接用；
- 需要重新编译 `cec15problems.c`、`cec15_test_func.c` 等文件。

### 15.4 `result/` 目录当前并没有随代码一起出现

`RUN_AutoSAEA.m` 最后会保存到：

- `result/NFE...`

但当前 `0AutoSAEA/AutoSAEA/` 目录里没有看到现成的 `result/` 文件夹。  
因此如果直接运行，通常还需要先手动创建这个目录。

---

## 16. 如果把论文和代码放在一起看，应该怎样理解这项工作

### 16.1 它不是在发明新的 surrogate

这篇工作并没有提出全新的 GP、RBF、PRS、KNN 模型。

它真正做的是：

- 把若干成熟 surrogate model 组织成一个模型池；
- 把若干成熟 infill criterion 组织成一个策略池；
- 再用一个在线学习机制去自动配置它们。

所以它属于：

**algorithm selection / online portfolio / auto-configuration**

这一路的思路，而不是纯建模论文。

### 16.2 它的关键价值是“适配不同问题”

为什么这篇论文有意义？

因为在 expensive optimization 里，经常会遇到下面这个现实：

- 一个方法在某类函数上极强；
- 换一类函数就不一定行；
- 人工事先决定“永远用 GP-EI”或者“永远用 RBF-local-search”，往往太死。

AutoSAEA 的回答是：

- 不提前承诺；
- 优化过程中自己试、自己学、自己调。

### 16.3 代码实现透露出的作者思路

从代码能明显看出，作者真正想保留的核心是：

- 双层选择结构
- 低层 reward 的排名化设计
- 借助 DE 统一产生候选或优化 surrogate

而不是死守论文里每个统计量的形式。

因此，代码呈现出一种很典型的研究型实现风格：

- 核心思想保留；
- 数学上最“漂亮”的部分做了简化；
- 目标是把实验跑通并验证思想有效。

---

## 17. 对后续阅读这个文件的人，我建议重点抓住什么

如果你只想快速抓主线，我建议记住下面 5 点。

### 17.1 AutoSAEA 的对象不是“单个方法”，而是“方法组合”

它挑的不是一个 surrogate，也不是一个 infill，而是两者的组合使用方式。

### 17.2 高层是模型选择，低层是准则选择

这就是论文标题里 “model and infill criterion auto-configuration” 的真正含义。

### 17.3 low-level reward 比 high-level reward 更关键

因为代码里真正稳定落地的部分，是低层基于排名的 reward；而高层更多是基于这些 reward 的统计汇总。

### 17.4 DE 是整套系统的统一候选生成器

很多 arm 最后都离不开 DE：

- 要么 DE 产生 offspring 供 surrogate 筛选；
- 要么 DE 直接在 surrogate 上做 local search。

所以 surrogate 决定“如何评估候选”，DE 决定“候选从哪里来”。

### 17.5 这份代码更像“论文实验实现”，不是开箱即用平台

它非常适合理解思路和复现实验逻辑，但如果要继续做研究或工程扩展，通常还需要：

- 补齐依赖；
- 整理接口；
- 明确高层 reward 的正式实现；
- 清理一些原型代码残留。

---

## 18. 最后的总评

### 18.1 从论文角度看

AutoSAEA 的贡献不在于发明了新的 surrogate，而在于：

- 把 surrogate model 选择和 infill criterion 选择统一成一个分层在线决策问题；
- 用 TL-MAB、TL-UCB、TL-R 给出了一个结构化的解决方案；
- 强调了“模型和准则的耦合适配”这件事。

### 18.2 从代码角度看

代码很好地体现了论文主线：

- 4 个模型
- 8 个合法组合
- 每轮 1 次真实评估
- 用 reward 和 UCB 做在线选择

但代码并不是论文公式的逐字实现，而是一个偏实验原型的简化版落地：

- low-level reward 很忠实；
- high-level reward 做了经验均值化简；
- 工程依赖和命名细节还有一些粗糙处。

### 18.3 从研究启发角度看

如果你把它放到更大的研究背景里看，这篇工作最有价值的启发是：

**在昂贵优化里，真正需要自动化的往往不是“参数微调”，而是“方法选择本身”。**

这也是 AutoSAEA 最值得学的地方。
