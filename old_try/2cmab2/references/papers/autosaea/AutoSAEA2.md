# AutoSAEA 论文与代码详细解读

## 一、论文概述

**论文标题**: Surrogate-Assisted Evolutionary Algorithm With Model and Infill Criterion Auto-Configuration
**发表期刊**: IEEE Transactions on Evolutionary Computation, 2024
**核心贡献**: 提出 AutoSAEA 算法，通过"两层多臂老虎机"（TL-MAB）自动选择代理模型和填充准则的最佳组合。

---

## 二、背景知识详解

### 2.1 什么是昂贵优化问题（EOP）？

在很多工程场景中（如汽车碰撞分析、航空结构设计），我们要优化一个目标函数 f(x)，但这个函数**没有解析表达式**，每次计算都需要运行昂贵的仿真（可能一次要几小时甚至几天）。这类问题叫做 **昂贵优化问题（Expensive Optimization Problem, EOP）**。

因为计算成本极高，我们不能像普通优化那样随便调用几万次目标函数。通常只允许几百到一千次评估（Function Evaluations, FEs）。

### 2.2 什么是代理辅助进化算法（SAEA）？

**进化算法**（如遗传算法、差分进化）是一类模拟自然进化的优化方法，它们维护一个"种群"（一组候选解），通过变异、交叉、选择等操作不断迭代改进。但进化算法通常需要大量函数评估，不适合直接解决 EOP。

**代理模型（Surrogate Model）** 是一个"替身"——用已有的评估数据训练一个计算成本很低的近似模型来代替昂贵的真实函数。这样，进化算法的大部分评估都在代理模型上进行，只有少数"有价值的"候选解才用真实函数评估。

**SAEA = 代理模型 + 进化算法**，核心流程是：
1. 用少量初始样本训练代理模型
2. 用代理模型预测并筛选好的候选解
3. 对选出的候选解进行真实的昂贵评估
4. 将新数据加入训练集，更新代理模型
5. 重复直到预算用完

### 2.3 什么是填充准则（Infill Criterion）？

填充准则决定了**如何从代理模型的预测中选出下一个要真实评估的候选解**。不同的准则有不同的倾向：

- **利用（Exploitation）型准则**：选代理模型预测值最好的点。容易陷入局部最优。
- **探索（Exploration）型准则**：选代理模型最不确定的区域中的点。能发现新区域但收敛慢。
- **平衡型准则**：综合考虑预测值和不确定性（如 LCB、EI）。

### 2.4 什么是多臂老虎机（MAB）？

**多臂老虎机（Multi-Armed Bandit, MAB）** 是强化学习中的经典问题。想象你面前有 K 台老虎机（赌博机），每台有不同的（未知的）奖励分布。你每次只能拉一台的摇臂，获得一个奖励。目标是在有限次尝试中最大化总奖励。

核心困难在于**探索-利用权衡（Exploration-Exploitation Tradeoff）**：
- **利用**：总是拉目前看来最好的摇臂（可能错过更好的）
- **探索**：尝试不太了解的摇臂（可能浪费次数）

#### UCB（Upper Confidence Bound）策略

UCB 是解决 MAB 问题的经典方法。选择摇臂 a 的公式为：

```
选择 a = argmax [ Q(a) + sqrt( α * ln(t) / T(a) ) ]
```

其中：
- `Q(a)` 是摇臂 a 的平均奖励（利用项）——选奖励高的
- `sqrt(α * ln(t) / T(a))` 是置信上界（探索项）——选尝试次数少的
- `t` 是当前总轮次，`T(a)` 是摇臂 a 被选择的次数
- `α` 控制探索程度

直觉：如果一个摇臂很久没被拉，它的探索项会变大，最终会被选中尝试。这保证了每个摇臂都能被充分探索。

#### 层次化多臂老虎机（Hierarchical MAB）

当摇臂数量很多时，可以把摇臂组织成层次结构。本文采用**层次聚类结构**：
- 高层摇臂代表不同的"类别"
- 每个高层摇臂关联若干低层摇臂
- 先选高层摇臂，再在其关联的低层摇臂中选择

好处：减小搜索空间，降低选到差摇臂的概率，理论遗憾界更低。

---

## 三、AutoSAEA 算法详解

### 3.1 核心思想

现有的 SAEA 要么只自适应选模型，要么只自适应选填充准则，没有**同时协同选择两者**。AutoSAEA 的关键创新是：

> 把"选代理模型"和"选填充准则"建模为一个**两层多臂老虎机（TL-MAB）**问题：
> - 高层摇臂 = 代理模型（GP, RBF, PRS, KNN）
> - 低层摇臂 = 填充准则（LCB, EI, prescreening, local search, L1-exploitation, L1-exploration）

### 3.2 四个代理模型

| 模型 | 类型 | 特点 | 关联的填充准则 |
|------|------|------|----------------|
| **GP（高斯过程）** | 回归 | 不仅给出预测值，还给出预测的不确定性（方差） | LCB, EI |
| **RBF（径向基函数）** | 回归 | 计算效率高，用基函数的线性组合拟合 | prescreening, local search |
| **PRS（多项式响应面）** | 回归 | 用二次多项式拟合，简单但表达能力有限 | prescreening, local search |
| **KNN（K近邻）** | 分类 | 把解分成好/差几个等级，预测新解的等级 | L1-exploitation, L1-exploration |

#### GP 模型详解
GP 假设目标函数是一个高斯过程的一个采样。给定训练数据后，对任意新点 x，GP 给出：
- **预测均值** μ(x)：对 f(x) 的最佳估计
- **预测方差** σ²(x)：估计的不确定性

方差大意味着该区域数据稀疏，模型不确定。

#### RBF 模型详解
RBF 用一组"基函数"的加权和来拟合数据：
```
f_RBF(x) = Σ wᵢ · φ(||x - xᵢ||) + 线性项
```
其中 φ 是基函数（本文用立方函数 φ(r) = r³），wᵢ 是权重。直觉上，每个训练点贡献一个"影响范围"，离该点越近影响越大。

#### PRS 模型详解
二阶多项式响应面：
```
f_PRS(x) = β₀ + Σ βᵢxᵢ + Σ βᵢⱼxᵢxⱼ
```
类似多元线性回归，但加了交叉项和二次项。用最小二乘法求系数。

#### KNN 模型详解
本文的 KNN 是分类器（K=1）。把当前种群按适应度分成 5 个等级（Level 1 是最好的 20%），然后预测新解属于哪个等级。预测方法很简单：新解的等级 = 离它最近的训练样本的等级。

### 3.3 八个填充准则

| 编号 | 模型 | 准则 | 描述 |
|------|------|------|------|
| a₁ᴸ | GP | **LCB** | 选 `f̂(x) - w·ŝ(x)` 最小的。w=2。预测值低且不确定性大的点得分高。 |
| a₂ᴸ | GP | **EI** | 选期望改进最大的。综合考虑预测值可能超越当前最优的概率和幅度。 |
| a₃ᴸ | RBF | **prescreening** | 用 DE 生成 N 个子代，选 RBF 预测值最小的。 |
| a₄ᴸ | RBF | **local search** | 在当前种群的包围盒内，用 DE 优化 RBF 模型（最多 100D+1000 代）。 |
| a₅ᴸ | PRS | **prescreening** | 同 a₃ᴸ，但用 PRS 模型预测。 |
| a₆ᴸ | PRS | **local search** | 同 a₄ᴸ，但用 PRS 模型。 |
| a₇ᴸ | KNN | **L1-exploitation** | 从预测为 Level 1 的子代中，选离现有 Level 1 解**最近**的（深入开发已知好区域）。 |
| a₈ᴸ | KNN | **L1-exploration** | 从预测为 Level 1 的子代中，选离现有 Level 1 解**最远**的（探索新区域）。 |

#### LCB 填充准则详解
LCB（Lower Confidence Bound）：
```
选 x* = argmin [ f̂_GP(x) - w · ŝ(x) ]
```
- `f̂_GP(x)` 是预测值（越小越好）
- `ŝ(x)` 是预测标准差（越大表示越不确定）
- `w=2` 控制探索-利用平衡

减去 w·ŝ(x) 意味着：**预测值好的点得分高（利用），不确定性大的点也得分高（探索）**。

#### EI 填充准则详解
EI（Expected Improvement）：
```
EI(x) = (f_min - f̂(x)) · Φ(Z) + ŝ(x) · φ(Z)
```
其中 Z = (f_min - f̂(x)) / ŝ(x)。

直觉：EI 计算新点**期望能比当前最优改进多少**。如果预测值远好于当前最优（第一项大），或不确定性大（第二项大），EI 都会大。

#### Prescreening 详解
最简单的策略：用 DE 算子从当前种群生成 N 个子代，用代理模型预测它们的适应度，选预测最好的那个做真实评估。相当于纯利用。

#### Local Search 详解
在当前种群的变量范围（包围盒）内，用 DE 算法**优化代理模型本身**（最多 100D+1000 代），找到代理模型的最优解做真实评估。比 prescreening 更深入但计算量更大。

#### L1-exploitation 详解
从 KNN 分类器预测为 Level 1（最好的 20%）的子代中，找到与现有 Level 1 解**最大距离最小**的那个。也就是选一个"深入已知好区域中心"的点。

#### L1-exploration 详解
从预测为 Level 1 的子代中，找与现有 Level 1 解**最小距离最大**的那个。也就是选一个"远离已知好解、在新区域但可能也好"的点。

### 3.4 两层关联结构

高层和低层摇臂的关联关系（不是所有模型都能搭配所有准则）：

```
GP   ──→ {LCB, EI}
RBF  ──→ {prescreening, local search}
PRS  ──→ {prescreening, local search}
KNN  ──→ {L1-exploitation, L1-exploration}
```

总共 8 种合法组合（组合臂 Combinatorial Arm）。

### 3.5 算法主流程

```
输入：种群大小 N=100, 最大评估次数 MaxFEs=1000, α=2.5

1. 初始化：
   - 用拉丁超立方采样（LHS）生成 N 个初始解并真实评估
   - 所有摇臂的 Q 值和选择次数设为 0

2. 热身阶段（前 8 轮）：
   - 依次尝试 8 种组合臂，确保每种至少使用一次
   - 记录每次的奖励

3. 主循环（第 9 轮起，直到 MaxFEs 用完）：
   3.1 选 N 个最优历史解作为当前种群 P
   3.2 用 TL-UCB 选高层摇臂（模型）和低层摇臂（准则）
   3.3 用选中的模型+准则协同产生一个新候选解 x_t
   3.4 真实评估 x_t，加入数据库
   3.5 用 TL-R 更新奖励和 Q 值

4. 输出数据库中的最优解
```

### 3.6 TL-UCB（两层上置信界）详解

TL-UCB 分两步选择：

**第一步：选高层摇臂（模型）**
```
a_t^H = argmax_a [ Q_a^H(t) + sqrt(α · ln(t) / T_a^H(t)) ]
```
- Q_a^H(t)：高层摇臂 a 的当前价值（利用项）
- sqrt(α·ln(t)/T_a^H(t))：探索项，选择次数少的摇臂会有更大的探索奖励
- α=2.5 控制探索力度

**第二步：选低层摇臂（准则）**
在选中的高层摇臂关联的低层摇臂中，用同样的 UCB 公式选择：
```
a_t^L = argmax_{a ∈ A_{a_t^H}^L} [ Q_a^L(t) + sqrt(α · ln(t) / T_a^L(t)) ]
```

### 3.7 TL-R（两层奖励）详解

这是 AutoSAEA 最精妙的设计之一。

#### 低层奖励
新解 x_t 在当前种群中的排名为 I(x_t)（1=最好，N+1=比所有都差），低层奖励为：
```
r_{a_t^L} = -I(x_t)/N + (N+1)/N
```

| 排名 I(x_t) | 奖励 r |
|---|---|
| 1（最好） | 1.0 |
| 51（中等，N=100） | 0.5 |
| 101（最差） | 0.0 |

奖励特点：
- 范围 [0, 1]，与排名线性相关
- **非稀疏**：即使解不是最优也有正奖励，不像"0/1 奖励"那样大部分时候给 0
- **相对稳定**：基于排名而非绝对值，在优化过程中不会因为适应度量级变化而剧烈波动

#### 低层 Q 值更新
```
Q_{a_t^L}(t+1) = [ T_{a_t^L}(t) · Q_{a_t^L}(t) + r_{a_t^L} ] / [ T_{a_t^L}(t) + 1 ]
```
就是**增量平均**：新 Q 值 = 之前所有奖励的平均值加上新奖励后的均值。

#### 高层奖励
```
r_{a_t^H} = [ Q_{a_t^L}(t+1) - Q_{a_t^L}(t) ] / |A_{a_t^H}^L|
```
- 分子：低层 Q 值的变化量。如果低层表现变好（Q值上升），高层获得正奖励；反之获负奖励。
- 分母：该高层摇臂关联的低层摇臂数量（本文都是 2）。

#### 高层 Q 值更新
```
Q_{a_t^H}(t+1) = Q_{a_t^H}(t) + r_{a_t^H}
```

论文证明了一个重要性质：**高层 Q 值始终等于其关联的所有低层 Q 值的平均值**：
```
Q_{a_t^H}(t+1) = Σ Q_{a^L}(t+1) / |A_{a_t^H}^L|
```

这意味着：如果一个模型的两个填充准则都表现好，这个模型的 Q 值就高；如果一个准则好一个差，模型的 Q 值就中等。

#### 奖励传播示例

假设第 t 轮选了 GP+LCB，新解排名第 21（N=100）：
1. 低层奖励：r = -21/100 + 101/100 = 0.8
2. 假设 LCB 之前被选了 4 次，Q值为 0.55：
   - 新 Q_LCB = (4×0.55 + 0.8) / 5 = 0.6
3. 高层奖励：r_GP = (0.6 - 0.55) / 2 = 0.025
4. 假设 GP 之前 Q 值为 0.675：
   - 新 Q_GP = 0.675 + 0.025 = 0.7

### 3.8 差分进化（DE）算子

DE 是本文中产生子代的基础算子。对种群中的每个个体 x_i：

**变异**：
```
v_i = x_i + F · (x_best - x_i) + F · (x_r1 - x_r2)
```
- x_best：当前最优个体（引导方向）
- x_r1, x_r2：随机选的两个个体（提供随机扰动）
- F=0.5：缩放因子

这是 DE/current-to-best/1 变异策略：从当前个体出发，向最优个体靠近，同时加随机扰动。

**交叉**（二项交叉）：
```
o_{i,j} = v_{i,j}  如果 rand < CR 或 j == j_rand
         x_{i,j}  否则
```
- CR=0.9：交叉率（90%的维度来自变异向量）
- j_rand：保证至少一个维度来自变异向量

**边界修复**：越界时随机重置到合法范围内。

---

## 四、代码结构与实现详解

代码用 MATLAB 实现，主要文件及其功能：

### 4.1 文件清单

| 文件 | 功能 |
|------|------|
| `RUN_AutoSAEA.m` | 入口函数，多次运行实验并统计结果 |
| `AutoSAEA.m` | **主算法**，实现整个流程 |
| `DEoperator.m` | DE 变异+交叉算子（生成子代） |
| `DE_optimizer.m` | DE 优化器（用于 local search） |
| `TL_UCB.m` | TL-UCB 计算 |
| `Low_level_r.m` | 低层奖励计算 |
| `GP_lcb_arm.m` | {GP, LCB} 组合臂 |
| `GP_ei_arm.m` | {GP, EI} 组合臂 |
| `GP_fit.m` | GP 模型训练（DACE工具箱） |
| `predictor.m` | GP 模型预测 |
| `RBF_pre_arm.m` | {RBF, prescreening} 组合臂 |
| `RBF_ls_arm.m` | {RBF, local search} 组合臂 |
| `my_rbfbuild.m` | RBF 模型训练 |
| `my_rbfpredict.m` | RBF 模型预测 |
| `PRS_pre_arm.m` | {PRS, prescreening} 组合臂 |
| `PRS_ls_arm.m` | {PRS, local search} 组合臂 |
| `KNN_eoi_arm.m` | {KNN, L1-exploitation} 组合臂 |
| `KNN_eor_arm.m` | {KNN, L1-exploration} 组合臂 |

### 4.2 AutoSAEA.m 主函数逐段解读

```matlab
function [hf, MaxFEs, gfs, s_l_s] = AutoSAEA(FUN, D, LB, UB)
```
- 输入：目标函数 FUN、维度 D、变量下界 LB、上界 UB
- 输出：历史适应度 hf、最大 FEs、每步最优 gfs

#### 初始化
```matlab
initial_sample_size = 100;  % 种群大小 N=100
MaxFEs = 1000;              % 最大评估次数
apha = 2.5;                 % UCB 探索参数 α
level = 5;                  % KNN 分级数

% LHS 初始采样
sam = LB + (UB-LB) .* lhsdesign(100, D);
% 真实评估所有初始样本
for i = 1:100
    fitness(i) = FUN(sam(i,:));
end
```

#### 奖励存储
```matlab
% 每个低层摇臂的奖励历史
Save_rp = [];  % RBF prescreening 的所有历史奖励
Save_rl = [];  % RBF local search
Save_gl = [];  % GP LCB
Save_ge = [];  % GP EI
Save_pp = [];  % PRS prescreening
Save_pl = [];  % PRS local search
Save_ki = [];  % KNN L1-exploitation
Save_ko = [];  % KNN L1-exploration

% 每个高层摇臂的奖励历史
rbf_model = [];  % RBF 的所有关联低层奖励
gp_model  = [];  % GP 的所有关联低层奖励
prs_model = [];  % PRS 的所有关联低层奖励
knn_model = [];  % KNN 的所有关联低层奖励
```

**注意**：代码中高层模型的 Q 值通过 `mean(rbf_model)` 等计算，即所有关联低层奖励的平均值。这与论文证明的"高层 Q 值 = 关联低层 Q 值的均值"略有差异——代码实际是对所有历史奖励取均值，而不是分别对每个低层臂取均值再平均。但在实践中效果类似。

#### 热身阶段（前 8 轮）
```matlab
if id <= num_arm  % num_arm = 8
    if id == 1     % {RBF, prescreening}
    elseif id == 2 % {GP, LCB}
    ...
    elseif id == 8 % {KNN, L1-exploration}
end
```
前 8 轮按固定顺序各尝试一种组合臂。这是为了确保 UCB 公式中 T(a) ≠ 0（避免除以零）。

#### TL-UCB 选择阶段
```matlab
% 第一层：选模型
for i = 1:4
    q_value_m = mean(model_rewards);  % 模型的 Q 值
    U_model_value(i) = TL_UCB(sum_reward, id, q_value_m, apha);
end
idx = argmax(U_model_value);

% 第二层：选准则（以 RBF 为例）
if idx == 1  % RBF 被选中
    for i = 1:2
        U_rbf_value(i) = TL_UCB(Save_rp_or_rl, id, q_value, apha);
    end
    idx2 = argmax(U_rbf_value);
    % idx2==1 → prescreening, idx2==2 → local search
end
```

### 4.3 TL_UCB.m 解读

```matlab
function [U_value] = TL_UCB(sum_reward, id, q_value, apha)
    U_value = q_value + sqrt((apha * log(id)) / length(sum_reward));
end
```

- `q_value`：摇臂的 Q 值（平均奖励）
- `length(sum_reward)`：该摇臂被选择的次数 T(a)
- `id`：当前迭代轮次 t
- `apha`：α=2.5

完全对应 UCB 公式：Q(a) + sqrt(α·ln(t)/T(a))

### 4.4 Low_level_r.m 解读

```matlab
function [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm)
    N = length(ghf);
    ghf_sum = [ghf, candidate_fit];  % 当前种群 + 新解
    [~, index] = sort(ghf_sum);
    in = find(index == (N+1));       % 新解在排序后的位置
    reward = -1/N * in + (N+1)/N;   % 线性映射到 [0, 1]
end
```

把新解加入当前种群后排序，找到新解的排名 `in`，然后用公式 `r = -in/N + (N+1)/N` 映射到 [0,1]。排名第 1 得 1.0，排名第 N+1 得 0.0。

### 4.5 各组合臂函数通用模式

每个 arm 函数（如 `RBF_pre_arm.m`）遵循相同模式：

```
1. 训练代理模型（用当前种群 ghx, ghf）
2. 用对应的填充准则选出一个候选解 candidate_position
3. 检查候选解是否已在数据库中（去重）
4. 如果是新解：
   - 真实评估得 candidate_fit
   - 加入数据库 hx, hf
   - 调用 Low_level_r 计算奖励
5. 如果是重复解：reward = 0
```

#### RBF_pre_arm.m（RBF + prescreening）
```matlab
% 训练 RBF
srgtSRGT = srgtsRBFFit(srgtOPT);
% 预测所有子代
fitnessModel = my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, offspring);
% 选预测最好的
[~, sidx] = min(fitnessModel);
candidate_position = offspring(sidx, :);
```

#### RBF_ls_arm.m（RBF + local search）
```matlab
% 训练 RBF
srgtSRGT = srgtsRBFFit(srgtOPT);
% 用 DE 优化 RBF 模型（最多 100D+1000 代）
Max_NFE = 100*Dim + 1000;
[candidate_position, ~, ~] = DE_optimizer(Dim, Max_NFE, srgtSRGT, minerror, ghx, 1);
```
注意 local search 不需要外部生成子代，而是直接在代理模型上运行完整的 DE 优化。

#### GP_lcb_arm.m（GP + LCB）
```matlab
% 训练 GP
[dmodel, ~] = GP_fit(ghx, ghf, @regpoly0, @corrgauss, theta);
% 对每个子代计算 LCB 值
w = 2;
for i = 1:size(offspring, 1)
    [tempobj, ~, MSE, ~] = predictor(offspring(i,:), dmodel);
    OffObj(i) = tempobj - w * sqrt(MSE);  % LCB = 预测值 - w*标准差
end
[~, I] = min(OffObj);  % 选 LCB 最小的
```

#### GP_ei_arm.m（GP + EI）
```matlab
% 训练 GP
[dmodel, ~] = dacefit_3(ghx, ghf, @regpoly0, @corrgauss, theta);
Gbest = min(hf);
for i = 1:size(offspring, 1)
    [y, ~, mse, ~] = predictor(offspring(i,:), dmodel);
    s = sqrt(mse);
    % EI = -(Gbest-y)*Φ((Gbest-y)/s) - s*φ((Gbest-y)/s)
    EI(i) = -(Gbest-y)*normcdf((Gbest-y)/s) - s*normpdf((Gbest-y)/s);
end
[~, I] = min(EI);  % 注意取的是 min（因为 EI 带负号）
```

#### KNN_eoi_arm.m（KNN + L1-exploitation）
```matlab
% 分级标签
train_label1 = ceil(sidx * level / N);  % 分成 5 级
Parents_L1 = ghx(find(train_label1==1), :);  % Level 1 的解

% 训练 KNN 分类器
mdl = ClassificationKNN.fit(ghx, train_label1, 'Distance', 'minkowski');

% 预测子代标签
label = predict(mdl, offspring);
select_pp = find(label == 1);  % 预测为 Level 1 的子代

% L1-exploitation：选最大距离最小的（最靠近已知好区域中心的）
for 每个预测为 L1 的子代:
    计算它到所有 Level 1 父代的距离
max_dist = max(dist, [], 2);  % 每个候选到最远 L1 父代的距离
[~, in] = min(max_dist);      % 选这个距离最小的
```

#### KNN_eor_arm.m（KNN + L1-exploration）
与 exploitation 几乎相同，但最后一步反过来：
```matlab
max_dist = min(dist, [], 2);  % 每个候选到最近 L1 父代的距离
[~, in] = max(max_dist);      % 选这个距离最大的（最远离已知好解的）
```

### 4.6 DE_optimizer.m 解读

这是用于 local search 的 DE 优化器。在代理模型上运行（不消耗真实 FEs）：

```matlab
function [bestP, bestFitness, bestP_position] = DE_optimizer(Dim, Max_NFEs, srgtSRGT, minerror, ghx, flag)
```

关键特点：
- 搜索范围限定在当前种群的包围盒 `[min(ghx), max(ghx)]` 内
- 使用 DE/best/1 变异策略：`V = bestP + F*(P1-P2)`
- 收敛停止：如果连续 10 代最优值改进小于 1e-20 则停止
- `flag=1` 时用 RBF 预测，`flag=0` 时用 PRS 预测

---

## 五、算法设计的直觉与优势

### 5.1 为什么需要自动配置？

不同的 EOP 有不同的适应度景观（landscape）：
- 光滑问题 → PRS 可能就够了
- 多峰问题 → GP+LCB 的探索能力更好
- 高维问题 → RBF 计算效率更高
- 局部结构明显的问题 → KNN 分类+局部搜索效果好

没有一种模型+准则能通杀所有问题。AutoSAEA 让算法**在运行过程中自己学习哪种组合最有效**。

### 5.2 为什么用两层结构而非扁平结构？

如果把 8 种组合臂看作 8 个独立臂（扁平 MAB），每个都需要充分探索。两层结构的好处：
- 如果 RBF 模型效果差，整个 RBF 分支（包括 prescreening 和 local search）都会被降低优先级，不需要分别验证
- 减少了探索空间，加速了收敛
- 理论遗憾界更低

### 5.3 奖励设计的巧妙之处

- **基于排名而非绝对值**：优化过程中适应度值的量级可能变化很大，排名更稳定
- **非稀疏奖励**：即使新解不是最优也有正奖励（只要不是最差），信号更丰富
- **双向传播**：低层奖励向上传播到高层，实现了模型和准则的协同评价

---

## 六、实验设置关键参数

| 参数 | 值 | 含义 |
|------|---|------|
| N (种群大小) | 100 | 初始 LHS 采样数，也是每轮的种群大小 |
| MaxFEs | 1000 | 最多真实评估 1000 次 |
| α | 2.5 | UCB 探索参数 |
| F | 0.5 | DE 缩放因子 |
| CR | 0.9 | DE 交叉率 |
| level | 5 | KNN 分级数 |
| w (LCB) | 2 | LCB 中不确定性的权重 |
| Local search 代数 | 100D+1000 | DE 在代理模型上的最大优化代数 |

---

## 七、总结

AutoSAEA 的核心创新可以用一句话概括：**用层次化多臂老虎机在线学习最佳的"代理模型+填充准则"组合**。

算法框架清晰：
1. 初始化 → 2. 热身（每种组合试一次） → 3. UCB选择+评估+奖励更新 → 4. 输出最优

它解决的核心矛盾是：不同问题需要不同的模型和准则，但我们事先不知道哪种最好。通过在线学习（MAB），算法能在优化过程中自适应地将更多资源分配给表现好的组合，同时保留对其他组合的探索。
