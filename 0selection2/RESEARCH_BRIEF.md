# 研究简报

> **用于 `/idea-discovery` 或 `/research-pipeline` 的文档化输入模板。**

## 问题陈述
当前研究问题是：能否训练一个**监督学习的 neural solver selector**，让**一个模型**同时支持多个 routing problem，并在实例级别从多个已有 solver 中选出最合适的方法。通俗的讲，就是，输入一个实例，模型根据实例的特征选择一种方法来解这个实例

这项工作的重点不是设计一个新的 `TSP` 或 `CVRP` solver，而是把已有 solver 看成候选方法池，对每个实例做 solver selection。

与 NSS（论文见Neural Solver Selection for Combinatorial Optimization.md，代码见literature/nss代码） 不同在于，nss是对tsp和cvrp问题分别训练一个selector模型，而当前目标不是“每个问题单独训练一个 selector”，而是训练一个**跨问题共享**的 selector，在统一框架下同时支持 `TSP / CVRP / ATSP / 15 种 MVRP 变体` 共 18 个问题。

这个问题的难点有三个（我暂时能想到的，你可以想更多）。
第一，不同问题支持的方法的数量不一样且不同（例如，tsp支持的方法很多，而mvrp支持的方法很少），面临方法分布不平衡的问题；并且不同问题间的方法可能存在交叉（例如tsp和cvrp问题都支持lehd方法）。我暂时的思路是使用mask掩码排除不支持的方法。
第二是如何区分实例属于什么问题，例如mvrp中，某些变体问题（OVRP）的输入格式和cvrp完全相同，当模型接到一个实例输入时，无法区分他是什么问题
第三，如何统一表示不同问题的实例，同时又能识别不同问题的实例的结构差异。

重点：对于问题二和问题三，导师给我的建议是可以参考URS论文和CoEKS论文，你可以参考，也可以进一步自己搜索扩展。可以参考推荐算法他们是怎么解决用户适配和输出不均衡问题的，不一定局限于组合优化领域的论文!


## 背景

- **领域**: 组合优化 / 推荐算法 / routing
- **子方向**: 多问题统一的监督学习 solver selector
- **已读关键论文**:
  - `NSS (Neural Solver Selection for Combinatorial Optimization)`（论文见Neural Solver Selection for Combinatorial Optimization.md，代码见literature/nss代码）
  - `URS`（论文见literature/urs.md，代码见literature/URS代码）
  - `CoEKS`(论文见literature/CoEKS.md，代码见literature/CoEKS代码) 对应的跨问题专家化思路
## 约束条件

- **算力**:
  - 当前机器：`2x RTX 3090 24GB`
  - 默认环境：`conda activate easynco_zhoucl`
  - 长任务默认放在 `tmux`
- **目标会议/期刊**:
  - ICML

## 期望方向
- 改进现有方法：扩展到多个问题上，统一监督学习的多问题 neural solver selector
- 在NSS论文的做法基础上改，并且在写代码的时候，将NSS的代码复制一份，然后在他的基础上改。

## 非目标
- 不把 `time` 作为当前阶段的监督目标，只考虑cost

## 已有结果（如有）

## 其他要求和限定
### 需要关注的指标
指标文档在：
- `0selection/观测指标`
对于每个问题，都至少应该关注观测指标.md中的指标

### 训练数据说明
训练数据在data文件夹，对数据的详细介绍在 data/README.md 文件