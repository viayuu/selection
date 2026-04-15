# Idea Discovery Report

**Direction**: 在不改变问题锚点的前提下，继续推进“一个统一监督学习的 neural solver selector，同时支持多个 routing problem，并在实例级别从已有 solver pool 中选择最合适 solver”的主线。
**Date**: 2026-04-15
**Workflow status**: landscape + current-result diagnosis + idea refinement completed
**This round does not do**: 新一轮大规模 pilot / 新模型训练；本轮重点是基于你已经跑出来的 strong baseline，重新收敛更强、更深的 paper idea。

## Executive Summary

我对你当前工作的核心判断是：

1. **simple unified selector 这条主线已经成立**，不需要再证明“统一 selector 能不能 work”。你现在最强的 full-scale run 已经在标准 `test` 上给出很强证据：`test_macro_problem_oracle_ratio ≈ 1.003554`，并且 `macro gap vs best single = -0.018162`，说明 unified selector 在 mean cost 上已经稳定优于 best single per problem。
2. **下一篇 paper 不应该再把重点放在“做更大、更复杂的统一架构”上**。从你自己的实验看，复杂条件化/平衡化增强并没有稳定压过当前 simple unified selector，这意味着继续堆 `expert / adapter / MoE / 更复杂条件化` 很容易变成 overdesign。
3. **真正值得写成下一层 scientific story 的，是为什么这个简单统一 selector 在 IID 标准测试上已经成立，但在 benchmark / OOD 上仍然失效，尤其是 `CVRPLIB`。**

基于这个判断，我认为本轮最值得推进的单一 strongest direction 是：

## 🏆 Final Recommendation

### 环境不变残差后悔建模（Environment-Invariant Residual Regret Learning）

一句话概括：

> 与其继续学习一个 pooled `score(instance, solver)`，不如让 unified selector 去学习 **跨环境可迁移的 solver 优势规律**：预测每个 solver 相对安全 anchor solver 的 **residual regret / residual advantage**，并显式约束这些 solver preference 在不同 environment / regime 下尽量稳定。

这条线比当前 simple unified selector 更深的地方，不是结构更大，而是问题更准：

- 你已经证明了 unified selection 在 IID 条件下能 work；
- 下一步真正的问题是：**什么样的 solver preference 是可迁移的，什么样的 preference 只是训练分布里的脆弱相关性？**
- 这个问题直接对准你现在最大的弱点：**CVRPLIB / benchmark OOD gap**。

---

## 1. 我对你当前工作的重新理解

### 1.1 你现在已经完成的，不是“idea 阶段”，而是“主方法已基本成型”

从 [RESEARCH_BRIEF_TEMPLATE_CN.md](../RESEARCH_BRIEF_TEMPLATE_CN.md)、[HANDOFF_CONTEXT_FOR_NEXT_AI_2026-04-15.md](../HANDOFF_CONTEXT_FOR_NEXT_AI_2026-04-15.md)、以及当前 run 结果看，你已经不是在做一个 vague proposal，而是已经完成了一个很完整的 first-paper baseline：

- 统一实例表示；
- 全局 solver pool；
- per-instance feasible mask；
- 显式 `problem_id / problem_features / scale / instance_stats`；
- 共享 hierarchical encoder；
- masked pair scoring；
- ranking-based objective；
- 完整 train / val / test / benchmark / observation metrics pipeline。

所以当前最重要的认知更新是：

> **你不是缺一个“能 work 的 unified selector”**；你缺的是一个更 sharp 的第二层 scientific claim。

### 1.2 当前 baseline 的真正价值

当前 simple unified selector 的价值，不在于“结构复杂”，而在于它已经回答了一个本来就很难的问题：

- 不同问题有不同 candidate solver set；
- 输出空间不是固定多分类；
- 统一表示必须跨问题共享；
- 共享模型又必须显式知道问题语义；
- 用 ranking 比 flat classification 更符合 solver selection 本质。

换句话说：

> 你现在这个 simple unified selector，已经是 **paper-worthy baseline plus core claim** 级别，不再只是“预实验”。

---

## 2. 当前结果真正说明了什么

### 2.1 整体 test 结论：主线已经成立

来自当前最重要 full-scale run：

- `val_macro_problem_oracle_ratio = 1.003591`
- `test_macro_problem_oracle_ratio = 1.003554`
- `test_macro_problem_gap_vs_best_single = -0.018162`

这意味着：

- 标准 `test` 上 unified selector 已经非常接近 oracle；
- 而且在 mean cost 维度上已经稳定优于 best single per problem；
- 所以“one shared selector 是否可行”这个问题，本质上已经答出来了：**可行，而且有收益。**

### 2.2 真正困难的不是 average，而是 competitive / shifted regimes

从 [analysis_test/per_problem_summary.csv](../neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256/analysis_test/per_problem_summary.csv) 和 observation metrics 看，有两类问题完全不同：

#### A. 真正有选择难度的 competitive 问题

代表：

- `CVRP`
- `TSP`
- `ATSP`
- 部分 `OVRP/VRPL` 类变体

它们的特征是：

- 没有单一 solver 绝对统治所有实例；
- top1 accuracy 不是接近 1；
- selector 的价值来自对 solver complementarity 的真实利用；
- 这些问题才是 unified selector 应该真正被审查的核心场景。

特别是 `CVRP`：

- `selector_mean_cost = 18.4663`
- `best_single_mean_cost = 18.5028`
- `selector_top1_accuracy = 0.383`
- `best_top1_accuracy = 0.396`

也就是说：

- 你的 selector **在 mean cost 上已经赢了**；
- 但在 top1 纯命中率上还没有完全压住 `OMNI`；
- 这说明它已经学到了“soft ranking improvement”，但对 hardest CVRP cases 的 **winner discrimination** 还不够稳。

#### B. 近乎 trivial 的 low-entropy 问题

代表：

- 多个带 `TW` 的变体，例如 `OVRPTW / OVRPBTW / VRPTW / VRPBLTW` 等。

它们的特征是：

- `top1_accuracy` 基本接近 1；
- selector 与 best single 几乎完全持平；
- 本质上是一个 solver 已经几乎统治整个问题族。

这类问题不是没价值，但它们有一个危险：

> 如果把它们和 harder problems 混在一起看 overall average，会让 paper 看起来“结果很强”，但 reviewer 会质疑：你到底是在解决真正困难的 solver selection，还是在大量 easy cases 上刷平均数？

### 2.3 最值得作为 paper 新切口的 weakness

根据你的简报和交接文档，当前最核心 weakness 是：

- `benchmark_macro_problem_oracle_ratio = 1.017672`
- 主要弱点集中在 `CVRP/LIB`

这非常关键，因为它说明：

- 你不是缺 IID performance；
- 你缺的是 **distribution shift / benchmark realism / environment transfer**。

这恰恰是下一层 paper story 最适合切入的地方。

---

## 3. 我认为应该明确推翻的旧方向

这一部分很重要：不是所有之前看起来“更高级”的想法，今天还值得继续。

### Kill 1. 把“大 unified architecture”当成 headline

不建议下一篇 paper 再把主线写成：

- shared/private experts
- PLE / STAR / MMoE / HAMUR 风格大改造
- 更复杂 problem conditioning
- 更重的 solver-conditioned interaction

原因很简单：

- 你现在的 strongest baseline 已经证明 simple model 足够强；
- 你自己也已经观察到更复杂条件化方案没有稳定优于 simple baseline；
- 继续堆结构很容易让 paper 看起来像 “NSS + URS + recommender tricks”，而不是一个真正更强的问题。

### Kill 2. 把“one selector for all problems”继续当成主 claim

这个 claim 对你当前阶段已经不够 sharp 了。

因为你已经有证据说明：

- 一个 selector for all problems **确实能 work**；
- 所以下一篇 paper 再重复这个 claim，会显得力度不足。

更强的说法应该是：

- unified selector **在什么条件下真正能 transfer**；
- unified selector **在什么 environment 上会 fail**；
- 怎么让 learned solver preference 更 transportable，而不是只拟合 pooled correlations。

### Kill 3. 过度依赖 easy TW-heavy variants 来支撑主结论

这些问题可以保留在 full result table 里，但不该成为 narrative center。

正确做法是：

- 把 paper 的重心放在 harder competitive subsets；
- 尤其是 `CVRP / TSP / ATSP` 和 OOD benchmark；
- easy variants 只作为 completeness，而不是 novelty 载体。

---

## 4. Ranked Ideas

下面是我基于你当前项目状态重新排序后的更深 idea。排序标准不是“谁结构更花”，而是：

- 是否真正解决了当前最关键 scientific weakness；
- 是否比 simple unified selector 更深；
- 是否能在你现有代码和 2x3090 环境上 reasonably 验证；
- 是否更容易说服 reviewer。

---

### Idea 1: Environment-Invariant Residual Regret Learning

**结论**: `RECOMMENDED`
**Score**: 9.4 / 10
**Risk**: Medium

#### One-sentence thesis
统一 selector 不应只学习 pooled `score(instance, solver)`，而应学习 **跨 environment 可迁移的 solver advantage / residual regret**。

#### 它回答的科学问题
> unified selector 学到的 solver preference，哪些是跨环境稳定的，哪些只是训练分布里的脆弱相关性？

#### 为什么它比当前 simple unified selector 更深
当前方法已经回答了：

- one unified selector 能不能训练出来；
- masked compatibility ranking 合不合理。

这个 idea 往前走了一步：

- 不再只问 “能不能统一训练”，
- 而是问 “**怎样的 solver preference 才能真正 transfer 到 benchmark / OOD**”。

这是一个明显更强的问题，而且正好对准 `CVRPLIB` weakness。

#### 最便宜的验证实验
在尽量不改 encoder 主体的前提下做一个轻量版本：

1. 选择一个 safe anchor solver（例如 per-problem strong single，或 global robust baseline）；
2. 把每个 solver 的监督目标从绝对 cost / rank，改成相对 anchor 的 residual regret；
3. 用已有 `problem_features + scale + instance_stats` 构建 environment group，例如：
   - problem family
   - scale bin
   - clustering/statistics bin
   - asymmetry level
   - TW tightness / route-limit tightness
4. 增加一个简单的不变性正则：让同一 solver preference 在不同 environment 上更稳定；
5. 重点比较：
   - standard `test`
   - hard competitive subset
   - `CVRPLIB` benchmark

#### 最强 reviewer objection
“你的 environment grouping 是人工设计的；如果 gain 只依赖你定义得巧，并不能证明学到了真正 invariant 的 solver preference。”

#### 为什么我仍然推荐它
因为它：

- 完全对准你最重要 weakness；
- 不依赖大规模新架构；
- 容易做 clean ablation；
- 很适合把当前 simple unified selector 升级成第二篇更深的 paper。

---

### Idea 2: Competition-Aware Unified Selection

**结论**: `STRONG BACKUP`
**Score**: 8.9 / 10
**Risk**: Low-Medium

#### One-sentence thesis
统一 selector 应显式区分“需要 selection 的 competitive instances”和“本质上近乎 trivial 的 one-solver-dominant instances”。

#### 它回答的科学问题
> 什么时候 solver selection 真有信息价值，什么时候 selection 其实只是形式上存在？

#### 为什么它更深
当前结果混合了：

- 很难选的 competitive cases；
- 几乎不用选的 trivial cases。

这个 idea 的价值在于，它把 paper 从“平均性能更高”转成：

- unified selector 的收益主要来自哪些 instance regimes；
- 什么场景下 solver diversity 真正 matters。

这比继续堆架构更有 scientific clarity。

#### 最便宜的验证实验
1. 利用已有 solver cost labels 定义 `competition score`：
   - best vs second-best cost gap；
   - gain over safe baseline；
   - top-k regret spread。
2. 训练一个 auxiliary head 预测 `trivial vs competitive`；
3. 用 competition-aware reweighting 强化 hard cases；
4. 汇报：
   - 全量结果；
   - hard subset 结果；
   - benchmark 结果。

#### 最强 reviewer objection
“这更像 evaluation reframing，不一定构成一个足够强的新方法。”

#### 我对它的判断
如果单做主线，可能略偏“分析型 paper”；但如果作为 Idea 1 的 supporting protocol，它非常强。

---

### Idea 3: Selective / Abstaining Unified Selector

**结论**: `STRONG BACKUP`
**Score**: 8.7 / 10
**Risk**: Low

#### One-sentence thesis
在 OOD benchmark 上，统一 selector 不应该总是强行 top-1；当不确定时应该允许 fallback 到 top-k 或 safe solver。

#### 它回答的科学问题
> 在 distribution shift 下，selection 应该是强制分类问题，还是风险受控决策问题？

#### 为什么它更深
你现在 CVRPLIB 的问题，很可能不是“完全不会分辨”，而是：

- 某些 hard instances 上会做 **错误且自信的 top-1 决策**。

所以这个 idea 不是简单后处理，而是把 selection 目标改成：

- 在高置信区域 top-1；
- 在低置信区域 fallback / abstain；
- 用 risk-coverage 或 cost-runtime tradeoff 来评价。

#### 最便宜的验证实验
1. 基于现有 logits 做 calibration；
2. 做 confidence-based top-k / safe-fallback inference；
3. 重点看：
   - CVRPLIB cost 改善；
   - 风险-覆盖曲线；
   - 相同或受控推理预算下的收益。

#### 最强 reviewer objection
“这更像 selection strategy paper，而不是 selector representation 本身的突破。”

#### 我对它的判断
如果你想以最小实现代价拿到更强 benchmark story，这是非常现实的一条线；但它更适合作为 Idea 1 的 inference-time companion，而不是单独 headline。

---

### Idea 4: Constraint-Compositional Solver Preference Modeling

**结论**: `GOOD SECOND-LINE IDEA`
**Score**: 8.3 / 10
**Risk**: Medium

#### One-sentence thesis
solver preference 不应被看作 flat problem labels 的函数，而应被看作对基础约束属性及其组合的 compositional response。

#### 它回答的科学问题
> unified selector 是否能对组合式 routing constraints 学到 compositional solver preference，而不只是记 problem id？

#### 为什么它更深
你的 15 种 MVRP 变体并不是 15 个完全无关的类别，而是：

- open route
- backhaul
- time window
- route limit
- load / priority / asymmetry 等

这些约束的组合。

如果 paper 能证明：

- selector 的 preference 可在约束维度上 factorize / compose，
- 那么你对“跨问题共享”的解释会更科学，不再只是 empirical pooling。

#### 最便宜的验证实验
1. 保留当前 instance encoder；
2. 把 problem id / flat problem embedding 替换为 constraint-compositional descriptor；
3. 做 leave-combination-out 实验：
   - 某些 constraint 组合不参与训练，只在测试出现；
4. 比较 flat problem id vs compositional descriptor。

#### 最强 reviewer objection
“真实 solver preference 可能高度非线性，你的 compositional assumption 未必成立。”

#### 我对它的判断
这是个很有 paper 味道的方向，但它更偏“compositional generalization”主题，不如 Idea 1 那样直接击中你眼前最强 weakness。

---

### Idea 5: Counterfactual Shift Augmentation for Benchmark Robustness

**结论**: `SUPPORTING IDEA`
**Score**: 7.9 / 10
**Risk**: Medium

#### One-sentence thesis
当前 benchmark gap 的核心不是 model too small，而是 train-test support mismatch；通过有针对性的 counterfactual environment augmentation，可以提高 selector 的 OOD robustness。

#### 它回答的科学问题
> 什么样的 synthetic shift 是与 real benchmark gap 对齐的，且能帮助 unified selector 学到更 robust 的 solver preference？

#### 为什么它更深
这条线不是继续堆模型，而是把问题转向：

- 数据支持域为什么不够；
- benchmark gap 来自哪些可识别 shift；
- 哪些 perturbation 能保持 solver ranking，哪些不能。

#### 最便宜的验证实验
在当前数据上构造一批 perturbation：

- 更大规模；
- 更强 cluster；
- depot relocation；
- demand concentration shift；
- asymmetry / route-limit style shifts。

然后做 consistency regularization 或 group-aware training。

#### 最强 reviewer objection
“augmentation 是 heuristic，不能保证对应真实 benchmark distribution。”

#### 我对它的判断
适合作为 Idea 1 的配套实验或 supporting mechanism，不太建议单独作为 headline。

---

### Idea 6: Regime-Prototype / Support-Aware Unified Selection

**结论**: `ANALYSIS-HEAVY BACKUP`
**Score**: 7.5 / 10
**Risk**: Medium

#### One-sentence thesis
unified selector 的成功与失败，取决于 test instance 落在训练分布中的哪个 latent routing regime；CVRPLIB 之所以难，是因为它更接近低支持 regime。

#### 它回答的科学问题
> unified sharing 到底在什么 latent routing regimes 上有效，在哪些 regimes 上会失败？

#### 为什么它更深
它不是再发明一个更强网络，而是试图解释：

- 统一共享何时真正 beneficial；
- 哪些 regime 是 train support 充足的；
- 哪些 regime 是 benchmark-only 的 sparse support region。

#### 最便宜的验证实验
1. 用 `instance_stats + current encoder embeddings + solver win patterns` 做 clustering；
2. 分析各 regime 的 solver entropy、selector regret、benchmark density；
3. 增加 lightweight prototype regularization。

#### 最强 reviewer objection
“如果它主要是 post-hoc analysis，那 novelty 可能不够。”

#### 我对它的判断
非常适合作为 analysis section，但单独做 headline 可能偏弱。

---

## 5. Final Ranking

综合考虑 novelty、可写性、与当前 weakness 的贴合度、实现成本和 reviewer 说服力，我的最终排序是：

1. **Environment-Invariant Residual Regret Learning**
2. **Competition-Aware Unified Selection**
3. **Selective / Abstaining Unified Selector**
4. **Constraint-Compositional Solver Preference Modeling**
5. **Counterfactual Shift Augmentation for Benchmark Robustness**
6. **Regime-Prototype / Support-Aware Unified Selection**

---

## 6. 我建议的 paper 重构方式

如果你接受我的判断，那么接下来不建议把 paper 继续写成：

- “我们提出一个更复杂的 unified selector 架构”

而建议重构成：

### 新问题锚点（不改变总体方向，只 sharpening）

> 一个 simple unified selector 已经可以在多问题 routing 上获得很强 IID test 表现；但其 learned solver preference 在 benchmark / OOD，特别是 CVRPLIB 上仍不够稳定。核心问题不是能否统一训练，而是如何学习 **跨环境可迁移的 solver preference**。

### 新主 claim

> Unified cross-problem solver selection 的下一层关键，不是更大架构，而是 **environment-robust preference learning**。

### 新 supporting claims

1. average metric 会被 low-entropy variants 稀释，必须区分 hard competitive instances；
2. 对 benchmark / OOD，更合理的目标是 residual regret / invariant preference，而不只是 pooled score fitting；
3. selective fallback / top-k safe inference 可以作为 risk-controlled decision layer，进一步降低 hard OOD mistakes。

---

## 7. Minimal Next-Step Plan

如果只做最值得做的一步，我建议：

### Step 1. Freeze baseline
把当前 [round2_full_true_baseline-0415-040256](../neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256/) 定为主 baseline，不再继续 architecture lottery。

### Step 2. Define hard / competitive subset
从已有 labels 中定义：

- solver entropy
- best-vs-second-best margin
- gain-over-safe-baseline

形成一个 `competitive subset`。

### Step 3. Implement residual regret target
在当前 scorer 基础上改成：

- 相对 anchor solver 的 residual regret / residual advantage；
- 尽量不先动 encoder trunk。

### Step 4. Add simple environment-invariance regularization
environment 先用已有 metadata 近似构造：

- problem family
- scale bucket
- stats bucket
- constraint intensity bucket

### Step 5. Report three blocks only
1. standard IID test 不掉；
2. hard competitive subset 更强；
3. `CVRPLIB` / benchmark 更稳。

这 5 步比继续扩 expert / adapter / solver-conditioned heavy interaction 更像下一篇真正值得写的 paper。

---

## 8. What I would archive for now

建议明确归档，不再把它们当主线：

- heavy expertized unified selector as headline
- generic “one selector for all problems” framing
- 继续靠 TW-heavy easy variants 撑 narrative
- 继续做不带更强问题意识的 architecture sweep

它们可以保留为：

- 参考文献
- 小 ablation
- appendix / future work

但不该再是现在的 main idea。

---

## 9. Final Verdict

如果你问我一句最直接的话：

> **你现在不缺一个更复杂的 unified selector；你缺一个更 sharp 的、能解释 benchmark/OOD failure 的 scientific question。**

因此，本轮我给出的最终建议是：

### 保持总方向不变，但把下一篇 paper 的中心改成：

> **面向 benchmark / OOD 的 unified solver preference transfer**

其中最值得优先推进的具体方法是：

> **Environment-Invariant Residual Regret Learning for Unified Cross-Problem Solver Selection**

这比“继续加 MoE / adapter / 更重交互”更深，也更像 reviewer 会真正买账的下一步。

---

## Source Notes

本轮判断主要基于：

- [RESEARCH_BRIEF_TEMPLATE_CN.md](../RESEARCH_BRIEF_TEMPLATE_CN.md)
- [HANDOFF_CONTEXT_FOR_NEXT_AI_2026-04-15.md](../HANDOFF_CONTEXT_FOR_NEXT_AI_2026-04-15.md)
- [IDEA_REPORT.md](../IDEA_REPORT.md)
- [model.py](../neural-solver-selection_unified_cost/model.py)
- [dataset.py](../neural-solver-selection_unified_cost/dataset.py)
- [loss.py](../neural-solver-selection_unified_cost/loss.py)
- [analysis_test/summary.txt](../neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256/analysis_test/summary.txt)
- [analysis_test/per_problem_summary.csv](../neural-solver-selection_unified_cost/train_logs/round2_full_true_baseline-0415-040256/analysis_test/per_problem_summary.csv)
- [test_problem_metrics.md](../neural-solver-selection_unified_cost/train_logs/observation_metrics/round2_full_true_baseline-0415-040256/test_problem_metrics.md)
- local literature anchors: NSS / URS / RouteFinder

本轮还参考了一个独立研究子代理的 second opinion，用于避免我只是顺着旧思路做增量优化。