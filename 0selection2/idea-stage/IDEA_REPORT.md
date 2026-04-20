# Research Idea Report v2 — Selector Effect-First Improvement

**Direction**: 提升跨 18 个 routing problems 的监督式 solver selector 效果，重点解决 `top1 低` 与 `很多 solver 从不被选中` 两个问题。  
**Date**: 2026-04-19  
**Trigger**: `$idea-discovery`  
**Scope**: 以效果为第一目标，弱化 novelty 优先级；重点参考 `CoEKS`、`URS`，并补充推荐系统、算法选择、长尾学习文献。  
**Sources used**: local literature + arXiv + Exa + web + Codex reviewer。`Semantic Scholar` 多次触发 `429`，`DeepXiv` 本机缺少 CLI，因此用 arXiv 直连与 web/Exa 补齐。

---

## Executive Summary

当前 selector 的主要问题不是“再多调一点 loss/seed 就会好”，而是 **结构性地漏召回了很多本来应该被考虑的 solver**。

以当前最强 completed run `R12_init_gaprank_manual_adapter_soft_risk_seed2` 为例：

- `macro_top1 = 0.5326`
- `macro_top2 = 0.8788`
- `macro_top3 = 0.9863`
- `macro_vs_sbs = -0.1188%`
- `macro_vbs_gap_closed = 7.67%`

但更关键的是：

- `82` 个 solver arms 里有 `42` 个最终 **从未被选中**
- 其中有 `27` 个属于 **hidden winners**：oracle `top1` 非零，但 selector `pick rate = 0`
- 代表性例子：
  - `VRPL / MTPOMO`: oracle `12.7%`, pick `0`
  - `VRPBTW / MVMOE`: oracle `11.8%`, pick `0`
  - `VRPTW / MVMOE`: oracle `9.9%`, pick `0`
  - `ATSP / GLOP`: oracle `2.4%`, pick `0`
  - `TSP / INVIT`: oracle `0.7%`, pick `0`

这说明当前主瓶颈是：

1. **candidate recall 不够**  
   真正好的 solver 经常在进入最终决策前就被压没了。
2. **head-arm collapse**  
   模型过度偏向常见强势 solver，长尾 solver 即便在局部区域更优也学不出来。
3. **shared representation 过于均质**  
   不同约束组合、不同 solver niche 的判别信号被共享 trunk 冲淡。

因此，最值得投入的主线不是继续堆单段 softmax，而是：

**两阶段 selector：shortlist retrieval -> solver-aware reranking**  
再叠加：

- **constraint-aware shallow experts**
- **solver-conditional scoring**
- **winner-coverage aware long-tail training**

这也是本轮 literature + reviewer 一致给出的最高 ROI 方向。

---

## Grounded Failure Modes

### F1. 单阶段分类头在做“压缩选择”，不是“找对候选”

当前模型虽然 `macro_vs_sbs` 已经能到负值，但 `top1` 仍明显不高，说明它更多是在做 conservative 的安全选择，而不是准确恢复 oracle winner。

这个症状和推荐系统里“retrieval 不够，rerank 再强也没用”是同一个问题。

### F2. 真实存在的 solver niche 被彻底淹没

从 `R12` 的 `arm_distribution` 和 `method_top1` 看，存在很多 solver：

- 在某些问题上 **确实会赢**
- 但模型永远不选

这不是 calibration 小问题，而是 **支持域没被建出来**。

### F3. 跨问题共享是有用的，但当前共享方式太粗

`CoEKS` 和 `URS` 都说明了一个核心事实：

- 全共享 dense model 容易负迁移
- 完全细粒度 node-level routing 又可能看不见 task/constraint 级结构

你的 selector 现在更像“共享太多、条件化不够细”，因此 solver niche 容易被主流问题分布吞掉。

### F4. 一条 pooled embedding 不够表达实例的“多解倾向”

有些实例并不是只有一个明显 solver family，而是存在多个 plausible solver mode。  
推荐系统里这会被建模成 `multi-interest` 或 `multi-generator`；放到 selector 里，对应的是：

- 先生成多个候选 solver 假设
- 再做更精细的 evaluator / reranker

---

## Literature Landscape

| Paper | Domain | 关键结论 | 对 selector 的直接启发 |
|---|---|---|---|
| `CoEKS` | cross-task VRP solver | constraint-specific experts 有效；浅层 mutual distillation 有益，深层会过度同质化 | scorer 侧要做 constraint-aware experts，且只做浅层共享/蒸馏 |
| `URS` | unified routing solver | UDR + mixed bias module + problem-conditioned parameters 提升跨问题泛化 | encoder 侧要强化 problem-conditioned bias，而不是只靠 problem ID |
| `MMoE` | recsys MTL | 多任务之间需要 gated experts，而不是单一共享 trunk | 不同 problem family / solver family 需要条件化专家 |
| `PLE` | recsys MTL | progressive shared+task-specific extraction 比硬共享更稳 | shared low-level + problem/constraint-specific high-level 更适合你的场景 |
| `LongRetriever` | recsys retrieval | 先做高召回 candidate retrieval，再做后续精排，长序列/多兴趣对召回重要 | 先 shortlist solver，再 rerank；单段 classifier 不够 |
| `Comprehensive List Generation for Multi-Generator Reranking` | reranking | 多个互补 generator 比单 generator 更能覆盖候选空间 | shortlist 不应只有一个 head，而应有多个互补 proposal heads |
| `AW-MoE` | personalized ranking | MoE + 针对长尾/稀疏用户的辅助训练可提升长尾表现 | rare-arm solver 需要独立 coverage pressure 与 tail-aware routing |
| `Class Incremental Learning for Algorithm Selection` | algorithm selection | solver vocabulary/label set 会扩张，rehearsal-based CIL 有效 | 未来增加 solver 时可做 rehearsal / memory bank，而不是完全重训 |
| `Bi-linear Learning to Rank for Algorithm Selection` | algorithm selection | `instance × algorithm` 的双线性打分优于 one-hot algorithm encoding | scorer 要显式建模 `(instance, solver)` 对，而不是只做全局多分类 |
| `Balanced Group Softmax` / `Balanced Meta-Softmax` | long-tail learning | 组内平衡与 prior-shift 修正能缓解 head/tail 压制 | 需要 availability-conditional 的 group-balanced 训练，而非全局均衡 |

---

## Ranked Ideas

### 1. Two-Stage Solver Retrieval -> Solver-Aware Reranking
**Status**: `RECOMMENDED MAIN PATH`

#### Hypothesis
先解决“把该考虑的 solver 找出来”，再解决“在候选里选最优”。  
单段 masked softmax 的问题不只是 ranking error，而是 **many winners never enter the effective decision set**。

#### Architecture
- **Stage 1: shortlist retrieval**
  - shared recall head
  - tail-aware recall head
  - constraint-aware recall head
  - 取 union top-k / threshold，目标是高 oracle winner recall
- **Stage 2: solver-aware reranker**
  - 输入 `(instance embedding h(x), solver embedding e_s, pair features phi(x,s), constraint signature)`
  - 对 shortlist 内 solver 做 pairwise/listwise reranking

#### Why it directly targets both failures
- 对 `top1`：reranker 只在小候选集内做细粒度判别，更容易学准
- 对 zero-picked arms：retrieval stage 明确优化 rare winners recall，不再让它们在全局 softmax 里被淹没

#### Literature basis
- `LongRetriever`: retrieval stage 的高召回必须单独建模
- `Comprehensive List Generation`: 单 generator 不够，多 generator 的互补性决定上限
- algorithm selection 的 `Bi-linear LTR`: `(instance, algorithm)` 对建模比全局类头更自然

#### Recommended metrics
- shortlist winner recall@1 / @3 / @5
- hidden-winner recovery count
- oracle-win-rate-weighted recall of nonzero-win arms
- final reranker top1 conditioned on winner in shortlist

#### Notes for current codebase
当前 `support head` 是这个方向的弱化版，但还不够强，因为它：
- 只有单个 shortlist generator
- 没有显式互补性目标
- 最终 scorer 仍然不够“真正 pairwise reranker”

---

### 2. Constraint-Aware Shallow Experts + URS-Style Conditional Bias
**Status**: `HIGH PRIORITY`

#### Hypothesis
你的 18 个问题本质上是由不同 constraint composition 诱导出来的 solver preference shift。  
如果 trunk 只做硬共享，rare solver 的 niche 很容易被主流问题分布冲淡。

#### Architecture
- shared low-level encoder
- constraint-specific or problem-family-specific experts
- shallow mutual distillation only on lower/shared layers
- URS-style problem-conditioned bias / adapter / parameter modulation
- mixed bias on geometry / asymmetry / relation priors

#### Why it helps
- 对 `top1`：减少不同问题间负迁移
- 对 zero-picked arms：给约束局部 niche 保留单独表达空间

#### Literature basis
- `CoEKS`: experts 按 constraint 组合更合理；深层蒸馏会伤害 specialization
- `URS`: problem-conditioned bias 与 unified representation 能更稳地建模跨问题差异
- `MMoE` / `PLE`: shared + task-specific extraction 优于纯共享

#### Caution
不要做全层 heavy distillation。  
这一点 `CoEKS` 已明确给出负例：深层 homogenization 会把 niche 抹平。

---

### 3. Multi-Generator Shortlist with Coverage / Comprehensiveness Objective
**Status**: `HIGH PRIORITY`

#### Hypothesis
很多 rare winners 消失，不一定是模型完全不知道它们，而是 **单一 shortlist policy 不愿意提它们**。  
更合理的做法是让多个 generator 从不同视角提案，再由 evaluator 统一裁决。

#### Architecture
- generator A: strong shared winner prior
- generator B: tail-arm aware proposal
- generator C: constraint-sensitive proposal
- generator D: uncertainty / diversity oriented proposal
- union candidates -> reranker

#### Training target
- recall objective
- inter-generator complementarity
- list comprehensiveness / coverage objective

#### Why it helps
- 对 `top1`：更容易把真 winner 送进 reranker
- 对 zero-picked arms：允许 rare arms 通过“专门的 generator”进入候选集，而不是依赖主流 head 的仁慈

#### Literature basis
- `Comprehensive List Generation for Multi-Generator Reranking`
- industrial two-stage recsys pipeline

#### My view
如果只能做一个大改，我会把它并到 Idea 1 里，而不是单独作为支线。

---

### 4. Solver-Conditional Scoring Instead of One Global Multiclass Head
**Status**: `HIGH PRIORITY`

#### Hypothesis
当前很多 rare arm 被压死，根本原因之一是它们在全局 softmax 中永远输给强头部 arm。  
把问题改成“给定 instance，单独判断 solver s 是否值得上榜/击败其他候选”，每个 arm 才有真正自己的 decision surface。

#### Architecture
- scorer `g(x, s)` 而不是 `g(x) -> all class logits`
- solver embeddings
- pair features: cost prior, problem family, constraint bits, support stats, geometry stats
- pairwise/listwise losses on shortlist

#### Why it helps
- 对 `top1`：更适合学习 solver-specific preference regions
- 对 zero-picked arms：每个 solver 至少有自己的一条 decision boundary，而不是只能抢一个全局 logit 位置

#### Literature basis
- `Bi-linear Learning to Rank for Algorithm Selection`
- item-aware / target-aware ranking in recsys

---

### 5. Winner-Coverage Aware Long-Tail Training Package
**Status**: `MEDIUM-HIGH PRIORITY`

#### This is not the main fix, but it should be stacked on the main fix

推荐组合：
- availability-conditional balanced group softmax
- winner-balanced minibatching
- support/focal loss on shortlist recall
- oracle-near-winner soft targets
- rare-arm rehearsal buffer
- pick-entropy / coverage regularization

#### Why it helps
- 对 `top1`：减轻 head-arm dominance
- 对 zero-picked arms：明确给 rare winners 生存空间

#### Literature basis
- `Balanced Group Softmax`
- `Balanced Meta-Softmax`
- `AW-MoE` 的长尾/稀疏样本辅助训练

#### Important caveat
这一类方法如果单独使用，容易只把分布“抹平”而不真的提升 cost。  
所以它更适合作为 Idea 1/4 的配套，而不是主方法本体。

---

### 6. Multi-Interest Instance Encoding
**Status**: `MEDIUM PRIORITY`

#### Hypothesis
有些实例天然对应多个 solver family 的 plausible mode。  
用一个 pooled embedding 压成一条向量，会让这种多模态偏好消失。

#### Architecture
- K 个 latent interest tokens
- 每个 token 产生一个 solver preference view
- top views merge into shortlist or reranker features

#### Why it helps
- 对 `top1`：让 ambiguous instances 不再被硬塞到单峰表示里
- 对 zero-picked arms：某些 rare arms 可能只在某个 latent view 下出现

#### Literature basis
- multi-interest retrieval / matching
- `MMoE` / `PLE`

#### Caveat
这条路比前四项更容易 overbuild。  
应该在 two-stage 主线站稳后再加。

---

### 7. Continual Solver Vocabulary / Rare-Arm Memory
**Status**: `BACKUP IDEA`

#### Hypothesis
如果以后还会加入新 solver 或者极少见 solver，selector 不应该每次都全量重训。  
可以保留 rehearsal buffer，做 solver-class incremental updates。

#### Why it matters
这条路对当前 `top1` 提升不是最快，但对未来系统扩展性有价值。

#### Literature basis
- `Class Incremental Learning for Algorithm Selection`

---

## De-Prioritized / Rejected Directions

### A. 继续放大 dense backbone / 纯粹拉长训练
当前 failure mode 更像结构性偏置，而不是明显欠拟合。  
更长训练很可能只会把 head-arm bias sharpen 得更厉害。

### B. 把 8-fold augmentation 当主解决方案
数据增强可以作为 regularization，但它并不直接解决“oracle 会赢但模型永远不选”的根问题。  
在你现有实验里，这条路也没有体现出足够强的信号。

### C. 深层全局 mutual distillation
`CoEKS` 已经很明确地指出深层蒸馏会让专家过于同质化。  
对你这种 zero-picked arm 问题，风险更大。

### D. 只靠 problem ID / prompt token
这可以帮助 disambiguation，但无法解决 hidden winners 和 head-collapse。  
它只能是条件输入，不是主方法。

### E. 继续围绕单段 softmax 做小损失修补
这些改动可以作为配套，但大概率不够把 `top1` 从 `0.53` 推到你想要的量级。

---

## Reviewer Cross-Check

本轮额外请 Codex reviewer 复核后，结论与上面的排序高度一致：

1. `two-stage shortlist + solver-aware reranking`
2. `constraint-aware shallow experts`
3. `solver-conditional scoring`
4. `winner-coverage aware long-tail training`
5. `multi-interest encoding`

reviewer 还特别强调：

- 不要把“更大 backbone / 更长训练”当主修复线
- 不要把 generic augmentation 当第一优先级
- 最有说服力的证据必须是 **hidden-winner recovery**

---

## Recommended Next Experiment Sequence

### P0. 先把评价协议补全
未来所有主实验至少固定输出：

- `macro_top1 / top2 / top3`
- `macro_vs_sbs`
- `macro_vbs_gap_closed`
- `arm_coverage`
- `pick_entropy`
- `hidden_winner_count`
- `oracle-win-rate-weighted recall`
- `shortlist_recall@k`
- `final top1 | winner in shortlist`

### P1. 实现真正的两阶段 selector
- multi-generator shortlist
- solver-conditional reranker
- full-length run，不只 quick val

### P2. 在 reranker 上接 constraint-aware experts
- shared low-level encoder
- constraint/problem-family experts
- shallow knowledge sharing only

### P3. 叠加 long-tail coverage package
- group-balanced loss
- rare-arm rehearsal
- winner-balanced batches

### P4. 如果 P1-P3 仍然受限，再上 multi-interest encoder

---

## What Would Convince Me

如果后续实验想证明“真的在解决问题”，最关键的不是只看一个更低的 `macro_vs_sbs`，而是要同时看到：

1. `hidden winners` 明显减少  
2. `zero-picked arms` 明显减少  
3. `shortlist recall` 上升  
4. `macro_top1` 和 `mean cost` 同时改善  
5. rare-arm 的提升不是靠瞎探索，而是：
   - switch precision 高
   - harm rate 低
   - benefit rate 高

---

## Final Recommendation

如果只保留一句话作为本轮结论：

> **下一阶段最值得做的不是继续打磨单段 classifier，而是把 selector 改造成“多提案 shortlist retrieval + solver-aware reranking”的两阶段结构，并在 scorer 侧加入 constraint-aware shallow experts 与 long-tail coverage training。**

这是当前最贴合你真实 failure mode，也最有希望同时提升 `top1` 和 rare-arm coverage 的路径。

---

## Source Notes

- 重点本地文献：
  - `literature/CoEKS.md`
  - `literature/urs.md`
  - `literature/LITERATURE_REVIEW_v2.md`
- 本轮新增 arXiv PDF：
  - `literature/downloads/2506.01545.pdf`
  - `literature/downloads/2508.15486.pdf`
  - `literature/downloads/2504.15625.pdf`
  - `literature/downloads/2306.05011.pdf`
  - `literature/downloads/2006.10408.pdf`
- Exa 与 web 用于补充推荐系统/算法选择最新线索
- Semantic Scholar exact fetch 因 `429` 未稳定可用
- DeepXiv 在本机未安装 CLI，因此未作为主来源使用
