# Research Idea Report

**Direction**: Unified supervised neural solver selection across multiple routing problems with one model  
**Generated**: 2026-04-14  
**Pipeline**: local literature + code reading -> idea generation -> novelty screening -> reviewer-style critique -> local refine pipeline  
**Ideas evaluated**: 7 generated -> 5 survived filtering -> 0 piloted -> 1 recommended

## Executive Summary

The strongest current direction is **a unified supervised selector that scores problem instance and candidate solver jointly**, instead of training a separate selector per problem as in NSS. The key opportunity is not merely "one model for all problems", but **compositional compatibility learning**: use a unified instance representation plus explicit problem/solver capability descriptors so the selector can handle heterogeneous problem types and heterogeneous feasible solver sets in a single framework.

At the current stage, the main blocker is **label data generation**, not architecture complexity. The label dataset still needs to be generated with `EasyNCO`, so the most valuable immediate next step is to build a small but reliable mixed-problem label-generation pipeline and use it to compare a few safe baselines before committing to a heavier model.

## Literature Landscape

### 1. What NSS already solved, and what it did not solve

`NSS` established the right meta-problem: neural solvers exhibit strong instance-level complementarity, so selector learning is worthwhile. It also showed that ranking-based supervision and stronger instance encoders help selector quality. However, the practical formulation remains close to **per-problem solver selection**, mainly on `TSP` and `CVRP`, with a problem-specific modeling mentality.

For the new direction, NSS should be treated as a **starting point**, not the target end state. Its biggest reusable lessons are:

- selector learning is meaningful
- ranking is often better than plain top-1 classification
- hierarchical / graph-aware instance encoding matters

Its main limitation for the new agenda is:

- it does not directly solve **cross-problem unified selector learning**

### 2. Cross-problem routing literature focuses on unified solvers, not unified selectors

Recent routing papers already moved toward **single-model multi-problem routing**, but they mostly study **one unified constructive solver**, not a selector over a heterogeneous solver zoo.

The local `URS` material and code are especially important because they expose three mechanisms that are directly relevant to the new direction:

- unified data representation
- active-feature / multi-hot problem representation
- lightweight problem-conditioned parameter modulation

Similarly, `MVMoE`, `CaDA`, `RouteFinder`, `CoEKS`, and `URS` all push toward multi-problem routing, but they answer a different question:

- "How can one neural solver solve many routing problems?"

The new question is:

- "How can one selector choose among many heterogeneous solvers across many routing problems?"

That gap looks real and publishable if framed carefully.

### 3. Recommendation systems already solved several structural subproblems

The recommendation literature is highly relevant because it routinely handles:

- one shared model across many domains
- output spaces that differ by domain or candidate set
- negative transfer between related but non-identical tasks
- domain imbalance and minor-domain degradation

The most relevant patterns are:

- `MMoE`: shared experts with task-specific gates
- `PLE`: progressively separated shared/private experts to reduce negative transfer
- `STAR`: one shared center plus domain-specific parameters
- `AdaSparse` / `AREAD`: multi-domain adaptation and domain imbalance handling
- large-scale retrieval/ranking systems: query-item scoring instead of separate heads

The strongest conceptual translation into the current problem is:

- treat the **routing instance as query**
- treat the **candidate solver as item**
- learn a shared compatibility scorer over `(instance, solver)`
- use a **feasible mask** for solver support mismatch

### 4. The actual open gap

After combining the local reading and external literature, the clearest open gap is:

> There appears to be no well-established paper that directly studies a **single supervised selector** over a **heterogeneous solver pool** across **multiple routing problem families**, while explicitly handling **non-uniform solver support** and **cross-problem shared representation**.

This is stronger than merely saying "train one model instead of many". The paper must show why unified selection is structurally hard and why the proposed representation/conditioning strategy is the right way to address it.

## Ranked Ideas

### Idea 1: Compositional Problem-Solver Compatibility Selector (RECOMMENDED)

- **One-sentence summary**: Learn a single supervised selector that scores `(instance, solver)` pairs across multiple routing problems by jointly encoding a unified instance representation, a problem attribute representation, and a solver capability representation.
- **Hypothesis**: A selector that models explicit compatibility between problem attributes and solver capabilities will outperform both per-problem selectors and naive pooled selectors, while naturally handling variable candidate-set size via masking.
- **Core mechanism**:
  - unified instance input
  - coarse problem family token plus active-feature multi-hot
  - solver metadata / capability vector plus learnable solver embedding
  - shared masked scorer `s(x, a)`
- **Minimum viable experiment**:
  - generate a small label dataset with `EasyNCO` on `tsp`, `atsp`, `pctsp`, and 4 representative CVRP variants
  - pick a small solver zoo with partial overlap
  - compare:
    - per-problem selector
    - pooled selector with only problem id
    - pooled masked scorer without solver metadata
    - full compositional compatibility model
- **Expected contribution type**: new method plus diagnostic empirical finding
- **Novelty**: 8/10
  - **Closest work**:
    - NSS: instance-level solver selection, but essentially problem-specific
    - URS / RouteFinder / CaDA / CoEKS: multi-problem unified solvers, not selectors
    - recommender MTL: structurally similar, but not in solver selection for routing
- **Feasibility**: medium
  - technically feasible with current codebase
  - bottleneck is label generation, not model coding
- **Pilot result**: SKIPPED
  - label dataset has not yet been generated
  - per skill rules, this should be marked as "needs manual pilot"
- **Reviewer's likely objection**:
  - "This looks like NSS plus URS plus recommender-style masking. Why is this not just engineering?"
- **Why it is still worth doing**:
  - because the paper can make a sharper claim than "one model for all":
    - heterogeneous solver support is an **open-set candidate problem**
    - problem and solver are both **compositional objects**
    - compatibility learning gives a principled route to extension and masking

### Idea 2: Default-Anchored Universal Selector

- **One-sentence summary**: Use a robust per-problem default solver as anchor, then learn a single cross-problem model that decides when to stay, when to switch, and optionally which solver to switch to.
- **Hypothesis**: Anchoring all problems to a stable default policy simplifies output imbalance and may outperform more complex full-ranking approaches under limited labels.
- **Minimum viable experiment**:
  - on the same mixed-problem dataset, compare:
    - per-problem single best
    - pooled stay/switch model
    - pooled one-vs-default chooser
- **Expected contribution type**: strong baseline or safe first paper
- **Novelty**: 5.5/10
- **Feasibility**: high
- **Pilot result**: SKIPPED
- **Reviewer's likely objection**:
  - "This is practical, but maybe too incremental unless it reveals a surprising scaling law or generalization finding."
- **Why it matters**:
  - it is the most reliable baseline and the fastest way to get signal before investing in the full compatibility model

### Idea 3: Shared-Private Universal Selector with PLE / STAR Style Adapters

- **One-sentence summary**: Start from the unified selector, then introduce shared/private experts or domain-conditioned adapters to reduce negative transfer across problem families.
- **Hypothesis**: If naive pooled training underperforms due to cross-problem interference, lightweight shared/private routing will recover performance without losing the one-model property.
- **Minimum viable experiment**:
  - use the full unified selector as base
  - compare:
    - shared-bottom
    - problem-conditioned residual/adapters
    - PLE-like shared/private expert split
- **Expected contribution type**: architectural refinement
- **Novelty**: 4.5/10 standalone, 7/10 as a validated extension after Idea 1
- **Feasibility**: medium
- **Pilot result**: SKIPPED
- **Reviewer's likely objection**:
  - "This is a direct transfer of recommender-system MTL ideas. Why is routing-specific insight needed?"
- **Recommendation**:
  - keep as stage-2 plan, not the first thesis

### Idea 4: Zero-Shot Solver / Problem Extension via Capability Tokens

- **One-sentence summary**: Train the selector to consume explicit problem and solver capability descriptors so that new solvers or newly added problem types can be inserted with minimal or no full retraining.
- **Hypothesis**: Explicit compositional descriptors can support extension to new solver/problem combinations better than fixed IDs alone.
- **Minimum viable experiment**:
  - hold out one solver during training and test whether metadata-aware scoring can still place it reasonably
  - optionally hold out one problem subtype
- **Expected contribution type**: generalization / extension capability
- **Novelty**: 8/10 if it works
- **Feasibility**: medium-low
- **Pilot result**: SKIPPED
- **Reviewer's likely objection**:
  - "This may be too ambitious before the base selector is stable."
- **Recommendation**:
  - keep as secondary claim or follow-up experiment for Idea 1

### Idea 5: Label-Efficient Universal Selector with Partial Solver Evaluation

- **One-sentence summary**: Reduce expensive label generation by evaluating only a subset of feasible solvers per instance, then train the selector from partial rankings or uncertain pairwise comparisons.
- **Hypothesis**: Most of the benefit of full solver-zoo supervision can be retained with much cheaper labeling if solver evaluation is scheduled adaptively.
- **Minimum viable experiment**:
  - compare full labels vs partial labels generated by:
    - random subset
    - anchor-plus-challengers
    - uncertainty-guided evaluation
- **Expected contribution type**: data efficiency / systems contribution
- **Novelty**: 7/10
- **Closest work**:
  - active learning for expensive algorithm selection
  - algorithm portfolios with partial performance data
- **Feasibility**: medium
- **Pilot result**: SKIPPED
- **Reviewer's likely objection**:
  - "This may become a second paper or a systems appendix unless tied tightly to the unified selector thesis."
- **Recommendation**:
  - excellent backup or secondary contribution after baseline unified selector exists

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| Purely transplant PLE / MMoE / STAR into the selector as the main paper | Too easy for reviewers to dismiss as architecture borrowing without a selector-specific scientific question |
| Runtime-aware top-k universal selector as the main thesis | More natural as a later extension after the basic one-model selector is proven |
| Continue RL / bandit / contextual bandit work | No longer aligned with the current research pivot |

## Novelty Check Summary

### Closest Prior Work

| Paper | Year | Venue / Source | Overlap | Key Difference |
|-------|------|----------------|---------|----------------|
| Neural Solver Selection for CO | 2025 | ICML / arXiv | instance-level solver selection | mainly per-problem, not unified cross-problem selector |
| Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization | 2024 | arXiv | single model across routing tasks | unified solver, not selector |
| RouteFinder | 2024 | arXiv / workshop | multi-problem routing foundation model | unified solver, no heterogeneous solver selection |
| CaDA | 2025 | PMLR | cross-problem routing solver | unified solver, no solver-zoo selection |
| URS | 2025 | arXiv / local code | unified data representation for many VRPs | unified solver, not selector |
| CoEKS | 2026 | ICLR | structured expert decomposition across constraints | unified solver architecture, not solver selection |
| MMoE / STAR / AREAD / AdaSparse | 2018-2025 | recsys / KDD / arXiv | shared/private modeling, domain adaptation, imbalance handling | recommendation setting, not routing solver selection |

### Overall Novelty Assessment

- **Idea 1 overall**: `PROCEED WITH CAUTION`
- **Reason**:
  - The space is promising, but "one shared selector" by itself is not enough for novelty.
  - The paper must emphasize:
    - heterogeneous solver support
    - compositional problem and solver descriptors
    - shared compatibility learning rather than naive pooling
    - optionally extension to new solver/problem combinations

## Reviewer-Style Critique

### What a strong reviewer is likely to ask

1. Why is the main contribution more than "training one selector across many problems"?
2. Why not simply use:
   - a pooled model with problem id
   - separate heads
   - a recommender-style baseline with masking?
3. What is the strongest evidence that explicit problem/solver compositional descriptors matter?
4. Does the shared model actually beat or at least match per-problem specialists?
5. If some problem families degrade, why is the one-model constraint still justified?

### Minimum package that would satisfy the reviewer

For Idea 1, the minimum convincing package is:

- a reliable mixed-problem label dataset generated with `EasyNCO`
- a clear solver-support mask and solver metadata table
- strong baselines:
  - per-problem selector
  - pooled selector with only problem id
  - pooled selector with mask but no solver metadata
- one key ablation:
  - with vs without compositional problem/solver descriptors
- one transfer-style experiment:
  - either new solver insertion or held-out problem subtype

## Pilot Experiment Results

| Idea | Status | Reason |
|------|--------|--------|
| Idea 1 | SKIPPED | mixed-problem label dataset not generated yet |
| Idea 2 | SKIPPED | same blocker |
| Idea 3 | SKIPPED | same blocker |

Following the skill rules, these should be treated as **needs manual pilot** rather than "negative".

## Suggested Execution Order

1. Build a small `EasyNCO`-based mixed-problem label pipeline before any full modeling.
2. Start with `Idea 2`-style safe baselines:
   - per-problem selector
   - pooled masked ranker
   - default-anchored switch baseline
3. Move to `Idea 1` as the main thesis:
   - compositional problem-solver compatibility selector
4. Only add `Idea 3` shared/private experts if the shared selector shows negative transfer.
5. Treat `Idea 5` partial-label generation as a second-stage efficiency extension.

## Recommended Next Steps

- [ ] Verify the exact initial `16` CVRP variants to include
- [ ] Build an `EasyNCO` problem-method coverage matrix
- [ ] Generate a small pilot label dataset on 4-6 problems first
- [ ] Implement safe baselines before the full compatibility model
- [ ] Use the resulting signal to decide whether Idea 1 stays the main thesis

## Key Sources

- Local notes:
  - `/public/home/zhoucl/shiys/已有文献/nss.md`
  - `/public/home/zhoucl/shiys/已有文献/urs.md`
  - `/public/home/zhoucl/shiys/已有文献/moe.md`
- Local code:
  - `/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv`
  - `/public/home/zhoucl/shiys/EasyNCO`
  - `/public/home/zhoucl/shiys/old_try/2l2r`
- External papers:
  - NSS: https://arxiv.org/abs/2410.09693
  - MTL Routing (MTPOMO-style): https://arxiv.org/abs/2402.16891
  - RouteFinder: https://arxiv.org/abs/2406.15007
  - CaDA: https://proceedings.mlr.press/v267/li25bi.html
  - URS: https://arxiv.org/abs/2509.23413
  - GOAL: https://openreview.net/pdf?id=z2z9suDRjw
  - MMoE: https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
  - STAR: https://arxiv.org/abs/2101.11427
  - AREAD: https://arxiv.org/abs/2412.11905
  - AdaSparse: https://arxiv.org/abs/2206.13108
  - PLE: https://doi.org/10.1145/3383313.3412236
  - Active learning for expensive algorithm selection: https://arxiv.org/abs/1909.03261
