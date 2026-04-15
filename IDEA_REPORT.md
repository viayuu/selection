# Research Idea Report

**Direction**: unified supervised neural solver selection across multiple routing problems with one model  
**Date**: 2026-04-14  
**Status**: FINAL VERSION

## Executive Summary

The final recommended direction is:

> Learn a single **set-aware, compositional, cross-problem solver selector** that ranks feasible solver entries for each routing instance by jointly modeling:
>
> - a unified instance representation
> - an explicit problem descriptor
> - a structured solver descriptor
> - instance-solver compatibility

This final version sharpens the original idea in three important ways.

First, the output space should **not** be treated as a fixed multiclass label space. Different problems support different solver sets, so the selector should score feasible `(instance, solver)` pairs under a mask.

Second, the model should **not** rely on one shared pooled instance vector alone. It needs a representation stack that combines:

- `NSS`-style hierarchical graph representation
- `URS`-style unified raw schema and problem descriptor
- solver-conditioned instance views

Third, the real scientific contribution is **not** “train one model instead of many”. The paper should argue that unified solver selection is structurally hard because it jointly faces:

- non-uniform candidate sets
- long-tail solver entries
- cross-problem negative transfer
- problem-discrimination requirements
- instance-representation requirements

The final recommended method is:

## Final Recommended Method

**Set-Aware Compositional Universal Selector**

with:

1. unified raw instance schema
2. hierarchical graph-native instance encoder
3. explicit problem descriptor
4. solver descriptor and solver embedding
5. solver-conditioned interaction module
6. shared/private/semantic expert stack
7. masked ranking head over feasible solver entries

This is the cleanest design that incorporates the most valuable lessons from:

- `NSS`
- `URS`
- `CoEKS`
- recommendation systems
- multi-task learning

while still remaining experimentally realistic.

## Double-Check Adjustment

After a final consistency check, the main idea itself remains reasonable and does **not** need to be changed.  
The only adjustment worth making is to separate:

- the **core method that should be implemented first**
- the **full extension version**

to avoid over-designing v1.

### Core method

The first implementation should contain only:

1. unified raw instance schema
2. hierarchical graph-native encoder
3. explicit problem descriptor
4. solver descriptor
5. masked `(instance, solver)` scorer

This is already enough to test the main claim:

- unified cross-problem solver selection should be modeled as masked compatibility ranking

### Full extension

Only after the core method is stable should we add:

1. solver-conditioned interaction
2. shared/private/semantic experts
3. stronger imbalance-specific training tricks

So the final report still recommends the same direction, but the implementation order is now more disciplined.

## Problem Anchor

Construct one supervised selector that can choose among heterogeneous solver entries across multiple routing problems, instead of training one selector per problem.

The current target problem family includes:

- `tsp`
- `cvrp`
- `atsp`
- `pctsp`
- `16` CVRP variants

The selector must work under:

- variable feasible solver sets
- heterogeneous problem attributes
- heterogeneous solver support
- scale diversity

## Why This Problem Is Hard

The full problem is harder than standard per-problem solver selection because it combines three difficulties.

### 1. Output-space difficulty

Different problems do not share the same valid solver set.

So the model cannot safely assume:

- one fixed global softmax over all solvers

The actual challenge is:

- **dynamic feasible candidate sets**

### 2. Optimization difficulty

The pooled model must learn from:

- head problems such as `TSP/CVRP`
- smaller problem families such as `ATSP/PCTSP`
- very sparse solver coverage for some variant families

So the training problem includes:

- solver-frequency imbalance
- problem imbalance
- cross-problem gradient conflict

### 3. Representation difficulty

The model must learn instance representations that are:

- shared enough to transfer across problems
- discriminative enough to distinguish problem families
- rich enough to reflect solver-relevant structure
- adaptable enough to support candidate-dependent scoring

That means the right answer is not:

- hand-crafted instance features only
- one shared pooled graph vector only
- problem id only

## Final Literature Conclusion

### What NSS contributes

`NSS` is the correct starting point for selector design because it demonstrates that:

- instance-level solver complementarity is real
- ranking supervision is strong
- graph-native instance encoding is better than relying only on hand-crafted features
- hierarchical graph pooling improves generalization

Its most reusable representation lesson is:

- **instance representation for solver selection should be learned from the instance graph itself**

Its main limitations for the current agenda are:

- it is still essentially per-problem
- it is built around `TSP/CVRP`
- its representation is solver-agnostic

### What URS contributes

`URS` is the strongest local reference for unified routing representation because it contributes:

- a unified raw feature schema
- explicit problem descriptors
- problem-conditioned modulation

Its key lesson is:

- a unified model still needs explicit problem semantics

For the selector, this translates to:

- use a unified raw schema
- use problem descriptors
- but go one step further and make the representation also **solver-conditioned**

### What CoEKS contributes

`CoEKS` contributes the strongest architectural lesson about cross-problem representation:

- lower layers should share
- deeper layers should specialize
- semantic expert organization is better than naive full sharing

Its key lesson for the selector is:

- the latent space should separate transferable geometry from problem-specific semantics

### What recommendation systems contribute

Recommendation systems contribute more than a general analogy. After a deeper pass through the multi-domain and cross-domain recommendation literature, there are four reusable patterns that are directly relevant here.

`Query-item scoring`:

- routing instance = query
- solver entry = item
- selector = compatibility scorer

`Candidate-aware representation`:

- a good instance representation often depends on which solver we are evaluating

This is the main reason the selector should move from:

- `h(x)` only

to:

- `h(x, a)` or solver-conditioned pooling

`Shared-private / domain-aware representation`:

- `MMoE` shows that when domains/tasks are only partially related, different gates should read different mixtures of shared experts instead of forcing one universal path.
- `PLE` is even more relevant because it was designed to reduce negative transfer and the `seesaw phenomenon`; its key idea is to progressively separate shared and task-specific information rather than mix them once at the bottom.
- `STAR` shows that a strong industrial compromise is not "all shared" or "all separate", but a shared center plus domain-specific parameters.
- `HAMUR` sharpens this further: static domain-specific parameters are often too rigid, and domain-shared hypernetworks can dynamically generate adapters for each domain.

Translation:

- different routing problems should not be forced through one identical representation path
- shared geometry can stay low-level
- higher-level semantics should be problem-conditioned
- lightweight conditional adapters may be a strong first implementation before full expert decomposition

`Minor-domain / long-tail imbalance handling`:

- `AdaSparse` uses domain-aware neuron weighting and adaptive sparsity so each domain keeps only the subnetwork capacity it truly needs.
- `MAMDR` is also useful because it explicitly separates:
  - domain conflict in shared parameters
  - overfitting in sparse-domain specific parameters
  and addresses them with domain negotiation and domain regularization.
- `AREAD` is especially relevant because it explicitly studies dozens of domains, uses hierarchical expert masks per domain, and augments minor domains with counterfactual assumptions.
- `PAAC` is valuable from the long-tail side: it argues that head items should not only dominate training but should provide supervisory signal to improve tail representations, and it explicitly tries to reduce representation separation between head and tail items.

Translation:

- rare solver entries should not merely be reweighted; they should borrow structured signal from frequent related solver entries
- small problem families should get protected capacity and protected gradient budget
- minor problems may need augmentation or prototype-based transfer rather than naive pooling
- when shared parameters and sparse-domain-specific parameters fail for different reasons, they should be regularized differently rather than treated as one generic imbalance problem

`Shared-vs-specific preference disentanglement`:

- `CoPD` explicitly extracts shared attributes to guide the learning of shared preferences, while disentangling the domain-specific part, and even uses popularity-weighted objectives.
- the more recent `Joint Identifiability` line argues that perfect disentanglement is unrealistic, and instead proposes organizing representations by depth: shallower levels capture cross-domain stable structure, deeper levels capture domain-variant structure under an identifiable relationship.
- `MIND` remains the clearest candidate-aware representation paper: one user vector is insufficient, so the model keeps multiple interest vectors and lets the candidate item decide which one matters.

Translation:

- instance representation for unified solver selection should probably have:
  - a shallow shared subspace
  - a deeper problem-variant subspace
  - multiple semantic view tokens
- and the final useful instance view should be selected or mixed by the candidate solver

### What multi-task learning contributes

Multi-task learning gives the training recipe:

- `GradNorm` / `IMTL`: loss balancing
- `PCGrad`: conflict mitigation
- `Recon`: privatize highly conflicting upper layers
- `Deep-AMTFL`: asymmetric transfer from rich tasks to sparse tasks

These are not the main scientific novelty, but they are the most plausible tools to stabilize the unified selector.

## Final Method Thesis

The paper’s main thesis should be:

> Unified cross-problem solver selection should be formulated as a **masked compatibility-ranking problem over solver entries**, and solved by a representation that is simultaneously:
>
> - graph-native
> - hierarchy-aware
> - problem-aware
> - solver-aware

This is stronger and more defensible than:

- “we trained one selector on many problems”

because it gives a structural reason for:

- why naive pooling fails
- why fixed-head classification is wrong
- why a compositional representation is needed

## Final Model Design

The final recommended architecture is:

## Set-Aware Compositional Universal Selector

with seven modules.

### Module A: Unified raw schema

Every instance should be mapped into a common schema containing as many of these fields as relevant:

- coordinates or asymmetric identifier
- demand
- prize
- penalty
- earliest time
- latest time
- service time
- node-role indicators
- depot / route indicators

This is inherited from `URS`.

### Module B: Graph-native instance encoder

Use a graph-native encoder, not a hand-crafted-feature pipeline.

Core requirements:

- node-aware
- edge-aware
- variable-size support
- asymmetric support
- attribute-rich support

This is inherited from `NSS` and reinforced by graph-based algorithm-selection work.

### Module C: Hierarchical multi-scale pooling

Use hierarchical pooling to summarize the instance across levels instead of only flat pooling.

Desired behavior:

- capture cluster representatives
- capture boundary/global shape
- capture large-scale organization

This is the strongest reusable `NSS` design.

### Module D: Problem descriptor encoder

Build a problem descriptor using:

- problem family token
- constraint multi-hot
- symmetry/asymmetry flag
- depot style
- route organization
- continuous global statistics

Recommended statistics:

- scale
- demand mean/std
- TW tightness
- route-limit tightness
- asymmetry intensity
- prize density
- spatial clustering statistics
- support-set size

This descriptor should condition:

- expert routing
- score calibration
- residual adaptation

The deeper recommendation literature suggests this problem descriptor should not be just a flat one-hot or multi-hot.

Better interpretation:

- shallow components capture cross-problem stable structure
- deeper components capture problem-variant semantics

This is closely aligned with:

- `PLE`
- `HAMUR`
- `CoPD`
- `Joint Identifiability`

### Module E: Solver descriptor encoder

Each solver entry should be represented by:

- a learnable id embedding
- a structured capability descriptor

Recommended capability fields:

- supported problem families
- supported constraints
- checkpoint scale range
- constructive vs iterative
- neural vs heuristic
- symmetry compatibility
- prize/TW/asymmetry support

This helps the model generalize better than using solver ids alone.

### Module F: Solver-conditioned interaction

This is the most important new representation upgrade beyond `NSS`.

Instead of using only one global instance vector, compute a solver-conditioned representation through:

- cross-attention from solver token to instance tokens
- or gated pooling over semantic instance views

That lets the model ask:

- which aspect of this instance matters for this solver?

This is the recommendation-style candidate-aware step, and it is the selector-side analogue of:

- `MIND`-style multi-interest routing
- item-aware user representation refinement

### Module G: Shared/private/semantic experts

The representation trunk should not be fully shared from bottom to top.

Recommended structure:

`Lower layers`:

- shared
- learn transferable geometry and coarse structure

`Middle layers`:

- semantic experts by constraint family

Suggested semantic experts:

- geometry
- capacity/load
- time windows
- prize/penalty
- asymmetry
- pickup-delivery
- route organization
- scale/distribution

`Upper layers`:

- problem-family-private adapters or expert heads

This is the cleanest integration of:

- `CoEKS`
- `PLE`
- `STAR`
- `HAMUR`

## Output Formulation

The selector should **not** use one fixed multiclass head over the global solver union.

Instead:

- build feasible solver set `A(x)`
- compute `s(x, a)` for every `a ∈ A(x)`
- rank feasible solvers
- choose top-1 or top-k depending on strategy

So the prediction object is:

- a ranked feasible candidate set

not:

- a flat class over all solvers

## Final Training Objective

The main training objective should be ranking-based, not plain classification.

Recommended priority:

`Primary loss`:

- masked listwise or pairwise ranking loss over feasible solver entries

`Secondary loss`:

- masked top-1 classification loss

`Auxiliary losses`:

- problem-family prediction
- constraint reconstruction
- solver-support prediction
- instance-statistics regression

The auxiliary losses make the representation more discriminative and stable.

## Final Imbalance Strategy

The output imbalance issue should be split into three subproblems and solved separately.

### Candidate-set imbalance

Solve with:

- feasible mask
- masked ranking loss
- candidate-set-size-aware normalization

### Solver-frequency imbalance

Solve with:

- solver-entry-balanced weighting
- rare-entry regularization
- family-level prototype sharing for sparse solver entries

The closest recommender analogue is:

- `PAAC`

where frequent entities are used as structured supervisory signal for rare entities instead of merely being down-weighted.

### Cross-problem optimization imbalance

Solve with:

- problem-balanced batching
- `GradNorm` or `IMTL`
- `PCGrad` if gradients conflict
- privatized upper layers if conflict persists

The closest recommender analogues here are:

- `PLE` for reducing the seesaw phenomenon
- `STAR` for shared-center plus domain-specific adaptation
- `AREAD` for domain-specific hierarchical expert masks

## Final Representation Answer

The direct answer to “how should an instance be represented?” is:

> Not as one hand-crafted feature vector, and not as one flat graph vector only.
>
> It should be represented as a **bundle** containing:
>
> - unified node/edge schema
> - hierarchical graph summaries
> - explicit problem descriptor
> - continuous instance statistics
> - semantic view tokens
>
> and this bundle should be refined conditionally by the candidate solver.

The deeper recommendation literature suggests one more nuance:

- do not insist on perfectly disentangling "shared" and "specific" information everywhere

The better target is:

- shallow shared structure
- deeper problem-variant structure
- solver-conditioned readout over multiple semantic views

## Final Scientific Claim

The paper should make one main claim and two supporting claims.

`Main claim`:

- Unified cross-problem solver selection is best formulated as **masked compatibility ranking over solver entries**, not fixed-head multiclass prediction.

`Supporting claim 1`:

- The right instance representation for unified selection is **graph-native, hierarchical, problem-aware, and solver-conditioned**.

`Supporting claim 2`:

- Explicit handling of:
  - candidate-set mismatch
  - solver long-tail imbalance
  - cross-problem gradient conflict
  is necessary to make one shared selector work well across heterogeneous routing problems.

## Minimal Publishable Method Package

To make the final paper convincing, the method package should include:

1. a mixed-problem labeled dataset with feasible masks
2. a masked `(instance, solver)` scorer
3. a strong instance representation based on:
   - graph-native hierarchical encoding
   - problem descriptors
   - solver descriptors
4. evidence that this beats:
   - per-problem selectors
   - naive pooled selector with problem id only
   - pooled masked scorer without solver descriptors

The paper does **not** need to prove every advanced extension at once.

## Recommended Baselines

The baseline ladder should be:

1. per-problem selector
2. pooled selector with problem id only
3. pooled fixed-head selector with feasible masking
4. masked `(instance, solver)` scorer without solver metadata
5. masked scorer with solver metadata
6. masked scorer + problem descriptor
7. full model with solver-conditioned interaction

## Recommended Ablations

The most important ablations are:

`Representation ablations`:

- flat graph encoder vs hierarchical encoder
- no problem descriptor vs problem descriptor
- no statistics token vs statistics token
- no solver-conditioned interaction vs solver-conditioned interaction

`Architecture ablations`:

- fully shared trunk vs shared/private trunk
- unguided experts vs semantic experts

`Training ablations`:

- classification vs ranking
- no balancing vs balanced batching
- no gradient conflict handling vs `GradNorm` / `PCGrad`

## Recommended Metrics

The main metrics should be:

- normalized regret
- top-1 hit rate
- macro average across problems
- macro average across solver-frequency buckets
- performance by candidate-set size bucket

The paper should avoid focusing only on:

- overall micro average

because that would hide minor-problem degradation.

## Double Check Notes

I re-checked the final version against the local `NSS`, `URS`, and `CoEKS` materials and made three cleanups:

1. removed dependence on old supplementary idea files so this report is self-contained
2. kept only one final method direction, deleting earlier backup / fallback ideas from the main report
3. kept the final method grounded in the strongest verified local evidence:
   - `NSS`: hierarchical graph instance representation + ranking
   - `URS`: unified raw schema + explicit problem descriptor
   - `CoEKS`: shared/lower, specialized/upper, semantic expert organization

Remaining uncertainty is not conceptual but empirical:

- whether the full solver-conditioned interaction is already needed in v1
- or whether the masked `(instance, solver)` scorer plus problem/solver descriptors already gives most of the gain

So the correct implementation order is still:

1. masked scorer baseline
2. problem/solver descriptors
3. hierarchical encoder
4. solver-conditioned interaction

## Risks

The main risks of the final direction are:

`Risk 1`:

- The full architecture may look like “NSS + URS + recommender systems”.

Response:

- the paper should emphasize the new formulation:
  - masked cross-problem compatibility ranking

`Risk 2`:

- The richer model may overcomplicate the first experiment.

Response:

- implement in stages and make the masked scorer baseline strong first

`Risk 3`:

- Data generation may remain the bottleneck.

Response:

- keep the first solver pool high-overlap and platform-native

## Final Recommendation

Proceed with:

- the unified selector line

but **not** with the vague formulation of “one selector for all problems”.

Proceed with the sharper formulation:

> one masked compatibility-ranking selector over `(instance, solver)` pairs, using a hierarchical graph-native, problem-aware, solver-conditioned representation.

This is the strongest and cleanest final synthesis of the current idea-discovery round.

## Immediate Next Steps

The next concrete steps should be:

1. freeze the first solver-entry pool and feasible masks
2. build the first mixed-problem labeled dataset
3. implement the masked `(instance, solver)` scorer baseline
4. add:
   - problem descriptors
   - solver descriptors
   - hierarchical encoder
5. only then add solver-conditioned interaction and expert decomposition

## Source Map

Key local references:

- [nss.md](/public/home/zhoucl/shiys/已有文献/nss.md)
- [urs.md](/public/home/zhoucl/shiys/已有文献/urs.md)
- [moe.md](/public/home/zhoucl/shiys/已有文献/moe.md)
- [model.py](/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/model.py)
- [dataset.py](/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/dataset.py)
- [Model.py](/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv/Model.py)
- [Model_LIB.py](/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv/Model_LIB.py)
- [multi_hot_set.py](/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv/multi_hot_set.py)

External anchors:

- NSS: https://arxiv.org/abs/2410.09693
- URS: https://arxiv.org/abs/2509.23413
- CoEKS: local summary in [moe.md](/public/home/zhoucl/shiys/已有文献/moe.md)
- MMoE: https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
- STAR: https://arxiv.org/abs/2101.11427
- AdaSparse: https://arxiv.org/abs/2206.13108
- PLE: https://recsys.acm.org/recsys20/session-4/
- AREAD: https://arxiv.org/abs/2412.11905
- MAMDR: https://arxiv.org/abs/2202.12524
- MIND: https://arxiv.org/abs/1904.08030
- HAMUR: https://arxiv.org/abs/2309.06217
- PAAC: https://arxiv.org/abs/2405.20718
- CoPD: https://arxiv.org/abs/2410.20580
- Joint Identifiability for CDR: https://arxiv.org/abs/2411.17361
- GOAL: https://openreview.net/forum?id=9E158CD5hD
- Prompt Learning for Generalized Vehicle Routing: https://www.ijcai.org/proceedings/2024/771
- BQ-NCO: https://arxiv.org/abs/2301.03313
- GINES 2025: https://www.scitepress.org/Papers/2025/131534/131534.pdf
- Seiler et al. 2020: https://arxiv.org/abs/2006.15968
