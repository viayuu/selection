# Unified Selector Reference Code Index

**Root**: `/public/home/zhoucl/shiys/reference_repos/unified_selector`
**Date**: 2026-04-14

This directory collects codebases that are useful for implementing the current final idea in [IDEA_REPORT.md](/public/home/zhoucl/shiys/IDEA_REPORT.md).

## 1. Local code that already existed

These were already present locally and are still the most important direct references:

- `NSS` local code:
  - `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection`
  - use for:
    - hierarchical graph encoder
    - ranking-based selector
    - instance feature extraction baseline

- `URS` local code:
  - `/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv`
  - use for:
    - unified raw schema
    - problem descriptor
    - problem-conditioned modulation

- `EasyNCO` local code:
  - `/public/home/zhoucl/shiys/EasyNCO`
  - use for:
    - solver execution
    - dataset generation
    - label generation

## 2. Downloaded routing / CO code

### `goal-co`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/goal-co`
- Commit:
  - `f9bd6ae5cc540d664f6da908d375a4d45ab82805`
- Repo:
  - `https://github.com/naver/goal-co.git`
- Reference:
  - `GOAL: A Generalist Combinatorial Optimization Agent Learner`
- Why useful:
  - generalist CO backbone
  - multi-problem input/output organization
  - lightweight task specialization patterns

### `rl4co`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/rl4co`
- Commit:
  - `7a1539678b6a81cc68b4069440aa5e98f6931f6c`
- Repo:
  - `https://github.com/ai4co/rl4co.git`
- Reference:
  - `RL4CO`
- Why useful:
  - broad routing/CO environment and model infrastructure
  - useful for checking task abstractions and routing problem interfaces

### `routefinder`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/routefinder`
- Commit:
  - `cc3ab078650e8db62c0562a60f421defe38d5ae3`
- Repo:
  - `https://github.com/ai4co/routefinder.git`
- Reference:
  - `RouteFinder`
- Why useful:
  - cross-problem routing foundation model
  - useful for problem representation, prompt/task conditioning, and routing abstraction

### `bq-nco`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/bq-nco`
- Commit:
  - `b7f927faf1d87dca699361312471c10ecc8dd346`
- Repo:
  - `https://github.com/naver/bq-nco.git`
- Reference:
  - `BQ-NCO`
- Why useful:
  - invariance / quotienting ideas
  - useful for thinking about nuisance variation in instance representation

## 3. Downloaded recommendation / multi-domain code

### `HAMUR`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/HAMUR`
- Commit:
  - `53d8dd588282bc288f2621b8fa85e2df9b910e10`
- Repo:
  - `https://github.com/Applied-Machine-Learning-Lab/HAMUR.git`
- Reference:
  - `HAMUR: Hyper Adapter for Multi-Domain Recommendation`
- Why useful:
  - hypernetwork-generated domain adapters
  - clean reference for lightweight conditional adaptation

### `MAMDR`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/MAMDR`
- Commit:
  - `0226c7f196700b08585b11e738c6d70cdbc9de18`
- Repo:
  - `https://github.com/RManLuo/MAMDR.git`
- Reference:
  - `MAMDR`
- Why useful:
  - domain conflict vs sparse-domain overfitting separation
  - useful for imbalance-specific training design

### `FuxiCTR`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/FuxiCTR`
- Commit:
  - `b7dff736885fdb8f59387d82d08219ad2e4cae50`
- Repo:
  - `https://github.com/reczoo/FuxiCTR.git`
- Reference type:
  - mature implementation library
- Confirmed useful modules:
  - `MMoE`
  - `PLE`
- Why useful:
  - direct code reference for shared/private multi-task architectures
  - easier to borrow module structure than from paper pseudocode alone

### `DeepCTR-Torch`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/DeepCTR-Torch`
- Commit:
  - `f6854257b1fc29caab655ce72c6b5528b7432574`
- Repo:
  - `https://github.com/shenweichen/DeepCTR-Torch.git`
- Reference type:
  - mature implementation library
- Confirmed useful modules:
  - `MMOE`
  - `PLE`
- Why useful:
  - compact PyTorch reference implementations
  - good for quickly prototyping multi-task shared/private blocks

### `MMLRec`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/MMLRec`
- Commit:
  - `61e3f1906970d51065ca37590c8e2d4abcf99527`
- Repo:
  - `https://github.com/alipay/MMLRec-A-Unified-Multi-Task-and-Multi-Scenario-Learning-Benchmark-for-Recommendation.git`
- Reference type:
  - official benchmark implementation
- Confirmed useful modules:
  - `STAR`
- Why useful:
  - stable implementation source for the `STAR` mechanism
  - broader benchmark for multi-task / multi-scenario recommendation designs

### `AREAD-Multi-Domain-Recommendation`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/AREAD-Multi-Domain-Recommendation`
- Commit:
  - `eb0a5aa5becb2e75d71a7aa4e8b3967f8f1b2f7d`
- Repo:
  - `https://github.com/Chrissie-Law/AREAD-Multi-Domain-Recommendation.git`
- Reference type:
  - official implementation
- Confirmed useful modules:
  - `AREAD`
  - hierarchical expert structure
  - domain-specific expert mask learning
- Why useful:
  - direct code reference for many-domain expert masking
  - useful for thinking about sparse-domain protection and counterfactual augmentation

### `KDD2024-PAAC`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/KDD2024-PAAC`
- Commit:
  - `ae440e5e48757ade2f166420d141109e177cf7e4`
- Repo:
  - `https://github.com/miaomiao-cai2/KDD2024-PAAC.git`
- Reference type:
  - official implementation
- Confirmed useful modules:
  - `PAAC`
- Notes:
  - the repository name and code use `PAAC`
  - the README title says `PCCA: Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias`
- Why useful:
  - concrete long-tail / popularity-bias reference
  - useful for translating head-to-tail supervision ideas into solver-tail support

### `CoPD`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/CoPD`
- Commit:
  - `c0861a329f5fc8a3b729def2e29ad4ed3770bac3`
- Repo:
  - `https://github.com/XiangZongyi/CoPD.git`
- Reference:
  - `Coherence-guided Preference Disentanglement for Cross-domain Recommendations`
- Reference type:
  - implementation repo
- Confirmed useful modules:
  - `CoPD`
- Why useful:
  - useful reference for shared vs specific preference decomposition
  - relevant to problem-distinction and representation-separation design

### `MIND`

- Path:
  - `/public/home/zhoucl/shiys/reference_repos/unified_selector/MIND`
- Commit:
  - `fe249a1c4b55163accd483263d94d969b6b51121`
- Repo:
  - `https://github.com/Wang-Yu-Qing/MIND.git`
- Reference:
  - `Multi-Interest Network with Dynamic Routing for Recommendation at Tmall`
- Reference type:
  - third-party standalone PyTorch implementation
- Confirmed useful modules:
  - `MIND`
  - dynamic-routing multi-interest encoder
- Why useful:
  - direct reference for candidate-aware / multi-interest user representation
  - closest ready-to-read code for the `solver-conditioned instance view` intuition
## 4. Still not cleanly resolved after a second pass

After a second search pass across GitHub, arXiv pages, search engines, and benchmark repos, two items still do **not** have a stable standalone code repo that I can confidently mark as confirmed:

- `AdaSparse`
  - I did not confirm a public repo for the recommendation paper itself.
  - GitHub search returns an unrelated `AdaSparse` project for distributed learning, not recommendation.
  - Search-engine results also point to paper index pages that expose a `request code` style entry rather than a stable repository link.
  - I also checked practical substitute libraries already downloaded here (`MMLRec`, `FuxiCTR`, `DeepCTR-Torch`, `HAMUR`) and did not find an `AdaSparse` implementation inside them.

- `Joint Identifiability for CDR`
  - I did not confirm a public repo for `Joint Identifiability of Cross-Domain Recommendation via Hierarchical Subspace Disentanglement`.
  - I checked GitHub search, arXiv / paper pages, CatalyzeX, Papers With Code, and authors' public pages, but did not find a stable downloadable implementation.

Important nuance:

- `STAR` is now resolved in practice via `MMLRec`, which contains a usable implementation, even though I still did not confirm a dedicated standalone official `STAR`-only repo.
- `MIND` is now resolved in practice via a standalone PyTorch repo, but I have **not** confirmed that this particular repo is the original official release from the paper authors.

When these remain unresolved, the best practical substitutes are still:

- `MMLRec`
- `FuxiCTR`
- `DeepCTR-Torch`
- `HAMUR`

because together they already cover the most reusable mechanisms:

- shared/private experts
- domain-aware routing
- conditional adaptation
- `STAR`-style multi-domain sharing

## 5. Suggested usage order for implementation

If you start implementing the current final idea, the best order of reference is:

1. `NSS` local code
   - for hierarchical graph encoder and ranking selector baseline
2. `URS` local code
   - for unified raw schema and explicit problem descriptor
3. `HAMUR`
   - for lightweight conditional adapters
4. `FuxiCTR` / `DeepCTR-Torch`
   - for `MMoE` / `PLE` style expert blocks
5. `MMLRec`
   - for `STAR` and benchmark-grade multi-scenario organization
6. `MAMDR`
   - for imbalance-aware regularization ideas
7. `AREAD` / `PAAC` / `CoPD` / `MIND`
   - for specialized recommendation-side mechanisms that inspired the final selector design
8. `RouteFinder` / `GOAL`
   - for broader task generalization and problem abstraction patterns
9. `BQ-NCO`
   - for invariance-aware refinements
