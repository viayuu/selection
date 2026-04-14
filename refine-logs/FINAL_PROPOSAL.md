# Final Proposal

**Date**: 2026-04-14  
**Status**: READY FOR DATASET + BASELINE BUILDING  
**Based on**: `IDEA_REPORT.md`

## Problem Anchor

Given a routing instance drawn from multiple problem families and a heterogeneous solver zoo with partially overlapping support, learn **one supervised selector model** that chooses the best feasible solver for each instance better than naive pooled selectors and competitively with per-problem selectors.

## Final Method Thesis

Represent both the **problem instance** and the **candidate solver** as compositional objects, then learn a shared masked compatibility scorer `s(instance, solver)` across problems rather than training one selector per problem.

## Dominant Contribution

The dominant contribution should be:

- **a unified cross-problem solver selector based on compositional problem-solver compatibility**

This is stronger than merely saying "we use one model for many problems". The selector should explicitly address:

- non-uniform feasible solver sets
- problem identity and problem structure
- shared representation across heterogeneous routing problems

## Optional Supporting Contribution

- solver metadata / capability descriptors that allow cleaner extension to new solvers or problem subtypes

## Explicitly Rejected Complexity

For the current stage, reject the following:

- RL / contextual bandit / LinUCB as the main story
- end-to-end solver training
- a very heavy expert system as the first implementation
- full zero-shot claims on unseen problem families before the base selector works
- runtime-aware dynamic top-k selection as the main paper thesis

## Proposed Method Skeleton

### Input side

- unified routing instance representation
- coarse problem family token
- active-feature / multi-hot problem representation

### Solver side

- learnable solver embedding
- solver metadata / capability descriptor

### Scoring side

- shared scorer over `(instance, solver)`
- feasible mask to restrict scoring / normalization to supported solvers

### Training side

- supervised ranking objective as the current leading candidate
- alternative losses remain open for later validation

## Key Claims To Validate

1. A single cross-problem selector can outperform naive pooled baselines under a shared training setup.
2. Explicit compositional problem/solver descriptors matter beyond a plain shared model with problem ID only.
3. A masked pair-scoring formulation is a better fit for heterogeneous solver support than per-problem output heads.

## Main Risks

### Risk 1: "This is only engineering integration"

Mitigation:

- center the story on compositional compatibility, not just one-model pooling
- run strong ablations against simpler pooled alternatives

### Risk 2: Negative transfer across problem families

Mitigation:

- first measure it explicitly
- only then consider shared/private adapters or experts

### Risk 3: Label generation is too expensive

Mitigation:

- start from a small solver zoo and a small problem subset
- build pilot labels first
- treat partial-label generation as backup extension

## First Decision Gate

Before implementing the full method, answer:

1. What exact `16` CVRP variants are in the first paper?
2. Which `EasyNCO` methods have working inference and pretrained checkpoints for those problems?
3. What is the smallest solver zoo with enough overlap to make unified selection meaningful?

If these are not fixed, do not overbuild the model yet.
