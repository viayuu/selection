# Review Summary

## Overall Verdict

**Proceed with caution**, but proceed.

The direction is promising because the gap appears real:

- unified multi-problem routing solvers exist
- per-instance solver selection exists
- but a unified cross-problem supervised selector over a heterogeneous solver zoo is not yet clearly established

## Strongest Positive Point

The problem is structurally meaningful:

- different problems activate different solver subsets
- problem identity is partly discrete and partly compositional
- this naturally motivates pairwise compatibility learning instead of per-problem output heads

## Strongest Concern

The paper can easily look like:

- NSS + URS + recommender masking

unless the contribution is framed as:

- compositional compatibility between problem structure and solver capability

## What Must Be Proven

1. One shared selector is not merely possible, but useful.
2. Explicit problem/solver descriptors are necessary.
3. Masked pair scoring is better than naive pooling.

## What Does Not Need To Be Proven Yet

- full zero-shot generalization to completely unseen problem families
- heavy MoE systems
- large-scale runtime-aware deployment

## Review Recommendation

- Use a conservative first implementation.
- Keep the story compact.
- Make the label-generation pipeline a first-class engineering milestone.
