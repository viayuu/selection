---
type: idea
node_id: idea:ood_invariant_residual_regret
title: "Environment-Invariant Residual Regret Learning for Unified Solver Selection"
stage: proposed
outcome: unknown
based_on:
  - paper:nss_anchor
  - paper:urs_anchor
  - paper:routefinder_anchor
target_gaps:
  - gap:G1
  - gap:G2
  - gap:G3
  - gap:G4
created_at: 2026-04-15T19:13:45+08:00
updated_at: 2026-04-15T19:13:45+08:00
---

# Hypothesis

A unified selector should generalize better to benchmark/OOD regimes if it learns each solver's residual regret relative to a safe anchor solver, and if the learned solver preferences are regularized to remain stable across routing environments defined by problem family, scale, and instance statistics.

## Proposed Method

Keep the current simple unified selector backbone largely intact, but replace absolute pooled score fitting with residual-regret prediction relative to an anchor solver. Build lightweight environment groups from existing metadata (`problem_features`, `scale`, `instance_stats`) and add an environment-invariance regularizer on solver preference relations.

## Expected Outcome

- Preserve strong IID `test` performance.
- Improve hard competitive subsets.
- Reduce `CVRPLIB` / benchmark regret relative to the current baseline.

## Why This Idea Matters

This sharpens the project from “can one unified selector work?” to “what solver preferences actually transfer across environments?” It directly targets the currently dominant weakness instead of adding more architecture.

## Cheapest Validation

1. Freeze the current best baseline.
2. Define a safe anchor solver.
3. Re-label supervision as residual regret / residual advantage.
4. Bucket environments by problem family, scale, and selected instance-stat dimensions.
5. Compare baseline vs residualized vs residualized+invariance on IID test, hard subset, and benchmark.

## Main Reviewer Objection

The environment grouping may look hand-designed, so gains must be shown to be robust and not dependent on one lucky partition.

## Failure Notes / Lessons

Do **not** let this grow into a heavy MoE/expert architecture paper. The value of this idea is that it turns the current benchmark failure into the main scientific question with minimal architectural inflation.
