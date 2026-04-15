---
type: idea
node_id: idea:competition_aware_selection
title: "Competition-Aware Unified Solver Selection"
stage: proposed
outcome: unknown
based_on:
  - paper:nss_anchor
  - paper:urs_anchor
target_gaps:
  - gap:G1
  - gap:G2
created_at: 2026-04-15T19:13:45+08:00
updated_at: 2026-04-15T19:13:45+08:00
---

# Hypothesis

The scientific value of unified solver selection comes primarily from competitive instances where solver complementarity is real; explicitly modeling competition intensity should improve both evaluation clarity and learning focus.

## Proposed Method

Derive a competition score from existing per-instance solver costs (e.g. best-vs-second-best gap, gain over safe baseline, solver entropy). Train an auxiliary head or reweight the ranking loss to focus on high-competition instances.

## Expected Outcome

- Stronger hard-subset performance.
- Cleaner interpretation of where unified selection truly helps.
- Potential benchmark gains if hard cases are emphasized during training.

## Why This Idea Matters

Current average metrics are diluted by many low-entropy variants where one solver dominates. This idea separates “selection is meaningful” from “selection is formally present but trivial.”

## Cheapest Validation

Use existing labels only. No new data collection is required.

## Main Reviewer Objection

This may look more like evaluation reframing than a sufficiently new method unless it also improves benchmark/OOD performance.

## Failure Notes / Lessons

This is strong as a supporting idea or analysis protocol. It is likely weaker as the single headline contribution than the invariant residual-regret direction.
