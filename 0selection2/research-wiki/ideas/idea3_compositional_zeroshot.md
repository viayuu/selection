---
type: idea
node_id: idea:3_compositional_zeroshot
title: "Compositional zero-shot MVRP selection"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:yu2026_coeks, paper:zhou2025_urs, paper:2024_prompt_vrp]
target_gaps: [G3']
tags: [zero-shot, compositional, mvrp]
---
# Hypothesis
Train on 11 MVRP variants (constraint combos), hold out 4 (e.g. OVRPBL, VRPBLTW, OVRPBTW, OVRPLTW). Constraint-bitvector conditioning lets the selector generalize to unseen combos even when underlying solvers' ranks shift.

## Minimum experiment
Built on Idea-1+2 base. 1 seed, 4h. Metric: macro mean-cost regret on held-out.

## Reviewer defense
Report both top-1 and regret; compare to problem-ID-only baseline (compositional baseline should degenerate to random on unseen combos).
