---
type: idea
node_id: idea:6_problem_id_dropout
title: "Problem-ID dropout sanity ablation"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:gao2025_nss]
target_gaps: [G3']
tags: [ablation, sanity, leakage]
---
# Hypothesis
Dropping problem-ID at train time (25/50/100%) preserves top-1 > random on OVRP vs CVRP when same fields are used — shows the model learns from instance features, not ID lookup. Pre-empts reviewer "lookup table" attack.

## Metrics
OVRP↔CVRP confusion matrix; per-problem top-1 with/without ID.
