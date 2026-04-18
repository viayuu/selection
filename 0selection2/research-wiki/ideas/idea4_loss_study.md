---
type: idea
node_id: idea:4_loss_study
title: "Loss-function diagnostic for selectors"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:gao2025_nss, paper:tornede2021_meta_as]
target_gaps: [G2]
tags: [ablation, loss]
---
# Hypothesis
Regret-weighted listwise loss dominates CE / pairwise / gap-regression on macro mean-cost regret. Provides a clean single ablation table for the main paper.

## Metrics
Pairwise win-AUC, Kendall τ vs oracle ranking, regret at margin bins, top-1/2/3.
