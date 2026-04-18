---
type: idea
node_id: idea:5_bs_mask
title: "Availability-conditional Balanced Softmax (BS-mask)"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:plm2021_partial_label_masking]
target_gaps: [G2]
tags: [ablation, loss, imbalance]
---
# Hypothesis
Adding logit shift −log p(s | s ∈ M(x)) lifts ATSP/MVRP top-1 without hurting TSP. Uses availability-conditional prior (global logit adjustment fails because unavailable solvers aren't negatives).

## Metrics
Macro-balanced top-1, arm entropy, ECE calibration.
