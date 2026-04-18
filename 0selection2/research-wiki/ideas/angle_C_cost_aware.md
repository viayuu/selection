---
type: idea
node_id: idea:angle_C_cost_aware
title: "Cost-aware cascade selector"
origin_skill: research-lit
stage: proposed
tags: [cost-aware, cascade, pareto]
---
# Idea
Predict (quality, runtime) Pareto; select one solver or top-k under budget.

## Target gaps
- G5 (cost-aware selection under budget)

## Experiment sketch
Budget sweep; metric = VBS-gap per second; baselines = single-best, round-robin, per-problem NSS, Poppy (upper bound).
