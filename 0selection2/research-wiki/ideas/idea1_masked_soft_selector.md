---
type: idea
node_id: idea:1_masked_soft_selector
title: "Unified masked selector with regret-soft labels"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:gao2025_nss, paper:zhou2025_urs, paper:yu2026_coeks]
target_gaps: [G1, G2]
tags: [selector, soft-label, masked-softmax, main]
---
# Hypothesis
A single encoder + problem-ID + constraint-bitvector token, trained with masked listwise loss on cost-gap-weighted soft labels p*(s|x) ∝ exp(-(c_s − c_*)/τ), matches 18 per-problem NSS specialists on macro mean cost.

## Why it works
Hard `ind` labels penalize near-ties; soft labels match the mean-cost metric. Masked softmax handles overlapping / variable pools. Constraint-bitvector disambiguates OVRP↔CVRP and related pairs.

## Minimum experiment
Extend NSS: UDR-style node features, problem+constraint tokens, per-solver embedding `e_s`, masked softmax over M(x), soft-label listwise loss. Train on all 18 at once. Pilot 1 seed, 10 epochs, 4h.

## Expected metrics
- Top-1 ≥ NSS on TSP/CVRP; > random on MVRP (25%) and ATSP (33%).
- Macro mean-cost regret ≤ per-problem NSS.
- Arm distribution ≠ single-best.

## Reviewer defense
Soft-label regret loss + masked-ID dropout + BS-mask ablation + compositional zero-shot experiment.
