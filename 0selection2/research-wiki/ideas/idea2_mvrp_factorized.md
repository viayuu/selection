---
type: idea
node_id: idea:2_mvrp_factorized
title: "MVRP solver × variant factorized scoring"
origin_skill: idea-creator
stage: proposed
outcome: unknown
based_on: [paper:sheng2021_star, paper:tang2020_ple, paper:yu2026_coeks]
target_gaps: [G2, G3']
tags: [factorized, mvrp, interaction]
---
# Hypothesis
Score = h(x)·e_s + b_s + u_{s,constraint_bit} + u_{s,coord_dist} beats a generic problem-ID model on MVRP and enables interpretable per-constraint solver preference.

## Why it works
All 15 MVRP variants share the same 4 solvers — the 15×4 label structure is a natural factorization (à la STAR/PLE), not a flat multi-class problem.

## Minimum experiment
MVRP-only pilot, compare to Idea 1 ablated. 1 seed, 3h.
