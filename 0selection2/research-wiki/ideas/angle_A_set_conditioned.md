---
type: idea
node_id: idea:angle_A_set_conditioned
title: "Set-conditioned solver scoring with solver-ID embeddings"
origin_skill: research-lit
stage: proposed
tags: [cold-start, selector, transfer]
---
# Idea
Learn f(instance, constraint_signature, solver_id) rather than a fixed classifier. Solvers get learnable IDs; a new solver is absorbed by adding an embedding, no retraining required.

## Target gaps
- G1 (cross-problem selector)
- G4 (solver cold-start)

## Why it works
Matches STAR/PLE-style shared-centre conditioning; extends CoEKS's constraint-expert logic to the *solver* axis.

## Experiment sketch
Leave-one-solver-out & leave-one-problem-out; metric = regret to VBS; baselines = 18 NSS specialists, RouteFinder, CoEKS, MTL-KD.
