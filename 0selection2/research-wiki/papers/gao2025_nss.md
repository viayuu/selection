---
type: paper
node_id: paper:gao2025_nss
title: "Neural Solver Selection for Combinatorial Optimization"
authors: ["Chengrui Gao", "Haopu Shang", "Ke Xue", "Chao Qian"]
year: 2025
venue: ICML
external_ids: {arxiv: "2410.09693"}
tags: [selector, nco, tsp, cvrp, per-instance-AS]
relevance: core
origin_skill: research-lit
---

# One-line thesis
First per-instance neural-solver-selector framework for COPs: extract instance features → classification/ranking head → robust selection strategy.

## Problem / Gap
Existing NCO works build single solvers; complementarity at instance level is untapped; ensembles run all solvers (expensive).

## Method
GAT encoder (+ hierarchical graph pooling) → MLP scoring head. Trained with either classification or listwise ranking loss. Inference: top-k + rejection threshold.

## Key Results
–0.82% gap on synthetic TSP, –2.00% on synthetic CVRP vs best individual solver; –0.88% on TSPLIB, –0.71% on CVRPLIB.

## Limitations / Failure Modes
One selector per problem (TSP and CVRP trained separately). No cross-problem sharing. Method pool is fixed.

## Reusable Ingredients
Hierarchical-pool encoder, rejection-based selection strategy, classification↔ranking loss comparison.

## Open Questions
Does a *single* selector generalize across problems? Can masking handle overlapping solver pools?

## Relevance to This Project
Direct base to modify: same encoder/head design, extended to 18 problems with mask + problem-conditioning.
