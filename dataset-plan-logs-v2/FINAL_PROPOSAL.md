# Final Proposal V2

**Date**: 2026-04-14  
**Status**: READY FOR FULL DATASET + FULL LABEL RELEASE  
**This proposal is separate from**: `dataset-plan-logs/FINAL_PROPOSAL.md`

## Problem Anchor

Construct the actual first full supervised dataset-and-label release for a future unified cross-problem solver selector.

## Core Thesis

The dataset program should no longer be anchored to `n=100` or to smoke runs.  
It should be anchored to:

- `10,000` instances per problem
- real scale diversity
- EasyNCO-native generation and evaluation
- initialization-only labels
- instance-level feasible masks

## Key Decisions

### 1. TSP / CVRP

- directly reuse `NSS` varying datasets
- keep the `50-499` size diversity

### 2. 16 CVRP variants

- use `MVRPGenerator` as the base
- move to solver-aware multi-scale generation
- recommended scale mix: `50 / 100 / 200`

### 3. ATSP

- use `ATSPGenerator` as the base
- move to a multi-family generator
- recommended scale mix: `20 / 50 / 100`

### 4. PCTSP

- use `PCTSPGenerator` as the base
- add geometry diversity plus prize/penalty regimes
- recommended scale mix: `100 / 500`

### 5. Label unit

- do not treat a paper name as the final label candidate unit
- treat `solver entry = method + checkpoint + decoding regime`

### 6. Label semantics

- initialization only
- `NoIteration`
- use per-instance export from `EasyNCO`

## Full-Release Solver Pools

- `TSP`: platform-native init-only neural solvers plus insertion heuristics
- `CVRP`: platform-native init-only neural solvers plus insertion heuristics
- `ATSP`: `MatNet`, `MatPOENet`, `GLOP`
- `PCTSP`: `DeepACO`, `GLOP`
- `16` CVRP variants: `MTPOMO`, `MVMoE`

## Main Deliverables

1. one 10k varying dataset per problem
2. per-problem manifest plus fixed-scale shards
3. full init-only label runs for all selected solver entries
4. merged supervision tables with instance-level feasible masks

## Explicitly Rejected Direction

- fixed `n=100`
- smoke-first official plan
- mixing init-only labels with iterative-search labels
- pretending all problems should use the same scale range

## Immediate Next Action

Implement the dataset generators and the full label-running scripts for the V2 release.
