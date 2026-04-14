# Refinement Report

## Starting Point

The original research direction had already moved through:

- RL
- bandit
- per-problem selector logic

The new direction pivots to:

- supervised learning
- one selector for multiple routing problems

## Why The Pivot Makes Sense

- The older line already validated that solver complementarity is real.
- The next meaningful question is not "is selection useful?" but:
  - "can selection itself be unified across problems?"

## Refined Problem Statement

Current refined problem statement:

> Learn one supervised routing-solver selector that works across `tsp`, `atsp`, `pctsp`, and a chosen set of CVRP variants, despite the fact that different problems support different candidate solvers.

## Refined Core Difficulty

The real difficulty is threefold:

1. heterogeneous candidate solver sets
2. problem identification and conditioning
3. unified input representation

## Refined Thesis

The selector should be framed as **compatibility learning** rather than simple classification:

- routing instance = query
- solver = item
- problem/solver attributes = compositional side information

## Complexity Rejected For Now

- no heavy MoE first
- no RL
- no runtime-aware scheduling first
- no extremely broad 100+ problem paper in the first iteration

## Immediate Next Move

Do not code the final selector first.  
First build the label and baseline pipeline.
