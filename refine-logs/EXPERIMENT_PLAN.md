# Experiment Plan

**Date**: 2026-04-14  
**Target**: validate the unified supervised selector thesis from `FINAL_PROPOSAL.md`

## 1. Precondition: Label Dataset Generation

This is the true stage-0 blocker.

### Goal

Use `EasyNCO` to generate per-instance solver outcomes for a mixed set of routing problems, then convert them into a selector supervision dataset.

### Why `EasyNCO`

- it already contains many routing environments
- it already contains many solver settings
- it already supports evaluation-style execution and flexible composition
- it is the natural place to build a unified labeling pipeline

### Immediate deliverables

1. a **problem-method coverage matrix**
2. a **small pilot label manifest**
3. a **joint dataset schema**

## 2. Problem Set

### Current target set

- `tsp`
- `atsp`
- `pctsp`
- `16` selected `cvrp` variants

### Important note

The exact `16` CVRP variants are still a project-level decision and should be frozen before large-scale label generation.

## 3. Candidate Solver Zoo

### Principle

Do not start with the largest zoo. Start with a **high-overlap, executable mini-zoo**.

### Current likely candidates from `EasyNCO`

- mostly shared or multi-problem:
  - `glop`
  - `deepaco`
- TSP-heavy:
  - `lehd`
  - `elg`
  - `difusco`
  - `t2t`
- ATSP:
  - `matnet`
  - possibly `glop`
- CVRP / multi-task:
  - `mtpomo`
  - `mvmoe`
  - possibly `omni`
- PCTSP:
  - `deepaco`
  - `glop`

### Freeze this before full run

For each candidate solver, record:

- supported problems
- checkpoint availability
- expected runtime
- whether inference is stable on current machine

## 4. Dataset Protocol

For each `(problem, instance, solver)` triplet:

- problem metadata
- instance id / scale / distribution
- solver id
- feasible flag
- objective value
- runtime
- optional extra diagnostics

From these, derive:

- feasible mask
- ranks among feasible solvers
- per-problem single-best baseline
- oracle best solver
- normalized regret labels

## 5. Baselines

### B0. Non-learning baselines

- random feasible solver
- single best global
- single best per problem
- oracle

### B1. Per-problem selector baseline

- NSS-style or `2l2r`-style supervised selector trained separately for each problem

### B2. Pooled naive selector

- single model over all problems
- only problem id
- no solver metadata

### B3. Pooled masked ranker

- single model
- feasible mask
- no explicit solver metadata

### B4. Full proposed selector

- unified instance representation
- problem family token
- active-feature multi-hot
- solver metadata / capability vector
- masked pair scorer

### B5. If needed: shared/private extension

- adapters or shared/private experts only if negative transfer is observed

## 6. Metrics

### Primary metrics

- mean selected objective
- mean normalized regret to oracle
- gain over single-best-per-problem

### Secondary metrics

- top-1 best-solver accuracy
- calibration / confidence quality
- per-problem performance variance
- fairness across problem families

## 7. Core Ablations

1. **per-problem vs unified**
2. **problem id only vs active-feature multi-hot**
3. **without solver metadata vs with solver metadata**
4. **plain pooled scorer vs masked pair scorer**
5. **shared model vs shared/private model** if needed

## 8. Run Order

### Run Block A: Label smoke test

- choose 4-6 problems
- choose 4-6 solvers
- choose 50-200 instances per problem
- verify end-to-end label generation and dataset conversion

### Run Block B: Baseline selector smoke test

- B0, B2, B3
- confirm that unified training is stable

### Run Block C: Per-problem baseline

- B1
- needed to answer whether one-model sharing is worth it

### Run Block D: Full method

- B4

### Run Block E: Negative-transfer mitigation

- only if B4 underperforms because of domain interference

## 9. Decision Gates

### Gate 1

If label generation is brittle, stop and stabilize the pipeline before modeling.

### Gate 2

If B3 already matches B1 closely, keep the first paper simple.

### Gate 3

If B4 does not beat B3, solver metadata / capability modeling is not yet justified.

### Gate 4

If shared training clearly hurts one or more families, only then add PLE / STAR / MMoE style structure.

## 10. Exploratory Secondary Experiments

These are optional and should come after the core result:

- held-out solver insertion
- held-out problem subtype
- partial-label training
- runtime-aware top-k inference
