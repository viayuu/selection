# Dataset Pipeline Summary V2

**Problem**: build the real full-scale dataset and init-only label release for the next unified supervised selector  
**Final Verdict**: READY FOR FULL DATASET GENERATION + FULL LABEL RUNS  
**Date**: 2026-04-14

## Deliverables

- Report: `DATASET_LABEL_PLAN_REPORT_V2.md`
- Proposal: `dataset-plan-logs-v2/FINAL_PROPOSAL.md`
- Plan: `dataset-plan-logs-v2/EXPERIMENT_PLAN.md`
- Tracker: `dataset-plan-logs-v2/EXPERIMENT_TRACKER.md`

## Core Decisions

- no more fixed `n=100`
- `TSP/CVRP` directly reuse `NSS varying`
- other problems become solver-aware multi-scale datasets
- labels stay `init-only`
- label candidates are `solver entries`, not just raw method names
- final supervision uses instance-level feasible masks

## Per-Problem Size Policy

- `TSP`: `NSS` varying `50-499`
- `CVRP`: `NSS` varying `50-499`
- `16` CVRP variants: `50 / 100 / 200`
- `ATSP`: `20 / 50 / 100`
- `PCTSP`: `100 / 500`

## First Full Label Pools

- `TSP`: platform-native init-only neural solvers plus insertion heuristics
- `CVRP`: platform-native init-only neural solvers plus insertion heuristics
- `ATSP`: `MatNet`, `MatPOENet`, `GLOP`
- `PCTSP`: `DeepACO`, `GLOP`
- `16` CVRP variants: `MTPOMO`, `MVMoE`

## Main Difference From V1

This V2 plan is no longer:

- fixed-size
- smoke-first
- coverage-first

It is now:

- multi-scale
- full-release oriented
- directly aligned with real label generation

## Next Action

- implement the new varying generators and the full init-only label-run scripts
