# Dataset Pipeline Summary V3

**Problem**: build the first executable multi-scale init-only selector-label release without editing `EasyNCO`  
**Final Verdict**: READY FOR REAL DATA GENERATION + REAL LABEL RUNS  
**Date**: 2026-04-14

## Deliverables

- Report: `DATASET_LABEL_PLAN_REPORT_V3.md`
- Proposal: `dataset-plan-logs-v3/FINAL_PROPOSAL.md`
- Plan: `dataset-plan-logs-v3/EXPERIMENT_PLAN.md`
- Tracker: `dataset-plan-logs-v3/EXPERIMENT_TRACKER.md`

## Core Decisions

- do not modify `EasyNCO`
- `TSP/CVRP` directly use `NSS varying`
- `ATSP` uses `20 / 50 / 100`
- `16` CVRP variants use `50 / 100`
- multi-scale datasets are represented as shard directories plus manifests
- labels are still strict `init-only`
- `PCTSP` is excluded from the first selector release

## Why V3 Replaces V2

V2 still assumed generator extension.  
V3 removes that assumption and only uses current code paths.

## Official First Selector Release

- `TSP`
- `CVRP`
- `ATSP`
- `16` CVRP variants

## Deferred / Auxiliary

- `PCTSP`

## Next Action

- implement the external shard-generation scripts and launch the full label campaign
