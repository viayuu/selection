# Dataset Pipeline Summary

**Problem**: build datasets and initialization-only labels for unified supervised solver selection  
**Final Verdict**: READY FOR COVERAGE AUDIT + SMOKE RUNS  
**Date**: 2026-04-14

## Deliverables

- Report: `DATASET_LABEL_PLAN_REPORT.md`
- Proposal: `dataset-plan-logs/FINAL_PROPOSAL.md`
- Plan: `dataset-plan-logs/EXPERIMENT_PLAN.md`
- Tracker: `dataset-plan-logs/EXPERIMENT_TRACKER.md`

## Core Decisions

- `TSP/CVRP` reuse `NSS`
- other problems generated through `EasyNCO`-centric extensions
- fix size `100` first
- maximize distribution diversity
- labels are initialization-only

## First 3 Actions

1. Build problem-method coverage matrix
2. Extend / reuse generators for smoke datasets
3. Verify `eval.py` init-only labeling with per-instance export

## Main Risks

- too little diversity in synthetic generation
- method support mismatches
- unstable init-only export for some methods

## Next Action

- implement Workstream A and B smoke tests before launching full 10k jobs
