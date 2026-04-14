# Final Proposal V3

**Date**: 2026-04-14  
**Status**: READY FOR ZERO-CODE-CHANGE FULL LABEL RELEASE  
**This proposal is separate from**: `dataset-plan-logs-v2/FINAL_PROPOSAL.md`

## Problem Anchor

Build the first truly executable multi-scale init-only dataset-and-label release without modifying `EasyNCO`.

## Core Thesis

The plan should now optimize for:

- multi-scale diversity
- real label-running feasibility
- zero `EasyNCO` code edits

not for:

- idealized generator redesign
- smoke tests
- coverage-only paperwork

## Key Decisions

### 1. TSP / CVRP

- directly reuse `NSS` varying datasets
- batch-run labels with `run_nss_eval_suite.py`

### 2. ATSP

- generate shard files by directly calling `ATSPGenerator`
- use scales `20 / 50 / 100`
- label with `MatNet`, `MatPOENet`, `GLOP`

### 3. 16 CVRP variants

- generate shard files by directly calling `MVRPGenerator`
- use scales `50 / 100`
- label with `MTPOMO`, `MVMoE`

### 4. PCTSP

- data can be prepared
- but it should not enter the first selector release under strict init-only

### 5. Dataset representation

- do not invent new loaders
- represent multi-scale datasets as shard directories plus manifests

## Main Deliverables

1. `TSP/CVRP` full varying-label release
2. `ATSP` multi-scale label release
3. `16`-variant `MVRP` multi-scale label release
4. merged supervision tables with feasible masks

## Explicitly Rejected Direction

- modifying `EasyNCO` generators
- modifying `EasyNCO` loaders
- official smoke-first stage
- including `PCTSP` as a fake multi-method selector task when it only has one feasible init-only method

## Immediate Next Action

Write the external dataset-generation scripts and full batch launchers outside `EasyNCO`, then launch the real label campaign.
