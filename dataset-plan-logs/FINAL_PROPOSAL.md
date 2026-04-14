# Final Proposal

**Date**: 2026-04-14  
**Status**: READY FOR IMPLEMENTATION  
**This proposal is separate from**: `refine-logs/FINAL_PROPOSAL.md`

## Problem Anchor

Construct mixed-problem supervised datasets and initialization-only labels to support later training of a unified cross-problem solver selector.

## Core Thesis

The first research bottleneck is not model design but **data and label infrastructure**.  
The pipeline must:

- reuse `NSS` where it already gives high-quality data
- extend the remaining problems through `EasyNCO`
- maximize diversity without leaving the platform ecosystem
- generate labels using **initialization only**

## Scope of the first dataset release

- `tsp`: use `NSS`
- `cvrp`: use `NSS`
- `16` CVRP variants: use `EasyNCO` MVRP-style generation
- `pctsp`: use `EasyNCO` PCTSP generator with diversity extensions
- `atsp`: use `EasyNCO` ATSP generator with diversity extensions

## First Design Decisions

### Fixed size

- use `n = 100` first

### Diversity driver

- prioritize **distribution diversity**
- defer multi-scale training to a later stage

### Label semantics

- no iteration
- initialization only
- export per-instance scores from `EasyNCO`

## Main Deliverables

1. problem-method coverage matrix
2. pilot dataset generation scripts / configs
3. label-generation configs with `NoIteration`
4. merged supervision dataset schema

## Explicitly Rejected Complexity

- full multi-scale data generation in v1
- full solver zoo in v1
- custom non-platform standalone data builders unless strictly necessary
- iteration-based labeling

## Main Risk

If the first release tries to cover too many methods and too many scales at once, the pipeline may become brittle before producing any useful labels.

## Immediate Next Action

Implement coverage audit and smoke dataset generation before any full 10k run.
