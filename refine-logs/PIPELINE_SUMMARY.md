# Pipeline Summary

**Problem**: unified supervised solver selection across multiple routing problems  
**Final Method Thesis**: learn one masked compatibility scorer over `(instance, solver)` using unified problem representation and solver capability descriptors  
**Final Verdict**: READY FOR DATASET + BASELINE BUILDING  
**Date**: 2026-04-14

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report: `IDEA_REPORT.md`

## Contribution Snapshot

- Dominant contribution:
  - unified cross-problem supervised selector via compositional problem-solver compatibility
- Optional supporting contribution:
  - solver metadata / capability descriptors for extensibility
- Explicitly rejected complexity:
  - RL / bandit
  - heavy MoE first
  - runtime-aware dynamic selection as main story

## Must-Prove Claims

- A single selector can work across multiple routing problems.
- Explicit problem/solver compositional descriptors help beyond naive pooling.
- Masked pair scoring is a better formulation than per-problem output heads for heterogeneous solver support.

## First Runs To Launch

1. Build `EasyNCO` coverage matrix for problems and solvers.
2. Generate a small mixed-problem label smoke dataset.
3. Train pooled baselines before the full compatibility model.

## Main Risks

- Label generation bottleneck:
  - mitigation: small pilot dataset first
- Novelty may look like engineering:
  - mitigation: emphasize compositional compatibility and strong ablations
- Negative transfer:
  - mitigation: add shared/private structure only if measured

## Next Action

- Proceed to dataset-generation engineering, not full model coding first.
