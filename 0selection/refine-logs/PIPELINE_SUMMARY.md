# Pipeline Summary

**Problem**: Train one supervised neural selector across 18 routing problems (TSP / CVRP / ATSP / 15 MVRP) that picks per-instance the best solver from a variable candidate pool (3–10 methods).
**Final Method Thesis**: A shared-encoder masked-softmax selector with **regret-soft listwise labels** and a lightweight MVRP **solver×constraint×coord-dist factorized** add-on beats SBS on macro mean cost across 18 problems and generalizes compositionally to held-out MVRP constraint combinations.
**Final Verdict**: READY (core plan fits in 58 GPU-h).
**Date**: 2026-04-18.

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant**: Unified masked selector with regret-soft listwise labels — 18 problems, one model, variable pool.
- **Optional supporting**: MVRP solver×constraint×coord-dist factorized scoring for compositional zero-shot.
- **Explicitly rejected**: cost-aware / time-aware selection, PLE hierarchical experts, set-conditioned solver cold-start, hypernetwork per-problem parameter generators, vanilla long-tail losses.

## Must-Prove Claims
- **C1** (main): unified selector matches/beats SBS on macro mean cost across 18 problems.
- **C2** (anti-leakage): gains come from instance features, not problem-id / availability-mask priors (**metadata-only baseline** is the pre-emptive defense).
- **C3** (loss): regret-soft listwise > CE / pairwise / gap-regression.
- **C4** (compositional zero-shot): factorized MVRP scoring generalizes to 4 held-out bitvectors.
- **C5** (imbalance): availability-conditional BS-mask lifts tail without hurting head.

## First Runs to Launch
1. **R0 — Evaluation & split audit** (0 GPU-h). Blocks everything. Produces SBS/VBS per-problem, mask sanity, paired-bootstrap scripts.
2. **R1-Full seed 0** (5 GPU-h). Template config; stability + sanity gates.
3. **R2-MetaOnly seed 0** (1.5 GPU-h). Can run in parallel after R1 template verified; delivers C2 first-look.

## Main Risks
- **R1 (medium)**: dominant-SBS problems leave little room for the selector. **Mitigation**: pre-compute per-problem oracle-SBS gap in R0; pre-register "matches SBS where gap < X%" phrasing.
- **R2 (low)**: regret-soft vs CE margin may be small. **Mitigation**: keep C3 claim narrow ("more stable / competitive"); drop from main if < 0.3% after 1-seed matched comparison.
- **R3 (medium)**: zero-shot may collapse on one held-out combo. **Mitigation**: report per-combo results; Fact must beat Flat by ≥ 0.7%.
- **R4 (implementation)**: masked-softmax leak / BatchNorm mix-up / cost-scale bug. **Mitigation**: `sanity_checks.py` gates before any real run.

## Next Action
- Implement `code/unified_selector/` per `idea-stage/RESEARCH_REVIEW.md §2`.
- Run `sanity_checks.py` (overfit-256, mask leak, cost-scale, tie).
- Launch **R0 → R1-Full seed 0** (see `EXPERIMENT_TRACKER.md`).
- Proceed to `/run-experiment`.
