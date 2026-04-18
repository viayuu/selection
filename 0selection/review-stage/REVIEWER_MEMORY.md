# Reviewer Memory

_Persistent across rounds. Loaded at the start of each round._

## Round 1 — Score: 7.2/10 (from prior session, oracle-pro)

- **Verdict**: Almost — "basic approach is sound but three concrete gaps to address before re-assessment"
- **Three prescribed fixes**:
  1. **SBS gate calibration** — confidence-gate the selector's pick against SBS so low-confidence cases collapse to SBS. Reason: selector occasionally ties or slightly under-performs SBS when uncertain; the calibrated gate closes the gap from +0.16% to near 0.
  2. **Combined cost loss** — weighted sum of CE (regret-soft) + expected-regret + switch-hinge. Reason: regret-soft CE alone under-penalises when selector chooses a dominated solver far from the oracle; the switch-hinge explicitly rewards picking the true winner *over* SBS whenever the instance-level VBS-SBS gap exceeds a threshold (2e-3 rel).
  3. **Per-problem per-solver learned bias + oversampling** — add a `problem_solver_bias ∈ R^{P×M}` table (init 0) and oversample the 10 under-performing problems (CVRP×4, VRPB×3, OVRPB×3, OVRP×2, rest ×1.5). Reason: the head-MLP shares weights across all 18 problems; some problem-solver-priors are hard to learn from pooled gradients; a tiny bias table + balanced sampling gives each problem a dedicated knob.
- **Suspicions flagged**:
  - Selector might be learning problem-ID priors rather than instance structure (already falsified by R2 MetaOnly baseline = +75.23% vs R1 +0.16%).
  - 2-epoch pilot may not generalize — oracle wants multi-seed + extended training.
  - ATSP pool has only 3 arms; risk that selector collapses to SBS there. This is OK if macro beats SBS elsewhere, but should be reported honestly.
- **Unresolved going into Round 2**:
  - Does the combined loss actually help vs regret-soft alone? (isolation run needed)
  - Does oversampling hurt under-sampled problems or help them? (check per-problem deltas)
  - Are all 观测指标.md metrics computed? (top1/2/3, per-method win/lose lists, arm distribution, loss curve, and the 3 bar-chart figures)
  - Gate calibration alone brought macro from +0.160% to +0.014% on the 2-epoch pilot; need to confirm this stacks with the other fixes.

## Round 2 — Score: TBD

_(to be filled after Round 2 review)_
