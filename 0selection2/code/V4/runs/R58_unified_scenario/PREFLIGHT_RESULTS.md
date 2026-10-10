# R58 Full Candidate Preflight: Original Budget

Date:2026-10-11. Implementation commit:`6c9bb2dc8`.

All128 problem/deployment pairs across all18 tasks passed the final frozen
TRAIN-only functionality, independent route feasibility and cost checks. The
final controller occupied the designated3090 for1401.04seconds (23.35minutes),
excluding preceding development probes. No new validation/test instance was
solved, no complete performance table was published, and no selector was trained.

This is NOT completion of R58. Per-deployment min/median/max-size witnesses
cannot certify the full dataset. Full checked cost vectors, one scratch baseline
and locked evaluation remain outstanding.

## Budget Decision

The size-weighted original-TRAIN projection is181.637GPUh for full label
generation alone, with a257.806GPUh adjacent-endpoint conservative reference.
The reference is not a statistical upper bound. Validation/test counts are
projected using the training size distribution; their inputs were not inspected.
Training/evaluation, loading, outer checks and I/O are additional costs.

- Four TSP diffusion deployments:83.096 projected GPUh.
- RouteFinder/MoSES family:61.522 projected GPUh.
- All other deployments:37.019 projected GPUh.

Therefore this exact full-generation recipe is SKIPPED under the user's100GPUh
limit. No label run or deployment lock has been started merely because preflight
passed. See`budget.json` and`BUDGET.md` for assumptions and per-method arithmetic.

## Next Bounded Step

At most3 additional single-GPU hours are allocated to a train-only throughput
pilot. Isolated singleton processes may share the same GPU, but inference batch
shape, random state, starts, augmentation and checking must remain unchanged;
speedup is credited only when the same train-example outputs are reproduced.

If measured acceleration is insufficient, a cheaper inference protocol must be
explicitly declared as a NEW deployment before any new validation/test labels
are generated. Its timing must support a complete run below the cap, with
margin for baseline training/evaluation and overhead. No candidate removal,
partial-release claim or scalar-archive substitution is authorized.

## Evidence

- `preflight/*.json`: current profiles, exact weights/configuration, TRAIN indices,
  saved route witnesses and independent costs.
- `preflight_final_controller.log`: actual successful full candidate traversal.
- `pipeline_state.json`: final status`preflight_complete`.
- `deployment_preflight.csv`, `progress.json`: all128 qualifications current.
- `single_customer_feasibility.json`: sufficient feasible constructions for all
 80000 B-task training inputs; not native solver-output certification.
- `VALIDATION.md`:53 implementation/history tests, plus15 budget tests passed.

The new scenario changes the feasible domain and potentially deployment
budgets. Future new/old score differences must not be claimed as matched model
improvements. No70% Top1 result exists.
