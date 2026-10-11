# R58: Unified scenario_v2

**Status: implementation and bounded qualification complete; full regeneration
and training SKIPPED because the measured whole-plan reference exceeds the
100 GPU-hour limit. R58 is not a completed benchmark/model experiment.**

## Completed Work

- Executable common task contract and independent route feasibility/cost checker.
- Classical backhaul adaptations with the original fixed weights, explicitly
  NEW deployments rather than historical or unmodified-paper reproduction.
- Fresh TRAIN-only qualification: **128/128 problem-method deployments passed**.
  All22 identities and original candidate orders are retained; no method or
  difficult instance was removed to obtain this result.
- One fixed final TRAIN-only timing refinement: **63/63 deployments,
  159/159 sessions**, covering all48 RouteFinder-family deployments under
  serial1, normal4 and private MPS4, plus15 other refined deployments.
- Exact RF-family route/cost equivalence and actual MPS client attachment were
  checked. MPS offered no global benefit: normal4 projects31.414276h versus
  MPS4's31.529182h. The measured plan therefore uses normal4 and credits
  **zero MPS acceleration**.
- Slurm1465.78 finished with exit0:0, elapsed5215s. The settled cumulative
  three-hour pilot ledger is8088.902146s; failed bootstrap and outer overhead
  are preserved and charged, not discarded or double-counted.
- Only the assigned gpu03 RTX3090 UUID
  `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` was used. The GPU baton is released.

The earlier original recipe was also over budget and remains archived under
`original_recipe/`. Its qualification is not substituted for the revised one.

## Final Budget Decision

The complete128-deployment projection covers1,536,000 solver-instance solves:
10000train +1000validation +1000test per deployment. Validation/test size
distributions are estimated from TRAIN; their data were not used to tune the
deployment or timing recipe.

| Fixed component | GPU hours |
|---|---:|
| Actual prior/preflight/pilot/refinement spend | 3.330399 |
| Remaining label reference, all128 deployments | 95.281179 |
| Measured63 session overhead, counted once | 1.109229 |
| Baseline/preparation/evaluation allowance | 3.000000 |
| Otherwise-unaccounted overhead allowance | 2.000000 |
| Fixed10% label contingency | 9.528118 |
| **Whole reference** | **114.248924** |

The63 refined references already include their session overhead; it is removed
before constructing the95.281179h label component and then added once. The
unchanged65 deployments contribute7.215483h using current conservative
preflight references. The allowance and contingency are planning assumptions,
not measured completion times or confidence bounds.

Decision: **`skip_whole_plan_at_or_over_100h`**. The actual CPU-only production
gate was exercised with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m code.V4.r58_pipeline --stage labels
```

It returned **exit3**, recorded `skipped_over_budget`, and did not create an
execution approval, execution plan, deployment lock, complete scenario release,
baseline checkpoint or test result. No further recipe search, candidate cut,
contingency reduction or legacy-cost training was started to evade this gate.

## Task Contract

- Deliveries precede pickups on each route; delivery and pickup sums separately
  obey capacity. Pure pickup routes are allowed. This is not signed-load or MB.
- Customer TW constrains service start; waiting and service consume time.
  Missing depot closing time means infinity, not an inherited deadline3.
- L limits travel distance, not waiting/service. Open routes exclude return
  distance and return time; all customers are visited exactly once.
- Every published cost must have a saved route passing independent simulation,
  not merely the solver's own action mask.
- Original instances, splits, labels, results and native solver source trees
  remain unchanged. Historical B costs are not certified scenario_v2 costs.

## Not Executed / Interpretation

No full new train/validation/test performance table has been generated or
published. Consequently the one from-scratch full18 R45A baseline, locked
validation selection, test evaluation, and new train-fitted references have
**not run**. There is no new Top1, Oracle, regret or model-improvement claim.
Small TRAIN route witnesses establish bounded deployment qualification, not
feasibility of all1,536,000 future solves.

Completing this fixed full18 plan requires a changed compute authorization or
an explicitly new scope decision. The measured114.25h is a planning reference,
not a guarantee or a claim that additional compute will improve Top1. The
post-R58 performance loop and70% target remain unachieved; they were not run
against the semantically inconsistent legacy cost table.

## Evidence

- `task_contract.json`, `deployment_preflight.csv`, `preflight/*.json`:
  common definition and current128 qualification.
- `BUDGET.md`, `budget.json`: complete arithmetic, TRAIN histograms, accepted
  current profiles, refined references and unchanged65 estimates.
- `timing_refinement/{selection,run_state,summary}.json`: fixed selection,
  all-mode records, global RF decision and settled spend.
- `timing_refinement_raw.tar.gz`: complete raw session/worker route, cost,
  equivalence and timing evidence; the unpacked directory remains on server.
- `final_refinement_*spent.json`, `final_refinement_result.json`,
  `final_refinement_sacct.log`: actual execution and one-time outer settlement.
- `bootstrap_failure/1465_77/`, `bootstrap_repair_evidence.json`:
  preserved premeasurement failure and narrow JSON-key repair.
- `pipeline_state.json`: observed production budget-gate exit, with no solve
  or training subprocess launched.

Source entry points and commands are documented in `../../R58_README.md`.

Primary and fresh artifact reviews found no actionable defect in this bounded
budget result (both9/10, ready for the skip conclusion only). These are
same-family/provisional reviews, not acceptance of whole R58 or a performance
claim. Full verbatim responses are in `final_evidence_review.md`.
