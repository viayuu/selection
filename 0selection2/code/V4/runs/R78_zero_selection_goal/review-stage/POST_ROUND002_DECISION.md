# R78 After Round002: Stop Offset Optimization, Await R77 Final Baseline

Date: 2026-10-15 (Asia/Shanghai). Research state: active.
Old application goal: paused and unmodified. STATUS.json belongs to the resource owner.
Research owner writes documents only; existing Sol/max owner handles execution.

## Decision And Evidence

Stop round002 optimizer updates now. Do not spend another round on R76 offset,
weight, temperature, margin, or prior-strength variants. The fixed-scale squared
offset proposal was not supported by observed TRAIN behavior. This is not a
claim that all calibration or all cost-sensitive learning is impossible.

Preserve execution `outcome=inconclusive` and the original numerical flags.
Record a separate research disposition: **strongly negative fitted rule;
no further optimization budget for this route**. The 1e-6 parameter tolerance
does not prevent a practical negative conclusion.

| TRAIN metric | R76 identity | Round002 fitted rule |
|---|---:|---:|
| Native Top1 % | 69.369645625 | 40.849743117 |
| Actual regret % | 0.460650092 | 2.826220296 |
| Raw macro mean cost | 11.373004840 | 11.653491213 |

All18 tasks worsen all three metrics, including all14 numerically converged
tasks. TSP and ATSP worsen as well as MVRP: this is not one ReLD binary problem.

Holding the14 converged FITTED rules fixed, even perfect native winner selection
and zero regret on the four remaining tasks give macro Top1 at most
**57.405479331%** and actual regret at least **2.485069032%**. Both still fail
identity badly. These are arithmetic bounds from the per-task TRAIN report,
not new experiments. They do not assume fitted and analytical-reference
predictions are identical on the14 tasks.

Small coefficient/objective errors cannot alone guarantee identical argmax
actions. Resolve that remaining question once with the known reference below;
do not extend optimization or introduce another tolerance ladder.

## One Permitted Sol Readout

Using existing TRAIN scores/costs, original identity, fitted b and the already
known closed-form b*, perform one CPU-only all18 metric pass:

1. Report identity / fitted / reference native Top1, raw mean cost and actual
   regret per task and with unchanged equal-task macro aggregation.
2. Report fitted-versus-reference changed predictions, corrected/broken Top1,
   and signed cost/regret differences. This measures decision relevance.
3. Optionally explain stability with the maximum perturbation of the full K
   class scores. With the same frozen h this is `epsilon=||b-b*||_inf` on the
   restored full K-vector, not an arbitrary reduced-coordinate norm. A fitted
   top-two margin strictly above `2*epsilon` cannot change winner under that
   perturbation. Keep the original global-ID tie policy; this is not a new gate.

Zero optimizer updates, zero new backbone forwards, no new fitting, no VAL,
no TEST/LIB, no coefficient or scale selection. b* is diagnostic only, never
a deployed candidate or retroactive convergence pass. No extra diagnostic
models, bootstrap campaign, or plotting package is required.

If b* remains comparably poor, close this route. An unexpected reversal returns
to research interpretation only: no automatic VAL access, deployment, or resumed
optimization. Stopping optimizer work does not wait for this confirmation.

## Single Next Mainline

**Finish the ALREADY RUNNING R77 all18 matched-update baseline and receive its
final TRAIN/VAL evidence before opening a new backbone or loss intervention.**
Do not duplicate R77, start another frozen-R76 head trial, or launch a loss sweep.
This existing dependency is an active research phase, not a blocked/completed goal.

The user reports both R77 arms have interim VAL Top1 around70.4%, whereas R76's
selected base had only2,592 updates at batch9,856. Those interim R77 numbers were
not independently inspected in this intake. They favor resolving optimization
adequacy and base-checkpoint choice before another intervention; they do not
isolate batch size, update count, data size, or schedule causally.

Request only a compact existing R77 handoff:

- Final checkpoint under R77's existing VAL rule, effective batch, seed, updates,
  TRAIN row counts by arm, and final TRAIN/VAL native Top1, mean cost and actual
  regret for all18 tasks.
- One all-method table of TRAIN/VAL winner support and selections, including
  R76-to-R77 cold-pair transitions. Report counts without inventing a near-zero
  threshold. Reuse saved predictions, not a new diagnostic model.

Keep all18 tasks, natural frequencies and one checkpoint/rule, without taskwise
assembly or ReLD-versus-rest reduction. A remaining zero pick is not proof of
unlearnability. No arbitrary new TRAIN loss is prescribed before this evidence.

R77 as an experimental reference and the accepted incumbent are different roles.
Original R76 identity stays incumbent until the fixed whole18 VAL Top1/raw-mean-
cost/actual-regret gate permits replacement. Higher Top1 alone cannot reset the
cost reference to a worse R77 rule. VS-SBS stays reported; TEST is not a selector.

The user-reported NSS new-CVRP gap0.7115% is a positive lead, not yet a matched
comparison to actual regret1.4183%. Confirm split/pool and mean-relative-gap
versus ratio-of-means definitions from reports before comparing the values.
This is not a new solver/semantic audit. Better performance with richer inputs
does not prove that old frozen R76 scores retain that information.

Any later new training remains TRAIN-only, all18, batch1024 and seed2, with the
fixed pre-TEST gate and identical locked TEST/LIB rule. No new optimizer training
or backbone modification is authorized by this handoff.

## Sources And Review

Numerical sources: [RESULTS.md](../round002/RESULTS.md),
[result_summary.json](../round002/result_summary.json),
[train_comparison.json](../round002/train_comparison.json).
The research owner did not run the reference readout.

One bounded actual Astra/max review completed and supports this decision,
same-family/provisional: [request](ROUND002_CLOSURE_ASTRA_MAX_REQUEST.md),
[verbatim response](ROUND002_CLOSURE_ASTRA_MAX_RESPONSE.md). Its disposition is
recorded in RESEARCH_REVIEW.md. No repeated review loop, new papers, code review,
or app-goal operation is part of this plan.
