# R78 Research Review Record

Date: 2026-10-15 (Asia/Shanghai).
Scope: research design and result interpretation only, never code review.

## Routing And Status

- Requested and CLI-confirmed reviewer: `gpt-6-astra`, reasoning effort `max`.
- `review_independence: same-family`.
- `acceptance_status: provisional` for the exploratory mechanism-screen design
  only, not a performance claim, code certification, or goal completion.
- Initial review completed and textual findings incorporated. A subsequent
  actual Astra/max review of round001 TRAIN/VAL evidence and the round002
  research decision is recorded below. Sol/max code reviews belong elsewhere.
- `app_goal_unmodified: true`; old application goal stays paused.
- Resource agent owns `../STATUS.json`; this record never overrides it.

## Round 0: Primary-Source Synthesis

The initial recommendation is R78-A: freeze the existing base, fit per-problem
candidate intercepts on natural-frequency TRAIN, batch1024, fixed unit-Gaussian
shrinkage, then one all-task VAL accept/reject decision against identity.
The latest user direction selects TRAIN fitting, replacing the earlier
validation-fit suggestion. No TEST-derived arm subset or coefficients.

Research deductions requiring no source-code inspection:

1. Under the stated pure-difference maximin map, the utility winner is the
   maximin winner. Global positive temperature cannot alter it; positive
   per-method rescaling of unshifted maximin scores also preserves it.
2. Prior-adjusted losses target a specified class-balanced or shifted objective,
   not automatically natural Top1 or expected solver cost.
3. Positive conditional winner support is insufficient when a method's losses
   overwhelm its gains. Constant selection is not automatically erroneous.
4. Intercepts can change boundaries but cannot invent new within-score ranking
   information. TRAIN/VAL cost-severity and separability summaries are required.

## Questions For Actual Astra/max Review

- Is the maximin sign/order deduction valid under the explicitly stated
  mathematical assumptions, without making a claim about uninspected code?
- Does the TRAIN-fit, VAL-assessed intercept proposal answer a useful mechanism
  question given in-sample base-model scores? What failure is non-identifiable?
- Are the fixed penalty normalization and one-candidate gate clear enough to
  avoid an implicit coefficient or checkpoint search?
- Does requiring macro Top1 improvement and non-increased macro regret plus
  original cost-vs-SBS measure hide taskwise harm or overstate significance?
- Is the conditional matched risk deletion R78-B justified as an isolation
  control, rather than a premature assertion that the risk term caused zeros?
- Are all 18 tasks and methods covered without TEST-selected exceptions?
- Are the original-paper assumptions, especially P3 balanced error and P8
  inference-time cost matrices, accurately limited?
- Does the persistent plan remain iterative without resuming the old app goal,
  running tests/GPU work, or treating the proposal as completion?

## Initial Evidence Intake (Historical)

Await actual TRAIN/VAL identity-versus-intercept results, fit convergence,
conditional score/cost summaries, and the existing matched-update comparison
when its owner supplies it. No result is fabricated in this record.

## Round 1: Actual Astra/max Review

Invocation evidence from the CLI header and completion:

- CLI: OpenAI Codex v0.162.1; provider: OpenAI.
- Actual reported model: `gpt-6-astra`; actual reported reasoning effort: `max`.
- Session ID: `01a13b2d-505c-7bc0-b6cf-e951edd2ffe5`.
- Working directory: `/public/home/shiys/0selection2`.
- Mode: read-only, approvals never, ephemeral session, process exit code0.
- Native delegation tools were unavailable; a single read-only CLI reviewer
  was used. Transport reconnect messages occurred but no second invocation,
  model downgrade, or parallel reviewer was launched.
- Inputs: [request](ASTRA_MAX_REQUEST.md), GOAL.md, RESEARCH.md, this record;
  primary-paper browsing allowed. No project-code or TEST-report access allowed.
- Output: [verbatim final review](ASTRA_MAX_RESPONSE.md), retained separately
  from the parent synthesis below. No edits or experiments by the reviewer.

Verdict: ready as an exploratory mechanism screen, subject to TRAIN-only
convergence declaration and explicitly limited interpretations. Not a causal
diagnosis, independent confirmation, or performance guarantee.

| Finding | Severity | Disposition |
|---|---|---|
| Passing offsets need not recover instance discrimination; failure does not rule out all useful offsets | High | Adopted. Narrowed result-to-claim wording; require constant/conditional reference comparisons; retained in-sample TRAIN caveat. |
| Macro acceptance does not guarantee taskwise safety or statistical non-inferiority; prior VAL use matters | High | Adopted. Expose adverse taskwise effects, preserve inherited metric, predeclare paired resampling unit, disclose adaptive VAL use. |
| A Top1/cost trade-off alone is insufficient to trigger risk-term deletion | Medium | Adopted. B remains deferred pending a separate detrimental-interaction hypothesis; frozen-reference fallback if no eligible checkpoint. |
| Maximin sign proof is sound but does not identify the uniquely correct intervention location or training dynamics | Low | Adopted. Restrict necessity to positive diagonal transformations of unshifted scores; separate fixed-score ranking from training gradients. |
| Fixed penalty is well posed but requires recorded numerical convergence | Low | Adopted. State frozen score units, inconclusive budget exhaustion, and the marginal stationarity equation without treating tiny gradients as a decision guarantee. |
| Paper assumptions are substantially correct but do not establish R78 guarantees | Low | Adopted. Explicitly reject transferring calibration/consistency guarantees to the mixed objective and macro gate. |

Decision: retain only A as the immediate mainline. No experiments, coefficients,
taskwise exceptions, TEST access, or old-goal resumption were authorized by this
review. Continue with incoming TRAIN/VAL evidence; the R78 goal remains ongoing.

## Round 2: Actual Results Review And Next Research Decision

Read-only CLI invocation: OpenAI Codex v0.162.1, provider OpenAI, model
`gpt-6-astra`, reasoning effort `max`, approvals never, ephemeral, exit code0.
Actual session ID: `01a13b71-ddf4-75a1-a7b2-fe4fe43e92a3`.
Request: [ROUND002_ASTRA_MAX_REQUEST.md](ROUND002_ASTRA_MAX_REQUEST.md).
Verbatim response: [ROUND002_ASTRA_MAX_RESPONSE.md](ROUND002_ASTRA_MAX_RESPONSE.md).
Scope was restricted to the listed round001 TRAIN/VAL reports and existing P7;
no code/TEST/raw-array access, edits, experiments, or app-goal operations.

The review confirms the observed gate rejection while keeping the exact CE
optimum unresolved. It selects one next mainline, not a menu: cost-sensitive
squared-offset fitting in the same K-1-parameter decision family. Completing
CE is not assumed useless; it is not scheduled concurrently with this screen.

| Research finding | Adopted consequence |
|---|---|
| Top1/CE improve but the actual cost gates fail on VAL; TRAIN has the same aggregate direction | Retain original identity; test cost magnitudes within the frozen decision class. |
| Only one task met numerical convergence | Keep the CE optimum unresolved; no intrinsic information-limit claim. |
| Zero-pick totals hide changed method membership | Report method identities and changed decisions; no coverage objective. |
| Native Top1, raw mean cost and relative regret are different objectives | Freeze the executed gate and expose the historical VS-SBS declaration discrepancy. |
| Squared loss can raise scores below -1 and depends on fixed score units | Declare this limitation and fix the scale, normalization and prior; no alternate-loss/temperature scan. |
| A tiny convex model permits an exact numerical reference | Require real minibatch1024 SVRG updates and a separate TRAIN-only optimum audit; no closed-form substitution. |

The reviewer supplied the exact finite optimizer, convergence criteria and gate;
they are carried into [ROUND002_RESEARCH_PLAN.md](ROUND002_RESEARCH_PLAN.md).
Same-family/provisional recommendation only, not implementation assurance or a
prediction of performance. No second optimizer, backbone change, coefficient
grid, class-frequency balancing, or TEST-guided selection is authorized.

Current persistent research state: **active**. Waiting for this internal plan
was not an objective blocker. The plan is handed to the existing Sol/max owner;
the research agent does not claim to have launched it or changed STATUS.json.

## Round 3: Practical Round002 Closure And R77 Intake

This is a research review round, not a third training experiment. One bounded
read-only CLI review completed: OpenAI Codex v0.162.1, provider OpenAI,
model `gpt-6-astra`, reasoning effort `max`, approvals never, ephemeral, exit0.
Actual session ID: `01a13bcc-c9b7-77f0-8a47-7f0cf56ccada`.
[Request](ROUND002_CLOSURE_ASTRA_MAX_REQUEST.md) and
[verbatim response](ROUND002_CLOSURE_ASTRA_MAX_RESPONSE.md).
Only the request and three specified round002 reports were read. No subagents,
new papers, code/TEST access, experiments, edits, or app-goal changes were part
of this review. No follow-up review loop was launched.

Verdict, same-family/provisional: stop optimizer updates now; complete the
already running R77 all18 baseline as the sole next mainline. Keep the old
round002 numerical outcome and 1e-6 criterion unchanged, but do not let them
prevent the practical negative resource decision.

The all18 degradation and fixed14-plus-perfect4 bounds support stopping.
Small parameter errors do not alone prove unchanged actions, so at most one
TRAIN-only fitted/reference decision readout is justified. The analytical
reference is not a deployed candidate. The optional margin bound uses the full
class-score perturbation; the full-K offset difference is equivalent here.

Adopted: [POST_ROUND002_DECISION.md](POST_ROUND002_DECISION.md), latest GOAL.md
and RESEARCH.md Section11. The proposal's fixed-scale restricted-family
assumption was not validated. It does not follow that all cost-sensitive learning,
all calibration, or input information is absent. No method-semantic or ReLD
binary explanation is established.

R77/NSS interim numbers remain user-reported leads. Receive the existing final
R77 TRAIN/VAL metrics and method-count table before choosing a new intervention;
do not replace the incumbent or reset cost thresholds on higher Top1 alone.
R78 remains active. Old app goal and resource-owned STATUS.json remain unchanged.

## Round 4: Final Full-B Residual Mainline

One bounded read-only CLI review completed: OpenAI Codex v0.162.1, provider
OpenAI, model gpt-6-astra, reasoning effort max, approvals never, ephemeral,
exit0. Session ID: `01a13c19-8089-72c1-b500-66e8c68edfe6`.
[Request](R77B_MAINLINE_ASTRA_MAX_REQUEST.md),
[verbatim response](R77B_MAINLINE_ASTRA_MAX_RESPONSE.md),
[final research specification](FULLB_MONOTONE_COST_PLAN.md).

The reviewer read the self-contained request, not project code or raw arrays.
The parent had read only the permitted TRAIN/VAL handoff and B report CSVs,
and revisited existing primary papers online. No R77 root results, TEST/status,
checkpoint contents, code review, experiments or new paper survey. No nested
reviewer or second review invocation. This is not independent metric verification.

Verdict: proceed with one finite, matched all18 trial of the monotone instance-
cost replacement. It is distinct from failed fixed offsets but remains an
empirical hypothesis. review_independence: same-family;
acceptance_status: provisional for the trial design only.

| Finding | Adopted consequence |
|---|---|
| Winner-bin signal and zero-method loss shares do not establish cost-beneficial switching | Treat them as motivation/attribution only, not recoverable gain or a target-method list. |
| Centered softplus has the intended direct-score gradient, but centering couples actions | No claim that all costly nonwinner scores fall; no extrapolation to neural Pareto improvement. |
| Relative regret, raw mean cost and native Top1 differ; task normalization changes shared-model weighting | Keep all three native gates, disclose normalization and label ties; no claim of optimizing unscaled macro regret exactly. |
| Mean cost-weight normalization does not match old gradients or bound tails | Fixed .02 is preregistered, not theoretically optimal; test the whole block, no scale/weight sweep. |
| Reset continuation needs a reset control; one seed and reused VAL limit conclusions | Same B weights and reset protocol,14904 updates each; do not use unmatched original continuation as control. |
| Beating B alone cannot establish a replacement benefit | Treatment must pass B's gate and exceed matched control Top1; report both cost differences against control. |

Final parent narrowing, before any new run: use continuation epochs4/8/12 as
the only three candidate VAL checkpoints for each branch, instead of all12
in the review draft. This limits evaluation/selection opportunities and follows
the user's few-checks requirement. The raw review is preserved; no claim that
this narrowing received a second review. No new metric or coefficient is chosen.

The user explicitly designated full-B as the new research baseline. This
supersedes earlier R76-incumbent/waiting wording without promoting a rejected
R76 rule. The persistent goal stays active; app goal and STATUS.json remain
unmodified. Sol owns deciding/scheduling execution and every code check. The
research agent has not launched either training branch or opened TEST.

Subsequent final native-name handoff: the user supplied the complete89-entry
B TRAIN/VAL supervision report and32-pair zero-set union, consistent with the
same locked B reference. The parent read those reports and clarified predicted
probability versus greedy argmax, true posterior versus winner-conditioned
statistics, and within-task raw-costloss versus macro-regret denominators.
The single mainline is now the proposed training Round003. No second reviewer
call or precision-check loop was added. Sol may prepare it; hardware allocation
must respect NSS priority before training starts. Goal stays active.

## Round 5: Round003 Final Verdict And One Bounded Next Proposal

One read-only CLI review completed: OpenAI Codex v0.162.1, provider OpenAI,
model gpt-6-astra, reasoning effort max, approvals never, ephemeral, exit0.
Session: `01a13cd9-0547-7e33-95ed-a84ff974ee4b`.
[Request](ROUND003_FINAL_ASTRA_MAX_REQUEST.md),
[verbatim response](ROUND003_FINAL_ASTRA_MAX_RESPONSE.md),
[single proposed next protocol](ROUND004_FROZEN_SCORE_COST_HEAD_PLAN.md).

The reviewer used the self-contained request, not independently checked code
or experiment outputs. The parent read only the five user-designated round003
reports and the original R77 TRAIN/VAL handoff as new result evidence, and
revisited existing primary papers. No TEST/LIB, raw arrays, checkpoints, source
code, numerical audits, experiments, subagents or follow-up review calls.

Verdict: round003 remains closed negative, B retained. Improved CE and the
zero-coverage accounting change do not establish better decisions. Selected
TRAIN rows are B fallback rows, so no fitted TRAIN gain or overfitting estimate
can be inferred. There is no reason to extend the failed risk-block trial.

The review supports one CPU-first frozen-B conditional-relative-cost head as
a distinct, limited hypothesis test, not an expected improvement. Adopted:

- The conditional-mean target treats B probabilities as features; it does not
  require them to be true calibrated winner posteriors.
- Complete candidate costs give supervision, not evidence that a useful
  cost/Top1 improvement region is represented or linearly predictable.
- Restricted regression, own-TRAIN scores, large MSE residuals and finite
  optimization limit the unrestricted conditional-mean argument.
- Relative regret, raw mean cost and native accuracy can disagree even under
  perfect cost prediction. Retain all three unmodified VAL gate conditions.
- A513-parameter head and a zero-centered prior do not ensure B-like decisions;
  the fixed Adam endpoint is not a certified numerical optimum.
- Keep batch1024, seed2,1380 updates/task, the fixed optimizer/penalty/LR,
  one final whole18 VAL candidate and no taskwise VAL substitutions or retries.
- Completed gate failure closes this finite route; an unfinished/unstable fit
  is incomplete. Both retain B without proving absent information.

review_independence: same-family. acceptance_status: provisional for one
bounded trial design only. R78 remains active. No app-goal or STATUS.json
operation, no execution by this research agent. The user's latest resource
note does not require filling idle3090s; this proposal needs no new GPU run,
NSS result or additional preflight statistics/audit pipeline.

## Round 6: Round004 Negative Closure, No Fifth Variant

One bounded read-only CLI review completed: OpenAI Codex v0.162.1, provider
OpenAI, model gpt-6-astra, reasoning effort max, approvals never, ephemeral,
exit0. Session: `01a13cf7-f078-7381-b063-4c6c435674dd`.
[Request](ROUND004_CLOSURE_ASTRA_MAX_REQUEST.md),
[verbatim response](ROUND004_CLOSURE_ASTRA_MAX_RESPONSE.md).

The request explicitly prohibited a fifth method, optimizer retry, additional
statistics/audit pipeline, paper search, code prescriptions and any execution.
The reviewer read only the self-contained request. The parent's new experiment
evidence was confined to the five user-authorized round004 report files. No
source, arrays, checkpoints, TEST/LIB, GPU/solver work or nested/follow-up review.

Disposition: `finite_frozen_score_cost_head_trial_closed_negative`; retain B.
The actual fitted candidate worsens TRAIN and VAL native decision metrics and
fails all three VAL gates. The earlier provisional recommendation did not
produce an accepted improvement. No fifth similar variant or rescue budget.

Adopted claim boundaries:

- Actual TRAIN degradation makes a solely VAL-overfitting account insufficient;
  approximation, objective/decision mismatch, regularization, optimization and
  information limits are not separately identified by these reports.
- Frozen-B CE does not assess the cost head, and unreported fitting MSE must
  not be described as improved. Isolated task/metric gains are not an accepted
  complete rule and do not authorize taskwise checkpoint/head assembly.
- Lower zero-pair counts and changed coverage do not demonstrate useful
  recovered decisions. Cost-category contributions use the existing task-
  normalized macro metric, not an unqualified cross-task physical-cost sum.
- The four negative R78 conclusions hold under their respective protocols;
  they are not a common-baseline causal ranking. R77's gains are separate, and
  round003 fallback TRAIN rows cannot reveal its fitted endpoint behavior.

The sole next research action is the existing NSS locked-model TRAIN/VAL
handoff, with comparison scope/definitions identified by Sol. No fresh pipeline
is requested. Coverage alone or a future quality advantage cannot identify a
specific architectural/input-role cause. R78 remains active, awaiting that
evidence, not complete or blocked. Old app goal and STATUS.json remain untouched.

review_independence: same-family. acceptance_status: provisional for this
research interpretation, not independent verification or code certification.
