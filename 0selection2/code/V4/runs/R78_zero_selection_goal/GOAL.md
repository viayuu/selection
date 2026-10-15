# R78: Zero-Selection Performance Research Goal

Updated: 2026-10-15 (Asia/Shanghai).
Current workflow phase: round006 closed negative after14904 successful updates
and full TRAIN/VAL; all three gates failed, no TEST, whole B retained. NSS
delivery is complete. Assigned4090 released; no R78 GPU process or automatic
next configuration remains. File goal stays active/unresolved, awaiting a
genuinely different mechanism, NOT awaiting NSS and NOT complete/blocked.
Solmax owns current/future non-paper planning, budgets, gates, code and result
checks. Astra's latest literature/ideas-only assignment ended; original notes
below retain historical authorship. Native/legacy goals were not operated.
See `round006/FINAL_RESULTS.md`, `round006/PROCESS_EXIT_RECEIPT.json` and
`final_delivery/USER_TASKS_SUMMARY.md` for the completed delivery.
Historical author: research-only agent. Canonical research path:
`/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/`.
Existing R76 reports are under `code/V4/runs/`.

## Historical Goal State And Ownership

The original agent split below is retained as provenance, not current routing;
current/future non-paper ownership is Solmax/6.1Sol max as stated above.

- `app_goal_unmodified: true`.
- `app_goal_state: paused` is the user's report about the old application goal;
  this document does not resume, replace, complete, or otherwise mutate it.
- The user reports that creating a new application goal was rejected because an
  unfinished goal exists. R78 therefore uses persistent project documents, not
  a new application-goal allocation.
- `persistent_goal_state: active`; a proposal or completed literature search
  is NOT completion of the performance goal.
- Waiting for the next internal research recommendation is not an objective
  blocker or a termination condition. A worker's blocked subgoal does not
  change this persistent research state; this agent does not operate app goals.
- This agent owns `GOAL.md`, `RESEARCH.md`, and research-only review notes.
  The resource agent owns `STATUS.json`; this agent must not write it.
- Research/results criticism: requested `gpt-6-astra`, effort `max`.
  Implementation and all code review: separate Sol/max owner, never this agent.
- No project source reading/review, tests, experiment execution, GPU access,
  solver execution, Git commits, or reversion of another agent's work here.
- R74/R58 and the historical broad performance goal remain outside R78.

## Objective

Determine when a method with positive winner support is avoidably never chosen,
and improve useful instance-level selection across all 18 tasks without buying
coverage by increasing cost. Zero selected frequency is a diagnostic, not the
optimization target. There is no per-method quota, uniform-frequency target,
or requirement to make every zero disappear.

The research base is now the R77 all18 TRAIN/VAL comparison, not continued
diagnosis or modification of R76. R76 offset exploration is stopped; its sole
TRAIN reference closure was already delegated to Sol and must not be duplicated.
The owner has confirmed both final R77 arms. The permitted diagnostic handoff
contains B at epoch28/update34776; **full-B is the new research baseline**.
Residual research uses its all18 TRAIN/VAL distributions, not an R76 repair list.
Do not attribute the joint R76-to-R77 training changes to batch alone.

## Fixed Scope

- Use the current R76 new-only scenario and its existing TRAIN/VAL/TEST split,
  candidate identities and deployment definitions. `reld_mt` means the retained
  MOEL deployment, not a new solver and not a best-of-two column.
- Do not infer additional candidate removals from shorthand such as "poolB".
  The candidate manifest supplied by the implementation owner is authoritative.
- Cover TSP, CVRP, ATSP, OVRP, VRPB, VRPL, VRPTW, OVRPTW, OVRPB, OVRPL,
  VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW, OVRPLTW, VRPBLTW, OVRPBLTW.
- All task4 new training uses logical batch **1024**, including a learned
  calibration layer. Keep natural within-task winner frequencies and equal
  task weighting. Keep seed2 for the first controlled comparison.
- Existing paired 10k/full, equal-update preparation belongs to its execution
  owner. Do not duplicate it or silently substitute it for a matched control.
- All fitting, diagnostics that influence design, coefficients, candidate
  inclusion decisions, checkpoint decisions, and rule selection use TRAIN/VAL
  only. The already disclosed R76 TEST numbers are background, not tuning data.

## First Mainline: R78-A (Historical Declaration)

Freeze one already locked base model. Fit only candidate intercepts to its
TRAIN scores, separately by problem, with `s[t,j](x) = z[t,j](x) + b[t,j]` and
`sum_j b[t,j] = 0`. This is post-maximin decision calibration, not a claim that
the original utilities or pairwise differences have been retrained.

Fit natural-frequency winner CE with a fixed zero-centered unit-Gaussian
penalty: mean CE plus `||b[t]||^2 / (2*N_train[t])`. Use batch1024 and zero
initialization. Fit one candidate by a TRAIN-only convergence rule; do not tune
the penalty, epoch, task-specific switches, or a temperature on VAL/TEST.
The implementation owner must record its numerical convergence/budget rule
before fitting; lack of convergence is not evidence that intercepts cannot help.

Compare this complete 18-task candidate once against identity on the same VAL.
Promote only if macro native Top1 increases and both macro actual relative
regret and the existing macro cost-vs-TRAIN-SBS metric do not increase. These
are empirical VAL gates, not statistical or per-task safety guarantees.
Report paired uncertainty and every task's costs. No taskwise patchwork.
If the gate fails, retain identity and record the mechanism-screen outcome.
Passing this gate means exploratory screen acceptance, not an all-task repair
or automatic formal release. Disclose each task's adverse changes, preserve
the inherited cost-vs-SBS definition, and predeclare the uncertainty resampling
unit. R76 already used VAL for checkpoint selection; one new comparison and
paired intervals do not remove this adaptive history.

The evidence and exact limitations are in [RESEARCH.md](RESEARCH.md).

## Round002 / R78-A2 (Historical Declaration)

One CPU frozen-score **cost-sensitive squared-offset** screen, still K-1
intercepts per task. No backbone update, vector-slope expansion, prior sweep,
or simultaneous CE-completion branch. Batch1024, seed2, all18 tasks. The exact
objective, fixed TRAIN-only cost normalization, finite SVRG fitting protocol,
convergence certificate, and acceptance criteria are in the
[round002 research plan](review-stage/ROUND002_RESEARCH_PLAN.md).

Use the executed round001 gate: native macro Top1 increases, raw macro mean
cost and actual macro relative regret do not increase, all unrounded and with
zero tolerance. VS-TRAIN-SBS remains reporting-only, as it was in the executed
round001 protocol. This differs from the historical research declaration above;
round001 failed both versions and is not retrospectively reclassified.

The exact CE optimum remains unresolved. Round002 asks whether cost supervision
can help in the same decision class, not whether CE has been disproved or the
backbone has no useful information. Passing the empirical macro gate is not
taskwise safety or independent confirmation. R78-B risk deletion stays deferred.

## Historical Decision: Round005 Closed, Await NSS TRAIN/VAL

This section records the earlier wait, not current pending work. NSS is now
complete and round006 is closed negative; use the current workflow phase above.

`persistent_goal_state: active`; `research_workflow_state: awaiting_nss_locked_train_val`.
Round005 is `fixed_equal_probability_route_closed_negative`: all three VAL
gates failed on the actual ensemble. Keep locked B ep28/u34776. No reweighting,
temperature, logits/voting alternative, taskwise rescue, new training or sixth
similar postprocessor. No formal TEST/LIB or dual-forward benchmark is released.

Round004 is `finite_frozen_score_cost_head_trial_closed_negative`. It completed
the approved24840 updates, batch1024, seed2, and one fixed whole18 VAL candidate.
All three gates failed. Keep locked B ep28/u34776; no TEST release, budget
extension, numerical-certification loop, coefficient retry or fifth similar
head/loss variant. The [head plan](review-stage/ROUND004_FROZEN_SCORE_COST_HEAD_PLAN.md)
is historical, not a new execution instruction.

The supplied reports distinguish actual fitted heads from B fallback. Actual
TRAIN Top1 falls71.176702703%->68.119776364%, and VAL70.894308943%->67.947511054%;
both mean cost and actual regret worsen on both splits. Fewer zero methods
(TRAIN25->21, VAL30->23) do not constitute a repair. VAL fixes3276 decisions
but harms5342, with1406 wrong-to-wrong changes; savings do not cover damage.

These results do not support a purely VAL-only overfitting explanation, but
do not identify a unique failure mechanism or establish absent information,
Bayes-optimal B, or impossibility of conditional-cost learning. The finite
route is closed without asking Sol for more optimizer or precision checks.
The earlier round003 risk replacement and the R76 offset routes stay closed;
none earned an accepted R78 candidate. R77's separate baseline improvement is
not evidence that an R78 intervention succeeded. See RESEARCH.md Section17.

The actual fixed A/B ensemble changes2444 VAL decisions:751 corrected,850
harmed and843 wrong-to-wrong. Native Top1 falls70.894308943%->70.753102268%,
mean cost rises11.395558343544->11.395610637023, and actual regret rises
0.401326745448%->0.402752678645%. Lower CE does not offset these failures.
Zero-with-winner pairs increase30->33, not decrease. These are actual candidate
metrics, not the unchanged B fallback. See RESEARCH.md Section19 and the
[closed historical plan](review-stage/AB_EQUAL_WEIGHT_SCREEN_PLAN.md).

The one70110-row CPU screen took0.290172s, with chunks1024 and no optimizer
updates. That is cached-score processing, not real two-model deployment
latency. The conditional formal-release and dual-forward measurement path
was not activated. The historical proposal still describes a two-model bundle,
not a single net; it is not an accepted artifact. No new TRAIN readout exists.

Existing A/B complementarity did not translate into an accepted equal-weight
rule. This result does not prove absent information, impossible ensembling,
or a unique error-correlation/overfitting mechanism. No further checks or
optimization are needed to close this finite route. Rounds001-005 produced
no accepted R78 intervention under their respective protocols.

The only next research action is NSS's already-planned locked-model TRAIN/VAL
method-coverage and quality handoff. Sol identifies comparable
instances, candidate pools and cost definitions in that handoff; this is not
a request for a new audit or statistics pipeline. Method counts alone are not
new mechanism evidence, and a future difference cannot automatically be
attributed to a particular input role or architecture. Keep the scope all18,
not LEHD-only. No role model, solver rerun, R58 restart or NSS TEST/LIB intake.

The user's bounded Sol update distinguishes the new TSP LEHD column from the
old NSS provenance. Export/profile configuration, including `first_mode='random'`,
is not yet bound to the effective decoder/batch adapter or pre-first-step
node order. Neither actual randomness nor a node0 start, missing role signal
or label noise is established. Preserve this as a deferred execution-conditions
prerequisite only, not a follow-up task or an all18/platform explanation.
No new model, solver, relabeling, source tracing or large audit is authorized.
Details are recorded in RESEARCH.md under Deferred Input-Role Context.

The last timing handoff estimated NSS still needed about3-4h; no job monitoring or hardware
allocation is performed here. Idle GPUs do not justify another experiment.
Waiting for existing results leaves R78 active, not complete or blocked.
The old app goal remains paused and unmodified; STATUS.json belongs to Sol.

## Round003 Historical Declaration And Run Notes

The following launch/interim notes are historical and are superseded by the
current final-result decision above; they do not authorize more continuation.

Stop further round002 optimizer updates and repeated R76 offset/weight tuning.
Its execution status remains `inconclusive`; its practical research disposition
is strongly negative, with no further optimization budget. Even giving the four
unqualified tasks perfect decisions cannot repair the other14 fitted rules.

The sole TRAIN-only reference comparison was already delegated to Sol by the
user. Do not relaunch it or use it to delay the R77 research intake. Its scope
remains zero new training/backbone inference and no VAL/TEST; it is diagnostic,
not a change to original tolerances or candidate selection. Historical scope:
[post-round002 decision](review-stage/POST_ROUND002_DECISION.md).

Use the owner's newly designated full-B checkpoint, epoch28/update34776, as
the reference for the next complete all18 rule. Its permitted VAL report gives
native Top1=70.89430894308944%, actual regret=0.40132674544775454%, and raw
macro mean cost=11.39555834354421. This explicitly supersedes the earlier
R76-incumbent wording for future R78 research, without reclassifying failed
round001/002 trials. Keep the same three-part, zero-tolerance VAL gate against B.

One bounded Astra/max review has completed. The user reports preparation
complete and approves Sol to execute the sole Round003 mainline, a
monotone instance-cost risk-block replacement, retaining native CE, comparisons,
R-Drop, architecture, and natural sampling. The
[final full-B research plan](review-stage/FULLB_MONOTONE_COST_PLAN.md) specifies
one matched original-objective control and one treatment from B's locked weights,
identical optimizer/RNG reset,14904 successful updates each, and just three
predeclared VAL endpoints. All18, batch1024, seed2; no quota, ReLD reduction,
new R76 fitting, parameter sweep, or new paper list. Sol owns any implementation
and execution; neither branch was launched by this research agent.

The latest native-name handoff is complete: all89 task-candidate entries,
TRAIN25 zero-with-winner pairs covering22458/1262052 rows, VAL30 covering
2648/70110 rows, intersection23 and union32. It does not create a32-method
upweighting list. Round003 is a whole18 intervention with no forced selections.
`persistent_goal_state: active`; `research_workflow_state: awaiting_matched_train_val`.
This records the user's latest handoff, not an independently checked job state.
Sol owns preparation, code, execution and code checks. Two RTX3090 GPUs are now
allocated to the independent Control/NewRisk branches, one GPU per branch,
each logical batch1024, seed2, fresh AdamW, sharing B ep28/u34776 starting
weights. This supersedes the previous ready_pending_hardware/NSS four-GPU DP
reservation. The user reports NSS DP rejected with0 updates and its original
forward restored stably on the single RTX4090 gpu05; the other hardware also
has assigned work. These are owner-reported execution facts, not code findings
verified here. No GPU probing, reallocation, job start or STATUS.json edit by
this research agent.

Latest user handoff: both Round003 branches have formally started on the two
allocated RTX3090 GPUs. The reported duration estimate is about80 minutes,
not a runtime checked by this agent. Both retain Bbest, seed2, fresh AdamW,
logical batch1024,14904 successful updates and VAL offsets+4968/+9936/+14904.
Other R77 controls and Sol's necessary implementation checks are complete;
this research agent does not repeat them or monitor the jobs. Await only the
owner-supplied whole18 TRAIN/VAL results for research interpretation and the
decision whether to continue. No new branch or loss is proposed while waiting.

The user has supplied the first two scheduled VAL snapshots at+4968/+9936;
their metrics are recorded in RESEARCH.md without additional report reads or
checks. These are interim observations only, not a request for a theoretical verdict,
new rule, early stopping or parameter changes. Continue the approved budget
through+14904 and await selected-TRAIN and the complete whole18 handoff before
research interpretation. Persistent state remains active and the gate is frozen.

Additional Sol-supplied input-role context is recorded in RESEARCH.md only as
a deferred research hypothesis. No explicit TSP start-role marker is reported,
but R57's candidate source evidence is not yet bound to historical deployment
and cost-column semantics, so it does not explain missed winners. Do not launch
a role model, rerun solvers, restore R58 or inspect code on that basis. Possible
instance-dependent calibration or verifiable role inputs remain unselected
mechanisms for later whole18 judgment after final Round003 TRAIN/VAL results.

The approved experiment is frozen: all18 tasks, batch1024, seed2, inherited
late LR,14904 successful updates per matched branch, VAL only at continuation
epochs4/8/12, the fixed gate, and unchanged core CE/compare/R-Drop. No new loss
proposal, coefficient search, extra precision check or research-review loop
while waiting. Resume research interpretation only on the planned TRAIN/VAL
feedback, still across all18 tasks and never as a local LEHD-only objective.
Do not read completed R77 TEST or LIB to revise strategy. A new formal release
locks one unique VAL-selected whole18 candidate before one complete TEST.

The treatment must pass the fixed B-relative three-metric gate and exceed the
selected matched control in native macro Top1 to support this replacement.
Cost differences against control remain explicit; passing B's gate does not
prove Pareto dominance over control. Negative results close this finite route,
not the continuing R78 goal. No TEST is read to make this decision.

Latest permitted diagnostic snapshot already lists B as ready: VAL native Top1
70.8943%, actual regret0.401327%, zero pairs30, and zero-method winner coverage
3.7769%. A's corresponding values are70.1997%,0.418613%,32,4.9337%; R76's are
69.2369%,0.464503%,54,20.3281%. A/B recovered22/24 of the old VAL zero pairs
without new zero pairs. These are observations, not a new rule or gate verdict.
Selected updates differ: A16146, B34776. The owner additionally supplied the
common-endpoint u49680 comparison: VAL Top1 A69.4808% / B70.8415%, with
TRAIN-to-VAL CE gaps A0.2952 / B0.0100. These support a full-data generalization
advantage under the paired protocol, not a batch-only causal explanation.

`persistent_goal_state: active`. Waiting for these existing results is not an
objective blocker. Old application goal and resource-owned STATUS.json stay
unmodified. R78-B backbone risk deletion remains deferred.

## Iteration Contract

1. Receive TRAIN/VAL reports from the execution owner; record provenance, base
   checkpoint, batch, updates, method order, scores, and split roles. Read
   reports only. Never substitute TEST winner counts for missing TRAIN/VAL.
2. Review conditional support, score separability, changed-decision gains and
   losses, and all 18 tasks. Separate calibration from information and cost.
3. Obtain actual Astra/max research criticism and record its route, response,
   unresolved issues, and same-family/provisional status. Do not invent PASS.
4. Require new mechanism evidence before a new intervention. Round003 and
   round004 are closed negative; the next action is the existing NSS locked
   TRAIN/VAL handoff, not a fifth similar variant. Historical risk deletion,
   offset/scaling retries and role models are not automatic alternatives.
5. Incorporate the next supplied results here and in RESEARCH.md. Rejection of
   A, unchanged zeros, completion of a report, or unavailable results is not
   completion of this persistent goal. Continue on new TRAIN/VAL evidence.
6. Before formal release, freeze ONE base checkpoint and ONE complete rule,
   including calibration weights, candidate ordering/masks, normalization and
   tie policy. Apply it to the complete 18-task TEST once. Use the same weights
   and rule on LIB, with no LIB-label fitting or LIB-specific calibration.

The previously observed TEST is not restored to statistical freshness by one
more frozen evaluation. Disclose the prior diagnostic exposure. Subsequent
method decisions stay TRAIN/VAL-only, even after the formal TEST report.

## Closure Criteria

The R78 head interventions have not earned promotion; R77 now supplies observed
TRAIN/VAL improvements and the new research reference candidates. Closing this
goal still requires a documented all-task TRAIN/VAL conclusion, actual research
review, a locked-rule evaluation if a formal version is delivered, and an
explicit outcome: useful improvement, or evidence-bounded lack of improvement
with the remaining gap stated. No claim that every zero is erroneous or that
all 18 tasks improve follows from a macro improvement. The user's ongoing
iteration requirement remains active until an agreed outcome or explicit stop.

Current research state: Round004 closed negative, actual fitted TRAIN/VAL
received, B retained; await the existing NSS locked TRAIN/VAL handoff.
Only the five designated round004 reports were read as new experiment evidence
for this closure; prior research context is retained without new source reads.
No R77 root results, TEST, LIB or statistics-status file was opened. Persistent
state remains active; neither a completed proposal nor this negative trial
completes the ongoing performance goal. See RESEARCH.md
and [research review record](review-stage/RESEARCH_REVIEW.md).
