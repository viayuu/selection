# Full18 Useful Method Recall Research

Updated: 2026-10-20. Application goal: active. Research objective: unresolved.

The user asks to analyze completed experiments that have not yet been analyzed,
then continue research and experiments that improve full18 Top1 while reducing
missed instances whose genuine-winning method is never selected. Four idle
authorized GPUs may run distinct experiments concurrently.

The preceding turn made verified progress: the new90k dataset was released and
all four R84 arms subsequently completed training, unique VAL selection, TEST,
LIB and PDF reporting. Those completed experiments did not achieve the research
goal. This continuation starts from their actual results rather than relaunching
them or treating an old tmux name as a live worker.

## Ownership

- Paper research and new improvement ideas: gpt-6-astra, reasoning max.
- Analysis, experiment planning, implementation, review, execution and delivery:
  gpt-6.1-sol, reasoning max.
- One brief implementation review per round; no repeated broad preflight audit.

## Fixed Data And Evaluation

Use `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`: 81,000
TRAIN, 4,500 VAL and 4,500 TEST per problem, all 18 problems. Reuse the frozen
R84 caches and TRAIN-only geometry statistics, SBS and fresh seed2 initializer.
No old 10k-prefix data, new solver runs, R58 relabeling or candidate changes.

Logical batch remains 1024. All new models start from the shared fresh weights;
old trained selectors are historical references, not warm starts. Record any
architecture, supervision, scheduling or checkpoint-selection change before
training. Existing TEST/LIB has influenced this research and is not fully blind.

Every formal version locks one checkpoint using VAL, then reports that same
checkpoint on complete TEST and both LIBs. Preserve native FP64 winner labels,
equal task weights, TRAIN SBS and the distinction between mean cost ratios and
per-instance actual regret. Deliver the paired chosen/winner distribution PDFs,
the four-dataset NSS table, full18/task metrics and machine-readable CSVs.

## What Counts As Progress

Track full18 Top1, actual regret, mean cost, genuine-winner recall, new false
picks, corrected and newly wrong decisions, and their paired cost differences.
Use all methods, with zero/near-zero groups as a diagnostic subset. Fewer zero
groups, a higher TRAIN score or a completed run is not completion of the goal.

Each round first reports its exact empirical VAL acceptance gate. TEST results
describe a locked version and cannot change its checkpoint or inference rule.
Substantial useful recall should occur on adequately supported methods across
multiple tasks; isolated picks of extremely rare methods do not resolve the
objective. Keep negative results and do not automatically scan coefficients.

## Current Evidence

R84's three modifications fail the matched VAL gate. Relative to its baseline,
direct attention, additive focal, and their combination recover 21, 10 and 46
of 4,614 missed target winners, respectively, with new false picks. Their TEST
Top1 changes are small and do not establish useful coverage repair. Historical
R80-R83 analysis and a fresh TRAIN/VAL coverage analysis are complete in
`prior_experiments/PRIOR_EXPERIMENTS.pdf`. TRAIN suppression is directly
observed: R84A correctly selects LEHD on only 7 of its 14,279 TRAIN winners;
MATPOENET has 6,476 TRAIN winners and no selections. This does not establish a
unique cause or prove that reweighting can improve the natural test objective.

## Pending Prototypes

The four recipes are locked in `protocol_lock.json`: natural OVR; its
EQLv2-inspired gradient-balancing counterpart; conditional nonwinner cost
profile KL while retaining the hard winner; and actual candidate-subset
training with labels derived from the existing exact comparisons. Implementation
and focused integration tests passed. Four fresh runs started on the assigned
two 3090s and two 4090s after the lock; initial full18 updates are finite with
zero AMP retries. A/B share the 3090 execution limits. C/D use 4090s; D's
TSP/CVRP physical microbatch is 256 to allow its auxiliary branch while
retaining the logical batch of 1024. No R85 optimizer update had started at
the protocol lock. The research objective remains unresolved.

## Live Progress

The bounded startup witness is `../R85_launch/launch_startup_check.json`: all
four real full18 cycles passed, with verified allocation/container/GPU UUID and
physical process identities. No cache copy or solver run was needed.

`tmux r85_goal_watch` updates `goal_progress.json` every 60 seconds. The first
full VAL evaluation has completed for each arm during warmup; these are not
final results or an acceptance decision. Training, unique VAL lock, full TEST,
LIB and per-model PDF delivery are automatic. The CPU watcher then produces
`comparison_artifacts/COMPARISON.pdf`, including the fixed R84 targets and the
matched B-versus-A OVR comparison. A small empirical pass does not complete
this research goal. Further iterations follow the completed evidence.

Next rounds follow the measured outcomes and new literature evidence. The
application goal remains active until its full objective is verified or the
user pauses it; it is not narrowed to launching these prototypes.

## Bounded Readout Geometry Diagnostic

On 2026-10-20, a CPU-only diagnostic of the frozen R84A update37440 used
input-stratified 64 TRAIN and 64 VAL examples per problem. It measured hidden
feature radius and angle to the existing shared scalar readout vector; it did
not train a classifier or inspect TEST/LIB. Artifacts are in
`../R86_readout_geometry_probe/PROBE.pdf` and the accompanying CSVs.

Removing feature radius as a frozen counterfactual corrected 8 and harmed 6
sampled TRAIN predictions, and corrected 6 and harmed 5 sampled VAL predictions.
Neither rule recovered any sampled fixed-target winners. These small changes
do not establish a useful remedy for zero selection or identify a causal
mechanism. Utility common-shift ambiguity and the absence of method-specific
classifier weights further limit the interpretation. No deployment rule or
R85 protocol changed, and no full normalized-head training is queued on this
evidence. A verdict on this round still requires the four locked R85
evaluations; the full research objective remains active and unresolved.

## First Completed Arm And Follow-Up Preparation

R85C completed all 57,600 updates and its locked TEST/LIB/PDF delivery without
restarting the other arms. Its selected update36000 has VAL Top1 71.008642%
versus R84A's 71.085185%, with lower actual regret (0.397969% versus 0.403494%).
The joint VAL gate therefore fails. Its fixed-target VAL rescues are 11, with
zero lost target winners and 10 new false picks; target recall remains 0.4757%.
TRAIN still has well-supported zero-selection methods. Full evidence is in
`C_cost_profile_analysis/ANALYSIS.pdf`. TEST's small gain does not override VAL.

The freed slot may host the separately locked R87 shared-versus-private
nonlinear-residual comparison once its code is ready. This tests parameter
allocation rather than another loss coefficient and is motivated by the
completed TRAIN evidence plus bounded primary-paper research. R85 A/B/D remain
unchanged and are not judged before completion. R87 is not a declaration that
all R85 mechanisms have failed; nor does it alter this round's gates or outputs.

## Completed Round And Continuing Goal

All four R85 arms completed their full training budgets, unique VAL locks,
TEST, LIB and per-model PDFs. The quiet watcher produced
`comparison_artifacts/COMPARISON.pdf`; the comparison protocol is complete.
None passes the predeclared joint VAL screen against R84A. D has slightly
higher VAL Top1 and slightly lower actual regret, but its mean_cost is higher;
that condition is not relaxed after observing the results. C improves costs
but not VAL Top1. The selected models are not promoted using TEST gains.

On the same 36 fixed VAL targets, A rescues0/loses6/adds2 false picks;
B rescues7/loses3/adds8 false picks; C rescues11/loses0/adds10 false picks;
D rescues8/loses2/adds13 false picks. These are small actual recovery counts,
not a substantial solution hidden by strict-zero group bookkeeping.

The requested research handoff has been carried out: R87 compares matched
shared/private instance-conditioned residual capacity, and R88 tests a
confirmed TSP execution-role representation omission against a geometric
anchor control. Both preserve full18 training/evaluation and the frozen labels.
Their independent locks are under `../R87_private_method_residual` and
`../R88_execution_role_inputs`. R85's finished receipts must not be mistaken
for still-running workers or automatically relaunched. No result here proves
random labels, an accuracy ceiling or a unique cause. The broader goal remains
active and unresolved while these follow-ups are evaluated.
