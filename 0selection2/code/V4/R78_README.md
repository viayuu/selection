# R78 Round 001: Frozen-Score Candidate Intercepts

## Final Release Status (2026-10-15)

Rounds001-006 are recorded in the [three-page retrospective PDF](runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/ZERO_SELECTION_RETROSPECTIVE.pdf)
and [round-by-round summary](runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/RETROSPECTIVE.md).
No candidate was accepted; retain whole R77B. Rounds001/002 remain numerically
inconclusive, not certified converged optima; the other routes closed negative
under their finite budgets. The long-term zero-selection goal is unresolved.
The waiting/preparation/authorization notes below are historical, not the final
status or permission to restart training, solvers or an app goal.

Canonical output: `code/V4/runs/R78_zero_selection_goal/round001/`.
Research-only documents and their original review remain in the parent directory.
Base commit: `5520b70741b7acba506b00370602709b40173108`. No commits requested.

## Fixed Protocol

R76 epoch18 is frozen. Read existing TRAIN `selected_best` and VAL `epoch018`
NPZ shards using the unmodified `r76_probability_diagnostics.load_shards` helper.
No backbone, solver, inference, or GPU execution is performed.

Each problem fits only post-maximin offsets `z+b`, with `sum(b)=0`, on natural
TRAIN winner CE plus `sum(b*b)/(2*N_train)`. The prior coefficient is exactly
one in summed-CE units, not a tuned alpha. There is no temperature, frequency
balancing, taskwise acceptance mixture, or parameter sweep. Seed is 2.

The predeclared numerical solver is CPU FP64 Adam, initial LR 0.05, quadratic
update-indexed decay to a floor of 1e-6, at most 30000 updates per task.
Every optimizer update has exactly 1024 examples. Shuffled TRAIN permutations
carry their tail into the next permutation; no rows are discarded or reweighted.
Full TRAIN checks occur at step0 and every 1000 updates, with no optimizer update
inside those diagnostic checks. Convergence requires projected full-objective
gradient infinity norm <= 1e-5 and objective change <= 1e-7 on two successive
checks. Budget exhaustion without convergence is inconclusive, not negative.
Identity at step0 is always retained. FP32 cached scores are promoted to FP64
before fitting and applying offsets; exact score ties use ascending global ID.

The complete fitted 18-task vector is locked before reading VAL labels. One full
VAL comparison uses all 18 fixed tasks with equal macro weight. The latest user
instruction sets the gate: native macro Top1 must strictly improve, macro mean
cost and macro actual relative regret must not increase, using unrounded values
and zero degradation tolerance. Macro cost-vs-TRAIN-SBS is reported separately,
not searched or substituted for the requested mean-cost gate. Taskwise harms are
disclosed; this is not an all-task safety or independent confirmation claim.

Paired intervals resample instances independently within each of the 18 fixed
tasks, seed2, 500 replicates. They do not affect acceptance or erase prior VAL
checkpoint selection. Generation-group dependence cannot be ruled out from
these NPZ files; intervals are conditional on the instance-independence assumption.

Only an accepted, fully converged VAL candidate may open the complete saved TEST
scores once. A persisted exclusive TEST-start marker prevents repeating that
operation in this round. Rejection retains identity and writes TRAIN/VAL research
handoff only. Historical TEST diagnostics are never fitting or tuning inputs.

## Run

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m code.V4.r78_zero_selection
```

Run the focused check with `python -m unittest code.V4.test_r78` after activating
the same environment. Long execution uses a dedicated `R78_round001_cpu` tmux
session. W&B uses the existing selector project, offline unless explicit
credentials are available; local JSON/CSV logs remain authoritative on failure.

## Outputs

`protocol.json`, TRAIN `identity_step0.json`, per-task `fit/*.json`, `loss.jsonl`,
`train_locked_candidate.json`, `train_comparison.json`, `val_comparison.json`,
`val_decision.json`, paired VAL intervals, method/task comparisons and observation
figures. `RESEARCH_HANDOFF.md` contains TRAIN/VAL only. TEST outputs exist only
after an accepted VAL gate. Runtime progress is persisted separately and merged
into the owned STATUS.json without touching the old main-thread app goal.

Implementation decisions: use the locked R76 checkpoint rather than select an
old epoch (research/user); use FP64 offset arithmetic with canonical ties
(implementation); keep exact batch1024 by permutation-tail carry (user constraint);
use a fixed numerical Adam budget and TRAIN convergence (implementation, declared
before fitting); use the latest mean-cost gate instead of the research draft's
SBS gate (user override); report per-task losses without taskwise rule assembly
(research). This is an in-sample head-fit mechanism screen, not a new model or
an unbiased out-of-fold calibration study.

## Outcome And Future Authorization

Round001 is a negative VAL candidate screen with no certified improvement:
Top1/CE improved, but mean cost and actual regret increased. The original
numerical fit/decision status remains `inconclusive` because only 1/18 tasks met
the declared convergence rule. These distinct facts are saved in round001's
`RESULTS.md`, `result_summary.json` and `result_schema.json`; original metrics,
fits and decisions are unchanged. TEST remains unopened and identity retained.

The policy prohibits **unplanned**, not all, broad retraining. An explicit user
authorization following Astra's next plan may include full18-task training.
The frozen-backbone/CPU-only restriction applies to round001, not the continuing
goal. Batch1024, authorized hardware and Sol execution/check routing still apply.
Waiting for that plan is normal workflow coordination, not permanent failure.

## Round002: Exact Authorized Squared Surrogate

Plan: `runs/R78_zero_selection_goal/review-stage/ROUND002_RESEARCH_PLAN.md`
relative to this code/V4 directory. Same R76 epoch18, no round001 CE completion
or backbone training. Construct FP64 relative gaps r and centered scores h;
c is the single TRAIN mean of r over N*K, or 1 only when all gaps are zero.
Fit `mean(w*(1+h+b)^2) + ||b||^2/(2*N)`, w=r/c, with sum(b)=0.
There is no trivial all-scores-minus1 solution in this row-centered score space.

Use the exact projected SVRG from the plan: zero initialization, per-task seed2,
1024 uniform-with-replacement TRAIN draws per update, eta=1/(8L), 30000-update
cap, full-gradient snapshot initially and every1000 updates. Snapshot row passes
and preparatory statistics are recorded separately from optimizer updates.
For this diagonal quadratic, the batch gradient difference is computed by its
algebraically identical diagonal expression; no analytical solution is assigned
to the trained intercepts. The analytic b* is TRAIN convergence evidence only.

Require two consecutive checks of gradient<=1e-5, objective change<=1e-7 and
||b-b*||_inf<=1e-6, with the exact quadratic objective-gap audit at FP64 rounding
tolerance. Initial check cannot count. If any task fails assumptions or budget,
retain identity, mark inconclusive, and DO NOT open this candidate's VAL or TEST.
Only all18 qualified fits become one locked candidate before one full VAL gate.
Gate, TRAIN-selected SBS identities, canonical ties and conditional TEST policy
are unchanged. All scales/margins/priors are fixed1; no coefficient scan.

Invocation after easynco activation:
`CUDA_VISIBLE_DEVICES='' python -u -m code.V4.r78_zero_selection --round 2`.
Outputs live only in `runs/R78_zero_selection_goal/round002/`; round001 files
are preserved. The app subgoal's actual workflow-wait flag is not faked as
resumed or completed; explicit user authorization allows file-based execution.

### Authorized Post-Round002 Readout

The existing execution outcome stays `inconclusive`. Astra's
`review-stage/POST_ROUND002_DECISION.md` stops this route's optimization budget
and authorizes only one TRAIN-cache identity/fitted/closed-form-reference
comparison, with prediction changes, signed costs/regrets and full-K margins.
The 1e-6 parameter tolerance is not a prerequisite for closing a poor route.

`python -u -m code.V4.r78_zero_selection --round002-train-reference-readout`
does not call an optimizer, load a backbone, read VAL/TEST or change the PLAN
gate. The reference is diagnostic only, never a deployed or retroactively
qualified candidate. New `train_reference_*` outputs and
`TRAIN_REFERENCE_READOUT.md` are separate from the original execution records.
The goal remains active while awaiting R77 final TRAIN/VAL distributions; no
further frozen-R76 intercept variants or duplicate R77 training are authorized.

## Round003: Matched Full-B Monotone Cost Continuation

Specification: `runs/R78_zero_selection_goal/review-stage/FULLB_MONOTONE_COST_PLAN.md`.
Implementation: `r78_fullb_cost_experiment.py`, launcher: `run_v4_r78_cost.sh`.
R77/R79 original scripts are reused read-only, not edited. The previous R76
CE-intercept and fixed-scale squared-intercept routes remain closed negative.
Round003 updates instance-dependent scores of the unchanged full18 network;
it is not another frozen-score bias trial or a LEHD/ReLD-only intervention.

Both arms load locked full-B ep28/u34776 with the same fresh seed2 AdamW and
R77 shuffle/prefetch/micro-accumulation pipeline. Each has14904 updates at
batch1024,828 updates/task, with the inherited shared late LR sequence and
R-Drop weight.35. Control retains the original objective. Treatment replaces
only risk by `.02 * mean_valid_K((r/a_t) * softplus(z-mean_valid_K(z)))`.
Original FP64 TRAIN costs define r and frozen a_t before loss weights become
FP32. Native CE labels, compare/R-Drop reductions, architecture and inference
stay unchanged. All supervision is stripped before both dropout forwards.

CPU preparation completed:18 tasks,70114 TRAIN rows/task,89 candidates.
`round003/late_lr.npy` stores all14904 fixed FP64 rates, first
`2.4543907940693327e-5`, last `1e-6`. `task_scales.json` and `train_weights/`
contain the TRAIN-only immutable target receipt and arrays. Actual model
forwards, continuation optimizer updates and TEST reads during preparation:0.
Five synthetic CPU tests passed; see `round003/CPU_CHECKS.md`.

Only VAL endpoints4/8/12 are candidates, with B as zero-update fallback.
Raw macro cost<=11.39555834354421, actual regret<=0.40132674544775454%, native
Top1 strictly above unrounded B; treatment-specific positive also requires
Top1 above selected matched control. No tolerance, taskwise assembly, scan or
TEST selection. Full protocol: `round003/EXECUTION_PLAN.md`.

Current workflow: `active / ready_pending_hardware`. NSS has priority.
`round003/hardware_allocation.pending.json` has approved=false and no assigned
cards. No GPU probes, GPU smoke or training have been run. One future GPU smoke
and each continuation require explicit user approval of arm/node/job/UUID,
using an existing-job exact Slurm step through the launcher, never SSH training.
Legacy/root app goal remains paused; the child's historical native app flag is
not misrepresented as resumed. No app-goal tools are part of this preparation.

## Round004: Frozen-B Conditional Cost Head

Current workflow: active /waiting_Astra_evidence_direction. The authorized CPU
trial completed all18 tasks,1380 logical batch1024 updates/task,24840 total,
513 parameters. All three fixed VAL gates failed; the entire rule retains B
fallback. Actual fitted TRAIN/VAL metrics are preserved separately. No TEST/LIB,
new budget, taskwise mixture or temperature/accuracy variant is authorized.

Execution files: r78_frozen_score_cost_head.py,
run_v4_r78_frozen_score_cost_head.sh, test_r78_frozen_score_cost_head.py.
Complete output entry: runs/R78_zero_selection_goal/round004/README.md.
Final evidence: round004/RESULTS.md, results.json, final_receipt.json,
head_lock.json, locked_heads.npz and native/changed-decision CSVs.

PID4183038 and tmux R78_round004_cpu_head have exited. Launch-to-results27.48s,
fitting21.50s; original saved TRAIN70114 rows/task and VAL3895 rows/task were
used in full. Only two synthetic checks and the budget-included first real
batch finite check were performed. Native and legacy app-goal statuses were
not modified. Wait for Astra evidence direction, expected NSS TRAIN/VAL only.
