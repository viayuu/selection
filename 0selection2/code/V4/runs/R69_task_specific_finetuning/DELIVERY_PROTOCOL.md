# R69 Delivery Protocol: Preparation Only

This document and its three delivery sources prepare future operations. This
preparation phase authorizes NO freeze, TEST, analysis, model forward, prediction
replay, GPU query/job, TRAIN command, stop/restart, CPU-check rerun or Git action.
Only new Python/shell syntax is checked. Delivery runtime is unverified.

The SAME original trainer remains the only R69 training execution:
`r69_train`, original pane `%82`, pane PID505438, TRAIN step1465.101 on gpu03.
Parent's authoritative2026-10-12T20:32:23UTC observation was RUNNING/738s,
epoch8/global48/u1860/1398240 selected rows. This is an observation, NOT a
terminal receipt. Preparation neither queries nor controls that job.

## Locked Scientific Contract

Only B is newly trained. B is the complete18-branch ordinary AdamW bank, all
branches copied identically from the exact R68 pre-fine-tuning state: R67B fixed
terminal stage1 averaged-x trunk plus the entire fresh head670003. All selector
parameters train. NativeUtilitySelector architecture, inputs, native targets,
full legal pool order, global solver identities and directed ATSP stay unchanged.
Known original `problem_id` routes ONE complete branch. Costs and labels remain
targets only. No per-task selected-checkpoint assembly, head transplant,
optimizer transition, projector, SAM, averaging, new labels, solver or loss.

Fixed40E/global41-80 schedule,9600 logical updates/7.2M selected rows/14.4M
successful dropout-view rows. Each branch1600 active updates/400000 rows.
Three distinct tasks/update, real256/task/effective768;16/task/effective48 tails.
Native two-forward CE plus FP32 legal symmetric RDrop and the locked720-update
ramp. Each active branch's FP32 gradient/3, one combined clip1, one AdamW step;
inactive `grad=None`, no parameter/decay/moment/counter advancement. Whole-triple
AMP retries retain rows/RNG and do not advance the successful clock.

Fixed A is reused R68A28/global68/8644 validation correct. Its authoritative
identity is `R68_sharpness_aware_finetuning/final_test_lock.json/models.A`.
Read saved JSON reports only: no A weights load, training, recount, inference,
reselection or TEST. Its one historical original TEST is an already executed
test, not zero original tests or a new R69 test. Reuse establishes protocol
equivalence, not an independent replication. R60B remains the fixed historical
incumbent, not this matched control. Historical TEST exposure is disclosed;
new R69 B TEST never changes design, budget, checkpoint choice or planning.

## Gate 1: Actual Original TRAIN Terminal Success

Parent must observe actual step1465.101 `COMPLETED/0:0` on gpu03 AND owned
`r69_train` gone. `RUNNING/0:0`, a log line, elapsed estimate, process exit alone,
or a replacement step is insufficient. Parent writes
`training_terminal_receipt.json` with the original scheduler/tmux evidence.
The delivery implementation never queries the scheduler or tmux for this gate.

Required receipt shape follows the existing R68 parent receipt:

```json
{
  "state": "COMPLETED",
  "exit_code": "0:0",
  "slurm_step": "1465.101",
  "tmux_session": "r69_train",
  "tmux_terminal": true,
  "elapsed_seconds": 0,
  "scheduler_evidence": {
    "command": "sacct -j 1465.101 --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList",
    "returncode": 0,
    "stdout": "1465.101|COMPLETED|0:0|ACTUAL_ELAPSED|ACTUAL_START|ACTUAL_END|gpu03\n"
  },
  "tmux_evidence": {
    "command": "tmux has-session -t '=r69_train'",
    "returncode": 1,
    "stderr": "can't find session: r69_train"
  }
}
```

The example is a schema, NOT a factual receipt. Parent must replace every
example value, including elapsed_seconds, with actual original evidence.
Scheduler stdout uses the listed seven parsable fields; elapsed must match.

The actual R69 runner completion files are:

- `training_complete.json`: new_arms B only,40E/9600updates/7.2Mrows/14.4Mviews.
- `validation_candidates.json`: fixed reused A and one new B candidate.
- `B_task_specific_seed2/result.json`, `history.json`, `wall_time.json`.
- `B_task_specific_seed2/best.pt`, unique `best_checkpoints/epochNNN.pt`,
  `final.pt`, and `parameter_step_counts.json`.

There is NO R69 paired_training_complete.json or second newly trained arm.
All40 histories must be epoch1-40/global41-80, with full18*1000 VAL every epoch,
integer saved native counts, complete final TRAIN, and full selected-best TRAIN.
Result/raw-record counters must publish the actual fixed horizon and branch
budgets. Original launch receipt must still identify step1465.101/assigned UUID.

## Gate 2: Future Explicit Parent Freeze

Only after Gate1 and separate parent authorization, select B's earliest maximum
INTEGER full18 VAL correct count/18000. Published best.pt, selected unique bundle,
result and complete VAL history must agree. The entire saved bundle includes
all18 branches with canonical problem order from one common epoch.

Actual checkpoint schema is `model` with `branches.0.*` through
`branches.17.*`, `args`, zero-based `epoch`, `common_epoch`, `global_epoch`,
`branch_epochs`, `problem_names`, `native_correct`, `native_total`,
`successful_updates`, `presentations`, `model_state_sha256`,
`all18_complete_branches=true`, `parameter_mode='ordinary AdamW weights'`,
`checkpoint_mode='eval'`, `projector_included=false`. No SAM or averaging fields
are invented in the training checkpoint; unchanged saved args assert both false.
The future loader calls the actual `r69_model.make_bank`, then strictly loads
the ENTIRE saved bank before any forward. The factory's pre-fine-tuning
construction state does not replace any part of the final loaded bank.

Official B iff B VAL correct>8644; tie/lower fixed A. No task-wise checkpoint
selection, min-cost override, TEST-based choice, or substitution of R60B.
Freeze writes exclusive `final_test_lock.json` with actual B bundle paths/file
and state identities, unmodified A lock identity/JSON provenance, ordinary
unperturbed/noaveraging metadata, and:

```json
{
  "new_tests_required": {"A": false, "B": true},
  "new_test_attempts_allowed": {"A": 0, "B": 1},
  "historical_original_tests_reused": {"A": 1}
}
```

All future commands below require this environment first:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
```

Future CPU freeze command, separately authorized; DO NOT run during preparation:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r69_frozen_evaluation --stage freeze --authorize-test
```

## Gate 3: Future ONE New B TEST

After Gate1/2 and the explicit parent `locked-test` baton, use ONLY existing
allocation1465/gpu03/index1 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, owned tmux `r69_final_test`.
The launcher validates saved lock/evidence, makes one ordinary assigned-UUID-only
idle check immediately at actual launch, and waits on the same UUID if busy.
No other GPU/client/queue/allocation, device switching, client stopping,
environment/network preflight, sizing, reference inference or separate latency
benchmark. W&B settings stay offline selector/
yjkds-southern-university-of-science-technology; threads1/current bridge retained.

Future launch command, separately authorized; DO NOT run during preparation:

```bash
tmux new-session -d -s r69_final_test -c /public/home/shiys/0selection2 'R69_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r69_test.sh >> code/V4/runs/R69_task_specific_finetuning/final_test_pipeline.log 2>&1'
```

The exclusive `final_test_started.json` permits one new B TEST attempt only.
Any real failure or partial marker is preserved. No automatic retry, TEST
reselection, second attempt or relaunch after an observation timeout.
An existing tmux/marker is not permission to replace or restart anything.

B receives ONE original18*1000 fulllegalpool TEST, FP32 batch128 with seven128
batches plus one104-row tail per task. All144 bank and144 active-branch calls
must follow canonical known task IDs; one full-pool selector forward/query.
Evaluation runs ZERO solvers. No sampling, augmentation, prediction/NPZ replay,
A/reference inference or singleton timing pass. Original native winners,
input whitelist and candidate/global-ID utility tie rule are preserved.

`B_task_specific_seed2/test_predictions/<task>.npz` saves every original index,
logit, local-pool pred/native winner, raw FP64 costs, `nodes` sizes, original
`pool_ids` global identities and pool names. The proven full18 evaluator saves
these during the one live pass; they are not replayed for analysis.
`test_results_B.json` is new B only. `test_results.json/models.A` is the unchanged
saved R68A metric object, with separate reuse provenance and execution counts.
It never represents a new A TEST. Full evaluation walltime includes transfer,
metrics and NPZ saving; it is NOT singleton latency or solver time.

The launcher writes `test_launch_receipt.json`, assigned-only availability
records, and `test_process_receipt.json` for the evaluator exit. Process exit
is explicitly NOT scheduler terminal evidence. Parent/authorized executor must
separately observe the ACTUAL owned future TEST step `COMPLETED/0:0` and
`r69_final_test` gone, then write `test_terminal_receipt.json` with the same
scheduler/tmux evidence schema as Gate1, using the new owned step/session.
No waiting executor or automatic transition to analysis is created.

## Gate 4: Future ONCE CPU Analysis

Only after actual new B TESTterminal0 and separate parent authorization:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r69_analysis --authorize-analysis
```

An exclusive `analysis_started.json` permits one pass. Read saved JSON and raw
update records only; no model weights/forward, inference, NPZ replay, solver,
historical model selection, missing TRAIN inference or automatic analysis retry.
The lock/evidence checks for analysis do not load/hash model checkpoints.

Expected outputs include comparison/selected/availability/delta CSV, actual
learning/task/family/optimization/update curves, task optimization,
best/final/last5 summaries, actual common-epoch comparisons, per-method and
selection-distribution diagnostics, branch/counter CSV, figures, REPORT.md,
compute_summary.json, reference_provenance.json, analysis.json and
analysis/delivery receipts. A is named explicitly as reused R68A throughout.
Fixed historical R45A/R60B/R64-68 reports are descriptive saved JSON references,
not reselected or retested models. Missing historical TRAIN remains missing.
ALL remains the original equal18-task macro; MVRP aggregates15 tasks.

TRAIN curves contain actual every5/final/selected-best measurements only;
selected TRAIN points are distinct from dense epoch curves. VAL is complete
every epoch. Last5 means epochs36-40, not seed uncertainty or weight averaging.
Native CE/KL/objective/LR/grad/clip/views/AMP/runtime/memory and actual per-branch
activations/rows/parameter counters are disclosed from saved records. Initial
state, exact task/row/dropout schedule and inactive-parameter semantics use saved
receipts/records, without rerunning training or checks.

NEW B9600/7.2M/14.4M successful views and measured runtime/retries are separate
from HISTORICAL reused A9600/7.2M/3002.024237s/5 retries and inherited stage1
9600/7.2M counted ONCE. Actual B33173154 versus shared1842953 parameters;
capacity and per-parameter optimizer histories are NOT matched. A gain supports
the unsharing recipe, not a pure causal task-interference proof. No accuracy
ceiling/noise claim or70% promise. Numeric70% uses only the official validation-
frozen original18 TEST; supported completion additionally requires parent's
whole-goal audit and closure. Legacy B semantics are disclosed, not a metric veto.

Weights/NPZ/rawupdates/W&B/schedules/cache stay LOCAL under userignore. Git alone
is not all inference assets; nothing is force-added. R58 is fully deferred.
Eight legacy B-task feasible-domain mismatches stay disclosed/unfixed.
Parent owns reviews, memory/state, objective audit and one commit per completed
version. Preparation edits only these isolated delivery files and never the
six frozen training sources/snapshot/hashlist/protocol or any R64-68 artifact.
