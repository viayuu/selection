# R69: complete task-specific fine-tuning

One new `B_task_specific_seed2` arm is compared with the fixed reused R68A28
shared control (global68; 8644/18000 validation correct). The adopted actual
Astra/max plan and explicit control-reuse amendment are copied into the run
root. Review independence is same-family; acceptance is provisional.

## Locked implementation

The model bank contains eighteen independent, complete `NativeUtilitySelector`
copies in canonical original problem-ID order. Known `problem_id` activates
exactly one branch. Every branch starts from the exact saved R68
**pre-fine-tuning** state: R67B fixed stage1 terminal averaged-x trunk plus
the entire fresh utility head seeded670003. Trained R68 or R67 stage2 weights
never initialize B. All parameters train. Architecture, handcrafted34+ID,
original inputs, directed ATSP, legal pool order, global solver IDs and native
winners are retained. The input whitelist excludes targets and costs.

Each logical update uses three distinct tasks, real256/task (effective768),
or16/task tails (effective48), and two independent dropout forwards per task.
The objective is the task mean of native CE plus FP32 legal symmetric RDrop,
with coefficient `0.35*min(successful_update/720,1)`. Each task's unscaled FP32
gradient is divided by3 in its own branch coordinates. One combined global
clip at1 precedes one AdamW step. Inactive branches keep `grad=None`, and their
weights, weight decay, Adam moments and counters do not advance, including
branches with existing optimizer state. LR uses the global successful-update
clock; each parameter's Adam counter advances only with its own gradient.

Fresh ordinary AdamW uses `lr=3e-4*min(successful_update/720,1)`, WD1e-4,
betas(.9,.999), eps1e-8, foreachFalse. There is no SAM, averaging, projector,
new loss, labels, augmentation, pruning, task grouping, early stop or LR search.
Whole-triple AMP recovery restores pre-update RNG, clears all gradients,
lowers the scale and retries the same rows. No LR, optimizer, row cursor or
successful-update counter advances until all three gradients are finite.
Twenty attempts are the fixed limit; there is no automatic job restart.

## Schedule, evaluation and accounting

Regenerate the original R67/R68 seed2 80-epoch paired schedule and take exact
global epochs41-80; use original `dropout_seed(9600+successful_before)`.
Fixed40 epochs mean9600 logical updates,7.2M selected rows and14.4M successful
dropout-view rows. Each branch is active1600 times and sees400000 rows.

Full18 validation is FP32 eval-mode batch128 including tails every epoch.
Full TRAIN evaluation is every5 epochs, final and selected best. Select the
maximum **integer** native correct count/18000, retaining the earliest exact
tie. Every saved candidate is all18 complete branches from ONE common epoch;
no per-task best stitching. `best_checkpoints/epochNNN.pt` uniquely identifies
the bundle; `best.pt`, `final.pt` and optimizer/scaler/RNG `last.pt` are local.
Official B requires strictly more than8644 validation correct; tie/lower keeps
fixed A28. Inference remains one task-routed selector forward over the full
legal pool, direct utility argmax with global-ID ties, and one completed solver.

Expected capacity is33173154 B parameters versus1842953 shared A parameters;
actual counts are recorded at formal initialization. Example/view budgets
match the reused control, but capacity and per-parameter update counts do not.
Unused parameters on irrelevant input paths can have fewer Adam steps than
their branch's1600 activations; actual counters are recorded on completion.

New B runtime/retries are measured separately. Reused A's9600 updates/7.2M
rows/3002.024237s/5 retries are historical expenditure. Inherited R67B stage1's
9600 updates/7.2M rows are historical pretraining counted ONCE separately.
This is one new arm compared with a reused control, not two newly trained
arms or fresh encoder training. Reuse establishes protocol equivalence,
not an independent replication.

## Execution and handoff

`bash code/V4/run_v4_r69.sh cpu-check` performs the one focused synthetic CPU
check: initialization, known-ID routing/target isolation, gradient/3,
inactive immutability (also previously active branches), injected whole-update
AMP retry, and full18 common-epoch bundle save/load. Invocation/failure receipts
are append-only. No real-data pilot, GPU sizing, replay or diagnostic campaign.

`bash code/V4/run_v4_r69.sh train` runs only B through existing allocation1465,
gpu03, assigned index1 UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`.
Use owned tmux `r69_train`. A normal assigned-UUID-only idle check immediately
precedes launch; if busy, wait on that same UUID. Thread counts are1, conda is
`easynco`, and the existing `python_compat` bridge is retained. W&B is offline
in project `selector`, entity `yjkds-southern-university-of-science-technology`.

The run root is `code/V4/runs/R69_task_specific_finetuning/`. Protocol, args,
source snapshot, initialization/schedule identities, reuse/inheritance budget,
CPU receipt, history, raw updates, AMP/LR/gradient/runtime/memory, W&B,
`current_status.json`, `pipeline.log`, launch and first-update handoff receipts
remain local. The executor returns once a successful finite formal update is
visible and leaves fixed40E training live. Monitor existing status/logs every
180-300s; observation timeout never authorizes a duplicate or restart.

There is no TEST, analysis, freeze or Git execution in this phase. Parent later
freezes one validation-selected common-epoch B bundle and the official arm,
then runs ONE original full18 TEST for new B only. A's existing TEST is reused
unchanged, without inference or recount. Historical TEST exposure is disclosed;
new R68 TEST does not guide this experiment. R58 and eight legacy B-task
feasible-domain mismatches remain deferred/unfixed, and do not veto native70.
The single-completed-solver70% objective remains active and unachieved.
Parent owns R68 review/publication and the one commit per completed version.
No existing sources, data, protocols, results, checkpoints, user work or Git
index are modified. Weights/NPZ/rawupdates/W&B/cache/schedules are not added to Git.
