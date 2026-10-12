# R70 Conditional Delivery Protocol

Parent authorizes ONE combined CPU preparation and conditional future delivery,
owned CPU tmux `r70_delivery`. Preparation performs syntax checking only. New
delivery runtime is UNVERIFIED during preparation; no preparation GPU query,
job, checkpoint load, forward, metric replay, freeze, TEST or analysis occurs.
The original R70 training sources/gradient/runner/parameters/source_launch and
all R69 sources/data/protocol/TEST/report files stay frozen and unedited.

Original TRAIN identity: tmux `r70_train`, pane `%88`, pane PID870566 on admin01;
Slurm1465.103/gpu03/trainer PID286375; sole physical GPUindex1 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`. The existing successful CPU check and
discarded capacity update are never repeated. Training is never stopped/restarted.

## Conditional Gates

The CPU controller reads current_status/pipeline/tmux and sacct for ORIGINAL
step1465.103 only, every240 seconds. `training_terminal_receipt.json` records
actual sacct command/stdout/stderr/returncode/timestamps/elapsed and separately
actual owned-tmux absence evidence. Require COMPLETED/0:0 on gpu03 AND
`r70_train` gone; RUNNING/0:0, process exit or log completion alone never suffices.
An actual failure preserves its original evidence and stops without retry.

Published training must be complete40E/global41-80, full18x1000 VAL each epoch,
9600 successful logical updates/7.2M selected rows/14.4M encoder-view rows and
14.4M mixed-head rows. Read saved records only. No dense TRAIN measurement is
invented; existing TRAIN is every5/final and selected-best.

Parent preauthorizes mechanical VAL freeze only after these gates. Choose the
earliest MAX INTEGER full18 VAL count for B, saving/loading ALL18 branches from
one selected common-epoch complete bank. Official B strictly>8604; otherwise
fixed A20. Costs, per-task performance and TEST never select. Exclusive
`final_test_lock.json` precedes any new TEST; exclusive `final_test_started.json`
allows exactly one new B TEST and no automatic retry/reselection.

The actual R70 checkpoint has `model` with `branches.0.*` through
`branches.17.*`, `args`, zero-based `epoch`, `common_epoch`, `global_epoch`,
`branch_epochs`, `problem_names`, `native_correct`, `native_total`, `macro`,
`successful_updates`, `presentations`, `model_state_sha256`,
`all18_complete_branches`, `parameter_mode`, `checkpoint_mode`,
`projector_included` and `inference_rule`. The existing actual
`r69_model.make_bank` factory constructs the bank, then strictly loads its
ENTIRE saved state before any forward. No transplant, averaging, mixed-feature
TEST or per-task checkpoint assembly. Native FP32 clean utility/full original
pool/global-ID tie behavior and one active branch/query stay unchanged.

## Fixed Control and ONE New TEST

R70 A is EXACT R69B20/global60 complete18 bank, original VAL8604/18000.
Its metadata comes from R69 `final_test_lock.json/models/B`; its original TEST
object comes from R69 `test_results.json/models/B`, reused unchanged.
A weights are never loaded/hashed/trained/recounted/reselected/inferred/tested.
New TEST counts A0/B1; original A already1. Reuse is protocol equivalence, not
replication. R69's official arm and all its files stay unchanged.

After training terminal success and VAL lock, create ONLY owned tmux
`r70_final_test`, calling `bash code/V4/run_v4_r70_test.sh`. This dispatches ONE
new step within existing allocation1465/nodegpu03, binding only the assigned
UUID. Immediate assigned-UUID-only idle check once; if occupied, wait on that
same UUID. No other GPU/client/queue/allocation or live process control.

Exactly ONE original full18/full legal pool native FP32 batch128 TEST for NEWB,
regardless of VAL improvement:18x1000 rows,104-row tails,144 wrapper and144
active-branch batches. No augmentation/sampling/pruning/ensemble/specialist,
mixed-feature forward/solver/singleton latency benchmark. Zero solvers execute.
The live evaluation saves NPZ logits, predictions, native winners, rawFP64
costs, ordered indices, sizes and global pool IDs/names, plus official TEST JSON.

Independently require the actual new TEST step COMPLETED/0:0 and owned
`r70_final_test` gone, saved as `test_terminal_receipt.json`. Process0 is separate
`test_process_receipt.json`, never scheduler evidence. Failure is preserved;
there is no automatic TEST retry, selection change or extra evaluation.

## ONCE CPU Analysis and Parent Handoff

Only after actual TEST terminal0, invoke `r70_analysis` ONCE, with an exclusive
analysis marker. Read saved JSON/CSV/update logs only; no NPZ metric replay,
inference, representation audit or further experiment. Export full18 equal
macro, per-task and TSP/CVRP/ATSP/MVRP TRAIN/VAL/TEST Top1/2/3, CE, mean cost,
vsSBS and actual regret. Best/final/last-five VAL/common-actual-epoch comparisons
keep sparse TRAIN missing; last-five means are not seed uncertainty or averaged
checkpoints. Existing plot helpers render method/selector comparisons and pick
distributions from saved JSON. Historical R45A/R60B/R64 references use saved
locked official JSON only; missing historical counts remain missing.

Preserve raw cleanCE/mixCE/KL/total/LR/gradient/clip/AMP/retry/successful versus
attempted encoder and mixed-head rows/wall/memory tables and curves. New formal
B:9600/7.2M/14.4Mencoder+14.4Mmixedhead; reused A historical R69B:
9600/7.2M/3554.574466s/5 retries/14.4Mcleanviews; inherited stage1 counted ONCE:
9600/7.2M/2764.217798s. Capacity4.474874s/ONE discarded TRAIN update is EXTRA,
not part of formal9600. Both A/B have33173154 parameters and the same per-parameter
update recipe; extra head compute is unmatched. Exact actual retries/attempts
come from saved formal records.

Write RESULTS/REPORT/comparison CSV/curves/reference provenance/analysis JSON,
analysis_receipt and actual delivery_receipt. Whole evaluation walltime is not
singleton latency or solver runtime. Checkpoints/NPZ/rawupdates/W&B/cache/
regenerable schedule stay LOCAL; Git alone does not contain all inference
assets. This executor performs no Git operation or staged-index write.

No model/budget/rule/design changes follow NEWB TEST. Historical TEST exposure
is disclosed; parent's future planning explicitly excludes NEW R70 TEST.
Goal70 is ACTIVE/unachieved; numeric70% threshold is12600/18000 and parent
whole-objective audit/actual Astra reviews/fresh review/memory/state/publication
remain required. R58 is fully deferred. Eight legacy B feasible-domain semantics
mismatches remain disclosed/unfixed and are not a numeric70 veto.

The CPU executor uses conda easynco/current python_compat bridge/thread1/W&B
offline selector/entity yjkds-southern-university-of-science-technology. No
environment/network/package probes, agents, reviewers or Git. Parent owns the
outer CPU executor lifetime and its outer tmux exit log.
