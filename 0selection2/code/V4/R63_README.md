# R63: Native-Winner Classification With Recurrent Node Workspaces

R62 portfolio pretraining did not improve original full18 selection. R63 tests
distributed candidate-specific computation using only the original inputs,
complete legal pools and native winners. It introduces no solver execution,
expanded labels, route target, API call or R58 operation.

## Persistent Candidate-Node State

Keep the original four-layer128-dimensional instance encoder and ID/manual
solver encoder. Replace their shared two-layer joint stack with a shared cell
that maintains one128-dimensional state per candidate and input node.
The cell uses32-dimensional gated neighbor messages and a GRU for eight tied
rounds. All valid candidates participate; no Top2/Top3 gate is used.

The sparse input graph joins16 low outgoing and16 low incoming travel-distance
neighbors, plus true-depot and three special-token connections. Preserve the
original directed ATSP matrix and node masks; do not use routes, cost labels or
guessed feasibility to choose edges. The original dense encoder still reads
the full instance before these rounds.

A_static_memory_seed2 reads neighbors' initial candidate-conditioned states.
B_communicating_memory_seed2 reads their previous-round states. Both update
their own states, share the same parameter sets and retain gradients through
the source memory. They have identical initialization, common-prefix batches
and recurrence count. Their architectural difference is communication of
evolving information; their final training budgets differ as disclosed below.

Pool each candidate's workspace separately. The original shared decoder and
3d-input score head read its graph summary, context and solver token. The
result is a direct native-class logit, not a maximin score or predicted cost.
Inference runs only this selector and chooses one original solver.

## Objective And Budget

Both arms use only native-winner CE averaged across two dropout forwards,
plus existing symmetric R-Drop ramped to0.35 over three epochs. No pair/risk
loss, cost weighting, route target or class balancing is added.

Seed2; scratch initialization; AdamW1e-4/WD1e-4; dropout0.1/clip1/warmup3.
Fixed microbatch16 accumulates to each original effective batch128, weighted
by its sample count. There is one successful optimizer step per original batch;
AMP failure retries that entire batch/RNG rather than skipping samples.

The controller maximizes full18 validation native Top1 (patience3 LR halving,
patience8 stop, min15/max40, min_delta0.001pp). The original plan made A define
the common LR/horizon. A completed40 epochs/56880 successful updates. The
explicit user amendment in `runs/R63_recurrent_workspaces/budget_amendment.json`
stops B after15 complete epochs/21330 updates and full validation; B follows
A's LR trace only through that common prefix. A40/B15 is unequal compute,
not a matched-horizon causal comparison. A checkpoint-boundary interrupt and
zero-update finalization preserve B's completed history, optimizer and best
checkpoint; no epoch16 updates are authorized. Each arm retains its strict
maximum-Top1 checkpoint within its own completed budget, keeping the earlier
epoch on ties. Complete TRAIN evaluation uses eval mode.

The report discloses both actual budgets and first15-epoch validation history
values descriptively. It does not replace A's official40-epoch selection with
a prefix checkpoint or use TEST to repair the unequal-budget comparison.

The user target is70% native macro Top1 in the original full18/fullpool task.
Cost/regret remain honestly reported, not extra target vetoes. Reference+3pp
and B-A+2pp are intermediate interpretation milestones, not goal completion.
A validation target hit alone still requires locked held-out evidence.
There is no R63 test search or automatic test access.

## Locked Final Test

Both training arms must finish before the parent locks and authorizes TEST.
Each R63 arm keeps its own unique validation-Top1-best checkpoint. The official
R63 arm has the higher validation native macro Top1; an exact arm tie selects A.
TEST cannot change the official arm, epoch, candidate pool or decision rule.

Run one complete original18/fullpool TEST for each locked R63A and R63B model,
plus the already selected R60B `B_expanded_seed2/best.pt` at epoch39. Reuse
R45A's historical official full18 TEST without new inference. The R60B reference
TEST is an explicit addition, not a rerun of every past version; other R59--R62
no-test histories remain unchanged. Cost/regret remain reported outcomes, not
vetoes on the native-Top1 target. Validation alone does not establish70%.

The analysis consumes Pascal's actual artifact API under the R63 run root:

- `final_test_lock.json`: parent-frozen entries `A`, `B`, `R60B`, each with
  checkpoint `path`, `epoch` and `sha256`.
- `test_protocol.json`: execution receipt; only `status: complete` with
  `selection_frozen_before_test: true` and matching `locked_models` permits
  final TEST tables.
- `test_results.json`: complete evaluations keyed by `A`, `B`, `R60B`.
- `A_static_memory_seed2/test_predictions/<problem>.npz` and
  `B_communicating_memory_seed2/test_predictions/<problem>.npz`: R63 saved scores,
  predictions, identities and original FP64 costs for all18 tasks.
- `reference/R60B/test_predictions/<problem>.npz`: the fixed incumbent's newly
  authorized TEST predictions, not a replacement for its old validation records.

R45A is read from `runs/R45_solver_code_embeddings/test_predictions/A/`,
its official `test_results.json` and original `locked_checkpoints.json`.
Missing/pending/failed final TEST is marked explicitly; no zero scores,
validation substitution, inference or automatic retry is performed by analysis.

`comparison.csv` separates `val` and `test` and includes ALL plus every task,
with checkpoint path/epoch/hash. `family_comparison.csv` includes ALL, TSP,
CVRP, ATSP and MVRP. `decision_changes.csv` and `family_decision_changes.csv`
retain corrections, harms, both-wrong cost changes and net equal-problem-macro
cost/regret deltas for R45A/R60B to R63A/B, and R63A to R63B. Checkpoint identities
and completed/pending TEST status are also saved separately. Only existing
saved predictions are replayed; this reporting code never runs a model.

## Execution

Outputs: `runs/R63_recurrent_workspaces/`. Only the designated available3090
is authorized. Use easynco and tmux; preserve other GPUs, queues and all old
experiments. There is no GPU-hour cap. The12--24-hour planning estimate is
not a measured runtime or limit.

```bash
bash code/V4/run_v4_r63.sh plan
bash code/V4/run_v4_r63.sh train --resume
# Authorized B15 checkpoint-boundary recovery only; no optimizer updates:
R63_PARENT_GPU_BATON=b15-finalize bash code/V4/run_v4_r63.sh finalize-b15 --resume
# Only after both arms finish, locks are frozen and the parent authorizes TEST:
R63_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r63.sh test
bash code/V4/run_v4_r63.sh analyze
```

After a few focused CPU interface/accumulation tests, go directly to formal
training. No standalone GPU preflight, timing pilot, audit or review campaign.
Normal final validation records input-graph/selector latency and clearly
discloses cached-preparation exclusions; this is validation timing, not a new
TEST timing pass. Run the authorized locked TEST once, then analyze its saved
outputs once and commit code and completed results together once. Documenting
these commands grants no new GPU or TEST authorization.

Legacy B feasible-domain mismatches remain pending; this experiment does not
claim a repaired common-task benchmark. Failure rejects this computation
recipe, not the existence of unused signal or the possibility of reaching70%.

## Recorded numerical tie limitation

A epochs32 and34 have the same8729/18000 validation correct count. The original
floating-point macro comparison treated their averages as different by one
rounding unit and selected34. This violates the intended earliest-on-count-tie
rule, but does not change reported Top1. Preserve A34 and its completed TEST;
do not retrospectively replace the checkpoint or evaluate32. R64 will compare
integer correct counts prospectively. The R63 locks and history remain intact.
