# R104 Private Dispatch Handoff

Implementation only. Main alone authorizes readiness, queues/starts workers and
the watcher. No allocation, GPU job, queue, watcher or Git command was started
as part of this implementation. Import/status never start a process.

## Fixed Resources

| Arm | Parent | Slot / UUID | Existing resources | Exact CPUs |
| --- | --- | --- | --- | --- |
| A_width256 | R99_A | 3090_0 / GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | job1465 gpu03, CPU4, mem9GiB | 12,13,14,15 |
| B_depth8 | R99_B | 3090_1 / GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | job1465 gpu03, CPU4, mem9GiB | 16,17,64,65 |

Reuse container
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
Reuse the original R81 GPU mutex inodes and R98_parallel_dispatch/gpu03.lock;
do not allocate, substitute slots/containers, stop healthy work or retry failures.
CPU reservation checks remain shared with the existing dispatchers.

Directly reuse `/tmp/shiys_R84_new90k_cache` and its existing identity stamp.
The worker validates the original cache and fails if missing/inconsistent.
There is no preprocessing, feature stage, copy/rebuild fallback, teacher,
solver execution, trained initialization or new data.

## Stage And Ready Contract

Five stages only: train (unique whole-VAL lock and full TEST included),
idempotent TEST, timing, LIB batch2, CPU PDF. Both arms use seed2, logical1024,
57600 updates, warmup4320, VAL1440, TRAIN7200, TSP/CVRP micro128,
other tasks512 and eval128. Runner commands use positional train/test/timing
and exact --arm A_width256/B_depth8; only the final PDF runs on admin01 CPU.

Scientific protocol locked does NOT mean launch ready. Main explicitly calls
prepare with nonempty, nonsecret evidence and --runner-tests-passed after
the full runner/report/source team is final. No fixed test-count assertion.
Readiness lives under R98_parallel_dispatch/R104_<arm>/, not in this document.

Required ready.json fields: main_thread_ready=true, runner_tests_passed=true,
evidence matching main's CLI, plan_sha256 matching plan.json,
source_freeze_ready=true. plan.json requires main_thread_ready=true,
the same evidence, exact spec/commands, source_sha256, primary_cache_reuse_only,
cache_preprocessing=false, cache_copy=false, automatic_retry=false.
Named runtime entrypoints plus actual static local imports are fingerprinted;
tests are included by exact filename only. No r104*.py or test globs, no
future unrelated research files, no automatic refreeze after any source edit.

Parent root is R99_global_competition_input. Each arm waits independently for
its assigned parent's full train/TEST/timing/LIB/PDF rc0, not R99 comparison
or the other parent. A cached null selected-VAL checkpoint SHA is NOT accepted
or repaired by this dispatcher; main owns explicit metadata binding after rc0.
The R99 timing failures block the original parent gate. The only accepted
recovery contract is main's standalone finalize output at
`R99_global_competition_input/preparation/postprocess_recovery/<A|B>/recovery_complete.json`,
schema `r99_postprocess_recovery_v1`, complete=true/state=complete/exit_status=0.
Its exact five stage names, immutable checkpoint_lock/SHA/state hash and
original failure SHA map must match. TRAIN/VAL/TEST and assignment checks stay
mandatory; only the original timing-only failed worker/controller can be
superseded by the explicit recovery evidence, never rewritten or called rc0.
The original failure errors remain visible in preserved_original_errors.

The gate verifies hashed report_stage.json and bind_val_stage.json, five real
PDFs and their same-lock report protocol under recovery/report_view, official
timing file/all18 populations, main-owned original-reporter LIB receipts and
both native datasets. Official timing must disclose mixed/default, false
original_inference_equivalence and efficacy_reused_as_exact_online=false;
nonbitwise online timing is not relabeled as exact cached inference.
The VAL-binding proof must preserve the selected archive's null identity and
the original failure receipts. Its full before/after JSON must differ only
at val.identity.checkpoint_sha256, not numerical metrics or other fields.
No recovery execution, comparison of the other parent, global R99 comparison,
model/array loading or metadata binding occurs in this gate.

## Exact Main CLIs

Run long queue/watch commands in main-owned tmux sessions. The commands below
are for main to execute only, after its final source review and CPU evidence.
Use one stable, nonsecret evidence string for both arms and watcher.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1
export R104_READY_EVIDENCE='MAIN_REPLACES_WITH_FINAL_CPU_AND_SOURCE_EVIDENCE'

python -B -m code.V4.r104_wave A_width256
python -B -m code.V4.r104_wave B_depth8

bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_A_width256.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_B_depth8.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed
```

Queue each arm independently (separate main-owned tmux sessions):

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh queue --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_A_width256.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed --interval 60
bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh queue --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_B_depth8.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed --interval 60
```

Alternatively, start directly only once the corresponding parent is complete.
Do not execute both queue and direct start for an arm:

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_A_width256.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r104_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R98_parallel_dispatch/specs/R104_B_depth8.json --evidence "$R104_READY_EVIDENCE" --runner-tests-passed
```

Quiet whole-version watcher (main-owned tmux; explicit publication opt-in):

```bash
python -B -u -m code.V4.r104_goal --watch --publish-github --evidence "$R104_READY_EVIDENCE" --interval 120
```

The watcher requires both complete TRAIN/VAL/TEST/LIB/PDF/timing pipelines,
each bound to that arm's unique lock and the same main scientific protocol.
Capacity dimensions, parameter counts and full initialization hashes must NOT
match across axes. Only saved-result CPU compare may run after both arms.
The frozen compare contract is three saved-comparison CSVs plus same-lock
provenance; COMPARISON.pdf and a version-wide RESULTS.pdf are NOT required.
Each arm's five PDF reports remain mandatory. LIB rc0 is proved by the worker
stage; its reporter receipt binds the checkpoint and both native datasets.
Timing is bound by the exact embedded record (or a matching file SHA).
For R104 TRAIN captured before final file serialization, a null capture SHA
is accepted only through the runner's explicit selected-VAL checkpoint binding
and an identical TRAIN/VAL model-state hash, protocol hash and selected update.
The watcher does not edit that capture or any summary. R99's null-VAL parent
case is separate and remains main-only metadata binding, never this fallback.
Errors represented as a string or list are preserved as failures; no repairs,
retries, model actions, notifications or secret enumeration. Source changes
stop waiting before comparison/publication. Goal success is never inferred.

Publication uses the original global codex-publication.lock, main branch,
empty index precondition, explicit R104 source/report allowlist, no arrays,
weights, caches, teacher payloads, credentials or other-version results.
The only approved extra files are
`docs/research/capacity_scaling_for_solver_discrimination.md` and
`code/V4/bind_cached_val_checkpoint.py`; no other non-R104 files are staged.
One whole-version commit/push only; failures preserve receipts and index.

## Focused CPU Verification

```bash
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest code.V4.test_r104_launch code.V4.test_r104_goal -v
```

Fixtures use temporary paths and mock every compare/Git/worker operation.
Production readiness and source freeze are deliberately not created by tests.
Both real runner/report parsers also accepted all ten exact arm-stage CLIs on
CPU; parsing did not run any experiment or reporting command.
