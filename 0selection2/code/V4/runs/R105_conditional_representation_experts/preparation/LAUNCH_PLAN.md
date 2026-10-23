# R105 Administrative Handoff

Implementation and bounded CPU checks only, not a launch or readiness receipt.
Main owns PLAN.md, protocol_lock.json, main_ready.json, the final bounded Sol
review, source freeze, readiness, queue/start, watcher and publication opt-in.
No GPU probe, allocation/Slurm command, training, queue, production ready receipt
or Git mutation is performed by this preparation. Historical/live sources and
both R1024090 workers are untouched. No subagents are used.

## Fixed Resources And Scope

| R105 arm / run ID | Original completed parent | Slot / UUID | Exact CPUs |
| --- | --- | --- | --- |
| A_static_gate / R105_A_static_gate | R104_A_width256 | 3090_0 / GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | 12,13,14,15 |
| B_input_gate / R105_B_input_gate | R104_B_depth8 | 3090_1 / GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | 16,17,64,65 |

Reuse job1465/gpu03 and container
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`,
one original UUID, CPU4 and mem9GiB per arm. Preserve original R81 GPU mutex
inodes, R98_parallel_dispatch/gpu03.lock and shared CPU reservations. No new
allocation, ordinary SSH, slot replacement, healthy-run interruption, automatic
OOM shrink, failed-stage retry or wrapper migration.

Data: `/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
Prepared: `code/V4/runs/R84_new90k_attention/prepared`.
Cache source: `code/V4/runs/R84_new90k_attention/input_cache`.
Direct in-place input cache: `/tmp/shiys_R84_new90k_cache`.
Missing/inconsistent cache fails; never copy, rebuild, repair or add a teacher,
target, solver run or instance. Frozen full18 native FP64 winners/candidate pools
and existing TRAIN-only geometry remain unchanged. Query labels never enter forward.

Both arms use seed2, logical1024, updates57600, warmup4320, VAL1440, TRAIN7200,
TSP/CVRP micro128, other tasks512, eval128 and exact-row accumulation. Original
AdamW/loss/R-Drop/no-aux/method/head/inference stay runner-owned and unchanged.
Both have3425561 total parameters and identical complete fresh seed2/105002
states, not trained initialization. A feeds zero router inputs and LEARNS the
bias mixture; only its2048 feature-weight parameters are inactive. A is NOT a
fixed-uniform gate. B uses post-attention node states. Both compute all experts;
equal tensors do not mean equal active freedom or a sparse/equal-FLOPs claim.

## Readiness Contract

The launch gate checks only actual compact common main-lock keys: experiment,
locked, arms, seed, task_scope, data_root, prepared_root, primary_cache,
rows_per_problem, batch_size/logical_global_batch, max_updates, warmup_updates,
eval_every, train_eval_every, large_micro_cap, other_micro_cap, eval_batch_size,
current_query_labels_in_forward and external_pretraining. It invents no
architecture/routing schema. The real runner's required_lock_fields() validates
the detailed main scientific lock; the CPU parser test checks that exact API.

`r105_wave.make_spec` can describe/write the exact future spec while model,
runner or tests are pending. A spec is NOT ready. `prepare` requires nonempty
main evidence and --runner-tests-passed, all named final sources, the actual
stage parsers, and no prior readiness/claim. Main first completes model/runner
CPU evidence and its one bounded Sol review. This sidecar does not create or
rewrite main_ready.json, protocol_lock.json or scientific fields.

Exactly five stages, in this order:

1. `r105_experiment train`: complete TRAIN, strict minimum whole18 VAL cost
   checkpoint (first exact tie), unique lock and full TEST included.
2. `r105_experiment test`: idempotent saved full TEST, no reselection.
3. `r105_experiment timing`: actual standalone complete model, first128 TEST
   inputs/task, one warmup and three forwards, not cache latency or end-to-end.
4. `r105_reporting lib --lib-batch-size 2`: both native LIB datasets, same lock.
5. `r105_reporting report --device cpu`: all five per-arm PDFs on admin01 CPU.

Runner train/test/timing receive --root, --arm, --prepared-root, --data-root,
--cache-root, --protocol-lock, --gpu-type3090, --large-micro-cap128,
--eval-batch-size128 and --wandb-mode offline. LIB/PDF use reporter CLIs.
There is no extra feature/preprocess/teacher/routing stage. The runner records
the input-only routing diagnostic after the unique checkpoint lock.

Production ready.json under R98_parallel_dispatch/R105_<arm> must have
main_thread_ready=true, runner_tests_passed=true, nonempty evidence matching
main, matching plan_sha256, source_freeze_ready=true. plan.json must have
main_thread_ready=true, matching evidence, exact spec/commands, source_sha256,
primary_cache_reuse_only=true, cache_copy=false, cache_preprocessing=false and
automatic_retry=false. Exact named runtime entrypoints plus static local imports
are frozen; tests are exact names, not test/version globs. RUNNER_PREP.md,
MODEL_PREP.md, this handoff, PLAN.md, protocol_lock.json and original R84 prepared
identity files are included. Unknown added source declarations are rejected.
Any source edit, removal or set change blocks launch/watch; never auto-refreeze.

Each predecessor gate reads actual original R104 saved TRAIN/VAL/full TEST,
all18 timing bound to report, both LIBs and five PDFs, same main lock, exact
slot and train/TEST/timing/LIB/PDF stage order, original worker0 and controller0.
The assigned parent's completion is independent of the other parent or global
comparison/publication. R104 is already completed/published (main reports
cb7afd2a6); launch checks actual receipts, not that statement or resource-idle
memory. No R99 recovery schema, failed-receipt substitution or null-VAL-SHA
bypass is accepted. Nothing in the predecessor archive is rewritten.

## Goal And Publication

Quiet default120sec watcher: unchanged/nonactionable states produce no periodic
notification. It waits for BOTH full pipelines/reports, exact locks, full TRAIN,
VAL, TEST, timing, native LIB, PDFs and `routing_diagnostic.json`, then one saved
CPU `r105_reporting compare`. Comparison requires its three CSVs and same-lock
protocol; no new COMPARISON.pdf/whole-version RESULTS.pdf requirement is invented.
Success/failure prints once. Errors, including scalar/list/nested errors, are
terminal and preserved. Comparison/watch/publication each have one-shot receipts.
Pipeline completion never implies research goal success.

Routing handoff follows the final RUNNER_PREP.md and implemented runner/reporter
API: `routing_diagnostic.json` (singular, not the earlier requested plural names).
It requires complete R105 arm/state/checkpoint/selected_update/
model_state_sha256/inference binding, split=val, first_instances_per_problem=128,
input_only=true, selection_use=false; labels_read/targets_read/model_reselection/
routing_calibration/auxiliary_loss/solver_quotas=false. All18 per_problem rows
use sample_indices0..127, exact_n_counts summing128, layers0..3,
valid_physical_nodes, top1_instance_counts, four-expert graph means and
mean_graph_mixture_entropy_nats (NOT mean node entropy), and within_exact_n groups with
explicit estimability for singleton groups. Padded nodes cannot contribute to
physical-node means/coverage. It is descriptive only, not evidence of efficacy or absence
of coarse shortcuts. The model's capture_routing tuple/API is runner-owned;
the administrative checker never loads a model or redoes routing inference.

Selected TRAIN capture file SHA may be null ONLY through the proven R104 rule:
exact TRAIN/VAL model-state hash, protocol state and selected successful update,
plus the official selected-VAL-to-checkpoint binding. VAL itself must be bound
to the exact final checkpoint SHA; original archive_capture_identity remains
equal to evaluation/val/selected_best/identity.json. No administrative metadata
repair, model loading or historical archive edit occurs.

Publication is NEVER implicit. Only explicit --publish/--publish-github with
matching main evidence enables the original shared codex-publication.lock,
empty-index/main-branch preconditions and one curated whole-version commit/push
after both pipelines plus comparison. Exact frozen R105 source names, curated
R105 report/metadata files and the approved paper
`docs/research/post_r104_conditional_modularity.md` are eligible. Exclude arrays,
weights, raw labels/costs/predictions, caches, teacher payloads, logs, tokens,
credentials, symlinks, unknown future source files and unrelated dirty work.
Existing staged changes abort without reset/revert. Failed publication preserves
the index and first error, without retry.

Main-only eventual entrypoints, not run in this implementation:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1
export R105_READY_EVIDENCE='MAIN_REPLACES_WITH_FINAL_CPU_SOURCE_AND_SOL_REVIEW_EVIDENCE'
python -B -m code.V4.r105_wave A_static_gate
python -B -m code.V4.r105_wave B_input_gate
bash code/V4/run_v4_r105_launch.sh prepare --spec code/V4/runs/R98_parallel_dispatch/specs/R105_A_static_gate.json --evidence "$R105_READY_EVIDENCE" --runner-tests-passed
bash code/V4/run_v4_r105_launch.sh prepare --spec code/V4/runs/R98_parallel_dispatch/specs/R105_B_input_gate.json --evidence "$R105_READY_EVIDENCE" --runner-tests-passed
```

Main may then choose one-shot `start` or `queue --interval 60` for each exact spec,
never both. Watch separately in main's tmux; publication requires explicit opt-in:

```bash
python -B -u -m code.V4.r105_goal --watch --interval 120 --evidence "$R105_READY_EVIDENCE"
# Later, only if main explicitly authorizes whole-version publication:
python -B -u -m code.V4.r105_goal --publish --evidence "$R105_READY_EVIDENCE"
```

## Bounded CPU Verification

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest code.V4.test_r105_launch code.V4.test_r105_goal -v
```

Only small temporary JSON/PDF/text fixtures and masked CPU tensors are used.
No data reconstruction, dataset/label reads, real ready/queue/worker, model
training, Slurm command or Git action is tested. Real runner/reporter parsers
are parsed only, not executed. Full model tests are not repeated here.
Administrative suite:23/23 passed in2.314s after final filename alignment to
RUNNER_PREP.md. All ten actual arm-stage CLIs and the CPU compare parser passed.
Real R104 original predecessor checks passed for BOTH arms with errors=[];
the exact named source closure contained96 files and no missing declarations.
The wrapper also passed bash -n. These are administrative CPU checks, not a
GPU/resource witness, a model/runner review or permission to launch.
