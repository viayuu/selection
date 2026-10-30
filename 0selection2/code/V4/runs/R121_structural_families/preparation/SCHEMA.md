# Full18 Collection Schema And CLI

## Identity Layers

1. `selection.json`: frozen new90k scenario/hash;18 legal pools in stable global
   ID order; input-only TRAIN base choices; exact source/local indices and split
   mapping hashes. No individual cost/winner file is loaded for selection.
2. `protocol.json`: independent R121 version, two structural axes/task, numeric
   generation settings, native generator/R114/new adapter source hashes, seed
   semantics and unchanged B/R58 definitions. Altering these needs a NEW plan.
3. `deployments.json`:89 problem/method rows, source alias, old recorded profile
   identity, complete native budget, checkpoint path/SHA and dependency binding.
   CVRP uses the corrected cold lock. Current small-source drift/missing files
   are explicit blocks; large weight-content checks and ALL current runtime
   qualifications remain pending. Historical replay is never claimed.
4. `collection_plan.json`: prospective full18 counts, waiting state, no GPU
   assignment. Compact groups expand to endpoint x WHOLE legal pool; no historic
   winner-based filtering. Default256 bases/task gives4608 bases,23040 endpoints,
   113920 labels. This budget is planned, not executed or scientifically certified.
5. Prepared `manifest.json`: only the actually generated subset/prefix, native
   input file hashes, original base and shared-geometry group IDs, internal
   family partition, endpoint axis state, full candidate tasks and common
   initial per-base/method solver seeds. Unprepared tasks are not called complete.

Family partitions (`family_train` / `family_val`) both come from frozen TRAIN.
All descendants of a base and exact shared selected base geometry/matrix hash
stay together, including exact cross-task geometry aliases. Natural VAL/TEST
are unchanged. TRAIN/holdout source-index exclusion is checked per problem;
this is not a new exhaustive global holdout-geometry audit or a blind-test claim.

## Endpoint And Label Boundaries

An input JSON payload contains `item` with the EXACT native fields: TSP/ATSP
singleton tensor nested lists, CVRP `depot/loc/demand`, MVRP existing dictionary.
Everything else (base/family/group/axis/seed/source IDs) is metadata, never a
selector-forward input. Candidate costs/winners/solver routes are separate
receipts and TRAIN supervision only, never a query feature.

Each family is the original plus `z00,z01,z10,z11`. Both axes have nontrivial
input effects; collapsed/degenerate families fail rather than fabricate coverage
or silently replace the base. Routing geometry is explicitly constructed within
original singleton reachability. TW changes produce feasible but differently
compatible customer windows under the same native3.0 deadline and old service
durations; singleton feasibility does not imply the same multi-customer routes.
There is no family/cost reuse merely because some geometry was preserved.

Future per-task receipt fields include version/plan/manifest/input/deployment
identity, native cost/tour, independent route audit, common initial solver seed,
model seed, stage, device, explicit physical-binding hash, profile SHA, backend
audit, total wall time including loading, and first-error evidence. Failed
attempts are retained; no retries, repairs or base substitutions are automatic.
Preflight receipts are NOT production labels.

`full_pool_targets(pool, records)` requires one successful new-version LABEL
receipt per legal method on the SAME endpoint. It returns FP64-promoted native
costs, observed mask, all-best exact ties, stable-global-ID native winner and
`log C - mean(log C)`. Nonpositive/missing/invalid costs fail, with no epsilon
or zero filling. Differences between endpoint advantages are derived TRAIN
supervision, not extra independent observations or a cycle-loss novelty claim.

The CPU plan/preparation commands import only data-generation code. The future
`task --execute` branch lazily reuses R114 `NativeSession` initialization and
native routing solve/audit. Its new TSP/ATSP branch restores native tensor
packing and exact directed/geometric tour auditing, NOT historical MATNET role
draws. The independent R121 protocol uses newly seeded native roles. Recorded
profile mismatch fails instead of silently accepting current defaults.

## CPU Commands Now

Every command must first source conda and activate `easynco`. CPU affinity below
is an example of the one-core preparation budget, not a GPU binding.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 47 python -B -m code.V4.r121_structural_data plan --output /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/planned_v1 --bases-per-problem 256 --seed 2
```

After the plan exists, a bounded real-input CPU preparation prefix:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 47 python -B -m code.V4.r121_structural_data prepare --plan /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/planned_v1/collection_plan.json --output /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/cpu_smoke_v1 --problems TSP ATSP VRPBLTW --limit 1
```

Full CPU materialization, only when MAIN chooses its final population/protocol:
omit `--problems` and set `--limit 256`, using a NEW output directory and a
long-running CPU tmux/log. It still performs ZERO solver calls. Do not confuse
CPU endpoint preparation with completed candidate labels.

## Future Collection, NOT Launched Here

`task` is dry by default. Obtain an exact task ID from the prepared manifest.
Future execution requires ALL of these explicitly:

- `--execute --acknowledge-new-protocol --device cuda:0`.
- `--release MAIN_RELEASE.json`: `owner=main`, matching `r121_plan_sha256`,
  `TEST_complete/LIB_complete/report_complete/authorized_GPUs_released=true`,
  `authorized_gpu_uuids`, and actual R120 evidence path/SHA pairs.
- `--binding MAIN_BINDING.json`: actual process hostname, one physical
  `gpu_uuid` from that release, `cpu_ids`; `CUDA_VISIBLE_DEVICES` must be that
  exact UUID and `taskset` affinity must match. Logical `cuda:0` alone is not
  accepted as a physical-GPU identity.
- First `--stage preflight` with a NEW immutable receipt path. All current89
  deployments, especially TSP/ATSP13, remain runtime-unqualified in this handoff.
- Only then `--stage label --qualification PREFLIGHT.json`, matching the same
  plan, input manifest, deployment, physical binding and device. A single
  preflight does not establish successful execution of every family/scale;
  every production receipt must independently pass its native route audit.

CLI template for the later main-bound worker, NOT an instruction to run now:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES="${MAIN_GPU_UUID:?after R120 release only}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "${MAIN_CPU_SET:?main binding}" python -B -m code.V4.r121_structural_data task --manifest "${R121_INPUT_MANIFEST:?prepared manifest}" --task-id "${R121_TASK_ID:?exact task}" --output "${R121_ATTEMPT_RECEIPT:?new path under preparation}" --execute --acknowledge-new-protocol --device cuda:0 --release "${R120_RELEASE_RECEIPT:?main evidence}" --binding "${R121_PHYSICAL_BINDING:?main binding}" --stage preflight
```

Fresh-backend loading and all source/checkpoint verification are included in
future cost accounting. No GPU ETA is inferred from CPU generation. Warm
grouping can be a separately qualified execution version later; this preparation
does not transfer old16-task equivalence proofs to new full18 defaults.
