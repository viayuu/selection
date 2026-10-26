# Combined R114 Closeout Sidecar

CPU-only integration of **saved** original v1 labels for the other 15 problems
and separately locked CVRP host-audit-corrected labels. No solvers, models,
GPU probes, retries, frozen-module edits or publication are run by development
tests. Synthetic fixtures are not acquisition evidence. Corrected native labels
were not available at implementation time.

## Readiness And Scope

The parent must have its exact terminal
`preparation/grouped_execution/queue_v1/scheduler_progress.json`, an identical
immutable scheduler result, actual assembled labels and completed saved analysis.
The child plan is exactly:

`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/plan.json`

Its full CVRP scope is **10 methods / 32 bases / 96 endpoints / 64 pairs / 960
tasks**. Every pool, endpoint, task, seed and pair metadata field must match the
CVRP subset of the full parent manifest. Both actual assemblies, analyses,
child finalized invocation, launch receipts, actual RTX3090 hardware/UUID and
execution binding must be verifiable. CPU_READY alone never opens this gate.

The main resource owner launches the unchanged runner with `--execute
--finalize`. The builder's `run_native.sh` omitting `--finalize` does not prevent
this standard per-child assembly and analysis. Resource-owner scheduling may
overlap original other-15 execution on GPU0 with corrected CVRP on GPU1 after
R113 B completes; this sidecar does not schedule either.

The original ICAM/CVRP failure was a CUDA-to-NumPy **host audit** error, not
evidence of solver incapability. Original attempts and receipts stay untouched.
Only CVRP label endpoint/pair rows are replaced. The merged header retains the
full parent manifest identity, sets `lock_sha256=null`, and supplies per-problem
`execution_lock` and source provenance, including full manifest **file** hashes.
Coverage and FP64/stable-global-ID winners come from actual arrays. Missing
costs remain null and false-masked, not zero. An incomplete pool yields a
`partial_negative_closeout`, never labeling finished.

## CPU Commands

From `/public/home/shiys/0selection2`:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B -m unittest -v code.V4.test_r114_recovery_finish
python -B -m code.V4.r114_recovery_finish status --plan /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/plan.json
```

Main starts the following **only after CPU code review**, preferably in tmux:

```bash
python -B -u -m code.V4.r114_recovery_finish wait --publish --timeout-hours 336 --poll-seconds 60 --plan /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/plan.json
```

For reference, the resource owner's exact existing-runner stage command is
below. It is **not** a command for this CPU-only sidecar to execute. The owner
must already hold the GPU mutex inside the authorized container and assign the
actual host, physical UUID, 1-2 CPU IDs and unchanged timeout:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES="$R114_RECOVERY_GPU_UUID" python -B -u -m code.V4.r114_run_queue \
  --plan /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/plan.json \
  --manifest /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/manifest.json \
  --deployment-lock /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/deployments.grouped.lock.json \
  --host "$R114_RECOVERY_HOST" --gpu-uuid "$R114_RECOVERY_GPU_UUID" \
  --cpu-ids "$R114_RECOVERY_CPU_IDS" --timeout-seconds "$R114_RECOVERY_TIMEOUT_SECONDS" \
  --execute --finalize
```

`prepare` instead of `wait --publish` generates CPU-only reports without Git.
`verify` checks an existing combined delivery. Status/readiness returns 2 while
waiting; validation errors return 1, timeout 124. No automatic preparation,
solver, commit or push retry exists.

## Delivery And Publication

The monitor-required fixed output is:

`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/finish/tensor_boundary_combined_20261026`

It contains primary `RESULTS.pdf`, `RESULTS.md`, `counts.csv`, `closeout.json`,
`response_analysis/*`, input/source hash provenance and `finish_complete.json`
with `closeout_ready`, `kind`, `closeout_kind`, `labeling_complete` and `inputs`
including the exact parent progress file hash. Merged `combined_labels.json`
is server-only, hashed but not published. English PDF text avoids a new CJK
font cache/dependency. This is descriptive TRAIN response labeling, not a
trained function learner, TSP/ATSP response or full18 model result.

The baseline is prepared/verified through original `r114_finish.prepare(root)`
and preserved separately, never published alone. Future explicit publication
waits for R113's independent successful publication before using a **private
in-memory** copy of the existing R114 publisher with only prepare, verification
and exact allowlist hooks replaced. Its shared `.git/codex-publication.lock`,
one commit / one ordinary push maximum, unchanged upstream/index/dirty-external-
dependency guards, receipt and no-retry rules remain in force. No Git is used
by status, preparation, verification or tests.

The allowlist includes baseline and combined reports, the named sidecar/helper/
builder/launcher code, private corrected ordinary script, exact profile/lock
files and small provenance. Symlinked source bindings, full manifests, raw
labels/routes, weights, caches and dispatch-token `ready.json` are excluded.
Other source originals are represented by their bindings/hashes, not silently
added. Future publication receipt remains at
`preparation/finish/publish_receipt.json` and its `delivery` must match the fixed
combined directory exactly. Existing failed/uncertain attempts are never reset.
