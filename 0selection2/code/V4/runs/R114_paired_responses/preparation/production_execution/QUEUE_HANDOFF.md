# R114 Grouped Queue Handoff

## Current Status

Ready for resource scheduling, NOT label-complete. Actual GPU labels=0;
runtime-qualified deployments=0/76. All76 route hooks are source-only evidence.
No GPU launch, native protocol repair, or Git commit. User reports all GPUs busy
for about3h; this worker does not poll, allocate, or modify R113 transport.

This supersedes operational instructions for historical per-task queues
`label_queue/` and `label_queue_v2/`. Their100 gates/200 calls and one-process/task
topology are NOT the current scheduler. Old artifacts and cold source lock remain.

| Item | Frozen Current Contract |
| --- | --- |
| Scope | Research4.4 pilot;16 applicable problems x32 frozen TRAIN bases=512; no TSP/ATSP augmentation |
| Inputs |2,048 endpoints;1,536 pairs;9,600 entire-current-legal-pool production calls |
| Queue |76 runtime groups +76 label groups; one isolated problem/method backend per process |
| Qualification |100 exact deployment buckets;252 grouped probes +100 seeded replays +52 eager cold references=404 calls |
| Minimal isolation |Bucket original/changed pairs, two bases, original replay; one cold original per eager group; no extra base for dual-bucket groups or separate cold reference for cold-lazy groups |
| Lifecycle |52 eager backends persistent;24 RF/MoSES groups cold-reconstruct lazy subrunner each endpoint;3,072 production lazy reloads, not76 total weight loads |
| Numerics |Native float(cost) promoted to FP64; independent FP64 route recomputation is audit-only; exact tie by stable global ID |
| Seeds/budgets |Singleton batch1; reset Python/NumPy/Torch/CUDA each endpoint; same per-base/method seed; locked actual native profiles/budgets, no defaults |
| Failure |First invalid/missing route/profile/equivalence/timeout/incomplete attempt stops that problem; continue others; no auto retry, base replacement or protocol edit |
| Outputs |Production receipts only; stopped problems entirely masked; assemble partial/interrupted results, then read-only CSV/PDF analysis |
| CPU check |56 tests passed in7.960s,2 cores/numerical threads1/GPU hidden; solver execution mocked in temporary fixtures only |

Full manifest:
`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/production_inputs/manifest.json`

Current plan:
`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/plan.json`

Current grouped source/protocol lock:
`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/deployments.grouped.lock.json`

Unchanged independent cold reference source lock:
`/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/production_execution/deployments.production.lock.json`

Grouped lock retains native profiles/budgets/checkpoints and data/seed source;
adds queue, grouped worker, scheduler and main-owned r114_response_analysis.py at
freeze only. Changed frozen sources require a separately versioned plan/lock,
not overwriting this queue or silently continuing a changed native protocol.

## Exact Single-GPU Command

For Jason01a1727e-587d-7c13-8361-261044077ba1/resource worker: this is the fourth
independent job beside the three R113 arms, NOT a four-GPU R114 job. Resource
worker holds its existing external GPU mutex for this entire command and exports
CUDA_VISIBLE_DEVICES as exactly one assigned physical GPU-... UUID, plus
R114_CPU_IDS as its assigned1-2 CPU IDs. No GPU-name inference or auto allocation.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c "${R114_CPU_IDS:?resource worker must assign 1-2 CPU IDs}" \
  python -B -m code.V4.r114_run_queue \
  --plan /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/plan.json \
  --manifest /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/production_inputs/manifest.json \
  --deployment-lock /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/deployments.grouped.lock.json \
  --host "$(hostname -s)" \
  --gpu-uuid "${CUDA_VISIBLE_DEVICES:?resource worker must export one physical GPU UUID}" \
  --cpu-ids "$R114_CPU_IDS" --timeout-seconds 3600 --execute --finalize
```

Timeout is a per-active-endpoint/loading-stage safety ceiling, not a changed
solver search budget. Watchdog kills the full worker process group including
cold/native descendants on timeout/interruption. First execution permanently
binds host/GPU UUID/CPU IDs/timeout. Production is serial, with no GPU polling.

One full source/input preflight per scheduler invocation; no conversion rerun,
source inventory audit or extra comparison sweep required. Per-solve input hashes,
actual native profiles and independent route audits remain mandatory. Pending
work resumes with exactly the same command/binding; incomplete/failed attempts
remain stopped, not retried by this scheduler.

## Receipts And Analysis

Queue-local scheduler_execution.lock and queue_execution.lock do not replace the
resource worker's external mutex. Groups retain immutable request/log/cost/tour/
result evidence. scheduler_progress.json is atomic mutable status; invocation
history is scheduler_runs/run_NNNN/ with events.jsonl and result.json.

Every --finalize invocation, including interruption/partial failure, exports and
assembles under its own run directory. Stopped problems are excluded from export
and therefore all their winner rows are masked. Raw partial costs remain in
attempts; unrun tails are not fabricated. Qualification costs never become labels.
Root labels.json is created only for a fully completed queue, never for partials.

After actual export and assembly the scheduler runs the main-owned CLI:

```bash
python -B -m code.V4.r114_response_analysis --manifest FULLMANIFEST --labels ACTUAL_ASSEMBLED_JSON --output UNIQUE_RUN_ANALYSIS_DIR
```

Concrete argv is saved in analysis.command.json:
--labels .../scheduler_runs/run_NNNN/labels.json and
--output .../scheduler_runs/run_NNNN/response_analysis.
Analysis hides GPUs and writes read-only CSV/PDF without training claims. Analysis
failure preserves labels, records failure and never reruns solvers.

Exit codes:0 complete/dry-run,2 blocked partial,3 explicit bounded yield,
130 interrupted,1 infrastructure/analysis failure. One unsupported method does
not globally abort independent problems or falsely certify a full-pool winner.

## Coordination

Ready metadata/source hashes: grouped_execution/READY_RECEIPT.json.
This session has no Sol-agent messaging tool; bounded attempts to contact the
managed local daemon were rejected. No direct message delivery is claimed.
This addressed handoff is ready for main/resource worker consumption without
editing R113 transport or requiring another CPU audit/source pass.
