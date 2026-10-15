# R75 Large Complete-Label Data

## Release Status (2026-10-15)

Completed. The old-split run is negative against saved R65B; see
[final summary](runs/R75_large_complete_labels/FINAL_RUN_SUMMARY.md).
The current retained new-data model is R77B, not R75. The launch/status notes
below describe historical execution, not a running job or a restart request.

## Current Authorization

Capacity selection is complete: user approved per-task384 / effective1152, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. See the final
`runs/R75_large_complete_labels/BATCH_SELECTION.md` and `batch_recommendation.json`.
The one formal seed2 pipeline is launched in `r75_train`, with a separate
`r75_monitor` CPU observer. It prepares the full input cache, trains, locks one
validation-selected checkpoint, runs one complete TEST, then performs analysis.
See `runs/R75_large_complete_labels/status.json` for the live phase.

The user approved one seed2 full18 training/test run based on R65B after the
bounded capacity measurements. Formal authorization is recorded in
`runs/R75_large_complete_labels/user_authorization.json`. The original native70
goal stays paused; R75 is a separate explicitly requested experiment.
R74 has been terminated to release memory; all R74 disk assets are retained.

Data: `/public/home/shiys/训练数据_公共完整标签_20261013_1923`.
There are1,786,887 full-label TRAIN instances: TSP87904, CVRP98983, and100000
for each other task. Original VAL/TEST are unchanged. The eight legacy B-task
semantic discrepancies remain deferred; this is not a repaired benchmark.

## Batch Measurement

- Unchanged NativeUtilitySelector: original dual-stream128d/4 instance layers,
  2 joint layers, solver ID plus handcrafted34 features.
- Exact R65B update: two dropout forwards per task, native CE and symmetric KL,
  FP16 AMP, three unchanged-state FP32 task gradients, arithmetic mean, clip1,
  one official ScheduleFree AdamW update.
- Real new TRAIN rows, CPU cached input/geometry subsets, current batches only
  on the assigned GPU. No entire-data GPU cache and no duplicated dummy rows.
- Maximum-size TSP/CVRP/MVRP and TSP/CVRP/ATSP stress cases precede six
  representative mixed-size triples covering all18 tasks.
- At least3GiB conservative device headroom, at most8 batch values,80 successful
  scratch updates and600s GPU measurement. All sizing state is discarded.
- Recommend the largest safe candidate within95% of best measured mixed-size
  throughput. Timings include current-batch slicing/H2D and the complete update;
  one-time CPU input/geometry preparation is reported separately.
- Historical geometry statistics are used only for sizing. The eventual formal
  run needs new TRAIN-only statistics and sample-count-aware scheduling, not the
  original hardcoded10000-row schedules.

Commands, already limited to capacity stages:

```bash
conda activate easynco
bash code/V4/run_v4_r75_capacity.sh prepare
bash code/V4/run_v4_r75_capacity.sh measure
```

The measure launcher uses existing allocation1465 on gpu03, binds only UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, and refuses an occupied device.
Current physical index is recorded, not assumed. No other GPU is touched.
The two stages run in owned tmux sessions; W&B capacity logging is offline.

Results: `runs/R75_large_complete_labels/BATCH_SELECTION.md`,
`batch_comparison.csv`, `batch_recommendation.json`, per-batch receipts and logs.
Formal command: `bash code/V4/run_v4_r75.sh`; monitor command:
`CUDA_VISIBLE_DEVICES="" python -B -m code.V4.r75_monitor` in the same environment.
The R75 cache is node-local at `/tmp/shiys_r75_large_complete_labels_cache`;
raw data, normalization records, progress, weights and outputs remain separately
identified. There are1566 updates per full epoch,4698 successful-update warmup
steps and62640 maximum updates over40 epochs, with every real tail retained.
No interim Git commit is created. Checkpoint selection uses full18 validation
native correct count only; TEST never changes epochs or rules.

## Runtime Input Read-Ahead

The real full-data random mmap reads are slower than the small CPU-RAM capacity
fixtures. Capacity throughput is not the formal end-to-end ETA. An independent
CPU-only helper in `r75_prefetch` briefly read upcoming TRAIN input pages from the
exact existing epoch plan. It did not produce a sustained throughput gain and
was stopped at2026-10-13T17:16UTC (step1465.122 only). It never changed inputs,
labels, order, RNG, model, batch or budget, and did not restart the live trainer.
Its row-touch counters are IO counters, not additional training presentations.
See `runtime_io_note.json`, `prefetch_status.json` and `prefetch.log` inside the
run root. It used one CPU and2GiB in the existing allocation, with CUDA hidden.
At that time the formal trainer remained step1465.121 on the approved UUID;
the later user-authorized RAM continuation is described below.

## Lossless RAM IO Continuation

The user explicitly requested fixing GPU input stalls. A12.4s sample found2037
major faults and4.714GB of disk reads while the old16GiB step hit its memory cap.
The complete padded caches occupy22.11GB; lossless packing removes only zero
padding and reduces TRAIN+VAL resident inputs to14.03GB. A new19GiB step keeps
them in anonymous CPU RAM and uses bounded per-task pinned staging for H2D.

At logged/durable successful update16860, the old process was quiesced and its
complete checkpoint archived. Its model, optimizer, scaler, RNG, epoch and cursor
are resumed by `r75_fast_runner.py`; the original scientific source files and
protocol hashes remain unchanged. No logged successful update is replayed. At
most one in-flight, uncommitted group may have been discarded. Four real CPU
batches match every original field, dtype, order and padding exactly.

Initial RAM continuation tmux: `r75_train_fast1`; Slurm step1465.125;
monitor: `r75_monitor`. The subsequent pipeline continuation is below.
The restart command was `bash code/V4/run_v4_r75_fast.sh` (already launched; do
not start another process). Full provenance and timings are under
`runs/R75_large_complete_labels/runtime_io_continuation1/`, especially
`RUNTIME_IO.md` and `optimization_measurements.json`.

The observed balanced120-update window improved from2.3715s to0.4081s (5.81x).
Each window contains each of18 tasks20 times; batches are not paired. Separate
12-second GPU samples average18.17% before and94.25% after. This is not a promise
of constant100% utilization. The first mixed epoch11 reports post-resume epoch
wall only; use subsequent complete epochs for whole-epoch time comparisons.

## One-Ahead CPU Pipeline Continuation

The user requested further NSS-like CPU/GPU input optimization. The same formal
run resumes at logged/durable update25380 in tmux `r75_train_pipeline2`, step
1465.128, on the same GPU. The archived checkpoint includes the existing model,
optimizer, scaler, RNG and partial-epoch cursor. No logged update was replayed;
at most one uncommitted in-flight group at this transition may be retried.

`r75_cpu_prefetch.py` prepares exactly the next task group on one CPU thread.
The main thread retains pinned-staging/H2D-event protection and owns all CUDA
calls; no additional GPU group is prefetched. An isolated runtime copy of the
original training loop replaces only input supply and removes two unconditional
timing barriers. The original update arithmetic and all scientific sources are
unchanged. Batch384, optimizer, task/sample order, budget and VAL/TEST rules stay
fixed. This is not a new experiment or a resumption of the paused native70 goal.

The already-running command is `bash code/V4/run_v4_r75_pipeline.sh`; do not
launch a duplicate. Provenance and observed-window timings are saved under
`runs/R75_large_complete_labels/runtime_pipeline_continuation2/`. The resumed
epoch17 is mixed; its epoch wall-time field covers the latest segment only.
Use later complete epochs for whole-epoch timing. Input packing had already
provided the main speedup; this continuation only seeks a modest further gain.

The first balanced120-update comparison is0.4158s before versus0.4118s after.
Wall throughput, including intervening logging/checkpoints, is2754.7 versus
2715.3 rows/s; the after window has one recovered AMP retry versus none before.
These different-batch short windows do not establish a further material gain.
The12-second GPU sample averages92.5%, not sustained100%. No additional recipe
or batch search is opened; the same formal training continues.
