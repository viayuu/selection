# R77: Paired Sample Size, Batch 1024

## Release Status (2026-10-15)

Both arms completed 49,680 successful updates. Retain whole R77B, selected at
epoch28/update34776 with SHA256
`31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b`.
The same lock now covers VAL, TEST, TSPLIB and CVRPLIB; R76 LIB is not reused as
R77B performance. See the [22-page full report](runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/MODEL_STATISTICS.pdf),
[18-page distribution PDF](runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/selection_distributions.pdf)
and [current summary](runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/RESULTS.md).
The execution/resource notes below are historical; they do not authorize a rerun.

A uses exactly 10,000 TRAIN instances per task; B uses all 70,114. Both cover
all 18 tasks, with the original 3,895 VAL and 3,895 TEST rows/task unchanged.
Data: `/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`.
R76 caches and FP64 costs are read-only. No solver execution, cost regeneration,
R74 resume, historical selector weights, or extra seed is involved.

## Precommitted Pair

- A is a subset of B, sampled from real `input_metadata.npz`: proportional
  largest-remainder quotas at `scale`, then existing `distribution`, then existing
  `capacity_type`; uniform sampling without replacement within leaves. Never
  stratify on winner, labels or solver costs. Save native source row indices.
- R45A/R76 model: handcrafted34 plus ID, ScoreDifference/maximin, R43 supervised
  loss plus legal-candidate R-Drop. Both arms load the exact same named trainable
  tensors from fresh seed2 `identicalinit_seed2.pt`. Geometry mean/std buffers
  are fitted only on each arm's own TRAIN valid nodes and may differ.
- Exactly 49,680 **successful** optimizer updates per arm:
  `18 * ceil(70114 / 1024) * 40`. Each task receives 2,760 updates and 2,826,240
  sample exposures. A therefore cycles through its population more often.
  The ceil-based reference is 40 FULL-equivalent update rounds, not 40 exact
  data passes (B gets about 40.31 passes, A about 282.62).
- Every logical update has exactly 1,024 real rows. A continuous seeded per-task
  shuffle stream retains the tail and concatenates the next cycle's beginning;
  no drop-last, padded duplicate rows, or short optimizer updates. Both arms use
  the same task permutation each18 updates, with no consecutive identical task
  even across round boundaries. Sample permutations are population-specific.
- Identical **pre-fixed per-successful-update LR**: linear warmup through update
  3,726 to `1e-4`, then cosine to exactly `1e-6` at update49,680. AdamW WD=`1e-4`,
  dropout=`0.1`, clip norm=`1`, R-Drop weight ramp to `0.35` over 3,726 updates.
  Overflow retries do not advance the LR, stream or budget. No VAL-driven LR
  adjustment and no early stopping.
- Complete VAL every1,242 successful updates. Save the strict minimum
  `macro_vs_sbs_pct` using the **same FULL 70,114 TRAIN-fitted SBS identity** for
  both arms. Each arm's own TRAIN-fitted SBS is also reported on all splits,
  explicitly reporting-only, never used for checkpoint selection or scheduling.
- Evaluate each arm's own TRAIN every5 VAL points, then evaluate the locked best
  finally if it lacks TRAIN metrics. TRAIN populations differ; compare the arms
  on identical VAL/TEST populations, not their raw TRAIN metrics.
- After the entire fixed budget, lock the unique validation-selected checkpoint
  and evaluate the full TEST exactly once per arm. A one-shot `test_started.json`
  prevents automatic repeats, including after an interrupted TEST.

R76 is the architecture/loss/cache reference, **not a rigorous sample-size
control**: historical R76 used epoch control, global batch9,856, VAL plateau LR
and early stopping. R77 intentionally changes batch and update-axis LR/budget to
make A/B optimization exposure comparable. Checkpoint `epoch` filenames denote
VAL evaluation points, not an arm-specific data pass.

## Execution

Two independent single-GPU processes, no DDP synchronization. The user-approved
allocation is job1475 / gpu05 (resource agent confirmed idle cards and caches):

| Arm | GPU UUID | TSP/CVRP micro cap | Other tasks |
|---|---|---:|---:|
| A | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | 352 | 1024 |
| B | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d | 352 | 1024 |

Use threads2, R76 pinned prefetch2 and the existing node-local cache at
`/tmp/shiys_R76_R45A_single_reld_mt_cache`. TSP/CVRP accumulate
352+352+320 rows; micro losses are weighted by rows/1024 for a global mean.
Clip once and step once. Overflow replays the entire batch with restored RNG.
The reusable cap helper also supports3090 at most384, but the current launcher
accepts only these two approved4090 UUIDs. No new capacity sweep.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh prepare
# Only after the allocation is confirmed and the user approves:
R77_APPROVED=1 bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh start
bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh status
```

`start` creates a tmux session, launching two `srun --jobid1475 --overlap --exact`
workers with4 CPUs /9G each (20GiB allocation leaves room). Each worker finishes train, the locked TEST and saved
prediction analysis. It never starts four-rank global training. To resume the
same pair after an interrupted TRAIN, use `R77_APPROVED=1 R77_RESUME=1` with
`start`; last checkpoints retain complete VAL boundaries, shuffle cursors,
optimizer/scaler and RNG. Intervals not yet saved in a checkpoint replay from that boundary.
Already completed TRAIN/TEST stages are not rerun by `finish`.

Slurm20.11 otherwise assigns every job GPU to the first step, blocking the second.
`allocated` therefore requests exactly `--gres=gpu:1` per step within job1475;
UUID whitelisting and arm ownership additionally restrict CUDA to its approved
card. `--gres=none` is not usable here because device cgroups forbid GPU access.
This does not request another GPU or change the job's two-GPU allocation.

### This Run's GRES Recovery

A resumed from **this R77 run's own saved checkpoint at successful update2484**,
not from an R76/historical selector or a Git commit. The same R77 model, optimizer,
scaler, RNG and per-task shuffle cursors continued at update2485. A's first process
and B both began with the common fresh seed2 trainable tensors. B's unsuccessful
launch attempts performed zero updates before its single-GRES training started.
No architecture, supervised/R-Drop objective, gradient formula, logical batch,
optimizer, update-indexed LR or budget changed between A's early training and B.
Runtime fixes covered GRES launch/ownership, status/monitoring and reporting only;
the already predeclared arm-specific geometry buffers are intentionally different.
"Saved" here describes an experiment checkpoint, never a Git commit; none was made.

Focused CPU checks only:
`CUDA_VISIBLE_DEVICES='' python -B -m unittest code.V4.test_r77`.
W&B defaults to offline in project `selector`, entity
`yjkds-southern-university-of-science-technology`; `WANDB_MODE=online` is supported.

## Artifacts And Timing

Root: `code/V4/runs/R77_sample_size_batch1024/`. Preparation records the input
strata, A row IDs, separate geometry/SBS, common trainable hash and cache provenance.
Each arm records updates, actual successful counts, LR, loss components, gradient
norm, AMP retries, samples/sec, memory and complete per-task VAL/TRAIN metrics.
Saved VAL/TEST predictions support all18 top1/2/3, mean-cost, single-method
comparisons, arm-distribution and loss plots. `paired_test_comparison.csv` reports
the common-denominator A/B comparison after both one-shot TESTs finish.

Existing R76 4090 capacity timings were ~0.41s for CVRP352, ~0.42s for TSP443,
~0.29s for MVRP2464 and ~0.31s for ATSP2496. Extrapolating to this schedule gives
roughly **0.25-0.45s/update, 3.5-6.2h training/arm**, plus VAL/TRAIN/TEST evaluation
and filesystem overhead. These are estimates, not an R77 GPU benchmark; the two
arms run in parallel. The first72 successful updates record measured throughput
for a revised ETA. No GPU is probed or trained before approval.

`val_report` reads only saved validation scores/labels and writes
`VALIDATION_ONLY.md`, `validation_only_overview.json` and
`validation_class_history.csv`. Zero selection means a legal method/task with
positive VAL winner support but no predicted selection. Near-zero (<=0.1%) and
rare VAL support (<1%) are separate descriptive diagnostics, not sampling,
checkpoint or task4 acceptance rules. Counts alone do not imply a cost improvement.

The separate CPU `report_watch` writes root `STATUS.json` / `STATUS.md` every5
minutes, with actual counters, last complete VAL, latest common-update A/B table,
TRAIN/VAL curve gaps, read-only telemetry on the two approved GPUs, and stage gates.
It never controls or pauses the existing training or monitor processes. After both
one-shot TEST/analysis pipelines finish it refreshes saved-result summaries, including
ALL + four families +18 task rows in `TEST_ROWS.md` / `paired_test_comparison.csv`,
then writes `delivery_ready.json`. Root STATUS is the continuation entry after a
context compaction; existing locks/claims must be preserved, not retrained or retested.
The CPU `finalize_delivery` guard waits for the current controller, old30s monitor,
and5min reporter to exit before rendering the final23 scope rows. This prevents an
already-loaded older Python summary function from overwriting the new families table.
`FINAL_DELIVERY.json` is the final handoff marker; its counters, hashes and18 TEST
row counts are checked from saved artifacts only, without model inference or reTEST.

Canonical final paths are root `STATUS.json`, `SUCCESS_RECEIPT.json` and
`finalcomparison.csv`. The success observer does not stop or reassign any GPU;
it waits for both49680-update pipelines and observes both approved UUIDs idle
with no compute processes before writing SUCCESS_RECEIPT. The long comparison
contains all saved TRAIN/VAL/locked TEST points, ALL/four families/18 problems,
strict native Top1, regret, common and own TRAIN-fixed SBS, unique TRAIN counts
and `common_updates`. `COMMON_UPDATE_CURVES.png` places common successful
updates on the vertical axis. TEST is never recomputed for reporting.

## Locked TRAIN/VAL Diagnostic Handoff

`diagnostic_handoff` preserves the already saved final-selected TRAIN/VAL NPZ
and metric files using hard links under root `selected_train_val_predictions/`.
`diagnostic_watch` waits on the existing fixed-budget completion/VAL-best locks
and exports each arm as soon as its selected TRAIN evaluation is saved. Both
stages are CPU-only: no forward pass, TRAIN rerun, TEST read or rule change.
The final manifest records the checkpoint hash/update, source prediction hash,
per-problem row counts, common/own TRAIN SBS and geometry provenance.

Each NPZ uses `<problem>__` prefixes and contains `logits`, `pool`, `pool_ids`,
`nodes` (valid node counts), `costs` (native cached, unnormalized solver costs),
`winner`, `indices` and `source_indices`. Score/cost columns follow the local
`pool` slots; `pool_ids` maps those slots to global solver IDs. `source_indices`
maps A's TRAIN rows to the original FULL split; `indices` is the local row space.
Raw coordinates/features remain in the existing R76 cache, referenced by the
manifest. The score files are uncalibrated logits, not fitted probabilities.

`selected_train_val_predictions/manifest.json` has `complete=true` only after
both locked handoffs are ready. Its `A/` and `B/` directories each contain
`train.npz`, `val.npz`, both metrics JSON files and a checkpoint-specific manifest.
This preserves inputs for Feynman's separately owned
`r77_probability_diagnostics.py` / `train_val_selection_diagnosis/`; the R77
runner does not edit that code or directory. Root `DIAGNOSTIC_INPUTS.md` describes
the exact row/column conventions. No TEST prediction is included in this handoff.
