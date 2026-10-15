# R79: Original NSS on the New Large Dataset

## Final Release Status (2026-10-15)

Both task-specific selectors completed training and their locked TEST/LIB
evaluations. See the [completed report](runs/R79_NSS_new_data_batch1024/REPORT.md),
[two-page distributions](runs/R79_NSS_new_data_batch1024/selection_distribution/selection_distributions.pdf)
and [same-pool R77B comparison](runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/nss_comparison.csv).
The primary native/FP64 TSP TEST Top1 is 44.3388960%; the original FP32-argmin
compatibility Top1 in the historical NSS summary is 43.9537869%, not the same
label definition. The following runtime/ETA notes are historical provenance.

The final runtime used one approved RTX4090 per selector on gpu05/job1475,
with grouped micro64, global1024 and GPU-resident TRAIN inputs. DataParallel was
rejected by the single-TRAIN-batch numerical gate; there are zero formal DP updates.
The two gpu03 RTX3090s are released to the separately owned R78 branches.

## Original Recipe and Overrides

The authoritative baseline is `code/NSS_retrain/config_TSP_ours.yml` and
`config_CVRP_ours.yml`, verified against their historical training configs.
The original source files are not edited; their SHA256 values are in protocol.json.

| Setting | Original NSS `ours` | R79 |
| --- | --- | --- |
| Model | Original `Selection_model`, hierarchical pooling | Unchanged |
| Embedding / blocks / layers per block | 128 / 2 / 2 | Unchanged |
| Heads / QKV / FF hidden | 8 / 16 / 512 | Unchanged |
| Normalization / downsampling | ReZero / 0.8 | Unchanged |
| Manual / neural-solver representative features | Both disabled | Unchanged |
| TSP / CVRP candidate outputs | 8 / 10 | Same columns, same order |
| Loss | Original `RankingLoss(K, K)`, full ranking | Unchanged |
| Optimizer | Adam, lr=0.0001, weight decay=0.000001 | Unchanged |
| Epochs / augmentation | 50 / original eight-fold XY transforms | Unchanged |
| Scheduler / AMP / gradient clipping / early stopping | None | None |
| Seed | 2024 for both `ours` selectors | **2**, user override |
| Logical training batch | 64 | **1024**, user override |
| VALID / evaluation batch | 8 | Unchanged |
| TRAIN / VALID / TEST rows per task | 10000 / 1000 / 1000 | **70114 / 3895 / 3895**, new data |
| LIB | Original TSPLIB 49 / CVRPLIB 100 | Same real instances and costs |

For completeness, the upstream `*_original.yml` defaults are TSP seed2024,
CVRP seed1234, batch64, with 7/5 outputs. R79 preserves the repository's already
established `ours` 8/10 pools, not those upstream 7/5 pools.

Data root:
`/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`.
LIB is read through the original `code/NSS_retrain/datasets` symlink to
`/public/home/shiys/0selection2/data/TSPLIB` and `CVRPLIB`. No R76 loader, model,
library substitute, newly generated TSP sequence, or newly solved cost is used.

## Checkpoint Rule and Necessary Fixes

Both models use the original trainer's strict minimum **mean top-1 gap** on VALID.
Only if the whole gap table is zero is raw mean top-1 cost used instead.
The new VALID gaps are nonzero retained-pool Oracle regret, so both models use gap.
The historical NSS data had zero gap compatibility labels and therefore used cost.
This is an input-label difference under the same original criterion, not an
unreported switch to cost-based selection. New gaps are not optimal-tour gaps.

Initial untrained VALID is diagnostic only. Each of the 50 trained epochs is a
candidate; ties keep the earlier checkpoint. Original trainer did not save the
first trained epoch even when it remained best. R79 fixes this persistence gap.
After both models finish, `final_test_lock.json` freezes their corresponding
VALID-best paths and hashes before new TEST or LIB input loading. Each is evaluated
once, with exclusive started/completed receipts and reusable saved predictions.
Original TEST is never loaded. TEST/LIB cannot cause training or checkpoint changes.

Each improving trained epoch already saves `best_val_predictions.npz` alongside
its checkpoint. It contains all 3895 VALID rows: `scores`, `pred`, native FP64
cost `winner`, FP64 per-method `costs`, original `pool`, `indices` and
`source_indices`. At final freeze, these existing files are referenced and hashed
in `final_test_lock.json` and `locked_train_val_handoff.json`; VALID is not inferred
again. The latter contains only TRAIN/VAL references and VALID-best metrics for
the two uniquely selected checkpoints. TEST/LIB results must not be joined into
this research handoff. Best epochs use zero-based indices in the JSON records.

The only model execution fix explicitly places the original pooling padding mask
on the embedding device instead of depending on global CUDA tensor defaults.
Model operations, parameters, state-dict keys, loss and optimizer are unchanged.
CPU parity tests confirm identical outputs and sample-weighted accumulation
gradients, including an uneven final micro-batch. Data augmentation is lazy but
uses the original eight transformations exactly. CPU shuffle uses an explicit
seed2 generator; GPU-global-default loader mechanics are not retained.

CPU preprocessing uses original `process_instance_CVRP`; instances are packed
once, with FP64 source costs/gaps retained. In the initial path, the original NSS collate function runs
in persistent CPU workers with pinned transfer. Micro64 accumulates 16 times per
full batch1024, with sample-count weighting; the last logical batch is 784.
Only OOM can reduce micro-batch size; it cannot reduce the logical batch.
The fresh-first-update smoke condition requires pinned CPU input only for the
`cpu_pinned` path; `gpu_resident` instead requires the assigned CUDA device and
records `pinned=false`. This narrow cold-start fix does not restart the formal run.
W&B is enabled in offline mode for project `selector`, entity
`yjkds-southern-university-of-science-technology`; local logs remain available.

## Launch and Progress

tmux session: `R79_NSS_new_data_batch1024`.
Run root: `/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/`.
Logs: `pipeline.log`, `TSP_train.log`, `CVRP_train.log`; per-model `status.json`,
`updates.jsonl`, `epochs.jsonl`, `smoke_update.json` and `best_validation.json`.

Current controller: `bash code/V4/run_v4_r79_nss.sh single4090`.
TSP UUID: `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be`.
CVRP UUID: `GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`.
Both independent workers share srun step1475.21 on gpu05. The approved step has
4 CPUs and 9GiB host memory, with exact per-worker CUDA_VISIBLE_DEVICES.
The existing tmux controller automatically freezes VALID-best and evaluates
TEST/LIB after the full budget. Do not issue another launch or resource probe.

The initial command below is historical launch provenance, not the current layout:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r79_nss.sh pipeline
```

The wrapper lock prevents duplicates. The initial launch used allocation1465 on
gpu03, never SSH: TSP UUID `GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50` and
CVRP UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`.
Those two cards are no longer occupied by NSS.

This cluster's Slurm20.11 did not overlap GPU GRES between two separate steps:
the first step inherited both cards and the second stayed pending. That partial
launch is preserved in `aborted_slurm_launch/`, not used as a checkpoint or result.
The historical initial launch used one `srun --jobid=1465 --overlap --exact --nodes=1
--nodelist=gpu03 --ntasks=1 --cpus-per-task=4 --mem=9G` step containing two
independent GPU-bound workers. Each worker has 2 torch threads and 2 CPU loaders;
all workers remain within the step's CPU cgroup and combined 9GiB host-memory cap.

Each model: 560912 augmented rows/epoch, 548 updates/epoch, **27400 updates total**.
Original batch64 with 10000 rows and augmentation: 1250 updates/epoch, 62500 total.
CPU caches took about 29 seconds. Both first formal updates passed real batch1024
shape, finite loss/gradients and pinning checks, with no OOM (micro64).
Early steady rates: TSP about 2.04 seconds/update, CVRP about 2.47 seconds/update.
Estimated complete runtime including validation: **TSP 16h, CVRP 19h** in parallel;
these are estimates, not measured completion times.

Training completion automatically triggers checkpoint freezing, TEST/LIB
evaluation and `REPORT.md`, `summary.csv`, `results.json`, `completion.json`.
Per-model figures include loss, VALID-best/TEST/LIB top1 versus each solver,
mean cost versus each solver, and arm distributions. Top1/2/3, native FP64 winner
metrics, TRAIN-fixed SBS, descriptive split-SBS, Oracle and all per-solver costs
are retained. Report costs are FP64; original ranking loss uses FP32.

Verification: nine focused synthetic CPU tests in `test_r79_nss.py` passed.
No Git commit and no changes to original NSS, shared tools or other experiments.

## Performance Update: GPU-Resident TRAIN Inputs

Eight-fold means data augmentation/expansion, **not eight-fold cross-validation**.
The model, original RankingLoss, Adam and 50 epochs are retained. Seed2 and
logical batch1024 are user overrides, not tuned settings or a new NSS recipe.
Each base instance appears as the original eight deterministic XY transformations
per epoch, with unchanged demands and labels. TRAIN has 70114 independent base
instances and 560912 augmented samples per epoch. Logical batch1024 counts
augmented samples, not 1024 guaranteed-distinct base instances. The original
augmented-index random permutation is preserved; transforms are not grouped into
one batch per base instance. The last batch is still 784, so the 50-epoch schedule
still has exactly 27400 formal optimizer updates per model.

Active TRAIN input path is now `gpu_resident`: packed base coordinates/demands
and FP32 training labels live on each model's assigned GPU. Batch lookup,
swap/reflection and zero padding are vectorized. No eightfold GPU copy is stored.
CPU tests verify every transform, variable-length padding and identical sampler
RNG across two epochs. One real full batch1024 was compared exactly to the
original CPU NSS collate result before benchmarking. VALID/evaluation retain
the original non-augmented batch8 path.

Actual normalization is **ReZero**, not BatchNorm or LayerNorm. There are no
cross-sample statistics and no dropout. Thus micro changes only affect FP32 GEMM
and gradient reduction order, not RNG-consuming augmentation/layers. Candidate
gradients differed from micro64 by only 1.4e-7..2.8e-7 in relative norm.

| Task | Micro | Seconds / logical1024 | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| TSP | **64 selected** | **1.979** | 3.56 | 3.69 |
| TSP | 128 | 2.038 | 6.94 | 7.08 |
| TSP | 192 | 2.137 | 10.32 | 10.63 |
| CVRP | **64 selected** | **2.342** | 4.47 | 4.61 |
| CVRP | 128 | 2.421 | 8.67 | 9.06 |
| CVRP | 192 | 2.632 | 12.87 | 13.20 |

Larger micros fit safely but were slower, so both retain micro64. This was a
bounded TRAIN-only execution comparison, not a model/loss/hyperparameter sweep.
Each trial used one global-mean update, restoring the exact same model, Adam
and Torch RNG state; probe updates never advanced formal training counters.
Details and per-UUID telemetry are in each model's `performance_probe.json`.

Both resumed from their epoch0 checkpoint at update548, preserving model, Adam,
best VALID state and shuffle RNG. The interrupted suffix (TSP179/CVRP75 updates)
was archived in `uncheckpointed_updates_after_548_*.jsonl`, not treated as part of
the retained trajectory. Config and `execution_history.jsonl` record checkpoint
hashes, augmented-sample units, input path and numerical protocol.

Post-upgrade steady rates are roughly TSP2.03s and CVRP2.38s per update, with
input preparation around 0.8ms and sampled utilization81%/89%. Estimated remaining
runtime is about 15h/18h, not a completion claim. Spare memory was not the main
compute bottleneck. No TEST data was used for this decision.

The pre-grouping resident process's lifetime allocator peaks included the micro192 probe.
Use the per-candidate table above for micro64's measured peak, and
`gpu_telemetry.json` for current NVML usage (not Torch reserved memory).
Future optimize launches reset allocator peaks after probes and additionally log
current allocated/reserved values; that process was not restarted merely
to reset diagnostic counters.

## TRAIN-Only Runtime Gate: True-N Grouping and Cropping

The `grouped` launch resumes the existing epoch checkpoint, not a new run. It
compares exactly one next TRAIN logical1024 batch per model at the intermediate
weights. Original-order micro64/full logical padding is the reference; the
candidate stably sorts those same entries by their true node counts and crops
each micro64 to its maximum true N. Original forward, masks, pooling, true scales,
cost labels, solver pools and RankingLoss are unchanged. Output rows are restored
to the original order for comparison. No Adam step is taken by this check; model,
Adam and RNG states are restored, including the unconsumed shuffle RNG.

Tolerances are fixed before GPU measurement: scores atol/rtol3e-5, exactly zero
argmax changes, weighted loss error <=3e-5*max(1,abs(reference_loss)), relative
gradient L2 error <=5e-4, and maximum gradient error <=5e-5+5e-4*max(abs(reference
gradient)). Gradients must be finite with identical missing-gradient parameters.
Each model independently enables grouping only if every condition passes;
otherwise it immediately continues the original padded micro64 path. Results,
source hashes and resume checkpoint/update are saved in `runtime_equivalence.json`
and `config.json`. This is not a model or micro-capacity search.

Formal batches keep identical augmented-index membership and ordering across
logical batches. Only execution order inside a logical batch changes. Every
micro loss is weighted by its sample count divided by the actual logical batch
count, and the entire logical batch advances Adam exactly once. Tail784 and the
27400-update/50-epoch/eight-augmentation budget remain intact. VALID batch8 and
the strict VALID-best rule are untouched; TEST/LIB are not opened by the gate.

Any interrupted uncheckpointed suffix is retained in its archive rather than
silently counted twice; `execution_history.jsonl` records the exact handoff.
Real post-gate throughput is measured from formal `updates.jsonl` records, not
from the prior attention/linear operation-count estimates or gate timing.

Both real GPU gates passed on 2026-10-15; no additional candidate was tried.

| Task | Max score error | Argmax changes | Weighted mean loss error | Gradient relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| TSP | 1.66893e-6 | 0 | 1.90735e-6 | 1.49972e-6 |
| CVRP | 2.38419e-6 | 0 | 9.53674e-7 | 1.44489e-6 |

TSP resumed update3836 (epoch6 complete), CVRP update3288 (epoch5 complete),
restoring model/Adam/shuffle state. TSP discarded no uncheckpointed updates;
CVRP's 85 uncheckpointed updates were archived and replayed from the checkpoint,
not added to the retained 27400-update budget. Checkpoint hashes are recorded in
the gate and execution history. This historical grouped launch used step1465.149
and the original two3090 UUIDs, before the later approved migration to gpu05.
The tmux session name remains `R79_NSS_new_data_batch1024`.

| Task | Prior 20 updates: seconds/1024 | Grouped 40 updates: seconds/1024 | Augmented samples/s | Measured speedup | Remaining TRAIN+VALID estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP | 2.0513 | 1.2654 | 809.2 | 1.62x | 8.3h |
| CVRP | 2.4139 | 1.4227 | 719.8 | 1.70x | 9.6h |

The grouped windows are TSP updates3863..3902 and CVRP3308..3347. Baseline
windows are TSP3816..3835 and CVRP3268..3287. These are actual formal updates,
not gate timing, a FLOP proxy, or an end-to-end completion claim. ETA includes
the measured preceding two VALID durations, excludes later TEST/LIB, and may
vary with input node counts and resource load. The fixed measurement evidence
is in `grouped_runtime_report.json`.

Post-gate allocator peaks (allocated/reserved): TSP3.55/3.82GiB,
CVRP4.46/4.60GiB, with no OOM. Sampled per-card utilization was85%/87% and
NVML usage4234/5038MiB. Peak memory is not much lower because the largest-N
micro remains; the gain comes from not doing that padding work for every micro.

## Historical Preparation: Local Two-GPU DataParallel

Status: CPU preparation complete; **not launched or GPU-tested**. Current single
GPU grouped training continues unchanged. Do not stop it or use any R77 card
until the user explicitly licenses migration after both R77 training and TEST
have finished and released their resources. Both `R79_DP_APPROVED=1` and
`R79_R77_RELEASED=1` are required by the shell and Python GPU entrypoints. They
are intentionally unset during preparation. No delayed/pending GPU step is queued.

The planned fixed resource layout is:

| Selector | Allocation/node | Exact visible UUIDs, primary first |
| --- | --- | --- |
| TSP | job1475/gpu05, two4090 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be, GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d |
| CVRP | job1465/gpu03, two3090 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d, GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 |

Each model's two replicas are local to one node; this is not cross-node DP.
Each step is `srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4
--mem=9G --gres=gpu:2`, with 2 torch threads and the existing 2 VALID workers.
The listed allocations must still be held when migration is explicitly approved.
Only these resource tuples are accepted; reallocated jobs require a new approved
mapping, not automatic use of another device.

One source/device check found no registered buffers or unregistered Tensor
attributes at initialization. ReZero scalars are registered Parameters. The
temporary `instance_feature` is overwritten by each replica's forward and is
not part of the saved state. `ns_feature=false` avoids the unused token-list
branch. The existing `device_padding_concate` reads only its input tensors and
does not access its bound `self`, so the shallow-copied method is replica-safe.
The original attention scale is a CPU scalar, already supported by the current
CUDA path; it is not a persistent device-specific tensor. No original source,
class layout, parameter or preprocessing change is needed.

DP uses gathered logits and the **original** RankingLoss, not averaged per-device
loss scalars. The same global1024 is sorted by true N; each forward takes at
most128 entries, cropped to that group's maximum true N, and DP scatters at
most64 entries to each GPU. The mean RankingLoss is weighted by that forward's
count divided by actual logical batch count. There are 8 backward calls for
1024 and 7 for tail784 (last forward16, 8/GPU), then exactly one original Adam
step. Both replicas' gradient contributions are summed into the primary model.
`micro_batch` remains the per-GPU64 setting; `forward_micro_batch` records128.

After the explicit license and an approved checkpoint handoff only, `dual`
performs one TRAIN logical-batch equivalence gate per selector: current single
GPU sorted/cropped micro64 versus two-GPU sorted/cropped forward128, using the
same intermediate model, labels, pool and cloned next-batch shuffle plan.
It checks restored-order scores/argmax, weighted mean loss and all master
gradients at the same fixed FP32 tolerances as the preceding gate. No optimizer
step, VAL/TEST timing, capacity sweep or second candidate is used. All visible
CUDA RNG states, model/Adam state and shuffle RNG are restored. A failed gate
leaves that selector on the original single-GPU cropped path and records the
failure in `parallel_equivalence.json`; no tolerance adjustment is attempted.

All checkpoints save `unwrap_model(model).state_dict()` with the original keys,
never the DP `module.` prefix. Adam state, best VALID state, epoch/update budget,
CPU shuffle and CUDA RNG state remain resumable. VALID and final TEST/LIB use
the unwrapped primary model and original batch8, so neither the selection rule
nor their corresponding frozen checkpoint changes. Formal speed and memory
records distinguish per-GPU64 from gathered128 and include both GPU peaks.

Prepared command, **not executed**; only after explicit release/license and
handoff, in the existing tmux session/run directory:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
env R79_DP_APPROVED=1 R79_R77_RELEASED=1 bash code/V4/run_v4_r79_nss.sh dual
```

`dual_resume` retains a saved passing DP configuration (or recorded failed-gate
single-GPU fallback) without rerunning the one-batch gate. The wrapper lock
prevents launching alongside the existing NSS pipeline. Both dual modes retain
automatic completion -> VALID-best freeze -> unique TEST/LIB -> report.

Seven small CPU tests pass, including mean weighting on an uneven toy tail,
one Adam step, original-key strict checkpoint loading, and the two approval
guards. CPU DataParallel is an identity wrapper: these tests do **not** claim
real two-GPU numerical parity or throughput. Parameter counts remain
1219478/1219864 (about4.9MB FP32 each). Replication/gradient traffic is small, but
DP replication, gather and Python per-instance pooling may limit speedup.
No real DP speed estimate is presented before approved GPU execution.

## Migration Outcome: DP Rejected, Two Single4090 Selectors

The user explicitly released both R77 training+TEST resources and licensed the
migration. Real DP testing used the same TRAIN batch and epoch checkpoint on
the planned pairs. A PyTorch DP gradient detail required exact-zero gradients
on the four unused `modulate` parameters to be reset to original `None`, so Adam
would still skip them. No formal DP optimizer update was performed.

The subsequent numerical gate **failed materially**, not at FP32 tolerance:
TSP/CVRP max score errors were1255704.9/11373.3, argmax changed393/984 rows, and
gradient relative errors were972629.6/112868.6. The cause is the original ReZero
forward's `isinstance(self.norm, nn.Parameter)` dispatch: PyTorch DP replicas
expose their parameter copies as plain Tensors, selecting the unscaled residual
branch. A focused CPU test reproduces this dispatch failure. The original NSS
forward is deliberately not rewritten. DP is rejected/disabled for this run,
with evidence in `parallel_equivalence.json` and `dual_migration_outcome.json`.

Approved fallback: TSP uses only GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be;
CVRP uses only GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d, each as a separate original
model on gpu05/job1475. Both3090 are idle for R78, not used by NSS. The one shared
step has 4 CPUs/9GiB total and exact per-process UUIDs. No other user job or source
was changed, stopped or reverted.

Models resume the same run from TSP8220/CVRP7124, with original model/Adam state,
best VALID state and shuffle RNG. The handoff control had two disclosed failures:
the first watcher matched a forked DataLoader command line, and terminating the
srun wrapper subsequently caused immediate Slurm cancellation rather than a
Python graceful pause. Unsaved suffixes CVRP85 (legacy handoff), then TSP92/CVRP86
(single-GPU continuation after failed DP gate) are archived and replayed, not
counted twice in the retained27400-update trajectory. Their complete checkpoints
and VALID-best snapshots remain preserved in `migration_before_dual_*.pt` and
`migration_best_before_dual_*.pt`. `migration_handoff.json` corrects the watcher
claim; no claim of lossless mid-epoch handoff is made for those failed controls.

The resident sampler now materializes each canonical CPU RandomSampler epoch
once and supports saved plan/cursor/partial loss totals. Augmented-index sequence
and epoch-end sampler RNG are identical to the preceding loader; the unused
sampler tail draw is consumed at epoch start rather than at epoch end. Dedicated
CPU tests cover two-epoch identity and mid-epoch resume. New Python workers can
save at a logical-batch boundary after a direct SIGTERM; a controller must verify
that checkpoint before cancelling its wrapper. No further migration is planned.

The running single4090 path keeps original forward/loss/Adam, seed2, global1024,
micro64, tail784 and50 epochs x8 augmentations. Initial40-update measured means
were TSP1.345s and CVRP1.454s, estimated7.1/8.2h remaining; this is close to the
two single3090 rates, not a claim of substantial hardware acceleration. Sampled
4090 NVML usage was4242/5266MiB at60%/60% utilization. The two3090 were explicitly
queried within allocation1465 at2MiB/0% each. Runtime throughput may vary.

Active tmux remains `R79_NSS_new_data_batch1024`, initially step1475.21. Logs are
still `TSP_train.log`, `CVRP_train.log`, `pipeline.log`. The `single4090` wrapper
automatically finishes the full budget, freezes both strict VALID-gap best
checkpoints, evaluates TEST/LIB once per model and generates the report. TEST
and LIB were not opened for migration or runtime selection.
