# R64 adopted larger real-batch restart

The complete adopted plan is
`runs/R63_recurrent_workspaces/reviews/R63_closure_R64_plan.response.md`.
Its architecture/objective/PCGrad specifications remain in force. The newest USER
explicitly stopped the batch42 run and requires BOTH arms to restart fresh with a
measured larger real batch. The adopted amendment is
`.aris/traces/R64_large_batch_plan.response.md`: Astra/max text/context planning,
not a direct local-code audit. The immutable interrupted archive is
`runs/R64_pcgrad_full18_aborted_batch42_20261012T140928`; A completed14 epochs plus
partial15/21383updates, B never started. It is not a completed paired result and
none of its trained states, optimizer, schedule cursor or dropout progress is reused.
R63 is fully complete/pushed15363d0ec; its TEST and RESULTS are not tuning inputs.
Legacy B semantics remain unresolved and R58 is deferred.

Both arms use the fresh seed2 R45A handcrafted34+ID dual-stream model, four d128
instance layers, two joint layers, four heads, FFN512 and head256. The readout is
direct `encode_pairs + score_head`. Original TRAIN fits local geometry; task
conditions and directed ATSP matrices remain intact. Labels never enter forward.
The objective is nativeCE plus existing symmetric R-Drop, ramping to .35 over the
first three complete epochs' successful updates. No objective or architecture changes.

The existing `paired_schedule` really is cycle-based for these equal-sized tasks.
R64 verifies every18-task cycle and explicitly saves its six triples. For locked
per-task batch b, C=ceil(10000/b), updates/epoch=6*C, tail/task=10000-b*(C-1).
Every task uses its next b shuffled rows, retaining the complete real tail without
padding/duplication. Effective batch is3*b, final effective batch3*tail. There is
no accumulation or microbatching. Every epoch presents all180000 original rows.
Maximum per arm is40 epochs/7.2M presentations, with40*6*C optimizer updates.
The lower update budget versus aborted batch42/57360updates is explicitly adopted.
Three task gradients are computed at the same parameter state. A averages them;
B projects only the seven adopted shared dense prefixes against ORIGINAL other
gradients, using independent shuffled orders from RNG6402. Other parameters are
averaged. The actual matched names are saved in protocol/initialization receipts.

AMP loss scaling is explicit: scaled parameter gradients are divided once into
FP32 task buffers, then reduced, globally clipped once at1, and passed directly
to AdamW. `GradScaler.unscale_`/`step` are deliberately not called on already
unscaled buffers. Its growth interval/factors are retained with a saved explicit
successful-growth counter. Overflow retries the whole triple with restored
Python/NumPy/CPU/CUDA/projection RNG; no samples, optimizer states or successful
counts advance. Both arms seed every update's dropout stream identically,
independent of retry counts and W&B activity.

A controls the actual LR/horizon. Warmup is3*updates_per_epoch successful steps to1e-4; AdamW
WD1e-4, betas(.9,.999), eps1e-8. The existing controller observes negative native
validation Top1 in percentage points, min_delta.001pp, halve patience3, stop
patience8, min15/max40. `replay_schedule.json` stores every LR actually used;
B must reproduce that exact list and horizon. Checkpoints compare integer
validation correct counts out of18000; exact ties retain the earliest epoch.
Cost does not break ties.

## One Authorized TRAIN-Only Batch Sizing Pass

The same assigned-GPU tmux process performs exactly one bounded sizing pass,
then immediately starts formal A and B. At most three distinct per-task sizes:
start256; if eligible,512 then1024 if512 passes, otherwise384. If256 fails,128,
then192 only if128 passes. A genuine CUDA OOM is recorded as capacity evidence,
separately from ordinary whole-triple AMP overflow retries. If neither256 nor128
meets capacity/headroom, report the blocker with no further search.

For every size and each arm: fresh model/AdamW, two complete warmups and three
timed complete `update_triple` calls, with the three distinct largest padded-node
TRAIN tasks and their largest rows. Full resident original input/geometry caches
remain allocated. Measure max allocated/reserved memory, physical mem_get_info
sampled every20ms, and a conservative physical peak bound including external
baseline/context plus allocator peak reservations. Require at least15% physical
headroom in BOTH arms. Choose highest measured PCGrad presentations/second;
predeclared timing ties within3% favor the smaller batch. Memory fill is not an
accuracy criterion. Validation is resident but never forwarded or used for sizing.

Discard every calibration model, optimizer/scaler/projection state and RNG progress.
Record actual discarded updates, presentations, OOMs, retries and wall separately.
Reset seed2 and initialize both formal models fresh and identically. Keep LR1e-4:
no batch-based LR scaling. PLAN, batch decision, formal protocol, schedule and
source snapshot are written only AFTER sizing chooses b. Original inputs/labels
and full18/fullpools/native winners remain unchanged. Real throughput and the
changed optimizer-update budget, alongside matched A/B sample coverage, are explicit.

Validation covers the full original18 tasks/full legal pools every epoch; complete
eval-mode TRAIN runs every5 epochs, at termination, and for the selected model.
Predictions, full18/family/task metrics, gradient conflicts/norms/corrections,
clipping, retries, memory, throughput and actual execution are recorded. Final
selected validation measures singleton/batched selection in one normal evaluation,
with every query once. Its metrics are recorded separately from the epoch counts
used for checkpoint selection; batch-size numerical differences never reselect.

## Commands

Every invocation sources conda.sh and activates easynco, including tmux's shell.
The launcher uses the existing1465 allocation and gpu03/index1 only. It queries
only UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` at launch, waiting if occupied.
It does not modify clients, queues, allocations, MPS or environments. W&B is
offline, project selector, entity yjkds-southern-university-of-science-technology.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash code/V4/run_v4_r64.sh tests
tmux new-session -d -s r64_train 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && R64_PARENT_GPU_BATON=pcgrad-full18 bash code/V4/run_v4_r64.sh restart-large-batch >> code/V4/runs/R64_pcgrad_full18/formal_launch.log 2>&1'
```

`last.pt` commits the first update, every60 successful updates, and epoch
boundaries. It includes model/optimizer/scaler/growth state, sample/update counts,
partial epoch aggregates, controller, exact LR trace and all RNG states. Resume
replays only work after the last committed checkpoint. Append-only update/launch
records preserve the actual extra executed work if a process dies between commits;
logical schedule counts and actual execution counts are reported separately.
After a later interruption, `train --resume` reads the NEW root's locked batch;
it does not repeat sizing. Never point resume at the aborted batch42 archive.

There is no automatic TEST, analysis, git add/commit/push or reviewer gate. The one
explicitly authorized bounded sizing pass is the only calibration before training.
Normal training writes curve data to `curves/A_metrics.jsonl` and `B_metrics.jsonl`.
Explicit later reporting produces plots and tables:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash code/V4/run_v4_r64.sh analyze
```

## Later Parent Freeze And TEST

Training publishes `validation_candidates.json` with both unique selected bests
and the proposed official arm (integer validation correct counts, tie A). The
parent must freeze the models, inference rule and official arm, then authorize
ONE full original TEST for each. Training does not create this authorization.
The future `final_test_lock.json` schema is:

```json
{
  "authorized": true,
  "official_arm": "A",
  "inference_rule": "COPY THE EXACT validation_candidates.json inference_rule",
  "models": {"A": "COPY models.A OBJECT", "B": "COPY models.B OBJECT"}
}
```

Only afterward, `R64_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r64.sh test`
is accepted. The exclusive `final_test_started.json` receipt prevents automatic
repetition after any start. TEST never changes checkpoints, arms or inference.
R60B/R45A official TEST references must be reused, never re-evaluated. For later
analysis, `--reference-tests FILE.json` optionally maps their names to existing
official TEST JSON outputs. Historical TEST exposure must be disclosed in that
report; no new TEST evidence informed this R64 mechanism.
