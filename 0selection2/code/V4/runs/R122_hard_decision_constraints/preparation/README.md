# R122 D2 Runner Handoff

Only CPU preparation and CPU tests are authorized/executed here. No formal model
has been trained. No solver, GPU, Git, TEST inference, or LIB evaluation is run.
Scientific source: `docs/research/post_r119_global_reassessment_20261028/HANDOFF.md`, D2.

## Entry Points

Run from `/public/home/shiys/0selection2`. Every shell command activates easynco.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment describe --device cpu
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest -v code.V4.test_r122_experiment
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment prepare --device cpu
```

The following are unexecuted launcher entry points, not execution authorization.
Parent owns GPU assignment, tmux, queueing, and all reporting. Use the parent-assigned
visible `cuda:0` and the matching `--gpu-type` (`4090` shown, `3090` also supported).

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment train --arm model --device cuda:0 --gpu-type 4090
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment train --arm model --device cuda:0 --gpu-type 4090 --resume
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment test --arm model --device cuda:0 --gpu-type 4090
```

`train` stops after its unique VAL lock and selected TRAIN capture. It does not
auto-start TEST. `test` requires all 14,400 updates, exactly 800 per task, the
immutable single-checkpoint/model-state lock, unchanged source/preparation, and
the existing complete R84 full18 TEST cache. No cache build/copy fallback exists.

Default run: `/public/home/shiys/0selection2/code/V4/runs/R122_hard_decision_constraints/model`.
Preparation: the sibling `preparation` directory. Original R84 preparation is
read-only at `runs/R84_new90k_attention/prepared`. Its current persistent
`cache_source` is used because the historical `/tmp/shiys_R84_new90k_cache` is gone.

## Frozen Semantics

- Unchanged R45A/R84A ScoreDifferenceSelector, complete full18 pool, original
  scratch seed2 tensors and TRAIN-only geometry/SBS. No trained initialization.
- Natural TRAIN 81,000 per task; VAL/TEST 4,500 each; global/logical batch 1024.
  Exactly 14,400 successful updates; warmup 1080; VAL every 720. AdamW peak
  1e-4, cosine endpoint 1e-6, WD 1e-4, dropout .1, clip 1; one forward per
  microbatch, no R-Drop. Fixed caps 256 TSP/CVRP, 1024 otherwise.
- Native FP64 costs and int64 winners are restored after base cache gathering.
  Fixed membership is `(problem, pool[native_winner])` in the historical 36-item
  file, never a dynamic zero set or global-ID alias. Query inputs are allowlisted.
- TRAIN constraints are equal-task mean hard actual regret and unconditional
  fixed36 missed frequency. Row-weighted detached hard means are committed once,
  only after all microbatches and a successful AMP optimizer step. Retries replay
  identical rows/RNG without controller updates. Full18 cycles get equal task
  weight; multipliers/EMA/partial task sums/counts/counters are checkpointed.
- Separate saved R84A VAL thresholds select checkpoints only, never dual updates.
  Feasible candidates maximize full18 native Top1, then minimize actual regret,
  then earliest update. If none feasible, minimize summed positive violations
  divided by each VAL reference (floor 1e-12), then the same tie breaks.
- Inference remains original maximin with stable ascending global-ID score ties.
  Shared evaluation metrics stay unchanged: native Top1, mean cost, TRAIN-fitted
  SBS cost, Oracle cost, vs_SBS, vs_Oracle, actual regret, equal-task ALL.
  Optimization `ce` is the classification upper bound in bits; evaluation `ce`
  remains original R84 maximin-logit CE in nats. These are not interchangeable.

## CPU Reference Fit

Baseline R84A selected update 37440, checkpoint SHA256
`23fb9a5cf621045138fcdcfe5e6613a1bfd061c8d4dfeca108db3548dbf8031f`.
The capture model-state hash is matched to the actual checkpoint. TRAIN/VAL
shards are checked against native cache costs/winners, pools, row/source IDs and
saved original metrics. VAL archive hash is also bound to the fixed target file.

| Role | Hard actual regret (fraction) | Fixed36 missed frequency (fraction) |
| --- | ---: | ---: |
| TRAIN dual budgets | 0.003952068677254486 | 0.05691495198902606 |
| VAL selection-only thresholds | 0.004034942948096274 | 0.05696296296296296 |

These are baseline reference statistics, not R122 outcomes. `constraint_reference.json`
contains per-task counts, reference identities and file hashes; `prepared.json`
locks it. Any source/reference change is rejected, not silently re-fitted mid-run.

## LIB Factory And Remaining Work

Factory: `code.V4.r122_experiment.load_checkpoint(path, device='cpu')`, returning
`(selector, full_checkpoint_state)`. Aliases `load_model` and `from_checkpoint`
are provided. It restores the unchanged model/buffers and validates controller,
protocol, update counts and source cursors; it needs no R84 feature cache to load.
LIB must use the exact selected `model/best.pt`, hash/lock and `.eval()` inference.

Remaining: parent GPU execution/monitoring, fixed-budget VAL selection, explicit
locked fullTEST, LIB via this factory, requested distribution/metrics PDFs and
final analysis. TEST is development-exposed, not a new blind confirmation.
