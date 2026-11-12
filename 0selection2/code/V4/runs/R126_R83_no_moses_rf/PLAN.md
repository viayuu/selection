# R126: R83 model-only selector, no MoSES_RF

## Scope

One fresh seed2 full18 experiment, concurrent with but independent of R125.
Use the original frozen new90k B-filtered, single-reld_mt dataset, not R125's
no-BQ/LEHD scenario. Preserve every TRAIN81000/VAL4500/TEST4500 row per problem.
Remove only MoSES_RF wherever legal. Retain BQ and LEHD in TSP/CVRP. Keep the
existing B8 exclusions and the MOEL-only reld_mt alias. Never overwrite old data,
cost columns, labels or running R125 files. No solver runs or new Embedding calls.

Recompute native FP64 winner, rankings, Oracle, TRAIN-fitted SBS and conditional
baselines for the changed pools. Former MoSES_RF winner rows are kept, not filtered.
The changed tasks are CVRP plus the seven non-B MVRP variants. TSP, ATSP and the
eight B variants retain their existing candidate pools and labels.

## Architecture and Training

Use the actual R83 model-only source architecture: only encoder/decision nn.Module
code views; no inference/config/train/environment code. Frozen real vectors are
reused from R83/model_only_vectors.pt. Shared role projections and fusion plus
trainable method ID produce128-d method tokens. GLOP's insertion deployment has no
neural model, so it retains its explicit ID-only source exception.

Retain the original4-layer128-d instance encoder, two synchronous dual-stream
layers, QKV decoder, score MLP, method utility differences and maximin inference.
From-scratch common seed2 initialization, not an R83/R125 trained checkpoint.
Use unchanged R84 TRAIN-only input geometry statistics and node-local input cache.

Logical batch1024, microbatch256 for TSP/CVRP and1024 for other tasks, eval128.
AdamW peak1e-4/end1e-6 cosine, warmup4320 successful updates, weight decay1e-4,
dropout.1, global gradient clip1. Original CE.35/comparison.35/risk.02/RDrop ramp.35.
Fixed57,600 successful updates, full VAL every1440, full TRAIN diagnostics every7200.
Keep natural shuffled sample streams and carry tails across cycles.

Select strict minimum full18 VAL macro_vs_sbs_pct against this scenario's
TRAIN-fitted SBS; the first exact tie wins. Freeze one checkpoint and inference
rule before TEST/LIB. No confidence calibration, method quotas or test retuning.

## Execution and Delivery

Use the other idle allocated4090 UUID, disjoint CPU affinity and the original
per-UUID mutex. Do not restart, pause or modify R125. W&B is enabled, with an
explicit offline route if no credentials are available. Persistent tmux owns the
prepare -> train/lockedTEST -> timing -> LIB -> CPU PDF -> publish sequence.

Deliver VAL/TEST full18 per-task and macro metrics, TRAIN-fitted SBS and Oracle,
selection versus true-winner paired charts, TSP/CVRP TEST/LIB/NSS comparison,
learning curves and complete-model timing. Existing NSS scores may be restricted
to the new candidate pool as a fixed-model diagnostic, not as a retrained model.

R83 historical results used older data. R125 uses both a different candidate pool
and four source roles; neither comparison is a matched single-factor ablation.
Deleting a method changes Oracle and label difficulty. Interpret Top1 together
with same-pool baselines and actual cost; do not claim a causal imbalance fix.
