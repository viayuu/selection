# R123 runnable handoff

Status: CPU implementation only, not trained. The scientific mechanism is the
explicit astra_followup.md Brier multi-label correctness hard router, not the
superseded cost-weighted proposal. Existing immutable meta_data artifacts are
unchanged. GPU commands below are NOT executed and must wait for the parent's
R120 final TEST/LIB/report release.

## Completed minimum CPU checks

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r123_model code.V4.test_r123_experiment -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.r123_experiment describe --write-protocol
```

The minimal CPU tests passed. An earlier actual R84-input CPU spine loaded
the real three fixed historical models and a fresh 4x128 gate checkpoint via
load_checkpoint. On two actual certified TSP rows it produced [2,8] final
scores, exact cache/live equality on this CPU batch, nonzero new encoder
gradient, and original float64 evaluation costs. No optimizer update.
See learner_cpu_spine.json, which pins that check's exact earlier source
hashes. Per latest request it was not rerun after cache/cap consolidation;
final changes use only small CPU unit fixtures. This is not a bitwise guarantee across GPU types
or batch-shape kernels and is not a trained-model result.

learner_smoke_bank/train/TSP is intentionally only two rows and incomplete;
formal training rejects it. learner_cpu_smoke_checkpoint.pt is an untrained
zero-update fixture, NOT best.pt, a final lock or a historical expert.

## Deferred full bank prediction cache

Run from /public/home/shiys/0selection2. Bind an approved free GPU only AFTER
the parent's release. Ordinal 0 below is an explicit launcher choice, not a
claim that the device is currently free. These are full18 predictions, not
solver invocations or fine-tuning of any historical expert.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_prediction_cache --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --split train val --shard-index 0 --shards 2 --batch-size 128 --device cuda:0
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_prediction_cache --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --split train val --shard-index 1 --shards 2 --batch-size 128 --device cuda:0
```

Use tmux for these long commands. Parent controls physical GPU binding. The
two shards have disjoint problem sets, each processes TRAIN+VAL and loads one
expert at a time. Per problem/expert NPZ completion is atomic and locked;
completed archives are reused on restart. No expert sweep or alternate bank.
Default output is R123/prediction_cache. Learner --bank-root denotes the R123
root (or its prediction_cache directory); R124 reads the SAME R77B archives,
never duplicates them. See the frozen learner_cache_api.md.
--input-cache-root remaps the existing R84 input-cache tree read-only;
no input/features/geometry regeneration is performed here.

TRAIN caches contain exactly 10,886 certified current TRAIN81k positions per
task, totaling 195,948. VAL caches contain all 4,500 per task. Each records
[N,3,K] FP32 scores, exact current row/source references, qualified source IDs,
legal candidate names/global IDs, checkpoint/source hashes and precision.

## One Main Training Run

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_experiment train --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe/preparation/learner_run --bank-root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --device cuda:0 --gpu-type 4090 --predict-batch 128
```

Logical batch1024, seed2, 7200 successful updates, warmup540, eval720;
AdamW peak1e-4 -> 1e-6 cosine, dropout0.1, WD1e-4, clip norm1. For an explicit
3090 launch use --gpu-type 3090 (TSP/CVRP128, others512); 4090 uses256/1024.
Only an actual CUDA OOM permits one halving retry, with same batch/RNG and no
optimizer advance; a second OOM fails rather than silently shrinking further.
For an explicit
restart append --resume; source/model/protocol must match. Train uses only
metaTRAIN and all frozen expert predictions solely to construct supervision.
Every correct expert is target1 and every incorrect expert target0, including
multiple-correct and all-wrong rows. Brier updates the full new encoder/query
path. No cost/risk-weighted training loss and no load balancing quota.

W&B logs to selector / yjkds-southern-university-of-science-technology. Without
WANDB_API_KEY the default is offline logging; WANDB_MODE explicitly overrides.
Evaluation reports native Top1, FP64 raw cost/SBS/Oracle/ratios/actual regret
with historical full18 equal-task semantics. Best selection is full18 VAL
actual regret, then highest native Top1, then earliest update. One checkpoint
is used across every task. Runtime defaults beneath preparation/learner_run.

## Postlock TEST And LIB

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_experiment cache-bank --split test --device cuda:0 --predict-batch 128
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_experiment test --device cuda:0 --predict-batch 128
```

Both commands reject prelock execution. The input-side TEST cache must already
exist in the parent's R84 compatible cache; missing host arrays are not rebuilt
by this module. The R123 lock binds completed7200, unchanged selected best.pt,
protocol and inference. No TEST/VAL relabeling, test-driven tuning, per-task
checkpoint assembly or automatic TEST run at training completion.

LIB factory: code.V4.r123_experiment.load_checkpoint(path, device='cpu') returns
(HistoricalModelRouter, state). With an ordinary labels-free R84-compatible
query batch, model(batch)['logits'] computes the raw-input gate FIRST, groups
queries by selected expert, then runs only that expert for each query. It uses
each selected expert's original complete-pool scores, no cross-expert mixing.
pool_ids must describe the exact current legal deployment set; optional
solver_mask is inferred as all legal if omitted. Bank weights/normalizers are
frozen and force eval even if the outer router enters train mode.
Full TEST prediction caching is a reporting acceleration with explicit all-bank
precomputation cost, not the claimed sparse deployment execution path.

## Boundaries

The common-unseen certificate is source-record-level, NOT an exhaustive
content/base-copy certificate. Cheap metadata headers for all 18 tasks expose
only source_index/scale/coord_dist/distribution/capacity_type, no related/base
copy identity. The remaining generation cohort is not assumed representative.
Historical held-outs were development-exposed, not fresh blind sets.
No model accuracy gain, complementary-error gain or efficiency claim is made.
R122 objective/runner, R121, R120 and the original meta preparation were not
modified. No GPU, solver, formal training or Git operation was performed.
