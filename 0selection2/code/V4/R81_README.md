# R81B: Requested_R80B_cross_instance_auc

R80/B_batch512 is an existing, unrelated running arm. The only new group is
`B_cross_instance_rank_seed2` under `runs/R81_method_conditioned_auc`. There is
no R81A training, launcher, allocator, solver run, or embedding API call here.

## Queueworker CLI contract

From `/public/home/shiys/0selection2`, after `conda activate easynco`:

```bash
python -m code.V4.r81_experiment prepare --device cpu
python -m code.V4.r81_experiment describe --device cpu
python -m code.V4.r81_experiment train --group B_cross_instance_rank_seed2 --cache-root /tmp/shiys_R76_R45A_single_reld_mt_cache --device cuda:0 --gpu-type 4090 --wandb-mode offline --resume
python -m code.V4.r81_experiment test --group B_cross_instance_rank_seed2 --cache-root /tmp/shiys_R76_R45A_single_reld_mt_cache --device cuda:0
python -m code.V4.r81_reporting --group B_cross_instance_rank_seed2 --device cpu
```

These are interface commitments, not evidence that training has run. GPU binding
and queue ownership belong to the queueworker. `train` finishes the unique VAL
lock and then full TEST; the explicit `test` command safely resumes/reuses that
same lock. Reporting performs same-checkpoint LIB inference and saved-prediction
reporting, never training. `--root`, `--prepared-root` (R77), `--cache-root`,
`--data-root`, `--state-hash`, `--wandb-mode`, `--large-micro-cap` and
`--eval-batch-size` are runner options. `--arm` aliases `--group` for automation.

The root defaults to the absolute path
`/public/home/shiys/0selection2/code/V4/runs/R81_method_conditioned_auc`.
Use `--gpu-type 3090` instead if allocated a 3090; do not change the logical
batch or budget. The queueworker owns device binding and may make a smaller
precommitted physical micro cap with `--large-micro-cap`. This affects protocol
identity and must be fixed before the first update, not tuned on TEST/LIB.

The local-cache path above is the existing R80 warmed cache location inside
the allocated node/container. Admin01 need not have that path. Preparation
scanned the immutable SOURCE cache at
`/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt/cache`.
`prepared/prepared.json:cache_root` records that provenance; it is NOT a runtime
location restriction. `--cache-root` admits an existing local copy and performs
`open_cache` metadata/pool/rows checks plus exact original FP64 cost/winner
array-hash checks before the first optimization. No copying or rebuilding.

Queue dry-run, without requiring a GPU or a local cache on admin01:

```bash
python -m code.V4.r81_experiment describe --group B_cross_instance_rank_seed2 --cache-root /tmp/shiys_R76_R45A_single_reld_mt_cache --gpu-type 4090 --wandb-mode offline --device cpu
python -m code.V4.r81_reporting --help
```

The describe output includes the exact protocol state hash. Use that same GPU
type, micro cap, cache path and W&B mode for `train --state-hash HASH`. TEST uses
the saved protocol, so no `--gpu-type` or W&B override is needed there.

## Lossworker Relay

No send_input tool is exposed to this worker; the README and prepared receipt
are the relay to the main thread and Hooke. Integration follows the final
`R81_INTERFACE.md`: `CachedAUCObjective` and `utility_update`, with P/N bool
masks passed separately after removing their private keys from the batch.
Both the cache and replay forwards stay input-only. The complete original R77
CE/comparison/risk/RDrop loss is preserved. Lambda is
`.35 * min(1, successful_update / 3726)` on the successful-update axis.
Lambda=0 is a CPU numerical control, not a new A run.

## Locked Budget And Selection

Fresh identical R77 seed2 initialization; no trained weights. Full 18 problems,
70114 TRAIN rows/problem, ID+handcrafted34, dropout .1, logical batch 1024,
49680 successful updates, 3726 warmup, AdamW LR 1e-4 cosine to 1e-6, WD 1e-4,
clip 1. Full VAL every 1242 updates; full TRAIN evaluation every five VAL points
plus the selected checkpoint. Strict minimum full VAL macro_vs_sbs_pct against
FULL TRAIN-fitted SBS selects a unique checkpoint, first wins ties. Inference
remains maximin, stable ascending global-ID tie break. No TEST/LIB tuning.

P means a method is the unique original FP64 minimum. N means its original
FP64 cost is strictly greater than the per-row minimum. Tied-minimum methods
are neither. Eligible methods have both P and N in the entire TRAIN population,
fixed per problem. Batch FP32 costs are never upcast to invent these masks.
Original data, costs, winner labels and frozen caches remain untouched.

Historical R77B VAL zero/near-zero method lists are diagnostic only, not training
inputs, weights or sampling controls. Near-zero uses the existing <=0.1% rule.
R77B is historical reference only. Matched R81A has not run: no strict A/B causal
claim is supported. Prior TEST-informed R&D means TEST is not untouched blind.

## Delivery

Status/checkpoints/resume follow the R80 file conventions within the new group.
Full VAL/TRAIN-eval/TEST prediction archives additionally contain raw utility.
Postprocessing emits complete metric PDF, method-distribution PDF, per-method
AUC/TPR/score-quantile and rescue diagnostics, plus machine-readable CSVs and
the NSS/current four-dataset table. Missing NSS entries stay explicitly missing.

`metrics.csv` includes optimization-prefixed AUC/base components;
`optimization.csv` includes base_loss, auc_loss, auxiliary_loss, auc_weight,
lambda_auc, LR, pre-clip grad_norm and successful updates. Per-method batch
participation is in `method_participation.json` and checkpoints. The method
AUC CSVs include competitive, raw utility and maximin scores with exact ties
and P/N applicability; singleton/no-P/no-N results are explicitly unavailable.
Zero/near-zero CSVs include precision, recall, true rescue count, false-pick
excess cost, normalized false-pick actual regret and same-instance both-wrong
net savings. Totals use set unions rather than summing overlapping rows.

## Preparation And CPU Verification

Preparation ran on admin01, SOURCE cache only: 1262052 TRAIN rows and
49921168 FP64 cost bytes; 86 of 89 problem/method pairs eligible. Frozen R77B
VAL zero support is exactly 2648 using the specified rule. Original data/cache
and the active four R80 groups were not edited. No GPU job was started.
Preparation receipt: `prepared/prepared.json`, SHA256
`a07eee8a4131504173c7a52ccc713988fd22f5d4dd80c1697dc9efbd005dd700`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest code.V4.test_r81_auc code.V4.test_r81_experiment -v
```

Core witnesses include lambda0 all-parameter gradient/RNG equivalence,
positive-lambda true1024 versus 4x256 retained-graph equivalence, and AMP retry.
Runner witnesses include targets, local-cache override, extra utility archives,
optimization CSVs, a tiny CPU scratch-to-lock-to-TEST/reentry workflow, and a
full 18-problem synthetic PDF report. Synthetic results exist only in temporary
directories and are never presented as real experiment results. Queue readiness
is recorded in `prepared/code_ready.json`, never an R80 launch_ready file.
