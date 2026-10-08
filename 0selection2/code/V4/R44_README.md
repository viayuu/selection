# R44 Conditional Node Adapter

The three arms continue the same validation-selected R43A checkpoint. They
change only a residual immediately after `InstanceEncoder(node_only=True)`
and before the first joint node/solver block. The original model weights,
solver features, candidates, geometry, losses and decision rule are retained.

## Paper Mechanism

[PEPNet (KDD 2023)](https://arxiv.org/html/2302.01115v3), Section 2.2.1,
Equations 2-3: a two-layer gate, with a bounded `2 * sigmoid` output.
Sections 2.2.2 and 2.2.3 use such gates to multiply embedding/hidden channels.

R44 borrows this mechanism, not the entire EPNet/PPNet architecture. GELU,
metadata-only conditioning, a bottleneck residual, and zero-initialized Up
are project adaptations. No embedding/content stop-gradient is used.

## Arms and Budget

- C0: no adapter, original R43A continuation.
- C1: shared non-affine LayerNorm, bias-free Down/Up, GELU; rank39, 9,984 parameters.
- C2: rank32 residual, 17-field metadata gate; 9,824 parameters.

C2 reads the existing 16 `problem_desc` fields, in registry order, and the
train-normalized logarithm of real node count. CVRP/MVRP real nodes include
the depot. It does not read solver identities/pools, costs, labels, predicted
scores, or pooled node features. All nodes within one instance share the gate,
but their residuals depend on their individual encoded representations.

Each arm uses 14,220 successful updates (10 full data epochs), batch128,
fresh AdamW, all original parameters trainable, peak LR1e-5 for original
parameters and LR1e-4 for adapter parameters. A 711-update warmup is followed
by fixed cosine decay to 10% of each peak. The original R-Drop ramp to0.35
over three data epochs is preserved. Weight decay1e-4, dropout0.1, clip1.0,
FP16 AMP/SDPA, natural sampling, and tail batches are unchanged.

Stage A runs C0/C1/C2 with continuation seed2. It passes only when C2's
strictly best validation macro_vs_sbs_pct beats both controls by at least
0.001 percentage points and the last five common nonzero validation points
also favor C2. This threshold is an operational screen, not significance.
On failure: no test, extra seeds or automatic tuning. On success: repeat the
same three continuations with seeds17/42, lock all best checkpoints, then
test each once. These are continuations of one trained model, not independent
from-scratch replications.

## Launch on This Machine

The desktop shell is on `admin01`. The user's two allocated RTX3090s are
on `gpu03`, Slurm job1465; the RTX4090 is busy and is left untouched.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
srun --jobid=1465 --overlap -N1 -n1 bash code/V4/run_v4_r44.sh
```

Run this long command in tmux. The script runs one training process per3090,
then applies the locked stage gate. It can resume from completed epoch
checkpoints; completed runs are not retrained.

Results: `code/V4/runs/R44_conditional_node_adapter/`. The manifest records
the actual baseline, hashes, metadata normalization, train-only size bins,
source snapshot, parameters, exact budgets and hardware. Each arm stores
config, train log, history, best/last checkpoints, predictions and curves.
`comparison.md` answers whether C2 adds value beyond continuation/shared
capacity. `decision_changes.csv` includes both-wrong cost changes.

Historical RTX4090 and current RTX3090 TF32/SDPA evaluation can differ on
near-tied scores. This is recorded in baseline_replay.json; all three R44
arms must have the identical RTX3090 step0 result. Normalization is never
fitted on validation/test. Actual evaluation costs use the original FP64
labels, not FP32-upcast caches.
