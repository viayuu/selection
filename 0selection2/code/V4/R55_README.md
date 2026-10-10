# R55: Route-count cost-difference bottleneck

One seed=2 scratch run. Reuse the 300000 R54 training routes and fixed 1920
validation diagnostic routes; no solver execution, API calls or new labels.

## Model

The unchanged 4-layer d128 InstanceEncoder supplies nodes. A shared symmetric
edge MLP predicts D=A_MTL-A_MOEL for every valid undirected edge. Inputs are the
sum/absolute difference/product of 32-dimensional endpoint projections, raw and
input-mean-normalized distance, explicit constraint/descriptor fields and size.
The MLP has hidden width64 and dropout0.1. Its last layer uses small nonzero
Normal(std=0.001) weights and zero bias to avoid a random O(N^2) summed cost
offset. This is an initialization choice, not a learned gate or bypass.

The only scalar output is `sum(D_hat * true_input_distance)`, reduced in FP32.
Positive/zero chooses MOEL. There is no independent binary classifier. Counts
preserve multiplicity; open-route returns to depot are omitted before merging
edge directions. No feasible-route decoding is claimed.

Loss = per-instance positive/negative/zero group-balanced edge MSE + squared
cost-difference error normalized by 1% of the original full-pool Oracle cost.
Cost labels, true routes and Oracle denominators are loss-only. Predictions do
not use them. There is no BCE, risk, R-Drop or successor objective.

[Concept Bottleneck Models, ICML 2020](https://proceedings.mlr.press/v119/koh20a.html),
sections2-3, supplies the distinction between auxiliary supervision and a
mandatory supervised intermediate path. Signed billed counts and analytic
distance readout are this project's adaptation, not a reproduction of its
image tasks or learned concept-to-label network.
The analytic readout also consumes observed input distances, so this is not a
strict concept-only replica of that paper.

## Training and Selection

Fresh initialization; same initial Encoder values and 149976 non-tied training
queries/batch order as R53/R54. The 24 exact ties remain excluded for continuity.
Batch128, AdamW1e-4/WD1e-4, warmup3, clip1, max40/min15 epochs, LR patience3
(x0.5), early-stop patience8, min_delta0.001pp. Full18 validation actual regret
selects the strictly best checkpoint. Fixed R45A Top2={MOEL,MTL} gating remains.
Val-route subsets only diagnose edges and never determine gating or targets
for training. Full test runs once after the checkpoint SHA256 lock.

## Commands

From `/public/home/shiys/0selection2`, in `easynco`:

```bash
bash code/V4/run_v4_r55.sh precheck
# Only the authorized idle RTX3090; do not change bindings to occupied cards.
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G \
  env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d \
  bash code/V4/run_v4_r55.sh all
# After the locked evaluation, interpret saved edge predictions without GPU use:
python -m code.V4.r55_bottleneck_review
```

Long execution is placed in tmux `r55_train`. Its exact invocation and device
are saved in `runs/R55_route_cost_difference/execution.json` and `runtime.json`.
`prepare` verifies/converts targets without GPU training. `train` resumes only
this experiment's unchanged config/last.pt; it never loads trained Rxx weights.
`evaluate` requires an already completed/locked validation run.

The benchmark includes the Encoder and new edge decoder, with a separate
edge-only measurement clearly labeled. It excludes shared R45A ranking and
CPU I/O and is not complete routing latency.

## Artifacts

`runs/R55_route_cost_difference/`: protocol, source/target hashes, exact
conversion checks, sampling plan, geometry statistics, model/gradient checks,
history, curves, full18 comparisons, correction/harm costs, scalar predictions
and diagnostic bottleneck predictions. Best/last checkpoints, target caches and
offline W&B logs remain local and ignored by Git, consistent with prior runs.
Historical R45A/R53A/R54 results and original labels are never overwritten.
The optional read-only bottleneck review records its own source hash, nearest-
integer count diagnostics and constant-MTL accuracy reference. It never changes
model weights, predictions, checkpoint selection or training configuration.
