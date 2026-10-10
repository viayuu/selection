# R56: EGT-Inspired Directed Instance Encoder

One scratch seed2 experiment on the existing 15 MVRP tasks, with the R53A
unweighted BCE objective, node pooling/head and fixed R45A Top2 trigger. The
only architectural experiment is replacing the instance Encoder's four
node-only attention layers with four joint node/relation layers (128/32
dimensions, four heads). No routes, solver execution or auxiliary labels.

## Paper Mechanism

[EGT: Global Self-Attention as a Replacement for Graph Convolution, KDD 2022](https://arxiv.org/html/2108.03348v3),
section3.2, equations3-7:

- Pre-norm node Q/K/V and edge embeddings.
- Clip scaled node QK to [-5,5], add learned per-head edge bias.
- Multiply masked node softmax by a learned edge sigmoid gate, without
  renormalizing. Apply probability dropout0.1.
- Update nodes with the gated values and a node residual/FFN.
- Update persistent directed edge channels from the unmasked pre-softmax
  interactions, followed by their own residual/FFN. Both updates read the old
  node/edge states.

The original coordinate projection, local geometry, condition/stats/CLS
inputs are retained. Input helpers are reused, **not** the old attention
layers or fixed distance bias. Endpoints use separate source/target
projections of the original eight fields plus a real-node marker. Distances
are raw and mean-distance normalized using valid input-only off-diagonal
pairs. Conditions are existing cbits/problem_desc/log scale. Special-token
relations have zero distances/attributes and a false real-node marker.
All valid ordered pairs including self participate; only padding is masked.
No guessed local feasibility constraint removes an edge.

This is not complete EGT reproduction. GELU replaces ELU; probability
dropout replaces random attention masking; dynamic centrality and SVD
encodings are omitted. A terminal edge-only update is available for
diagnosis but has no BCE gradient under the requested node-only readout.
Earlier persistent edge updates and all four bias/gate projections have a
gradient path. No extra final edge pooling/head is silently added.

## Run

Activate `easynco` and run from `/public/home/shiys/0selection2`:

```bash
bash code/V4/run_v4_r56.sh tests
bash code/V4/run_v4_r56.sh prepare
```

Only the authorized idle RTX3090 UUID is allowed. On this machine the
compute node is reached through the existing Slurm allocation, not SSH:

```bash
tmux new-session -d -s r56_train 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && exec srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=offline bash code/V4/run_v4_r56.sh all > code/V4/runs/R56_relational_encoder/experiment.log 2>&1'
```

The run checks a full maximum-size batch optimizer update before training.
AMP overflow retries exactly the same planned batch and RNG. Batch128,
fresh AdamW1e-4/WD1e-4, clip1, warmup3, max40, min15/early-stop8 and
LR-patience3 match R53A. All pre-generated task/sample plan hashes are
compared to R53. Every epoch retains tails and covers all non-ties.

Every epoch evaluates complete validation; every5/final epoch evaluates
complete training in eval mode. Best selection is strict minimum full18
integrated validation actual regret, not specialist accuracy. Plateau
control requires 0.001 percentage points. The validation-best SHA is locked
before the single full test. Test cannot select a different checkpoint.

## Outputs

`runs/R56_relational_encoder/` contains RESULTS.md, protocol/config,
provenance, precheck, train/val curves, full18/group/per-problem comparisons,
binary metrics, corrections/harms, instance decisions, solver selections,
latency and saved val/test predictions. Large checkpoint/initialization/
sampling tensors and offline W&B logs remain local. Report R53A as a
historical reference, not a newly matched rerun. Do not infer effective
selection from changing relation states or training fit alone.
