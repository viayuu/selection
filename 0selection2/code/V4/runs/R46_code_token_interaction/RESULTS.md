# R46A final results

Only R46A was implemented and trained. R46B was not run. Seed=2, all18 tasks,
from scratch, on the authorized idle RTX3090 only. Existing jobs and other GPUs
were left untouched. No API calls or new source embeddings were used.

## Model and training

Each complete source role/deployment view remains a separate token. A shared
1024-to-128 projection and four shared role embeddings initialize the tokens.
There is no trainable solver ID, solver-specific bias/projection/head, candidate
position embedding, or handcrafted solver-feature branch. Two synchronous
node/source-token interaction layers precede instance-conditioned, per-solver
source pooling. The original shared decoder, utility head, score differences,
maximin decision, comparison objective, and R-Drop are retained.

Training completed40 epochs and56,880 successful optimizer updates,3160 per
task. The validation-cost-best checkpoint is epoch35. Full test was evaluated
once after this selection:18,000 instances,1000 per problem. Original FP64
costs and native winner labels are used throughout evaluation. All18 saved
prediction files reproduce the reported metrics with zero discrepancy.

## Frozen test comparison

ALL is a per-problem macro average; actual regret is the mean per-instance
relative selection loss and is not the ratio of ALL mean costs.

| Model | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---:|---:|---:|---:|---:|---:|
| R45A handcrafted | 47.9111% | 81.7444% | 92.4333% | 11.168366 | -1.0320% | 0.9432% |
| R45B static code | 47.4000% | 81.3889% | 92.3111% | 11.170249 | -1.0149% | 0.9634% |
| R46A source tokens | 47.7556% | 81.5833% | 92.0944% | 11.169184 | -1.0286% | 0.9496% |

R46A minus R45A: Top1 -0.1556 percentage points, mean_cost +0.000818,
actual regret +0.00633 percentage points.

R46A minus R45B: Top1 +0.3556 percentage points, mean_cost -0.001065,
actual regret -0.01382 percentage points.

The new path is slightly better than the static-code reference but does not
surpass the handcrafted reference. This is not a clear performance breakthrough.
Keep R45A as the reference. Without R46B, these results cannot establish a
benefit of correct source correspondence or generalization to unseen solvers.

## Identical-source limitation

The immutable R45 cache contains two pairs of identical complete source-token
packages within the TSP pool. With no solver ID and a shared network, equal
inputs cannot distinguish the members. Exact score ties are resolved by the
fixed global solver ID; no candidates were removed or secretly given identity
parameters.

| Pair | Native winners: first / second | R46A picks: first / second | Equal-logit instances |
|---|---:|---:|---:|
| DIFUSCO / DIFUSCO500 | 52 / 132 | 1 / 0 | 1000 / 1000 |
| T2T / T2T500 | 114 / 156 | 252 / 0 | 1000 / 1000 |

Consequently, R46A never selects the two higher-ID solvers that are the native
winner on288 TSP test instances. This is an input distinguishability limitation,
not evidence of label noise or a broken gradient path.

Paired TSP comparison with R45A makes its impact concrete:

| Native-winner subset | Instances | R45A correct | R46A correct | Sum of selected-cost changes, R46A minus R45A |
|---|---:|---:|---:|---:|
| DIFUSCO500 or T2T500 | 288 | 77 | 0 | +9.690369 |
| All other winners | 712 | 282 | 333 | -4.969599 |
| All TSP | 1000 | 359 | 333 | +4.720770 |

On this subset,77 previously correct selections become inaccessible; elsewhere,
net51 additional correct selections offset part of that loss. The overall TSP
Top1 decreases from35.9% to33.3%, and mean_cost increases by0.00472077.
This decomposition does not explain every difference on the other17 tasks.
The existing source cache does not encode these deployment distinctions;
previously mixed content cannot be reconstructed without new source evidence.

## Required checks and runtime

Nine focused tests and real128-instance TSP/CVRP/ATSP/OVRPTW preflight batches
passed. Source projection, instance Encoder, interaction, and dynamic pooling
receive finite gradients; fixed source buffers do not. Forward ignores costs
and winners, respects masks and candidate permutations, and reconstructs from
the checkpoint offline. Two dropout forwards produce one optimizer update.

On fixed pairs of validation instances, post-interaction source states, pooled
solver states, and pooling weights all change with the instance. This confirms
the dynamic path is connected, not that the learned semantics are effective.

Training and evaluation pipeline elapsed about72 minutes. Optimization loops
totaled about64.6 minutes; peak preflight training allocation was about9.1GiB.
Forward-only inference on the same128 validation instances, FP32/TF32/SDPA,
median of20 CUDA-event measurements after3 warmups:

| Model | TSP, max497 nodes | CVRP, max500 nodes | ATSP, max29 nodes | OVRPTW, max57 nodes |
|---|---:|---:|---:|---:|
| R45A | 54.369 ms | 54.919 ms | 6.080 ms | 6.252 ms |
| R45B | 54.831 ms | 55.401 ms | 7.195 ms | 7.335 ms |
| R46A | 55.487 ms | 57.385 ms | 6.429 ms | 6.786 ms |

These are fixed-batch measurements, not population-wide average latencies.
The26 previously assumed historical deployment records remain unverified;
this experiment does not restore those missing historical facts.

## Artifacts

- Complete per-problem results: `comparison.md`, `comparison.csv`.
- Curves: `curves/comparison.png`, `A_code_tokens_seed2/train_val_curves.png`,
  `A_code_tokens_seed2/optimization_curves.png`, and per-problem plots.
- Selected checkpoint: `A_code_tokens_seed2/best.pt`, epoch35,
  SHA256 `98ad6b2e135a61ff669e0155aaf9825888aa0ee8a10486e7c1da58e7f52f022f`.
- Final checkpoint/log/config/history: `A_code_tokens_seed2/`.
- Per-instance predictions: `test_predictions/A/`.
- Dynamic and identical-input checks: `dynamic_evidence.json`,
  `identical_source_predictions.csv`, `source_package_audit.json`.
- Timing details: `inference_latency.csv`, `inference_latency.md`.
- Replay and launch provenance: `prediction_replay.json`, `protocol.json`,
  `source_launch/`, `locked_checkpoints.json`.
