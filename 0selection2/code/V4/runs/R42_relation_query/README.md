# R42 Relation-Aware Solver-Query Selector

## Experiment

- A: unchanged effective R39A dual-stream classification architecture, trained from scratch with winner-cost and R-Drop.
- B: four sparse relation-message/global-attention layers; node memories from layers 2/4; four queries per solver and two cross-attention readout layers. No solver writes into the node encoder, and no common-context decoder.
- One selector seed (2). Full 18-task train/val coverage with native `ind`, natural sampling, no augmentation or label regeneration.
- Each epoch: 79 successful updates per task (78 batches of 128 plus a tail of 16); 1,422 global updates. Same saved task/instance schedule for A/B.
- Per update: two independent dropout forwards on identical input, averaged winner-cost plus symmetric KL; one AdamW update. AMP skipped steps retry the same batch and dropout paths with the reduced scaler.
- Winner-cost: `0.35 CE + 0.10 winner_pair + 0.02 scaled_risk`, `cost_scale=0.01`. R-Drop coefficient ramps to 0.35 over three epochs.
- LR 1e-4, three-epoch linear warmup; halve after three post-warmup validation plateau epochs. Minimum 15 epochs, maximum 40, stop after eight epochs without an effective validation-cost improvement.
- Effective improvement is fixed at 0.001 percentage point of macro vs SBS. Every strictly lower validation cost still saves `best.pt`, even below that stopping threshold.
- Dropout 0.1, weight decay 1e-4, gradient clipping 1.0; no gradient accumulation.

## Inputs and Isolation

The 34 solver fields and global IDs are unchanged. Existing 12-D local geometry uses the R39 train-only standardization, copied into checkpoint buffers. Costs, native winner and performance targets are stripped before model forward. Sparse edges depend only on inference-visible inputs. Padded candidate slots are supported at inference; this experiment trains homogeneous, fully valid problem pools, without padded cost slots.

Coordinate relations use Euclidean travel distance / speed=1, as in the original TW environments. Route limits apply to travel distance, not service time; open routes omit the depot-return term. Depot placeholder time windows are excluded from overlap/slack features. Edges are message neighborhoods, not feasibility pruning. ATSP retains outgoing and incoming low-cost neighbor types and forward/reverse matrix values.

Kth-distance ties are included, duplicate edges are merged, and source-grouped segment reduction avoids nondeterministic CUDA atomic summation in inference. Graph construction and degree counts are reused between the two dropout forwards and across message layers.

## Files

- `protocol.json`, `data_manifest.json`, `schedule_seed2.pt`: locked inputs, configuration and paired sample order.
- `runtime_witness.json`: actual 128-instance CUDA updates on large TSP/CVRP, ATSP and OVRPTW batches.
- `source_launch/`: source snapshot; hashes are checked before training and test.
- `dual_stream_rdrop_seed2/`, `relation_query_rdrop_seed2/`: args, train log, full train/val history, best/last checkpoint, per-epoch validation predictions and live curves.
- `locked_checkpoints.json`: both validation-selected checkpoints locked before any R42 test evaluation.
- `test_predictions/`, `test_results.json`, `comparison.md`, CSV/PNG files: generated after both training runs finish.
- `preflight_atomic_sum/`, `preflight_segment_sum/`: preserved pre-training witnesses; not additional selector training experiments.

The only available GPU is one RTX 4090 24 GB. The final preflight peaks were approximately A 8.86 GiB and B 15.89 GiB, so the two runs are queued rather than overlapped into an OOM. W&B is logged offline under each run because this environment has no configured online credentials. Local JSON and PNG files contain the same evaluation metrics.

## Evaluation

Full train/val use `eval()`, no dropped tails, FP32 inference and raw FP64 costs. ALL percentages are means of the 18 per-problem percentages, not ratios of ALL mean costs. SBS follows the historical split-specific best-fixed-method convention. Selection is classification argmax for both runs. Test is read once after both validation selections finish; test never chooses an epoch or a per-problem model.

The preset investment target is at least +2 Top1 percentage points and a 10% relative reduction of actual regret versus R39A validation, with MVRP improvement. This is an experimental target, not a promised gain or statistical-significance threshold. A single seed cannot establish multi-seed robustness; A/B change both encoder and readout, not Q/K/V in isolation.

## References

- [GraphGPS](https://arxiv.org/abs/2205.12454): local edge-aware messages and global attention.
- [Query2Label](https://arxiv.org/abs/2107.10834): category-associated queries read unpooled evidence.
- [R-Drop](https://arxiv.org/abs/2106.14448): symmetric consistency between dropout paths.

This is a project-specific PyTorch implementation, not a reproduction of those complete systems.
