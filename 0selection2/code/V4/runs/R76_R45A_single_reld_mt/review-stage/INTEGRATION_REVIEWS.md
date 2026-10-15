# R76 Bounded Integration Reviews

Requested reviewer: gpt-6-astra, reasoning effort max. Requested coding/cache
workers: gpt-6.1-sol, reasoning effort max. These are same-family, provisional
launch reviews, not independent performance acceptance or a resumed 70% goal.
The final reviewer did not independently verify its provider model identity.

## Initial Protocol Review

Agent: 01a139c5-ebcb-7bc2-919d-f34075ccbdb9.

**P1: Before Launch**
These are implementation requirements, not confirmed R76 bugs; the new files were not present during review.

- **Replace split-local SBS selection.** The inherited [evaluator](/public/home/shiys/0selection2/code/V4/performance_evaluation.py:32) reselects SBS on each evaluation split. Freeze each task's solver identity from TRAIN, then evaluate that solver's costs on VAL/TEST. Preserve the per-task ratio-of-means followed by the 18-task macro average. Save every strict validation minimum; `.001` applies to the controller, not checkpoint eligibility.
- **Weight the entire DDP objective.** The original [objective](/public/home/shiys/0selection2/code/V4/r43_experiment.py:65) is a local sample mean. Before backward, multiply it, including KL, by `4 * local_rows / global_rows`. Full-batch factors are `[8/7,8/7,6/7,6/7]`; tails require actual counts. Unscale and clip at `1.0` after gradient synchronization. All ranks must step or retry together with matching scaler state and restored rank-local RNG.
- **Retain FP64 costs through cache target construction.** [Comparison targets](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:17) depend on original precision, including exact ties. Upcasting cached FP32 is insufficient. Preserve per-instance comparison normalization and the original capped risk divided by `.01`.

**P2: Recipe And Speed**
- **Global1792 is an intentional recipe deviation.** Original [R45 fixes batch128](/public/home/shiys/0selection2/code/V4/r45_experiment.py:332). On this dataset, updates become **720 versus 9,864 per epoch**, approximately 13.7 times fewer. Keeping LR unchanged preserves its nominal value, not optimization equivalence or cumulative AdamW decay. Describe this as a large-batch R45A retrain.
- Shared preparation followed by node-local arrays is reasonable. The single bounded capacity check should exercise the actual two-forward/backward/optimizer path on both GPU types, then discard probe state. Capacity success alone does not establish that `[512,512,384,384]` balances throughput; no additional sweep is necessary.

**Overall Judgment**
**Provisionally acceptable protocol.** The requested architecture/objective, outer DDP wrapper, TRAIN-only normalization, full validation, and post-lock TEST boundary are coherent. Manifest/reader inspection confirms `reld_mt -> RELD_MOEL -> ID16`, retained 22-ID space, and manifest-aligned candidate ordering.

No files changed or tests/training run. Implementation acceptance remains for the single follow-up after code is ready.

`review_independence=same-family`
`acceptance_status=provisional`

Memory update, text only: R76 remains one seed2 freshfull18 run; R58 stays deferred; no additional experiments or iterative audit loop.

## Full Integration Review

Agent: 01a139e2-c72e-71a2-83b5-f50f8e4b8a08.

**P2 Warnings**
- **Checkpoint recovery:** [r76_experiment.py:623](/public/home/shiys/0selection2/code/V4/r76_experiment.py:623) commits `last.pt` before an improved `best.pt`. Interruption between those writes leaves resumed best metadata ahead of the saved model. Finalization can lock that stale model; TEST then rejects its epoch after consuming `test_started`. Minimum fix before using resume: publish best before last and reconcile checkpoint/best epochs during recovery, before locking TEST.
- **Failure propagation:** [run_v4_r76.sh:107](/public/home/shiys/0selection2/code/V4/run_v4_r76.sh:107) waits for children in launch order. If the 3090 launcher fails during startup, its failure remains unhandled while the 4090 launcher waits for rendezvous. Minimum fix: detect completion with `wait -n` and terminate remaining steps immediately on failure.

**Judgment**
**Provisionally ready for the fresh launch; 8/10.** These are nonblocking recovery warnings. No blocking defect found in sample-weighted loss/KL, FP64 targets, wrapped dual forwards, synchronized AMP retry/RNG, tails, aliases/masks, TRAIN-fixed SBS, strict selection, TEST marker ordering, or CPU prefetch.

Reviewed the supplied 13+7 test coverage; `bash -n` passed. No files edited, Git used, or GPU tests/training launched. No additional audit loop requested.

`review_independence=same-family`
`acceptance_status=provisional`

**Memory Update**
Handoff only, not persisted: final R76 integration review complete. Main owns launching, monitoring, and results; historical 70% goal and R74 remain paused; R58 remains deferred.

## Uniform-Batch Accumulation Review

Agent: 01a139f5-e373-7ad3-becf-56e9b13a9143.

READY: no blocking P1 found in the requested scope; provisional review.
DDP accumulation, microbatch weighting, common-round/dummy handling, single logical-step clipping, RNG replay, coverage, and checkpoint/test-lock logic are consistent.
Six targeted read-only CPU tests passed; distributed tests inspected, not rerun.
No writes, Git, GPU use, or new review loop. GPT6Astra/max identity cannot be verified here.

## Adopted Changes

The final logical batch is9856 uniformly across all18 tasks, not the initial
provisional1792. Measured per-device limits are microbatch caps. The code
retains actual sample weighting including KL and FP64 supervision; checkpoint
publication/recovery and child failure propagation warnings were fixed before
formal updates. One IPv6 automatic-interface startup failed before any update;
the successful launch uses bond0/AF_INET. No solver or TEST ran during review.
