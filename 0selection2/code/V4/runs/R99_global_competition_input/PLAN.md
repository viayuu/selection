# R99: Global Competition As An Input Transform

Status: implementation; no efficacy result and no GPU training launched.

## Fixed Comparison

A uses a row-normalized distance kernel. B alternately normalizes its rows and
columns for 64 Sinkhorn iterations. Both exclude self-arcs, include every other
valid node pair, use temperature 0.05 times the instance's mean off-diagonal
input distance, and have exactly the same trainable selector and initialization.
This compares local outgoing preference with global incoming/outgoing
competition, not two new attention architectures.

Each transform produces outgoing and incoming conditional averages of the
existing eight node attributes at one and two message steps (32 channels).
These per-node messages are projected into the original node stream before
the original four inter-node layers. The original geometry, constraints,
physical TSP anchor, method IDs/features, dual stream, score MLP and maximin
head remain. Messages are not a pooled graph summary or route labels.

The transform is input-only and deterministic. It is not an exact assignment
solver, a VRP lower bound, a feasibility test, a candidate-solver execution,
or a repair of B semantics. Finite-iteration marginal residuals must be recorded;
64 iterations are not a claim of exact convergence. Original solver outcomes
and routes never select arcs or enter the forward computation.

Both models start from the same fresh R84 seed-2 initialization, with an
identically initialized additional 32-to-128 projection (seed 990003).
Use the unchanged R77 objective and original successful-update schedule:
batch 1024, 57,600 updates, warmup 4,320, VAL every 1,440, TRAIN every 7,200,
AdamW 1e-4 to 1e-6 cosine, weight decay 1e-4, dropout .1, global clip 1.
Use the same natural task/sample order and TSP/CVRP microcap 352.

Select one complete18 checkpoint on minimum VAL macro_vs_sbs_pct, then
evaluate unchanged TEST/LIB. Do not assemble task-specific checkpoints.
R89B/R84A are historical references, not newly matched baselines. A/B is the
matched test of global column competition; include runtime and preprocessing.

## Scope

Use the existing NEW90k release: 81,000/4,500/4,500 rows per task, frozen
candidates and native FP64 labels. No instances, outcome labels, solver runs,
queues of outcome embeddings, or R58 work are added. Input-only messages may
be cached in FP16; online inference must perform the same transform and FP16
quantization. Cached-selector timing is not full inference timing.

This tests representational accessibility, not additional information. The
distance transform may explain common geometric difficulty without predicting
relative solver quality. Failure must be retained without a temperature or
iteration grid.
