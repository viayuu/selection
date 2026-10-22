# R101: Internalize Node Knowledge, Then Learn Full18 Selection

Two matched fresh runs, not a loss-coefficient grid. A learns from the frozen
random ReLD node cache already produced in R94A; B learns from R94B's pinned
pretrained cache. Only existing TRAIN node caches are reused. No routes,
candidate execution, new performance labels, query outcomes or R58 work.

## Stage I

Initialize the original R88B/R77 full18 selector from the same fresh R84 seed2
state and TRAIN-only geometry statistics. Optimize ONLY its instance encoder
on CVRP plus the fifteen MVRP tasks covered by the existing teacher interface.
TSP/ATSP have no fabricated teacher. Coverage of sixteen interfaces is not a
claim of sixteen-task donor pretraining; source data/overlap remain unknown.

Use direct per-node MSE after separate non-affine LayerNorm of the student's
and teacher's 128 channels. The student learns the teacher's channel basis;
there is no powerful learned adapter that could absorb the entire objective.
Average channels, then valid nodes within an instance, then the actual batch.
No graph-level winner target or method-conditioned output is used in Stage I.

Fixed 12,800 successful updates, logical batch1024, natural sixteen-task cycles
and per-task shuffle streams, seed2. CVRP micro352, others1024. One dropout
forward, dropout .1, AdamW 1e-4 to1e-6 cosine, warmup960, weight decay1e-4,
global gradient clip1. No VAL-driven selection of pretraining duration.

## Stage II

Transfer only this run's newly pretrained instance encoder. All other selector
weights retain the identical fresh R84 values. Reset optimizer/scaler/RNG and
use the unchanged full18 R77 objective, task/data order and 57,600-update
schedule: batch1024, warmup4320, VAL1440, TRAIN7200, AdamW1e-4 to1e-6.
No hint term, teacher feature input, extra head or donor is present in Stage II
or final deployment. Raw inputs, geometry, constraints, physical TSP anchor,
method IDs/features, dual-stream interaction and maximin head remain original.

Select one full18 checkpoint on minimum VAL macro_vs_sbs_pct; evaluate its
unchanged TEST/LIB. All reference metrics, paired distributions, NSS4 and
fixed-target true rescue/loss/false costs are required. Pretraining curves
are not a substitute for selector performance.

The random/pretrained teacher pair matches extra optimization and student
capacity. R89B/R84A/R94B are historical references, not newly budget-matched
controls. This is representation transfer/compression, not a novel algorithm,
proof of a root cause, or guaranteed rare-winner recovery. Prior negative
route/SupCon/pseudo-label transfer remains relevant.

## Resources And Provenance

Reuse immutable R84 input caches. A's existing teacher TRAIN files may be
stream-copied from the original gpu03 allocation container to a new shared
bridge and then node-local gpu05 storage. B's existing gpu05 teacher cache is
reused directly. No extraction is repeated. Record metadata, byte counts,
streaming hashes, transfer time, and the distinction between transported file
timestamps and input identity. Do not alter old R94 manifests to pass a gate.

Use existing authorized 4090_0 and4090_1 only. Healthy R99/R100 jobs remain
untouched; the second arm waits for R100's complete TEST/LIB/PDF pipeline.
Keep logical batch1024 and one GPU/arm. Final deployment is the original
student graph, but its actual latency must be measured rather than borrowed
from old cached-donor timings. Publish once when both arms finish.
