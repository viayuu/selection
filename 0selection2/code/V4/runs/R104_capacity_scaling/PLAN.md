# R104: Larger Shared Capacity, Existing Directions Retained

Date: 2026-10-23. Main-owned protocol; this is a preparation record, not a result.

## Question

Can substantially wider or independently deeper shared representations improve
full18 instance-level solver discrimination with the existing inputs and labels?
R99 input competition and R101/R102 pretraining directions continue unchanged.
No local ReLD expert, new solver execution, route target, or R58 operation is added.

| Configuration | Width | Instance / joint layers | Trainable parameters |
| --- | ---: | ---: | ---: |
| Historical original | 128 | 4 / 2 | 1,842,953 |
| A_width256 | 256 | 4 / 2 | 7,288,969 |
| B_depth8 | 128 | 8 / 4 | 3,426,057 |

Both retain four attention heads, FF width4d, original methodID+34 metadata,
node geometry and distance path, synchronous dual streams, original score MLP,
utility differences and maximin. No new normalization or training objective is
introduced along with capacity. Geometry's existing hidden64 is unchanged.
See MODEL_PREP.md for actual module dimensions and initialization details.

## Frozen Training

Use the frozen NEW90k full18 data: 81,000 TRAIN, 4,500 VAL and 4,500 TEST per
problem, native FP64 winner identities and existing legal candidate pools.
Eight B tasks retain the already filtered pool; original MOEL is reld_mt,
with no deleted candidate or alternate ReLD minimum restored. R58 stays deferred.

One fresh seed2 run per arm. Reuse only R84 TRAIN statistics and scratch
initialization; no trained selector, solver donor or historical label context.
Same-name/same-shape fresh original parameters are copied; resized/new capacity
tensors use isolated seed104002. Width-changed tensors are not padded or tiled.
Consequently the two architectures do not have identical complete parameter
initializations. This is not a parameter-matched or equal-FLOPs A/B experiment.

Logical batch1024; natural frozen task/row streams; 57,600 successful updates;
warmup4,320, full VAL every1,440, full TRAIN evaluation every7,200. AdamW
peak1e-4/end1e-6 cosine, weight decay1e-4, clipping1, dropout0.1. Original
CE0.35/comparison0.35/risk0.02 and two-forward R-Drop ramp to0.35.

Both arms use the same execution caps: TSP/CVRP micro128, all other tasks512,
evaluation128. These are execution splits, not smaller logical batches;
gradient accumulation is weighted by the actual row count. Original full
batch semantics remain intact. No automatic batch/LR shrink or terminal retry.
Any necessary capacity check uses only TRAIN, discards its temporary model and
optimizer, and starts formal training fresh; no result-based configuration search.

## Selection And Delivery

Choose one whole-model checkpoint by strict minimum full18 VAL
macro_vs_sbs_pct with the frozen full-TRAIN SBS reference; first checkpoint
wins exact ties. Lock inference before full TEST and LIB. No TEST/LIB-driven
checkpoint, threshold, per-task assembly or posthoc quota. Existing TEST has
participated in research and is not described as completely blind.

Historical R89A (original AdamW), R89B (best optimizer reference), and R84A
are not retrained. Hardware, microbatch, initialization and optimizer limits
must be disclosed instead of claiming a perfectly matched new control.

Deliver full18 and per-problem Top1/cost/TRAIN-SBS/Oracle/regret, fixed36
true-winner recovery, false picks and net cost, TRAIN/VAL curves, NSS comparisons
on TSP/CVRP TEST and LIB, and same-axis choice/winner distribution PDFs with
modelTop1 in every title. Time the actual larger model; parameter growth alone
or fewer zero-count methods is not success. Retain negative results.

## Resources And Research Roles

A uses original 3090_0 after R99A's complete TEST/timing/LIB/PDF pipeline;
B uses original 3090_1 after R99B. Both use existing allocation1465/gpu03,
one authorized UUID, four disjoint allocated CPUs and mem9GiB. Original node
cache is reused directly. R102A/B retain the two4090 slots. No healthy worker
is stopped, duplicated, or moved to unallocated SSH resources.

Astra/max was requested for papers/ideas only; Sol/max for implementation,
planning, execution and analysis. Independent backend attestation is unavailable.
The bounded paper review is docs/research/capacity_scaling_for_solver_discrimination.md;
its mechanisms motivate a hypothesis, not a measured benefit or theoretical
Top1 guarantee. Finish both arms and reports before one curated GitHub commit.
