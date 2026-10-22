# R94: Frozen Node Knowledge Before Trainable Graph Processing

This version is being prepared while R92A/B and R93A/B continue. It must not
interrupt healthy workers or use an unallocated GPU. A final main-ready receipt,
focused CPU tests and an available authorized slot are required before launch.

## Question

Does externally route-pretrained node knowledge improve the full18 selector
when its own trainable inter-node layers can recombine that knowledge with raw
attributes, rather than merely pool the frozen representations as in R50?

This is one representation-learning hypothesis, not a proposed solution already
supported by results. R50 remains negative evidence against easy transfer. No
additional routes, solver decisions, cost labels or instances are generated.

## Matched Pair

| Arm | Frozen node donor | Trainable part |
| --- | --- | --- |
| A | Randomly initialized, fixed ReLD-MOEL encoder architecture | Original full18 selector plus early node-fusion projection |
| B | R48-pinned pretrained ReLD-MOEL encoder, same architecture | Exactly the same trainable architecture and initial values |

Frozen native 128-dimensional node features are normalized separately and
projected into the original node input stream **before** the four trainable
inter-node layers. Raw attributes, local geometry, task constraints/size,
distance relations, method ID/34-dimensional features, the two synchronous
joint layers and the original score-MLP/maximin path remain available.

The donor has compatible static inputs for CVRP and fifteen MVRP tasks, not for
TSP/ATSP. Those two retain the original raw path. This is partial16-task
pretrained-feature coverage inside a full18 final selector, not a two-method
expert. No removed candidate is restored. No claim of repairing B semantics is
made; the frozen NEW90k scenario remains unchanged and R58 remains deferred.

Only exact real node sets enter the native encoder. Padding is introduced
afterwards for selector batching. Node-choice top2 routing, no normalization,
frozen eval and no gating noise are required. The decoder is never called.
O/L/service information omitted by the donor stays in the selector's raw path;
the donor must not be described as encoding every solver execution condition.

## Fixed Protocol

- Frozen NEW90k: every task TRAIN81,000 / VAL4,500 / TEST4,500; seed2.
- Original R77 objective: CE0.35 + comparison0.35 + risk0.02 + symmetric R-Drop
  ramped to0.35; no new auxiliary target or loss-coefficient search.
- 57,600 successful logical-batch1024 updates, identical task/sample order.
- AdamW1e-4, warm4320 then cosine to1e-6; WD1e-4, dropout0.1, clip1.
- TSP/CVRP micros352, other16 tasks1024; eval256. Real row normalization,
  same-batch/RNG AMP retry, no dropped rows.
- Full18 VAL every1440; full TRAIN every7200 and selected-point backfill.
- One checkpoint by strict minimum full18 VAL macro_vs_sbs_pct using the frozen
  TRAIN SBS; first tie wins. Lock before complete TEST and native LIB.
- TEST has participated in earlier development; no blind-test claim or
  TEST/LIB-based rule/temperature/checkpoint reselection.

External donor pretraining data/compute are additional resources, not matched
to learning from scratch. Original pretraining instance manifests and overlap
are unavailable and must remain disclosed. Historical R89B is the main
reference, R84A secondary; only R94A/B are the new matched representation pair.
GPU hardware may differ, so tiny differences are not causal proof.

## Execution and Delivery

Input-only donor features may be cached in FP16 using one fixed native FP32
extraction protocol. TRAIN/VAL preparation precedes training. TEST features are
prepared only after the formal VAL lock; LIB may use the identical online
donor path. Cache identities bind the original input release, node alignment
and donor, not the costs. No raw arrays, donor/selector weights or caches go to
GitHub.

Deliver one locked full18 model per arm: paired selection/native-winner PDF,
NSS comparison on TSP TEST/CVRP TEST/TSPLIB/CVRPLIB, TRAIN/VAL curves, all-task
cost/SBS/Oracle/regret metrics, the fixed R84 target36 recovery/precision/false
picks and net cost. Measure full donor-plus-selector inference, not just a
cache-hit readout. Fewer zero-selection groups alone does not satisfy the goal.

Promotion requires joint full18 VAL Top1 improvement with nonincreasing cost
and regret, including improvement over the random-donor control; report the
same locked models on TEST/LIB rather than select using them. Preserve negative
results and publish the completed version once under the shared Git mutex.

## Research and Roles

Astra/max supplied only the primary-paper/idea assessment in
`docs/research/pretrained_nco_representation_beyond_r50.md`. Sol/max performed
the bounded source feasibility in
`docs/research/nco_node_fusion_source_feasibility.md` and owns protocol,
implementation, review, execution, analysis and publication. This is early
fusion of external knowledge, not an exact paper reproduction or a promised
Top1/rare-method improvement.
