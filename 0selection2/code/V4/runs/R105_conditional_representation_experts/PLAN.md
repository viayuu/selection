# R105: Input-Conditional Representation Experts

Date: 2026-10-23. Main-owned prospective plan, not a result.

## Question

Do input-conditioned intermediate feature transformations learn more useful
solver-discrimination regions than the same experts with input-independent
mixing? This is not another score calibration, final-method ensemble, static
task bank, method-private head, or claim that capacity was the proven cause.
R102's original two4090 runs continue unchanged. R104's completed wider/deeper
models remain historical evidence, not initialization for this experiment.

The papers-only rationale is
`docs/research/post_r104_conditional_modularity.md`. It supports a different
sharing bias, not a prediction of project efficacy. Sol owns implementation,
planning, execution, review and analysis; Astra/max was requested only for
papers/ideas. Independent backend attestation is unavailable.

## Two Arms

- `A_static_gate`: four independently parameterized FF experts in every one
  of the four instance-Encoder layers, with a learned but input-independent
  softmax mixture in each layer.
- `B_input_gate`: the exact same experts and initial tensors, with the mixture
  conditioned on that layer's current post-attention node representation.

Both retain width128, four heads, original instance4/joint2, the geometry and
distance-bias paths, methodID+34 features, R88B physical TSP node0 role, original
score MLP, utility differences and maximin inference. Only the instance-layer
FF transformation changes. Joint layers, solver FFNs and score head do not.

Each of four experts is the original Linear(128,512)/GELU/dropout0.1/
Linear(512,128) FFN. A layer returns the softmax-weighted sum of all four
expert outputs, followed by the original residual and normalization. This is
dense conditional mixing, not a claimed sparse-compute MoE reproduction.
No top-k routing, capacity drop, balancing objective, solver quota, private
solver expert, new loss, teacher, historical-label context or solver execution.

Each router is Linear(128,4), initialized to all zeros. A feeds zeros into
that router, so only its bias can change; B feeds the current node state.
Thus both start from exactly the same complete parameter tensors, initial
function and initial uniform mixture. Both have3,425,561 total parameters;
A's2,048 router feature-weight parameters have identically zero gradients.
This inactive-control parameter difference must be disclosed: equal total
tensor count is not equal functional freedom or a strict active-parameter
match. B has only2,048 additional input-sensitive gate parameters, not an
additional expert bank. All four experts are computed in both arms.

At a fixed learned static mixture, A's weighted FF outputs can be absorbed
into one wider hidden FF layer by concatenating the expert hidden units and
scaling their output blocks. B's input-varying mixing cannot be absorbed as
constant output blocks in that way. This motivates the controlled comparison;
it does not prove that the new partition is useful or that a dense network
cannot learn similar behavior.

Reuse fresh R84 seed2 original weights, not a trained selector. Unchanged
parameters and expert0 copy their corresponding fresh original tensors.
Experts1-3 use isolated seed105002. Routers start at zero in both arms.
Both complete fresh states must match before training; no cost or winner
participates in their construction or forward computation.

## Frozen Data And Training

Frozen NEW90k full18 data, per task TRAIN81,000 / VAL4,500 / TEST4,500.
Use existing native FP64 winner identities and filtered legal candidate pools.
Keep the eight B-task exclusions and single original-MOEL `reld_mt` alias.
No new instance, solver run, route target, label change or R58 operation.
Reuse existing R84 TRAIN-only geometry, task/row streams and node caches
directly; do not copy or rebuild input/teacher caches.

Seed2, logical batch1024, 57,600 successful updates, warmup4,320; full VAL
every1,440 and full TRAIN evaluation every7,200 successful updates. AdamW
peak1e-4/end1e-6 cosine, weight decay1e-4, clip1, dropout0.1. Original
CE0.35 + comparison0.35 + risk0.02 + two-forward R-Drop ramp to0.35.
No coefficient, expert-count, gate-temperature, router or budget search.

Both arms use TSP/CVRP micro128, other tasks512 and eval128 initially;
the logical batch remains1024 with exact-row gradient accumulation. A bounded
TRAIN-only resource witness may reject this execution choice before formal
training; any change must be frozen for both arms before either formal start.
No automatic OOM shrink, terminal retry, restart or skipped accepted update.

## Selection, Diagnosis And Delivery

One whole-model checkpoint by strict minimum full18 VAL macro_vs_sbs_pct
using the common full-TRAIN SBS; first exact tie wins. Lock that checkpoint
and inference before full TEST and LIB. No TEST-driven change, per-problem
assembly or use of currently running R102 TEST results to tune this version.
Existing TEST is development-exposed, not claimed to be a fresh blind holdout.

Deliver full18/per-problem TRAIN/VAL/TEST Top1, cost, TRAIN-SBS, Oracle and
actual regret; fixed36 true-winner recovery, false picks and net costs;
same-axis choice/native-winner PDFs; NSS four-dataset comparison; full curves
and actual standalone model timing. Historical R89A/B and R84A are not rerun.
Acceptance still requires genuine system improvement, not gate diversity,
expert utilization or fewer zero-choice groups alone.

After locking the model, record a small input-only routing diagnostic on
the first128 VAL instances per task: per-layer expert weights/entropy and
variation between instances of the same exact problem/size, with coverage
and the small fixed sample clearly disclosed. It cannot establish efficacy
or absence of coarse shortcuts. No routing diagnostic changes selection.

Use the two naturally freed3090 slots in the existing job1465/gpu03 container,
same UUIDs/CPU quartets and GPU mutexes as completed R104. Keep both R102
4090 workers untouched. W&B remains enabled with explicit offline mode if
no credentials are available. Finish both full pipelines and reports before
one curated GitHub commit; preserve negative results and first fatal errors.
