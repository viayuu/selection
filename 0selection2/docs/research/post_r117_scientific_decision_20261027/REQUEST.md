# Scientific Decision After R117

Role: original-paper research and next scientific ideas ONLY, by
gpt-6-astra with reasoning_effort=max. Sol owns all experimental planning,
budgets, protocols, code, review, numerical checks, execution and delivery.
No GPU actions, operational plan, source edits outside your three files,
commits or subagents. Do not stop or start experiments.

## Verified Facts Supplied By Sol

R117 is now COMPLETE, not an interim negative. Published version:
f3a458ea292abbd0c7a48d449f92cf03e1eb1a12.
Both arms completed 28440 successful updates, identical fresh trainable
initialization and whole18 data/pools/objective. Each selected one checkpoint
at update27018 by VAL, then full TEST/LIB. Read the final RESULTS.md and PLAN.md
for the precise scientific contrast and limits, not to certify code/results.

| Model | VAL Top1 % | VAL regret % | TEST Top1 % | TEST regret % |
| --- | --- | --- | --- | --- |
| R117 A | 71.322222 | 0.398842 | 71.446914 | 0.385552 |
| R117 B | 71.341975 | 0.402931 | 71.512346 | 0.387063 |
| Historical R94B | see its report | see its report | 71.577778 | 0.379115 |

Both retain the SAME pretrained ReLD static base. Only added frozen native
computations differ, random in A and pretrained in B: LEHD six heavy decoder
layers at input-defined initial boundaries, ReLD decoder hidden states at
independent input-defined reference conditions, MatNet directed role encoder
and decoder glimpses. No actual query trajectory or solving. DIFUSCO diffusion
computation is not covered. This does NOT prove every native transfer fails.

The predeclared VAL gate failed. TEST B-A:1123 rescues,1070 harms, net53 out
of81000; regret increased0.001511 percentage points. Not a breakthrough.
Frozen R84 target36 TEST contains4598 true winners: R94B TP117, R117A TP127,
R117B TP129. B selects these methods257 times,129 correct and128 false.
TSP LEHD:762 winners, B selects4 and is correct1; A selects1/correct1.
DIFUSCO:334 winners, zero selections in both. Do not call a small count change
success or confuse fixed36 with the entire platform. Selected TRAIN and VAL
accuracy are both approximately71.5%; this version does not exhibit near100%
TRAIN fitting or the earlier huge TRAIN/holdout separation.

Sol's operational sidecar for the early-context idea is
docs/experiment_planning/early_graph_context_20261027.md. Its current NO-GO is
NOT a new scientific failure or a proof of hardware impossibility: a faithful
strong-late-local contrast has heavy support recomputation, unmeasured
throughput and a proposed336GPUh envelope. We will not weaken the late model
to pooled statistics or freeze only one side to make the comparison cheap.
Unknown base-family provenance is an explicit exclusion caveat, not discovered
leakage. Do not turn a metadata gap into the main scientific explanation.

## Read Existing Scientific Work Before Proposing More

- docs/research/post_r116_major_direction_20261027/REPORT.md and directions.csv
- docs/research/early_graph_context_followup_20261027/REPORT.md and sources.csv
- docs/research/comprehensive_direction_review_20261025
- docs/research/major_paradigm_review_20261025
- earlier deep_direction_review_20261023 and deep_direction_review_v2_20261023

Read the relevant reports/representation dataflows of the closest precedents
only as needed for scientific distinctions; do not audit code correctness or
recompute numerical results. R47 is trained retrieval with full performance
values, not a weak KNN. R101 is outcome-free hint pretraining, not a failed
inference-time label-bound context model. R102/R106, R109/R112/R113,
R114/R115 and R116 remain genuine counterevidence with their actual scope.
R54/R55/R61 route supervision and R51 probing must not disappear from history.

## Required Scientific Decision, Not Another Inventory

The two unresolved targets are true recovery of missed methods AND joint
whole18 Top1/cost improvement. Avoid a ReLD binary, a TSP-only rescue, quotas,
output biases, coefficient grids, or a minor architectural patch.

1. State exactly which hypotheses R117 weakens and which it did not test.
   Do not infer random labels, a Bayes limit, all-pretraining failure or that
   only one bottleneck remains. Do not recycle early transient gains.
2. Reconsider the learning object itself: current inputs, solver-specific
   computation, training supervision, shared-task inductive bias, optimization
   and final decision. Do not assume a bigger network or more process labels
   must help, but do not dismiss these broad dimensions without evidence.
3. Prioritize at most ONE main scientific direction and ONE reserve, or give
   a reasoned no-ready-direction conclusion. It must change something major
   and have a precise computation/knowledge source, closest negative precedent,
   strongest objection and falsifiable decisive contrast. Not a four-arm grid.
4. If considering TRAIN-native process learning, specify the scientific
   decision-relevant state/transition rather than vaguely asking for more
   trajectories. Confront R113 synthetic transition, R54 route auxiliary and
   R115 small-sample high-dimensional response negative; do not jump from
   teacher access to student transfer. Deployment may run only the student's
   own predicted states, never actual query solver traces/costs/routes.
5. If changing cost/winner decision learning, distinguish a genuinely new
   joint object from R39/R71 scalar performance regression, R78 cost-head
   postprocessing, R98 conditional mean gaps and R116 probabilistic rankings.
   Plain CE already targets winner probability; a new probabilistic name is
   not evidence. AUC and recall do not guarantee argmax or cost improvement.
6. If revisiting capacity/task sharing/algorithm-conditioned structure,
   explain the precise computational distinction from dual streams,
   candidate-private late branches, dynamic relation encoders and previous
   capacity attempts. Do not prescribe another layer-count sweep.
7. Inspect original methods and relevant experiments of the few papers that
   actually support the chosen mechanism. Prefer4-6 informative primary works
   over a long citation list; fewer is fine when directly pertinent. Record
   exact versions/sections and what was read. Separate evidence from analogy.
   Do not claim novelty or positive transfer merely because no equivalent
   implementation was found. Existing local paper sources can be reused.

## Immutable Boundaries And Handoff

Whole18 frozen new90k,81000/4500/4500 per problem, native/FP64 labels and legal
pools stay unchanged. R58 remains deferred. Existing external solver weights
and new TRAIN process supervision are allowed scientific knowledge sources;
new solving is user-authorized but must not be started by you. Historical
TRAIN labels may be context with self/base exclusion; query labels and all
VAL/TEST/LIB answers cannot enter forward. No new online rewards or blind-test
claims. TEST has participated in R&D; no retrospective reselection is allowed.

Only TWO24GB RTX3090s are authorized for subsequent implementation, not4090s.
This is a resource reality, not a request for your batch/LR/queue/budget design.
Sol must decide a finite executable protocol after the scientific handoff.
If the only faithful object requires disproportionate pairwise recomputation
or an unsupported local-to-global inference, explicitly say not ready.

Own ONLY REPORT.md,sources.csv,directions.csv in this directory. Write concise
Chinese with depth, roughly6-10PDF pages. Parent creates PDF and publishes the
completed scientific work unchanged. No general literature inventory or
automatic-next-experiment recommendation. Research completion does not finish
the unbounded improvement goal. Report model unavailability honestly.
