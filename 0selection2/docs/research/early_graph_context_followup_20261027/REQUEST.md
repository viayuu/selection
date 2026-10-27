# Focused Scientific Follow-Up: Early Graph Context

Research only, authored by gpt-6-astra with reasoning_effort=max. Sol remains
responsible for all experiment plans, budgets, protocols, code, numerical
validation, execution, monitoring and delivery. Do not take over those roles.

## Current Status And Scope

The completed scientific report is:
`docs/research/post_r116_major_direction_20261027/REPORT.md`, published as
6146fcaf5f227fb2309356c600aebff88273e5b4 with a 10-page PDF and15 primary sources.
Read its section7, the coverage table, strongest counterarguments, and sources.
The two previous broad reports are `comprehensive_direction_review_20261025`
and `major_paradigm_review_20261025`; the deeper earlier reports are
`deep_direction_review_20261023` and `deep_direction_review_v2_20261023`.

R117 is RUNNING on the only two authorized RTX3090s. Its finite matched
comparison uses shared pretrained ReLD static nodes plus random versus
pretrained native LEHD/ReLD/MatNet reference computation. It has NOT completed
its budget, final checkpoint lock, TEST, LIB or zero-choice analysis. Do not
treat interim scores as final evidence or infer that this direction failed.
Do not stop, modify, monitor, launch, or otherwise operate GPU experiments.

This request is a parallel scientific deepening of the report's second
direction, NOT approval to launch another experiment. Current R117 is the
sole GPU experiment. Completion of this paper research does not complete the
overall zero-selection/whole18 improvement goal.

## The Main Question

Can TRAIN-label-bound context influence the formation of fine-grained query
graph structure, in a way materially different from the existing negative
retrieval and tabular ICL attempts, when the solver performance functions and
data scenario are fixed? Explain why episodically partitioning one dataset
does or does not provide the needed learning signal; do not pretend arbitrary
episodes create new independent functions.

Only inference inputs and known deployment conditions are available for the
query. Historical TRAIN features, native winner and full relative cost vectors
may be support context. TRAIN queries must exclude themselves and associated
base-instance copies. VAL/TEST/LIB labels or query routes/costs cannot enter
context. There is no online reward feedback or unseen-test-label adaptation.
Maintain all18 problems and their frozen legal pools, not a ReLD binary or TSP
expert. R58 repair remains deferred. Context is not a quota or static bias.

## Required Scientific Work

1. Read original methods/assumptions/experiments, not only abstracts. Prior
   TNP, PRODIGY and CAGML coverage is already explicit, with serious caveats.
   Seek at least four additional informative primary works if available;
   do not pad the bibliography or force favorable conclusions. Record exact
   URLs, versions, sections actually read and transfer limitations.
2. Distinguish fixed-function nonparametric learning, episodic meta-learning,
   learned similarity, label-conditioned representations, early graph
   conditioning and ordinary late retrieval. Which object is actually new?
3. Confront R47 and the R97/R98/R100/R101 attempts. Use their reports or code
   only to understand the scientific learning objects; do not certify code
   correctness or recompute results. Identify the closest negative precedent.
4. Discuss how a support graph's graph-level label/cost vector can inform
   query nodes or directed relationships without natural cross-graph node
   correspondence. Avoid text encoders, arbitrary node-index alignment, or
   calling a pooled context residual an early structural mechanism.
5. Explain the strongest reason this might still fail: absent function
   variation, nonsmooth winner boundaries, geometry/advantage mismatch,
   duplicated priors, leakage, memorization, or an information bottleneck.
   State what the papers establish and what remains only a hypothesis here.
6. Give at most ONE main scientific formulation, ONE strong counterargument,
   and ONE decisive scientific contrast against a genuinely strong late
   context control with equivalent support information. No A/B/C matrix, no
   batch/lr/step budget, no launcher, no source changes, no actual experiments.
   If evidence is insufficient, say so rather than recommending by default.

## Delivery And Ownership

Own only REPORT.md, sources.csv and directions.csv in this directory; UTF8 LF.
Write concise Chinese, enough depth for roughly8-10 PDF pages rather than a
new giant literature inventory. Link original sources next to claims. Include
an explicit knowledge-source and failure boundary, and a clear handoff of
all finite operational decisions to Sol. Parent will mechanically render PDF
and publish this completed research separately. No commits or external task
messages. No new sub-agents. No claims of using a model other than the one
actually available; report inability instead of silently substituting.
