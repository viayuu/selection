# Full18 Useful Method Recovery: R89

The original research goal remains active and unresolved. R87's matched
private residual did not improve the joint VAL objective or materially recover
the fixed missed-winner set. Both versions still suppress well-supported
methods on TRAIN. This does not isolate a cause or prove an accuracy ceiling.

The bounded paper handoff declined an output-confidence penalty and
conditionally recommended an optimizer-geometry study. R89 tests that distinct
mechanism, not another output bias, loss coefficient or scoring architecture.
The fixed protocol is `protocol_lock.json`; the SD discussion is a declined
proposal, not an experiment performed under this version.

Both arms use the same complete original selector, same natural R77 objective,
same fresh initialization, same full18 data and logical batch1024. Both expose
the already-known physical TSP start role using the R88 execution input rule;
that input choice is based on deployment records, not R88 interim performance.
R88 itself continues untouched on the two3090s.

A is actual standard AdamW. B uses Muon only for explicitly named hidden
Linear matrices and actual AdamW for every other parameter. It adopts the
Moonshot RMS factor, not a mixture of incompatible scaling conventions.
Both use the same base LR/WD and update schedule, with no search. The proposed
relative importance of weak update directions is a hypothesis, not evidence
that rare winner information necessarily occupies those directions. Noise can
also be amplified; zero or missing information cannot be created by an optimizer.

All logical-batch gradients must be accumulated before the nonlinear Muon
transformation. AMP-failed attempts cannot advance either optimizer's state.
Runtime evidence must show the actual optimizer partition and matrix steps,
not merely a configuration label. Checkpoint/resume includes the complete
optimizer state. The shared environment, global Torch optimizer and frozen
source modules used by R88 must not change.

Use only the predeclared full18 VAL rule to lock each model, then produce full
TEST/LIB PDFs, NSS comparisons and task metrics. Natural Top1, actual costs,
true fixed-target recovery and false picks decide utility; a finite optimizer,
larger TRAIN accuracy or fewer zero groups does not complete the research goal.
Existing TEST/LIB have influenced research and are not blind confirmation.

Use the two released, existing allocation-owned4090s. No solver run, new cost
label, dataset rewrite, frequency quota, environment upgrade or current-job
restart is part of this study. Sol owns protocol/code/review/execution/results;
Astra's scope remains primary-paper research and improvement ideas only.
