# R106: Native encoder adaptation

Status: scientific protocol locked after bounded hardware evidence, not yet
launched. Main owns actual launch; engineering readiness is separate.

## Question

Does final selection supervision become more useful when it can update the
native pretrained solver representations, rather than only reading frozen
representations? This is not a new encoder family, extra supervision, a solver
execution experiment, or a promise of suppressed-winner recovery.

Astra's papers-only screen retains this conceptual distinction but DEFERs an
efficacy recommendation. Main/Sol chooses one falsifiable matched comparison;
idle hardware is not evidence of likely benefit. R105 remains untouched.

## Matched arms

- A_frozen: all three native pretrained encoders frozen; train the original
  full18 selector and the same three fusion projections.
- B_finetune: identical complete initial tensors, but allow the final original
  selection loss to update those same native encoder parameters.
- Both start from fresh R84 seed2 selector tensors, not a trained selector.
  Native weights are the original pinned R102-B ELG, MatNet and ReLD weights.
  The two arms have identical total tensors but different trainable freedom
  and training compute. External pretraining data and overlap remain unknown.

The raw-input/fusion/instance4/joint2/ID34/utility/maximin architecture remains
the R102 family. Donors execute online on both arms for TRAIN, VAL, TEST and
LIB. No cached nco feature may bypass the donor. Their output stays FP32 until
the normal fusion/selector precision path. Both native networks remain in eval
mode; eval does not disable B's parameter gradients. No decoder is present.
The deterministic native graph may be shared by the two dropout selector
forwards within one microbatch, with one combined backward. A scoped input and
parameter-state signature prevents reuse across batches or optimizer updates;
the scope clears on exit. Both arms use this same computation optimization.

To avoid thousands of tiny exact-N calls, any batched padding adapter must
mask attention and node-wise normalization correctly and preserve physical
node identities. It must not change weights, native roles, inputs, or valid
node semantics. Record its numerical agreement with the original exact-N
forward on a small necessary fixture; never claim bitwise equivalence. Both
arms use the same adapter. This is an explicitly adapted execution, not an
assertion that unread original implementations accept padding.

## Fixed training and evaluation

- Frozen NEW90k release: 81,000 TRAIN / 4,500 VAL / 4,500 TEST per task,
  all18 tasks and current filtered legal pools. No new split, labels, solver
  costs, routes, R58 repair, selection quota or current-query target input.
- Seed2; logical batch1024; identical natural task and row sequence.
- Original CE0.35 + comparison0.35 + relative risk0.02 + R-Drop ramp0.35;
  two dropout forwards, dropout0.1, original example-weight normalization.
- 57,600 successful updates; warmup4,320; VAL every1,440; TRAIN every7,200.
  AdamW common peak1e-4, end1e-6, weight decay1e-4, global gradient clip1.
- B native encoder effective learning rate is exactly0.1 times the common
  update-axis rate, including warmup and cosine. A's same group is frozen.
  This is one fixed transfer setting, not a coefficient or schedule search.
- Final common microcaps: TSP/CVRP256, ATSP128, other tasks1024; eval128.
  The initial smaller-cap witness and one final larger-cap witness both passed.
  Largest TRAIN input peaks were TSP18,655MiB, CVRP21,519MiB, ATSP19,144MiB,
  VRPBLTW14,740MiB, with one native call for two dropout forwards and finite
  nonzero native/core gradients. No optimizer step or TEST/LIB input was used.
  These hardware choices do not alter logical batch1024 or success budgets.
- Strict minimum whole18 VAL macro_vs_sbs_pct, first exact tie, common frozen
  full-TRAIN SBS; unique whole checkpoint then full TEST and native LIB.
  No TEST/LIB tuning, per-problem checkpoint assembly, or automatic retries.

The run protocol must distinguish frozen A trainables from total parameters,
and B's native versus selector update scales. Native parameter gradients,
state changes and source identities are necessary implementation witnesses,
not evidence of selection effectiveness.

## Delivery

Use the two free4090 UUIDs inside existing job1475/gpu05, four CPUs and9GiB
per process, disjoint bindings; no new allocation or ordinary SSH training.
Reuse frozen raw R84 cache and TRAIN-only statistics in place. No donor cache
generation, source-data rewrite or native solver labels. R105's two3090s are
not stopped or modified.

Report same-lock TRAIN/VAL/TEST, full18 per-problem cost/SBS/Oracle/regret,
fixed36 true recovery, false picks and net cost; NSS four datasets and native
LIB49/100. PDFs put choices and native/FP64 winners on the same axes with true
model Top1 titles. Time the real online donor-plus-selector call for all18;
do not present cache-read latency as total inference. Compare A/B and historical
R94B/R102B without claiming equal pretraining or historical compute controls.
Publish the complete version once; retain any failures unchanged.
