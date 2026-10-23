# R104 Reviewer Memory

## Round 01: 2026-10-23

- Contract: one bounded engineering pass, maximum five minutes; minimal CPU
  checks only. Own round01.md and this file. No model/protocol edits, GPU,
  data/weights/TEST-result reads, literature, agent messages or Git mutations.
- Verdict: ready, engineering only; 9/10; no remaining P1/P2 findings in the
  SHA-bound snapshot in round01.md. Same-family/provisional review; actual
  serving-model/reasoning attestation unknown, not attested Sol/max.
- Preserve live/frozen R99/R101/R102. Launch is main-owned and remains subject
  to source readiness and the assigned R99 full-pipeline predecessor gate.
- Locked capacities: A d256/instance4/joint2, 7,288,969 parameters; B
  d128/instance8/joint4, 3,426,057. Four heads, original score/maximin path.
  Fresh R84 seed2 same-shape reuse plus isolated seed104002 for new/resized
  tensors; not equal parameters/FLOPs or identical complete initialization.
- Private R84 train/evaluate globals resolve micro128 for TSP/CVRP, micro512
  for every other task and eval128; logical batch remains1024, row weighted.
  Full18 original stateless R-Drop objective and natural streams are retained.
- Unique strict-min whole18 VAL selection precedes TEST/LIB. Final VAL binding
  verifies model-state/protocol/update/checkpoint identities without rewriting
  capture archives; report readers restore the capture identity view.
- Arrival detail: the earlier goal required a non-null TRAIN checkpoint SHA,
  incompatible with best checkpoints on periodic7200 TRAIN evaluation steps.
  The arriving r104_goal.selected_train_ready handles null captures through
  exact model/protocol/update and official selected-VAL binding. Treat this as
  resolved, not an open issue. Added regression fixture was read. Reviewer ran
  three in-memory CPU assertions: valid null accepted, wrong model/update
  rejected. No full suite rerun and no files written by those assertions.
- Evidence boundary: model9/9 and runner/report7/7 passes are author-reported,
  not independently rerun. No GPU fit, measured efficacy or future success
  claim. Do not turn evidence limitations or cosmetic points into blockers.
