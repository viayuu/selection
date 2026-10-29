# Work State

2026-10-30 06:01 CST: R120 is complete and published. The original goal remains
active: the four principal original18 experiments R121--R124, locked evaluations,
analysis, PDF/CSV delivery and per-version publication are still required.

- Research: a focused original18 scientific supplement is available under
  docs/research/return18_major_routes_20261030. Its ROUTING.json does not verify
  the requested Astra/max execution identity; do not claim verified Astra use.
- R120: one 24-task model, VAL-lock update44640, early stop53280. Original18
  TEST native Top1=71.2321%; new6 any-exact-best Top1=79.1020%. Published commit
  2330198ced45baed6c7c72c95bc77d540cd7b50e; concise Chinese closeout PDF/CSVs
  under Return18_post_R120/R120_closeout. Not a routing breakthrough.
- R121: full18/89 native deployments, 23040 family endpoints and113920 planned
  labels. All89 deployment qualifications are covered, five through an explicit
  CPU input-constructor adapter. The actual CUDA-to-numpy failure was input
  placement after native backends changed the default device, not cost-output
  serialization. Some backends trigger it lazily on the second call. A frozen
  universal adapter applies the input boundary to all13 TSP/ATSP deployments;
  other backends retain the original call. All4 collection lanes are dispatched,
  retaining old live producers and their physical GPU mutexes. At06:01, unique
  contiguous deployment prefixes contain35744/113920 completed labels (31.4%),
  deduplicated across old and adopted records, not summed across copied roots.
  Existing failures are preserved; old benchmark cost/winner tables are unchanged.
  The single universal-root full-pool export and R121 training have NOT started.
- R122: all14400 successful updates complete. Unique VAL lock14400 is infeasible
  under its preset constraints. TEST-only resource metadata adapter preserves
  the trained source, checkpoint and original failure; formal TEST now completed
  with native Top1=68.6235%, actual regret=0.495953%. Same-lock LIB/PDF complete;
  published commit87038492989cd40c4a0ae260628dd00cc20cd64c.
- R123: all7200 successful updates complete; source-record-unseen metaTRAIN is
  10886/task, not an exhaustive content/base-copy certificate. Post-lock TEST
  bank initially failed before inference due redundant path-prefix remapping.
  A TEST-only CLI correction removed that remap, using identical existing inputs
  without training/model/reselection changes. Unique VAL lock1440: VAL native
  Top1=71.0605%, actual regret=0.403577%; TEST=71.3000%, regret=0.394871%.
  Actual TRAIN capture, same-lock LIB and live timing are complete. CPU-only
  NSS-schema normalization fixed the report without changing predictions or
  repeating GPU stages. The28-page PDF and CSVs are complete and ready to publish.
  The reused TSPLIB capture's lost original timing remains UNKNOWN.
- R124: all7200 successful updates and formal TEST complete, unique VAL lock5760.
  VAL native Top1=71.0185%; TEST=71.3284%, actual regret=0.396246%. TEST879
  corrections/823 harms against its frozen first stage: net56, not a breakthrough.
  Same-lock LIB/PDF complete; published commit
  b590ab34ef122f1a76730851bfdf0199611ae558.
- Source-model bank is R77B/R80A/R83, fixed by representation/objective diversity,
  with real complete metaTRAIN/VAL predictions. No current query labels enter
  forward. R58 remains deferred; existing frozen90k candidate rules remain.
- Controllers, original failures and exact resource plans under dispatch/ are
  evidence of attempts, not substitutes for completed TEST/LIB/report artifacts.
