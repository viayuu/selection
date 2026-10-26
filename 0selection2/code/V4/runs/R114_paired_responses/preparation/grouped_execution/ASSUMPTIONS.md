# Grouped Execution Decisions

No GPU authorization. Only R114-owned code/queue artifacts change. The existing
512-base inputs, native adapters, source locks, seeds, pools and labels stay intact.
No large CPU precheck, conversion rerun, model loading or GPU polling is added.

| ID | Decision | Source |
| --- | --- | --- |
| G01 | One process and backend per problem/method, singleton solves only. Reset Python/NumPy/Torch CPU and CUDA RNG to the original task seed before every endpoint. Preserve the original model-init seed and alias-safe confirmed factory. | user |
| G02 | 52 eager deployments may hold loaded weights; their actual equivalence is still unverified. RouteFinder/MoSES 24 deployments have lazy construction inside run, so discard only their local lazy subrunner/cache before each endpoint and retain the native cold path. Never fast-forward a random stream or silently enable warm-cache reuse. | default |
| G03 | This conservative lazy mode still pays 3,072 label subrunner reloads, plus 52 eager loads, rather than claiming 76 total weight loads. It reduces label worker processes to76. Future warm-lazy optimization needs separate evidence/protocol review. | default |
| G04 | Before labels for a problem, all legal methods must supply real independently valid routes. Use each bucket's original/changed pair, seeded replay after intervening solves and two distinct bases; only add a base when existing buckets do not already provide two. One cold original per eager backend checks equivalence. Known cold-lazy RF/MoSES needs no redundant cold reference. Total404 qualification calls. No CPU mock counts as qualification. | user |
| G05 | A missing route, invalid route, missing profile, failed equivalence, timeout or incomplete runtime stops that problem immediately; the remaining methods/endpoints of that problem are not scheduled. Other problems are not defaulted to success or needlessly marked failed. | user |
| G06 | Produce an independent execution/source-lock supplement for grouped topology, with parent cold lock and unchanged native budgets/seeds. Cold reference receipts retain the parent lock; grouped receipts use the supplement. Inputs and old locks are not rewritten. | default |
| G07 | Route availability is SOURCE evidence only: all76 adapters contain a route path, including return_actions=True in RF/MoSES. Runtime-qualified route exports remain0/76 until actual execution. | default |

Checks are restricted to tiny synthetic CPU worker/gate tests plus static source
and existing manifest metadata. No new full-instance CPU validation pass.

Current ready entrypoint is r114_run_queue with grouped_execution/queue_v1;
historical per-task queue counts are superseded. Stopped problems are masked in
released assemblies; independent problems continue. Assembly and the main-owned
read-only CSV/PDF response analysis execute even for partial/interrupted results.
Actual GPU labels and runtime-qualified deployments remain0 until real execution.
