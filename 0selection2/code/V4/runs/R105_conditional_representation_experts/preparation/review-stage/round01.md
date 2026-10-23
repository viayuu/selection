# R105 Engineering Readiness Review - Round 01

Date: 2026-10-23 (Asia/Shanghai).

## Judgment And Findings

**Engineering readiness: 9/10. Verdict: ready, provisionally.** This judges
the frozen implementation and its engineering handoff, not scientific merit,
efficacy, GPU feasibility, or permission to start work. Main still owns the
existing resource-witness, final readiness/source-freeze and launch decisions.
This review does not create a production readiness receipt.

| Priority | Result | Blocking status |
| --- | --- | --- |
| P1 | No actual finding identified. | None. |
| P2 | No actual finding identified. | None. |
| P3 | No actual finding identified. | None. |

No corrective implementation change or fix-check is requested. Runtime evidence
limits below are not relabeled as bugs. No additional parameter, objective,
model search, external research, SHA framework, or duplicate preflight is requested.

## Review Identity And Scope

- `requested_reviewer`: `gpt-6.1-sol`.
- `requested_reasoning_effort`: `max`.
- `actual_backend_attestation`: unavailable through this session; the requested
  model/effort is not presented as independently verified.
- `review_independence`: `same-family`.
- `acceptance_status`: `provisional`.
- No Astra, secondary agent, or spawned review was used.
- One initial bounded round; zero fix-check rounds used.
- Ownership is limited to this file and adjacent `REVIEWER_MEMORY.md`.

The current Git diff/status was inspected first. The six R105 implementation
files and five test files are untracked additions, so a tracked-only diff does
not contain their implementation. They were read directly in full. Unrelated
dirty work was left alone.

Primary contract: `PLAN.md` and `protocol_lock.json` in this experiment root.
Also read `MODEL_PREP.md`, `RUNNER_PREP.md`, `preparation/LAUNCH_PLAN.md`, and
`run_v4_r105_launch.sh`. Historical sources were inspected only along concrete
integration edges: original R84 training/evaluation, R88 roles, R76 objective,
private reporting adapters, and inherited dispatch/completion helpers.

The launcher owner's final-freeze notice and 23/23 CPU-test result were received
before reviewing the final launch/goal/wave sources. Main reports R104 completed
and pushed at `cb7afd2a6`; this is supplied operational context, not an independent
Git/remote audit performed in this round.

## Concrete Checks

Paths below are relative to `code/V4/` unless explicitly experiment-relative.

| Area | Evidence and conclusion |
| --- | --- |
| Actual private R84 integration | `r105_experiment.py:298` creates a private engine and rebinds its own `micro_caps` and JSON writer. `train` at line 332 supplies the R105 integration module. `r84_experiment.py:330` resolves initializer, protocol, objective, update, checkpoint and evaluation through those hooks; its direct global cap lookup therefore also uses the R105 caps. This is an actual execution path, not just descriptive configuration. |
| Model/control | `r105_model.py:75` replaces only instance-layer FFNs with four separate original FFNs and a dense FP32 softmax mixture. A feeds zeros and learns biases; B feeds the current post-attention state. All experts execute in both. `r105_model.py:107` retains the inherited R88B role/score/maximin path. Total count is 3,425,561 per arm, explicitly not equal active functional freedom: A's 2,048 router feature weights have zero gradients. |
| Fresh state and restoration | `r105_model.py:193` regenerates the exact seed2 original reference, maps original FFNs to expert0, initializes additional experts under isolated seed105002, and strictly restores caller state when requested. `initialize_shared` at line 252 accepts the original hashed scratch record and installs geometry buffers. `r105_experiment.py:103` compares complete cross-arm states, not counts alone. Its loader at line 221 validates config and exact full restored state, including buffers. No trained checkpoint is used for fresh initialization. |
| Objective and logical batch | `r105_experiment.py:123` returns the original stateless `RDropObjective`, not focal/teacher/auxiliary machinery. Its update hook at line 141 enforces logical1024 and 128/512 caps. `r84_experiment.py:83` weights each microbatch by its true row count and clips/steps once. The original bounded same-batch/RNG nonfinite-AMP replay is retained; it is not a terminal job retry or OOM cap search. |
| Budget and selection | `r105_experiment.py:60` and line 185 enforce the main lock and checkpoint protocol. Original R84 train uses 57,600 successful updates, warmup4,320, full VAL every1,440, full TRAIN every fifth VAL point (7,200), original AdamW/LR schedule and strict lower full18 VAL macro_vs_sbs selection. Exact ties retain the first checkpoint (`r84_experiment.py:462`). The selected model is restored and uniquely locked before TEST. |
| Label boundary | `r76_experiment.py:157` performs the original two forwards on `inputs_only(batch)`; the R88 role path also removes supervision. The new model/router uses node states, not winner/cost/target fields. `r105_experiment.py:343` has an explicit RAW-input allowlist; the routing reader at line 349 opens only frozen input arrays, never supervision arrays. It groups valid-physical-node routing means by exact real N on first128 VAL/task and names graph-mean entropy accurately. |
| Capture-to-checkpoint binding | `r105_experiment.py:237` checks the actual checkpoint file, complete model state, protocol state, selected update, arm/split and archived identity. The writer at line 283 binds the official selected VAL summary without rewriting the capture identity/archive. `r105_reporting.py:59` applies the same proof to selected TRAIN, so a capture-time null file SHA is not an unqualified bypass. `r105_goal.py:83` additionally requires the official VAL binding and unchanged selected archive identity. |
| Real postprocessor factory | `r105_reporting.py:91` calls the actual per-arm R105 loader and checks VAL/TEST model-state identity. LIB at line 260 passes `code.V4.r105_experiment` explicitly to the inherited LIB adapter, which restores the complete model/buffers. The private reporter copies at line 45 are rebound locally; historical/live reporter module globals are not replaced. |
| Pipeline/idempotence | `r105_launch.py:91` fixes train -> TEST -> timing -> LIB2 -> CPU PDF. Original train already completes locked full TEST; the explicit test stage reuses its saved same-lock result (`r84_experiment.py:573`), and R105 routing is idempotent (`r105_experiment.py:432`). Timing also validates/reuses an existing receipt. CPU reporting consumes saved predictions and can deterministically regenerate derived outputs, without reselection or automatic failed-stage retries. |
| Dispatch/resource boundaries | `r105_launch.py:111` fixes canonical data/prepared/cache paths, one original3090 slot/UUID per arm, exact CPU quartets and mem9GiB. Its cache stage at line 185 fails rather than copy/rebuild/repair. The predecessor gate at line 199 checks assigned original R104 evidence and worker/controller rc0. Private transport copies preserve existing mutexes/resources; neither R1024090 worker is reconfigured. Explicit main evidence and CPU-test confirmation precede prepare/start; exact source-set/content changes block later readiness checks. |
| Failure/completion/publication | Inherited `r84_launch.py:418` stops at the first nonzero stage. R105 queue/start/watch/compare are one-shot, not hidden restarts. `r105_goal.py:160` and line 204 require both complete same-lock pipelines, full TRAIN/VAL/TEST, routing/timing, native LIBs and five PDFs before whole-version saved comparison. Publication is explicit and curated (`r105_goal.py:334`), not an early/per-arm push. Pipeline completion leaves research-goal success unresolved. |
| Required delivery | `r105_reporting.py:352` retains five required PDFs per arm, saved NSS4 comparisons, full/per-task metrics with TRAIN-SBS and native legal-pool Oracle, task-equal ALL, same-axis choice/native-winner distributions, TRAIN diagnostics and fixed36 recovery/false-pick/net-cost tables. Historical R89A/B/R84A are saved references, not retrained controls; the B-versus-A comparison uses both locked whole models. |

## Tests And Evidence Limits

| Suite | Existing result | Reviewer evidence |
| --- | --- | --- |
| `test_r105_model.py` | 14/14 passed, model-owner handoff. | Read all 14 tests: exact tensors/counts, all18 initial-function/dropout parity, all-expert/router gradients, masking/capture, poisoned-label isolation, buffers/restoration, CPU autocast and RNG isolation. |
| `test_r105_experiment.py` + `test_r105_reporting.py` | 15/15 passed; main reports 29/29 combined with model tests. | Read all 15 tests: actual private-global cap binding, real trained-state factory, complete fresh equality, row weighting/replay, VAL/TRAIN proof, RAW-cache allowlist, real capture API, private adapters, explicit LIB factory, metadata and recovery arithmetic. |
| `test_r105_launch.py` + `test_r105_goal.py` | Final 23/23 passed in 2.314s, supplied by launcher owner/main. | Read all 23 tests: final main lock/real parsers, private resources, fixed commands, source freeze, R104 predecessor/failure gates, no-cache-repair, null-TRAIN proof, routing masks/exact-N, quiet one-shot comparison/publication and curated assets. Main also reports acceptance of both actual R104 predecessor receipts. |

Total supplied CPU evidence: **52/52**. These are producer-reported results,
with test bodies and call paths inspected in this review; they are not a new
independent execution claimed by the reviewer. No test suite was rerun and no
extra preflight was added.

This round did not use GPU/Slurm, launch jobs, inspect/run real datasets or
solvers, run training/TEST/LIB/timing/PDF production, commit/push, or spawn
another agent. Every shell command sourced conda and activated `easynco`.

CPU fixtures and source tracing do not prove CUDA FP16 behavior at maximum
frozen input sizes, GPU/host memory fit, wall time, complete production report
rendering, or predictive improvement. Those remain actual-execution evidence
for main's already-planned pipeline, not new audits requested by this review.
No efficacy claim is made. The 9/10 score deliberately does not represent
verified real-GPU end-to-end completion.

## Memory Update

Created adjacent `REVIEWER_MEMORY.md` with the bounded decision, concrete
integration anchors, supplied-test evidence status, runtime limitations, and
remaining one-fix-check allowance. No global Codex memory or other file was edited.
