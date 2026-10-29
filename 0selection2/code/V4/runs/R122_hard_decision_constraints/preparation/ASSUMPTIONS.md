# R122 D2 Implementation Decisions

Scope: only the new R122 runner, its CPU tests, and this preparation directory.
No solver calls, GPU execution, Git operations, or historical edits are authorized.
Base commit: not queried (the user explicitly prohibited Git operations).

| ID | Decision | Source |
| --- | --- | --- |
| A1 | Single group `model`; full18 unchanged R45A/R84A ScoreDifferenceSelector, scratch seed2. | user |
| A2 | TRAIN constraints use equal-task hard actual regret and unconditional fixed36 missed frequency, not conditional recall or selection quotas. | user / HANDOFF D2 |
| A3 | Fixed36 is the historical target file, indexed by task and deployment name using the native FP64 winner. | user |
| A4 | TRAIN budgets come only from the saved selected R84A TRAIN predictions. Separate VAL reference thresholds are selection-only. | user |
| A5 | Controller defaults: risk unit .01, dual learning rate .05, cap 20, EMA .9, initial multipliers 1; one equal-task feedback cycle per 18 successful updates. | existing parent objective defaults |
| A6 | VAL feasibility is componentwise <= reference (no slack). Fallback is the sum of positive violations divided by each reference threshold (floor 1e-12), then native Top1, actual regret, earliest update. | default, predeclared |
| A7 | Fixed microbatch caps: 256 TSP/CVRP, 1024 others; eval cap 256. No OOM-driven cap change. | default, implementation only |
| A8 | Reuse `prepared.json` cache_source, not the stale historical /tmp path; lock its current resolved path and TRAIN/VAL label identities. | verified local paths |
| A9 | `train` finishes and locks without auto-launching TEST. `test` is a separate explicit command after the complete budget and unique lock. Existing complete R84 TEST cache only. | interface default / user lock requirement |
| A10 | Checkpoints include controller buffers, optimizer, scaler, source cursors, RNG and pending VAL. Resume replays only work after a saved interval boundary. | user / existing R84 resume semantics |

## Acceptance

CPU-only describe, focused unit tests, and prepare must run. Prepare hashes and
binds reference protocol, checkpoint/model state, selected TRAIN/VAL captures,
native labels, source rows, candidate pools, fixed target list and source code.
It performs no network access, model forward, optimizer step or solver run.
Scientific effectiveness and GPU execution remain outside this implementation.
