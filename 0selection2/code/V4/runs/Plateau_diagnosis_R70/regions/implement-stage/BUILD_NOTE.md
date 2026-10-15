# Build Note

| Rung | Feature | Acceptance check | Tier | Status |
|---|---|---|---|---|
| F0 | Real FP64 loader to original-order inventory | `python -B -m code.V4.plateau_regions smoke` | MUST | passed |
| F1 | Bounded features, train-only bins/priors, LIB shift | `python -B -m code.V4.plateau_regions self-test` | MUST | pending |
| F2 | Full diagnostic run, rules/pairs/independence and replay | `python -B -m code.V4.plateau_regions verify` | MUST | pending |

## Run Record

Environment: easynco; CUDA hidden; at most 4 CPU affinity slots and numeric threads. No commits by explicit user instruction.

F0: real loader smoke exit 0; all 18 training label pools validated against registry. Artifact: loader_smoke.json. No stubs used.

## Deferred

Formal model checkpoint is unnecessary. Main agent owns model/error/permutation comparisons.

## Blockers

None at start. Cross-agent review availability will be reported, not simulated.
