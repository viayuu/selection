# R41 solver label stability

Sampling was locked before test. Train/val remain separate. Only complete original-config candidate pools enter winner statistics.
The raw NPZ keeps missing/unexecuted values as NaN. No minima across repeats replace historical labels.
A third scenario may reuse a deterministic cost only after two independent runs return exactly the same value. Observed and reused counts are separate; this is not a third solver invocation.

| Problem | Split | Complete / requested | Strict winner consistency | Numerical tie consistency | Old winner changed | Label regret | R39A regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | train | 0/64 | not measured | not measured | not measured | not measured% | not measured% |
| TSP | val | 0/64 | not measured | not measured | not measured | not measured% | not measured% |
| OVRPTW | train | 0/64 | not measured | not measured | not measured | not measured% | not measured% |
| OVRPTW | val | 0/64 | not measured | not measured | not measured | not measured% | not measured% |

Some full-pool audits are blocked. Do not infer label noise or stable winners from incomplete candidates.
See `solver_manifest.json` and `blocked_dependencies.md` for concrete provenance/dependency gaps.
