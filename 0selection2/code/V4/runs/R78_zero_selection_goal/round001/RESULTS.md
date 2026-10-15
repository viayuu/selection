# R78 Round001 Results

Candidate screen: **negative; no certified improvement**. Numerical fit:
**inconclusive**. Identity is retained; TEST was not opened. This round is not
completion of the continuing R78 goal, and does not reject all converged offsets.

R76 epoch18, checkpoint SHA256
`5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9`.
CPU only, 2 threads, seed2, exact batch1024; no new backbone/solver inference.
TRAIN-only natural CE intercepts `z+b`, sum-zero, fixed `||b||^2/(2*N)` shrinkage.
One complete18-task candidate, no alpha/temperature or taskwise rule search.

| Split | Rule | Macro Top1 % | Macro CE | Macro mean cost | Macro actual regret % | Macro VS TRAIN SBS % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | Identity | 69.369646 | 0.752703080 | 11.373004840 | 0.460650092 | -0.143845417 |
| TRAIN | Intercepts | 69.425982 | 0.743902973 | 11.373099773 | 0.462689801 | -0.139222429 |
| VAL | Identity | 69.236913 | 0.755865163 | 11.402474922 | 0.464503404 | -0.142069159 |
| VAL | Intercepts | 69.379546 | 0.746502800 | 11.402575925 | 0.467179029 | -0.137179498 |

All metrics equally weight the 18 fixed tasks. CE uses FP64 score arithmetic.
VAL Top1 improved by 0.142633 percentage points, but mean cost increased by
0.000101003 and actual regret by 0.002675625 percentage points. Both violate
the empirical zero-degradation gate. Macro VS is reporting-only. No candidate
was promoted and no TEST-based fitting, tuning or rule selection occurred.

All tasks used 30000 updates (540000 total); only OVRPBTW met the predeclared
two-check convergence rule. The original `val_decision.json` and `complete.json`
therefore retain `inconclusive`; the negative candidate screen does not certify
the fully converged CE optimum's performance. TRAIN/VAL zero-selected pairs
with true winner support stayed 55/54 for both identity and the candidate.
Taskwise adverse changes included TSP, ATSP and OVRPL. The paired intervals and
numerical boundary are documented in [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md).

Evidence: [TRAIN](train_comparison.json), [VAL](val_comparison.json),
[decision](val_decision.json), [frozen vector](train_locked_candidate.json),
[summary](result_summary.json), [summary schema](result_schema.json).
Original fits, scores, comparisons and decisions are preserved unchanged.
The main coordinator reports forwarding the TRAIN/VAL evidence to Astra.

The file-based goal remains active while awaiting the next plan. The child app
goal's historical blocked flag is recorded as workflow waiting, not completion
or a claim of permanent resource failure; no fake resume is claimed. The old
large app goal remains untouched. Later explicitly user-authorized all18-task
training based on an Astra plan is permitted: only **unplanned** broad retraining
is disallowed. Sol owns execution/checks, Astra the next research idea.
