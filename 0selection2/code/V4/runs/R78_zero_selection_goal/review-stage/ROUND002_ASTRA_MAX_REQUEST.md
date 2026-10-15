# R78 Round002: Research Decision Request

Requested reviewer: gpt-6-astra / max. Research/results critique only.
No code review, source reading, tests, training, inference, GPU, solver calls,
Git operations, STATUS.json writes, or application-goal operations. Old app
goal stays paused; the separate persistent R78 research goal stays active.

The user asks for exactly ONE concrete next mainline after round001, preferably
CPU reuse of frozen scores, with seed2 and actual training batch1024. No
coefficient/temperature grid, TEST input, forced coverage, taskwise cherry-pick,
or extra literature collection. You may recheck existing primary references.

## Allowed Local Reading

Only read this request and these round001 evidence reports, never linked code
or TEST files. All paths have prefix:
`/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/`.

- round001/RESEARCH_HANDOFF.md
- round001/protocol.json
- round001/val_decision.json
- round001/train_comparison.json and round001/val_comparison.json
- round001/changed_decisions_train.csv and round001/changed_decisions_val.csv
- round001/methods_train.csv and round001/methods_val.csv
- round001/fit/TSP.json, round001/fit/CVRP.json, round001/fit/OVRPL.json

Reports only. `jq`, `sed`, and `rg` limited to these reports are sufficient.
Do not read raw NPZ arrays, project implementations, code-review reports, or
additional experiment artifacts. Initialize conda using its shell hook and
activate easynco before any shell command. Do not edit files; return final text.

## Confirmed Facts

Base is locked R76 epoch18, pure-difference maximin scores. Round001 fits only
sum-zero per-problem intercepts `s=z+b`, `K-1` degrees per problem, natural TRAIN
frequency, objective mean winner CE + `||b||^2/(2*N)`, seed2, batch1024.
Identity is retained after an inconclusive, gate-failing result; TEST unopened.

All18 tasks exhausted 30000 updates each. Only one task satisfied the required
two-successive-check convergence rule (gradient infinity norm <=1e-5 and
objective change <=1e-7). Many final gradients are near that bound. Do not call
the exact CE optimum disproved, or infer missing information from non-convergence.

TRAIN: Top1 69.369646 -> 69.425982%, CE .752703080 -> .743902973;
actual regret .460650092 -> .462689801%, raw macro mean cost
11.373004840 -> 11.373099773, VS TRAIN SBS -.143845417 -> -.139222429%.
VAL: Top1 69.236913 -> 69.379546%, CE .755865163 -> .746502800;
actual regret .464503404 -> .467179029%, raw macro mean cost
11.402474922 -> 11.402575925, VS TRAIN SBS -.142069159 -> -.137179498%.

VAL paired candidate-minus-identity intervals: Top1 pp [.046998,.239695];
actual regret pp [-.000302806,.005706569]; VS TRAIN SBS pp
[.002239886,.007494755]. Adaptive VAL history prevents independent confirmation.

Zero-pick/positive-winner pair counts unchanged: TRAIN55, VAL54. But membership
changes: CVRP BQ picks0->32 while CVRP ELG picks281->0 on VAL. No quota is wanted.
Twelve tasks have no decision changes. Six tasks changed decisions:

| Task | VAL changes | Corrected / broken Top1 | Sum regret changes, pp over affected rows |
|---|---:|---:|---:|
| TSP | 82 | 26 / 18 | -0.154245817 |
| CVRP | 565 | 105 / 66 | -260.347914674 |
| ATSP | 302 | 160 / 125 | +154.418274477 |
| OVRP | 170 | 80 / 51 | -33.103641401 |
| VRPL | 481 | 160 / 132 | -33.756888653 |
| OVRPL | 755 | 209 / 248 | +360.532480445 |

ATSP and OVRPL also have increased TRAIN regret contributions. This supports
investigating how winner-CE decisions use cost magnitudes, but does NOT implicate
the old backbone's .02 risk term or authorize its deletion.

VAL TSP LEHD: identity mean probability .163229 vs winner rate .164570,
zero picks; mean margin on wins -.809682, on losses -.973858. Offsets yield mean
probability .169282, still zero picks. Means do not establish score separability.
TR/VAL marginals are not conditional probability calibration or information bounds.

There is a DOCUMENTED protocol distinction, not a code-review finding: the
original research proposal gated actual regret and VS-TRAIN-SBS alongside Top1;
the executed round001 protocol gates raw macro mean cost and actual regret and
labels VS-SBS reporting-only. Round001 fails all these cost measures, so the
rejection is unchanged. The next declaration must explicitly freeze the gate,
not retroactively relabel round001. Raw task costs have different scales.

## Decision To Review

Choose one next action; do not return a menu or propose concurrent experiments.
Decide whether numerical completion of the same CE fit must precede any new
method, or whether its still-inconclusive status can remain open while a
separate cost-aware decision-objective screen has higher value. Do not silently
relax convergence after VAL exposure or equate 1024-row I/O with batch1024 updates.

The parent's tentative research direction, if moving beyond CE, is to keep the
SAME intercept capacity and frozen score cache, change only the fitting target
to an explicitly cost-sensitive surrogate using TRAIN per-candidate relative
regrets, and retain an unchanged VAL Top1/cost acceptance rule. No backbone loss
change, no larger calibration matrix, no CE/risk mixture-weight search.

Existing primary P7, Avila Pires et al. ICML2013:
https://proceedings.mlr.press/v28/avilapires13.pdf
Sec.2 Eq.1-3 and Appendix C discuss nonnegative instance/action costs and
`sum_j c_j*phi(s_j)` on sum-zero scores. Examples include squared, exponential,
and logistic losses. The unrestricted calibration theory does NOT guarantee
success for frozen offsets or the joint accuracy/cost gate.

For a precise cheap hypothesis, examine the squared member with centered
frozen scores `h=z-mean(z)`, `s=h+b`, sum(b)=0, and TRAIN target
`r_j=(C_j-min C)/min C`. A single fixed task-wide scale (e.g. mean TRAIN r over
all instances/candidates, not row-dependent scaling) could normalize units.
Then minimize `mean_x mean_j (r_j/scale_t)*(1+h_j+b_j)^2 + ||b||^2/(2*N)`.
This is a small strongly convex quadratic, still rank-changing, with a
numerically auditable optimum; it could use seed2 minibatches1024. The point
is to test cost supervision within the same decision class, not to promise a
Top1 gain or a root-cause diagnosis. Reject this direction if its arbitrary
score-scale dependence or goal mismatch outweighs its value, and select a
better SINGLE action with exact objective and finite training protocol instead.

Address: positive cost tails, native Top1 versus expected regret, all18 scope,
convergence and coefficient selection boundaries, actual batch1024 compliance,
TASK/VALUE aggregation, why this is preferable to a scalar-temperature or
prior-weight sweep, and what outcomes would and would not support.

Return no more than 1200 words: findings first, then ONE recommended next
mainline with equations and a sufficient TRAIN/VAL declaration. Clearly label
assumptions and unsupported conclusions. Record actual review as same-family
and any positive assessment as provisional. No additional paper list is needed.
