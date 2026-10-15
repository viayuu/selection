# R76: Priors, Instance Signal, and Optimization

This is a post-hoc interpretation of existing predictions and training labels.
No neural network was trained or selected for this analysis. No solver was run.
All new-scenario TEST metrics cover 18 tasks with 3,895 rows each (70,110 total).
All methods below use the existing filtered pools and the fixed MOEL deployment
named reld_mt. Scores and costs are not combined across different deployments.

## TRAIN-Fitted Statistical References

| Policy | TRAIN rows per problem | TEST Top1 | TEST actual regret |
|---|---:|---:|---:|
| Per-problem majority winner | 70,114 | 66.727999% | 0.669031% |
| Per-problem, exact-size majority winner | 10,000 | 68.844673% | 0.515382% |
| Per-problem, exact-size majority winner | 70,114 | 69.215518% | 0.511213% |
| Per-problem, exact-size minimum mean relative gap | 70,114 | 68.556554% | 0.437735% |
| R76, locked epoch18 | 70,114 | 69.439452% | 0.464144% |

The first row is a majority-winner reference, not minimum-mean-cost SBS.
The size-conditioned references can choose different methods at different
sizes; they are not single-best-solver policies.

All lookup tables are fitted on TRAIN only. Within each problem and exact
actual size, count native FP64-derived winners and choose the largest count,
breaking ties by stable candidate order. The cost-oriented table chooses the
smallest TRAIN mean relative gap within the same group. There is no
TEST-dependent bin selection, threshold, or hyperparameter search.

The 10,000-row subset is sampled without replacement independently per task,
using numpy default_rng(2 + canonical_problem_index). No winner balancing is
used. The full and subset tables cover every TEST size; no fallback was used.
The predefined fallback for an unseen size is the corresponding per-problem
TRAIN statistic. This is one fixed subset, not a multi-seed equivalence study.

Size comes from input_metadata.npz/scale. Prediction node counts include a
depot for CVRP/MVRP, so subtract one for those tasks to recover customer count.
TSP and ATSP use node count directly. Targets come from targets.npz; raw costs
and relative gaps retain FP64. Each problem has equal macro weight.

## Conditional Instance Signal

Within each problem and exact size, uniformly permute complete prediction
vectors between TEST rows. Since ranking commutes with a row permutation,
the exact expected Top1 and regret can be computed from the predicted-method
frequencies without Monte Carlo or another model forward.

For a group g of n_g rows, let q_g(m) be the fraction predicted as method m,
p_g(m) the native winner frequency, and r_g(m) the mean actual relative regret
of always selecting m. Expected shuffled accuracy is sum_m q_g(m)*p_g(m);
expected shuffled regret is sum_m q_g(m)*r_g(m). Weight groups by their row
counts within each problem, then average all 18 problems equally.

Every TEST row belongs to a same-problem, same-size group with at least two
rows. No coarser size bins or cross-problem permutations are used.

| Prediction assignment | Top1 | Actual regret |
|---|---:|---:|
| R76's own prediction on each instance | 69.439452% | 0.464144% |
| Exact expectation under conditional permutation | 68.429704% | 0.518660% |

This supports useful instance association beyond problem and size, but not
causal understanding, an accuracy upper bound, or out-of-distribution transfer.
The approximately 1.01-point permutation difference and the approximately
0.224-point advantage over the size-majority table are different comparisons.
A model can have useful within-group information while using a less effective
coarse decision policy than another TRAIN-fitted baseline.

## Where Decisions Are Constant

On every one of the 3,895 TEST rows in each of the following 12 tasks, R76
chooses reld_mt:

VRPB, VRPTW, OVRPTW, OVRPB, VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW,
OVRPLTW, VRPBLTW, OVRPBLTW.

Those tasks account for two thirds of TEST rows, have mean Top1 82.642276%,
and contribute 48.471141% of R76's total macro actual regret. The final actions
are instance-independent on these rows; hidden representations or confidence
scores need not be constant. Constant decisions are not automatically bugs or
economically suboptimal without evidence about learnable alternatives.

| Problem | R76 Top1 | Size-majority Top1 | Conditionally shuffled expected Top1 |
|---|---:|---:|---:|
| TSP | 43.209243% | 37.637997% | 36.935356% |
| CVRP | 33.889602% | 26.957638% | 25.010232% |
| ATSP | 74.480103% | 76.251605% | 72.730531% |
| OVRP | 36.765083% | 39.332478% | 35.250443% |
| VRPL | 32.528883% | 34.608472% | 34.061197% |
| OVRPL | 37.329910% | 39.383825% | 36.039599% |

## Historical Comparisons

Using a consistently TRAIN-fitted per-problem majority-winner reference:

| Model and evaluation scene | Model Top1 | Majority Top1 | Net difference |
|---|---:|---:|---:|
| Old R45A, original old TEST and pools | 47.911111% | 39.650000% | +8.261111 points |
| R76, new TEST | 69.439452% | 66.727999% | +2.711453 points |
| R76, old TEST with current pools | 63.783333% | 60.761111% | +3.022222 points |

These are descriptive comparisons across changed scenes, not a causal
data-size ablation. The original 38.45% reference was cost-oriented post-hoc
SBS, whereas 66.728% is TRAIN majority-winner accuracy.

As an additional bookkeeping diagnostic, deleting unavailable candidates
from the original R45A saved rankings, and scoring them against the aligned
old TEST winners, gives 68.344444%. This does not re-encode instances under a
changed candidate pool and must not be called R45A's current-pool forward
result. It illustrates how much the target/pool change alone can alter a
headline score without training on additional data.

## Optimization Changed Substantially

| Setting at selected checkpoint | R45A | R76 |
|---|---:|---:|
| Logical global batch | 128 | 9,856 |
| Peak learning rate | 0.0001 | 0.0001 |
| Selected epoch | 38 | 18 |
| Successful optimizer updates | 54,036 | 2,592 |

R76 eventually stopped at epoch26 / 3,744 updates. Warmup, plateau reduction,
and early-stop rules remained epoch-based. Increasing batch approximately
77-fold while retaining LR and epoch-based control is a substantive optimizer
change, even though architecture and loss formulas are unchanged.

R76 selected TRAIN/validation Top1 is 69.369646% / 69.236913%, with CE
0.752703 / 0.755865. This is not evidence of the severe TRAIN/held-out
memorization gap seen in some earlier rounds. It is compatible with a
conservative policy, limited optimization, or weak residual predictive signal;
these explanations have not been isolated experimentally.

## Transfer Behavior

R76 chooses reld_mt on every old and new TEST instance of VRPTW and VRPLTW.
Yet VRPTW Top1 changes from 80.847240% (new) to 40.5% (old), and VRPLTW from
79.614891% to 41.0%. Together these two tasks account for 77.558218% of the
5.656119-point new-versus-old TEST accuracy difference. This is an accounting
fact about changed winner distributions with unchanged actions, not an
identification of why the underlying labels changed.

On TSPLIB, R76 chooses T2T for 25/49 instances, while T2T is the native winner
for 5/49. On CVRPLIB it chooses OMNI for 43/100 (native winner on 5/100),
ICAM for 52/100 (winner on 38/100), and RELD_CVRP for 0/100 (winner on 45/100).
TSP/CVRP candidate pools did not undergo the ReLD-MTL deletion. The LIB
degradation cannot directly be attributed to removing that MVRP candidate.
Changed instance coverage, deployment labels, and optimization remain
confounded. A larger generated dataset need not better cover these libraries.

## Interpretation and Next Control

The large headline Top1 is mostly attainable from coarse TRAIN statistics.
Useful instance signal remains, particularly in TSP/CVRP, but does not produce
a large overall advantage over the size-conditioned reference.

The 10,000-row lookup result supports the plausibility of a smaller-data
neural model achieving similar aggregate accuracy; it does not establish that
result. Fitting 128 training examples to 100% only establishes memorization
capacity, not the sample complexity of predicting unseen solver outcomes.

A focused next experiment would compare a fixed new-data subset and the full
new data under the same model, effective batch, initialization, optimizer,
update-indexed schedule, and successful-update budget. Use the same held-out
validation/test rows and preserve natural label frequencies. Fit checkpoints
using validation only. Comparing a revised full-data run with R76 would test
the changed training protocol jointly; it would not isolate batch size from
update budget. No such training was launched during this analysis.

## Sources

- R76 history.json, protocol.json, and evaluation/test/selected_best/rank*.npz.
- R76 old_test_transfer/predictions/*.npz and lib_evaluation/predictions/*.npz.
- R45 A_handcrafted_seed2/args.json and best_eval.json; test_predictions/A/*.npz.
- Original data/<problem>train/raw_label.pkl for the old TRAIN majority.
- New scenario <problem>train/{targets,input_metadata}.npz for all lookup fitting.
- [Shallue et al., JMLR 2019](https://jmlr.org/papers/v20/18-789.html): batch-size effects depend on optimization and compute budget; no universal large-batch penalty is established.
- [Hoffer et al., NeurIPS 2017](https://arxiv.org/html/1705.08741v2): adapting update budgets and optimization can close large-batch gaps in their studied settings; not proof of R76's cause or an AdamW scaling prescription.
- [Lipton et al., ICML 2018](https://proceedings.mlr.press/v80/lipton18a.html): pure label shift assumes invariant p(x|y); changed winner frequencies alone do not establish that assumption here.
