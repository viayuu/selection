# R71 Locked Delivery

VAL-frozen official arm A; original native TEST 8409/18000 = 46.716667%. Numeric70% threshold:12600/18000. Goal70 remains ACTIVE/unachieved; parent whole-objective audit and reviews remain required.

Exactly one new B trained and tested. Fixed A is EXACT complete R70B23/global63/8709 VAL correct. Its original R70 test_results.json/models/B object is reused unchanged; new TEST counts A0/B1, original A already1. No A weights/load/training/inference/recount/reselection/retest. Protocol equivalence is not independent replication.

B selected common epoch31/global71, VAL8371/18000: earliest maximum INTEGER original full18 VAL count. Official B requires strictly>8709; otherwise fixed A. All18 branches, normalizers, canonical IDs and fixed Sobol tables were locked before the new TEST. TEST, costs, NLL and task gains do not select.

| Model | Split | Top1% | Top2% | Top3% | Winner CE | Mean cost | vsSBS% | Actual regret% | NLL/D |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R71A_reused_R70B23 | train | 50.891111 | 83.271667 | 93.682778 | 1.042508 | 11.170485 | -1.116258 | 0.898297 | unavailable |
| R71A_reused_R70B23 | val | 48.383333 | 81.400000 | 92.194444 | 1.121139 | 11.176386 | -1.015037 | 0.988830 | unavailable |
| R71A_reused_R70B23 | test | 46.716667 | 80.866667 | 91.705556 | 1.132501 | 11.176412 | -0.976248 | 1.005218 | unavailable |
| R71B_joint_mixture | train | 46.063333 | 78.982778 | 91.238889 | 1.174186 | 11.185030 | -1.009822 | 1.012193 | -0.16085999903905776 |
| R71B_joint_mixture | val | 46.505556 | 78.855556 | 91.205556 | 1.175962 | 11.180571 | -0.975404 | 1.024641 | -0.12805792035659153 |
| R71B_joint_mixture | test | 45.105556 | 78.483333 | 90.777778 | 1.186001 | 11.180967 | -0.935260 | 1.047260 | -0.13406335359490995 |
| R45A | val | 49.116667 | 82.444444 | 92.761111 | 1.089768 | 11.168517 | -1.070041 | 0.926048 | unavailable |
| R45A | test | 47.911111 | 81.744444 | 92.433333 | 1.101119 | 11.168366 | -1.032003 | 0.943221 | unavailable |
| R60B | train | 50.770556 | 83.176111 | 93.585556 | 1.041664 | 11.167328 | -1.143916 | 0.872533 | unavailable |
| R60B | val | 49.361111 | 82.238889 | 92.722222 | 1.092423 | 11.168642 | -1.073544 | 0.922789 | unavailable |
| R60B | test | 48.244444 | 81.722222 | 92.511111 | 1.102036 | 11.168404 | -1.035671 | 0.940486 | unavailable |
| R64A | train | 48.524444 | 82.129444 | 92.793889 | 1.086541 | 11.178202 | -1.068215 | 0.950804 | unavailable |
| R64A | val | 48.694444 | 82.055556 | 92.577778 | 1.094214 | 11.169958 | -1.057617 | 0.938619 | unavailable |
| R64A | test | 47.377778 | 81.450000 | 92.216667 | 1.102747 | 11.173794 | -0.992692 | 0.981460 | unavailable |
| R64B | train | 48.615556 | 82.127222 | 92.780556 | 1.085773 | 11.178306 | -1.069022 | 0.949743 | unavailable |
| R64B | val | 48.761111 | 82.111111 | 92.488889 | 1.094646 | 11.170174 | -1.055205 | 0.940973 | unavailable |
| R64B | test | 47.411111 | 81.244444 | 92.250000 | 1.103343 | 11.173766 | -0.995764 | 0.977711 | unavailable |

Comparison CSVs include equal-weight full18 macros, TSP/CVRP/ATSP/MVRP families and every task for TRAIN/VAL/TEST. Histories retain actual best/final/last-five VAL and common actual epochs. Sparse TRAIN every5/final/selected-best stays sparse; no interpolation, five-epoch TRAIN mean, seed uncertainty or averaged checkpoint is asserted.

This contrasts the full joint-likelihood/readout/winner-probability package against R70, not covariance alone. Full continuous cost-difference supervision differs from winner CE/mixup. Joint NLL and NLL/D may be negative and are separate from native-winner probability CE. Lower density NLL alone does not establish selection benefit; continuous densities approximate exact cost-tie atoms and may fit irrelevant variation or overfit.

B retains the ORIGINAL pre-fine-tuning R67B terminal stage1 averaged-x backbone, replaces/deletes the scalar head, and trains all retained parameters/new density heads. It is not a from-scratch encoder. B has38231040 parameters versus A33173154; capacity and head/inference compute are unmatched. The retained-parameter active-update recipe remains1600 updates/400000 rows per branch.

Frozen inference uses one encoder pass/query, eight full-covariance components, the SAME2048 persistent Sobol-normal vectors/task for every component/query, streaming one component and256 vectors. It undoes TRAIN-only normalization, inserts reference zero, counts minima over all legal candidates, weights frequencies by mixture probabilities, and selects max probability/global-ID ties. Only reporting CE uses floor1e-12.16384 predicted vectors/query are not routes or solver executions. Reported density NLL reuses the same encoded parameters.

| Expenditure | Updates | Selected rows | Successful/attempted encoder views | Retries | Wall seconds |
|---|---:|---:|---:|---:|---:|
| new_B | 9600 | 7200000 | 14400000/14402560 | 4 | 4019.743262 |
| historical_reused_A | 9600 | 7200000 | 14400000/14404704 | 5 | 3668.194599 |

Inherited stage1 counted ONCE separately: 9600 updates/7200000 rows/2764.217798s. The ONE discarded capacity update adds768 TRAIN rows/5.965898s total (3.752883s update), with305848320 bytes of entire-bank Adam moments materialized. It is outside formal9600; mutated state was discarded and formal initialization recreated. New B has zero mixed-head rows; fixed A's historical mixup work stays historical.

New B peak allocated/reserved/physical memory: 15.748332/21.349609/21.675049GiB. Whole new TEST evaluation: 10.510779s including probability integration; not singleton latency or solver runtime.144 wrapper/active-branch/encoder batches cover all18000 rows including104-row tails. Zero solvers executed; deployment completes one selected solver.

Optimization CSVs/curves preserve actual jointNLL/NLL-D/total objective/LR/gradient/clip/AMP/retries/successful-vs-attempted encoder views/time/memory. No R71 cleanCE/mixCE/KL terms are invented. Raw formal counters determine actual expenditure; discarded capacity is extra.

Historical TEST exposure is disclosed. Saved locked historical references are descriptive only; no reference inference or NPZ replay occurred. NEW R71 TEST never changes this model/budget/rule and is excluded from successor planning. One seed supplies no seed-uncertainty or accuracy-ceiling claim.

Weights, NPZ probabilities/selection scores/predictions/native winners/raw FP64 costs/indices/nodes/global IDs/names, rawupdates, W&B/cache and regenerable schedules stay LOCAL. Git alone is not all inference assets; this worker performs no Git. Parent owns actual Astra reviews, fresh review, whole-objective audit, memory and one completed publication.

R58 and eight legacy B feasible-domain semantic inconsistencies remain fully deferred/disclosed/unfixed; they do not veto native70 numerics.

train B versus R71A_reused_R70B23: 1 task gains/17 regressions/0 ties.
val B versus R71A_reused_R70B23: 3 task gains/15 regressions/0 ties.
test B versus R71A_reused_R70B23: 5 task gains/13 regressions/0 ties.
train B versus R60B: 0 task gains/18 regressions/0 ties.
val B versus R60B: 2 task gains/15 regressions/1 ties.
test B versus R60B: 0 task gains/18 regressions/0 ties.

## Parent Review Closure

Actual primary TRAINVAL-only and fresh nightmare GPT-6-Astra/max reviews completed
0 with no actionable defects: integrity9/10 ready, original70 performance4/10 not
ready. Full raw responses and Memory updates are preserved; fresh new-TEST memory
was appended only after primary termination. No repair, extra evaluation or replay
was warranted. See `REVIEW_SUMMARY.md` and `reviews/`.

The full original objective remains unachieved (`objective_audit.json`: false).
Parent adopts ONE R72 fixed full18 consensus plan from completed TRAIN/validation
only; NEW R71TEST was excluded from that planning. One scoped completed-version
publication contains this R71 delivery; weights and prediction arrays stay local.
