# R70 Locked Delivery

VAL-frozen official arm: B. Official original18 TEST: 8409/18000 = 46.716667% native Top1. The numeric70% threshold is12600/18000. Goal70 remains ACTIVE/unachieved; parent whole-objective audit and reviews remain required.

Exactly ONE new R70 B was trained and tested. Fixed A is the unchanged R69B20/global60 complete18 bank with8604/18000 VAL correct and one already-executed original TEST. New TEST counts A0/B1; original A already1. A weights were not loaded, inferred, recounted, retrained or reselected. Its existing test_results.json/models/B object is reused unchanged. Protocol equivalence is not independent replication.

New B selected common epoch23/global63, VAL8709/18000: earliest maximum INTEGER full18 VAL count. Official B requires strictly>8604; tie/lower fixed A. All18 branches come from ONE complete bank checkpoint. Costs, task gains and TEST do not select.

| Model | Split | Top1% | Top2% | Top3% | CE | Mean Cost | vsSBS% | Actual Regret% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R70A_reused_R69B20 | train | 50.331111 | 83.065000 | 93.573333 | 1.050462 | 11.179513 | -1.062171 | 0.953169 |
| R70A_reused_R69B20 | val | 47.800000 | 81.361111 | 92.072222 | 1.119310 | 11.184301 | -0.965644 | 1.042269 |
| R70A_reused_R69B20 | test | 46.322222 | 80.800000 | 91.705556 | 1.127789 | 11.184552 | -0.922807 | 1.061529 |
| R70B_latent_mixup | train | 50.891111 | 83.271667 | 93.682778 | 1.042508 | 11.170485 | -1.116258 | 0.898297 |
| R70B_latent_mixup | val | 48.383333 | 81.400000 | 92.194444 | 1.121139 | 11.176386 | -1.015037 | 0.988830 |
| R70B_latent_mixup | test | 46.716667 | 80.866667 | 91.705556 | 1.132501 | 11.176412 | -0.976248 | 1.005218 |
| R45A | val | 49.116667 | 82.444444 | 92.761111 | 1.089768 | 11.168517 | -1.070041 | 0.926048 |
| R45A | test | 47.911111 | 81.744444 | 92.433333 | 1.101119 | 11.168366 | -1.032003 | 0.943221 |
| R60B | train | 50.770556 | 83.176111 | 93.585556 | 1.041664 | 11.167328 | -1.143916 | 0.872533 |
| R60B | val | 49.361111 | 82.238889 | 92.722222 | 1.092423 | 11.168642 | -1.073544 | 0.922789 |
| R60B | test | 48.244444 | 81.722222 | 92.511111 | 1.102036 | 11.168404 | -1.035671 | 0.940486 |
| R64A | train | 48.524444 | 82.129444 | 92.793889 | 1.086541 | 11.178202 | -1.068215 | 0.950804 |
| R64A | val | 48.694444 | 82.055556 | 92.577778 | 1.094214 | 11.169958 | -1.057617 | 0.938619 |
| R64A | test | 47.377778 | 81.450000 | 92.216667 | 1.102747 | 11.173794 | -0.992692 | 0.981460 |
| R64B | train | 48.615556 | 82.127222 | 92.780556 | 1.085773 | 11.178306 | -1.069022 | 0.949743 |
| R64B | val | 48.761111 | 82.111111 | 92.488889 | 1.094646 | 11.170174 | -1.055205 | 0.940973 |
| R64B | test | 47.411111 | 81.244444 | 92.250000 | 1.103343 | 11.173766 | -0.995764 | 0.977711 |

comparison.csv and selected_metrics.csv contain full18 equal macros, TSP/CVRP/ATSP/MVRP families and all per-problem TRAIN/VAL/TEST Top1/2/3, CE, mean cost, vsSBS and actual regret. Cross-task costs mix task units and are descriptive. comparison_deltas.csv/task_gain_regression.csv preserve task/family gains. Fixed R45A/R60B/R64 references use saved official JSON only; missing TRAIN/counts stay missing, with no recount/inference/NPZ replay.

TRAIN measurements are only every5/final and selected best. VAL measurements are every epoch. stage2_summary.csv contains best/final and epochs36-40 VAL mean/population SD/min/max; matched_common_epochs.csv uses actual common epochs only. A full last-five TRAIN mean is unavailable; summary_availability.csv lists the sparse actual TRAIN epochs, including a selected-best point in that interval when present. These are not seed uncertainty or averaged checkpoints.

| Compute | Updates | Selected Rows | Encoder Views Successful/Attempted | Mixed Heads Successful/Attempted | Retries | Wall s |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| NEW formal B | 9600 | 7200000 | 14400000/14404704 | 14400000/14404704 | 5 | 3668.194599 |
| HISTORICAL reused A | 9600 | 7200000 | 14400000/14405152 | 0/0 | 5 | 3554.574466 |

Inherited stage1 counted ONCE separately: 9600 updates/7200000 rows/2764.217798s. The one discarded capacity update adds768 TRAIN rows/4.474874s, with265385232 bytes of full-bank Adam moments explicitly materialized. It is extra work, excluded from the formal9600-update clock; its mutated state was discarded before formal recreation.

Both A/B have33173154 parameters and the same per-parameter update recipe:1600 active updates/400000 selected rows per branch. B adds14.4M mixed-head rows; head compute is unmatched. New B peak allocated/reserved/physical memory: 15.672294/22.306641/22.632080GiB. New complete TEST evaluation time: 7.988455s; this is not singleton latency or solver runtime.

B starts from the ORIGINAL pre-fine-tuning R67B terminal stage1 averaged-x trunk plus entire fresh head670003, never trained A20. It is not a from-scratch encoder. The unchanged fixed40E global41-80 schedule uses real256x3/tails16x3, fresh AdamW, FP32 active disjoint task gradients/3, one clip1 and one step. Inactive grad=None means no decay/moment/counter advancement. Whole-update AMP retries preserve rows/RNG and successful clock.

The sole objective package is0.5*CEclean+0.5*CEmix+.35*min(u/720,1)*KLclean. Both clean forwards occur first; one private seeded pi/lambda is shared across views, mixed features backpropagate through both partners, and auxiliary-head dropout restores ordinary CPU/CUDA RNG. No mixed KL, labels in features, coefficient search, new head or changed architecture. optimization/update CSVs and plots preserve actual component/LR/gradient/clip/AMP/retry/row/wall/memory records.

Native inference remains ordinary clean FP32/full original legal pool/global-ID exact ties, one active branch and one forward/query. Delivery executed zero solvers, exactly144 wrapper/active-branch batches with104-row tails. A deployed query completes one selected solver. No mixed-feature TEST, augmentation, sampling, pruning, ensemble or singleton benchmark.

Historical TEST exposure is disclosed. New B TEST does not change model, selection, budget, rules or successor planning; parent future planning explicitly excludes NEW R70 TEST. No result here establishes a70% claim, noise diagnosis or accuracy ceiling.

Weights, NPZ logits/predictions/native winners/raw FP64 costs/indices/sizes/global pool IDs/names, raw updates, W&B, cache and regenerable schedule stay LOCAL. Git alone does not contain all inference assets. This executor performs no Git operations.

R58 is fully deferred. Eight legacy B feasible-domain mismatches remain disclosed/unfixed and are not a native70% veto. Parent owns actual Astra reviews, fresh review, whole-objective audit, memory/state and one completed-version publication.

train B versusR70A_reused_R69B20: 12 task gains/6 regressions/0 ties.
val B versusR70A_reused_R69B20: 10 task gains/8 regressions/0 ties.
test B versusR70A_reused_R69B20: 9 task gains/8 regressions/1 ties.
train B versusR60B: 10 task gains/8 regressions/0 ties.
val B versusR60B: 4 task gains/14 regressions/0 ties.
test B versusR60B: 3 task gains/15 regressions/0 ties.
