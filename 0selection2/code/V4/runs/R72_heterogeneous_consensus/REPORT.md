# R72 Fixed Heterogeneous Consensus

Official arm B was locked before TEST from8923/18000 VAL, versus fixed R60B39 A's8885: +38 decisions (+0.211111pp).
ONE new original full18 B TEST: **8680/18000 = 48.222222%**; versus unchanged historical A: 8684/18000, -4 decisions (-0.022222pp). TEST never changes membership, checkpoint, aggregation or official arm.

| Policy | Split | Native Correct | Top1% | Top2% | Top3% | Mean Cost | vsSBS% | Actual Regret% | Borda-score CE | Historical Native CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R72A_reused_R60B39 | VAL | 8885/18000 | 49.361111 | 82.238889 | 92.722222 | 11.168642 | -1.073544 | 0.922789 | not applicable | 1.092423 |
| R72A_reused_R60B39 | TEST | 8684/18000 | 48.244444 | 81.722222 | 92.511111 | 11.168404 | -1.035671 | 0.940486 | not applicable | 1.102036 |
| R72B_fixed_consensus | VAL | 8923/18000 | 49.572222 | 82.383333 | 92.777778 | 11.168430 | -1.073296 | 0.925862 | 1.220912 | not applicable |
| R72B_fixed_consensus | TEST | 8680/18000 | 48.222222 | 81.888889 | 92.550000 | 11.168551 | -1.034889 | 0.940706 | 1.243185 | not applicable |
| R67B36_global76 | VAL | 8872/18000 | 49.288889 | 82.238889 | 92.716667 | 11.169950 | -1.062306 | 0.935440 | not applicable | 1.097729 |
| R67B36_global76 | TEST | 8658/18000 | 48.100000 | 81.633333 | 92.411111 | 11.169943 | -1.021719 | 0.955544 | not applicable | 1.107422 |
| R70B23_global63 | VAL | 8709/18000 | 48.383333 | 81.400000 | 92.194444 | 11.176386 | -1.015037 | 0.988830 | not applicable | 1.121139 |
| R70B23_global63 | TEST | 8409/18000 | 46.716667 | 80.866667 | 91.705556 | 11.176412 | -0.976248 | 1.005218 | not applicable | 1.132501 |

comparison.csv includes equal-weight full18 macros, every task and TSP/CVRP/ATSP/MVRP families for VAL and TEST. Historical component rows are saved descriptive records, not separately executed member metrics. R60 JSON has no native_correct fields; its counts in reporting tables are arithmetic from saved task Top1*n, not prediction recounts. Its original TEST object remains exactly unchanged.

The complete offline composite contains R60B39 original cost-selected score-difference/maximin weights, R67B36/global76 saved averaged-x weights without projector/optimizer transition, and R70B23/global63 all18 complete common-epoch branches. No historical checkpoint pointers are followed during inference. All36859060 stored parameters are frozen;5528859 parameters activate per query across three selectors.

Every query uses exactly three original returned FP32 logits, aligned over the complete legal pool by global ID. Doubled midrank q=2*strictly_higher+equal_including_self+1; integer Q sums all three. Minimum Q wins, with smallest global ID resolving final ties. Reporting-only z=-Q/6 defines Borda-score CE. Historical native CE is on a different score scale, so its difference is not a calibration gain. Density NLL is inapplicable. Targets never enter member inputs or aggregation. TRAIN-fitted normalization, geometry, ATSP direction and physical roles remain intact.

New R72 expenditure: zero TRAIN epochs/updates/TRAIN forward rows, zero optimized parameters, zero solver executions. VAL wall 6.995775s; ONE full18 TEST wall 23.786722s (23.173433s in three live forwards plus aggregation). TEST setup plus evaluation work was 30.246785s. These are ordinary whole-evaluation times, not singleton or completed-solver latency; no separate latency benchmark ran.

Each VAL/TEST pass covers18000 original rows,144 query batches,432 member forwards,54000 member-row forwards and144 active R70 branch calls, including eighteen104-row tails. No standalone A or component evaluation occurred. Deployment executes three selectors and completes one selected solver.

TEST peak allocated/reserved/observed physical memory: 2.258378/5.650391/5.960205GiB. One offline composite was restored; three historical constituent checkpoint files were not separately reloaded.

Historical R60 expansion: 56880 updates/40 epochs/5215.240736s, with6000000 extra historical rows across15 MVRP tasks. R67 historical total: 19200 updates/14400000 rows/5757.091320s, already including its stage1. R70 historical fine-tuning: 9600 updates/7200000 rows/3668.194599s, five retries. Shared R67B stage1 is counted ONCE: 9600 updates/7200000 rows/2764.217798s. Do not add that shared stage1 a second time. These are inherited historical costs, not new R72 training.

There is one fixed candidate and no R72 TRAIN curve, loss, epoch, best/final search, calibration, membership search or taskwise assembly. Three-selector inference and stored capacity are unmatched to A. R66 remains weak ensemble counterevidence, and shared R67/R70 pretraining does not establish complementary instance errors. A small aggregate change supports only this particular fixed bundle, not error independence or a70% forecast.

A reuses the EXACT original R60B39 TEST JSON traced through R64 official metadata to R63 reference/R60B/test_eval.json; new TEST counts A0/B1, original A already1. Historical TEST exposure is disclosed. The unique new TEST neither changes this fixed protocol nor goes to successor planning. Analysis ran ONCE on saved JSON only; no NPZ replay, model/solver forward, reference inference or additional experiment.

Original goal threshold:12600/18000 native TEST decisions. Observed official B:8680/18000. Goal70 remains ACTIVE/unachieved; parent actual reviews and whole-objective audit are pending.

R58 and eight legacy B semantic inconsistencies remain deferred/disclosed/unfixed, not a native70 numeric veto. Weights, NPZ/raw scores, batch logs, W&B and caches remain LOCAL; Git alone is not all inference assets. Parent owns actual Astra reviews, persistent memory, objective audit and one completed-version Git publication. This worker performs no Git.
