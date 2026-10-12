# R69 Locked Delivery

Validation-frozen official model: Reused R68A shared, epoch28, global68. Original18 TEST: 8435/18000 (46.861111% native Top1).

R69 executed ONE new arm: B, the complete18-branch ordinary AdamW bank. A is explicitly reused R68A28/global68, fixed at8644/18000 validation correct. A received no new training, weights load, recount, inference, reselection or TEST. Its one original historical TEST and saved TRAIN/VAL result/curves are reused unchanged. Reuse establishes protocol equivalence, not an independent replication.

B is selected by maximum INTEGER full18 validation native correct/18000 with the earliest exacttie. All18 branches come from ONE common epoch. Official B requires strictly>8644 correct; tie/lower retains fixed A. Costs and TEST do not select; individual task gains do not choose branch checkpoints. R60B is a descriptive fixed historical incumbent, not the matched control or a reselected model.

## Selected Metrics

| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reused_R68A_shared | train | 49.5844 | 82.4439 | 93.1717 | 1.065074 | 11.176339 | -1.079288 | 0.936865 |
| Reused_R68A_shared | val | 48.0222 | 81.2833 | 91.9833 | 1.119215 | 11.174142 | -1.024455 | 0.974169 |
| Reused_R68A_shared | test | 46.8611 | 80.4944 | 91.8444 | 1.133650 | 11.178087 | -0.960929 | 1.015239 |
| R69B_task_specific_new | train | 50.3311 | 83.0650 | 93.5733 | 1.050462 | 11.179513 | -1.062171 | 0.953169 |
| R69B_task_specific_new | val | 47.8000 | 81.3611 | 92.0722 | 1.119310 | 11.184301 | -0.965644 | 1.042269 |
| R69B_task_specific_new | test | 46.3222 | 80.8000 | 91.7056 | 1.127789 | 11.184552 | -0.922807 | 1.061529 |
| R45A | val | 49.1167 | 82.4444 | 92.7611 | 1.089768 | 11.168517 | -1.070041 | 0.926048 |
| R45A | test | 47.9111 | 81.7444 | 92.4333 | 1.101119 | 11.168366 | -1.032003 | 0.943221 |
| R60B | train | 50.7706 | 83.1761 | 93.5856 | 1.041664 | 11.167328 | -1.143916 | 0.872533 |
| R60B | val | 49.3611 | 82.2389 | 92.7222 | 1.092423 | 11.168642 | -1.073544 | 0.922789 |
| R60B | test | 48.2444 | 81.7222 | 92.5111 | 1.102036 | 11.168404 | -1.035671 | 0.940486 |
| R64A | train | 48.5244 | 82.1294 | 92.7939 | 1.086541 | 11.178202 | -1.068215 | 0.950804 |
| R64A | val | 48.6944 | 82.0556 | 92.5778 | 1.094214 | 11.169958 | -1.057617 | 0.938619 |
| R64A | test | 47.3778 | 81.4500 | 92.2167 | 1.102747 | 11.173794 | -0.992692 | 0.981460 |
| R64B | train | 48.6156 | 82.1272 | 92.7806 | 1.085773 | 11.178306 | -1.069022 | 0.949743 |
| R64B | val | 48.7611 | 82.1111 | 92.4889 | 1.094646 | 11.170174 | -1.055205 | 0.940973 |
| R64B | test | 47.4111 | 81.2444 | 92.2500 | 1.103343 | 11.173766 | -0.995764 | 0.977711 |
| R65A | train | 50.5561 | 82.4217 | 93.1906 | 1.063218 | 11.179115 | -1.063344 | 0.956103 |
| R65A | val | 47.7722 | 81.1556 | 91.4667 | 1.148225 | 11.183102 | -0.965331 | 1.044606 |
| R65A | test | 46.4722 | 80.4722 | 91.2611 | 1.159455 | 11.183157 | -0.918209 | 1.070499 |
| R65B | train | 49.3822 | 82.6967 | 93.1867 | 1.066195 | 11.173045 | -1.094776 | 0.926387 |
| R65B | val | 48.7222 | 82.1500 | 92.4833 | 1.090114 | 11.169640 | -1.056037 | 0.945528 |
| R65B | test | 47.6389 | 81.7000 | 92.3944 | 1.097159 | 11.171845 | -0.999267 | 0.978280 |
| R66A | train | 48.7900 | 82.0128 | 92.8400 | 1.181554 | 11.176315 | -1.077692 | 0.942771 |
| R66A | val | 48.7667 | 81.8944 | 92.7000 | 1.206949 | 11.169383 | -1.059812 | 0.938885 |
| R66A | test | 47.2167 | 81.2111 | 92.1111 | 1.223669 | 11.172895 | -0.993691 | 0.982136 |
| R66B | train | 49.5006 | 82.7406 | 93.2089 | 1.063500 | 11.172509 | -1.103415 | 0.917486 |
| R66B | val | 48.6667 | 82.0389 | 92.4667 | 1.088236 | 11.169790 | -1.059714 | 0.942704 |
| R66B | test | 47.6056 | 81.7889 | 92.4333 | 1.095159 | 11.171490 | -1.008410 | 0.968539 |
| R67A | train | 50.3750 | 83.2517 | 93.5883 | 1.044683 | 11.169400 | -1.118286 | 0.900189 |
| R67A | val | 48.7056 | 82.3389 | 92.6944 | 1.093810 | 11.170318 | -1.051980 | 0.951395 |
| R67A | test | 47.4889 | 81.6389 | 92.4056 | 1.102166 | 11.171380 | -1.001858 | 0.975968 |
| R67B | train | 51.2778 | 83.7906 | 93.9833 | 1.026101 | 11.166257 | -1.146737 | 0.871508 |
| R67B | val | 49.2889 | 82.2389 | 92.7167 | 1.097729 | 11.169950 | -1.062306 | 0.935440 |
| R67B | test | 48.1000 | 81.6333 | 92.4111 | 1.107422 | 11.169943 | -1.021719 | 0.955544 |
| R68B | train | 49.6978 | 82.5678 | 93.3000 | 1.063684 | 11.176063 | -1.056603 | 0.962807 |
| R68B | val | 47.8667 | 81.4833 | 92.1222 | 1.113930 | 11.177486 | -0.987798 | 1.013203 |
| R68B | test | 46.6444 | 80.7333 | 91.8167 | 1.126804 | 11.176961 | -0.942605 | 1.042468 |

comparison.csv/selected_metrics.csv contain full task/family/ALL selected TRAIN/VAL/TEST. ALL is the original equal18-task macro; MVRP includes15 tasks. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the original legal-pool oracle. Cross-task mean costs mix task units and are descriptive. Positive accuracy deltas are improvements; positive CE/cost/regret deltas are worse.

comparison_deltas.csv and task_gain_regression.csv compare new B with matched reused A and all fixed historical references, including R60B. Task and family gains/regressions are descriptive, never a checkpoint-selection rule.

- train B versus Reused_R68A_shared: 13 task gains, 5 regressions, 0 ties.
- val B versus Reused_R68A_shared: 8 task gains, 10 regressions, 0 ties.
- test B versus Reused_R68A_shared: 8 task gains, 10 regressions, 0 ties.
- train B versus R60B: 7 task gains, 11 regressions, 0 ties.
- val B versus R60B: 4 task gains, 14 regressions, 0 ties.
- test B versus R60B: 1 task gains, 17 regressions, 0 ties.

stage2_summary.csv reports selected-best/final TRAIN/VAL and last5 validation mean/population SD/min/max for epochs36-40. Last5 is neither seed uncertainty nor an averaged checkpoint. matched_common_epochs.csv uses only actually evaluated common epochs, including actual selected-best TRAIN. Every VAL epoch is full18; TRAIN is every5/final/selected-best. Curves contain only actual measurements, with missing historical TRAIN disclosed in metric_availability.csv. No dense TRAIN interpolation or reference inference fills gaps.

## Compute and Interpretation

| Component | Updates | Selected Rows | Successful View Rows | Attempted View Rows | AMP Retries | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEW R69B | 9600 | 7200000 | 14400000 | 14405152 | 5 | 3554.574466 |
| HISTORICAL reused R68A | 9600 | 7200000 | 14400000 | 14405632 | 5 | 3002.024237 |

Inherited R67B terminal stage1 pretraining counted ONCE separately: 9600 updates/7200000 rows/2764.217798s. New B only: 9600 updates/7200000 rows/14400000 successful views. Each B branch:1600 active updates/400000 selected rows. New B peak allocated/reserved memory: 15.661316/21.376953GiB. Reused A's3002.024237s and5 retries are historical, not new R69 expenditure.

Actual B capacity is33173154 parameters versus1842953 shared A parameters. Capacity and per-parameter optimizer histories are NOT matched. Per-parameter Adam steps advance only on active gradients; unused input paths may remain at0 steps. branch_compute.csv and parameter_optimizer_steps.csv preserve actual activation/row/counter evidence. compute_summary.json verifies saved initialization and actual task/row/dropout/LR order against the reused control; it does not rerun historical outputs.

All18 copies start identically from R67B terminal stage1 averaged-x trunk plus the entire fresh head670003, through the exact saved R68 pre-fine-tuning state. These are not from-scratch encoders. B trains all parameters, with no SAM/projector/averaging/optimizer mode transition. Same global41-80 schedule, native CE and FP32 legal symmetric RDrop, real256x3/tails16x3, global successful-update warmup clock, per-task FP32 gradient/3, one combined clip1 and one AdamW step. Whole-update AMP retries retain the same rows/RNG and do not advance the cursor or successful counter.

A gain supports this unsharing recipe; it does not isolate harmful task-gradient interference from increased capacity, fewer per-parameter updates, changed Adam histories or fitting18 models with10000 fine-tuning examples each. No result establishes an accuracy ceiling, label noise, or a70% promise. Historical TEST exposure makes cross-version comparisons descriptive; the new B TEST never changes architecture, budget, selection or planning.

B validation correct minus matched A: -40. Numeric official validation-frozen original18 TEST70% threshold reached: False. Parent whole-goal audit passed: False. Supported70% claim: False. Parent owns whole-goal audit and closure.

## Deployment, Provenance and Assets

Known original problem identity activates ONE complete branch, one full-pool direct-utility forward, ascending-global-ID exact ties, then ONE selected solver completes per deployed query. Delivery evaluation executes ZERO solvers. test_results.json distinguishes newly evaluated B from unchanged saved A metrics and provenance; A has one historical original TEST, zero new R69 tests. No reference inference, NPZ prediction replay, solver runs or separate singleton latency benchmark occur.

New B ordinary full evaluation time: 8.123364s. This includes transfer/metrics/NPZ saving and is NOT singleton latency. Other reference timings retain historical scope. Selected weights and complete NPZ logits/predictions/native winners/rawFP64 costs/indices/sizes/global pool IDs/names stay LOCAL with raw updates/W&B/schedules/cache. Git alone is not all inference assets; nothing is force-added.

The eight legacy B-task feasible-domain mismatches stay disclosed/unfixed and are not a native70 veto. R58 is fully deferred. Parent owns reviews, goal audit, memory/state and one Git commit per completed version. This report creates none of those approvals.
