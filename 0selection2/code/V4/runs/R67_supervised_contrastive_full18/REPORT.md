# R67 Locked Delivery

Official R67B, stage2 epoch36, globalepoch76: locked original18 TEST 8658/18000 = 48.100000% native Top1.

The official arm and checkpoint were frozen using maximum integer validation correct count, earliest exacttie and arm tieA. Costs, TEST and parameter mode do not select models. There is no task-wise assembly. TEST loads one complete saved stage2 averaged-x NativeUtilitySelector directly; no optimizer transition, projector or inherited trained selector.

Stage1 A uses native CE/RDrop; stage1 B uses only task-local supervised contrastive Eq.2. These objective values are different quantities. There is no stage1 validation curve. Both stage2 objectives are native CE/RDrop and their complete validation curves are comparable. Dropout-only views and fully trainable stage2 are project adaptations. Fixed contrastive multiplier.1 is a project loss scale, not the author-default temperature/base_temperature scale.

## Selected Metrics

| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R67A | train | 50.3750 | 83.2517 | 93.5883 | 1.044683 | 11.169400 | -1.118286 | 0.900189 |
| R67A | val | 48.7056 | 82.3389 | 92.6944 | 1.093810 | 11.170318 | -1.051980 | 0.951395 |
| R67A | test | 47.4889 | 81.6389 | 92.4056 | 1.102166 | 11.171380 | -1.001858 | 0.975968 |
| R67B | train | 51.2778 | 83.7906 | 93.9833 | 1.026101 | 11.166257 | -1.146737 | 0.871508 |
| R67B | val | 49.2889 | 82.2389 | 92.7167 | 1.097729 | 11.169950 | -1.062306 | 0.935440 |
| R67B | test | 48.1000 | 81.6333 | 92.4111 | 1.107422 | 11.169943 | -1.021719 | 0.955544 |
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

comparison.csv contains every original task, four families and ALL for available TRAIN/VAL/TEST. ALL is the equal18-task macro; MVRP is15 tasks. Original native winners and ascending-global-ID exact utility ties are preserved. Costs retain FP64. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Mean costs across heterogeneous tasks are descriptive across task units.

R45A38/R60B39 retain their historical cost-selected official policies. R64A/B, R65A/B and R66A/B retain original validation locks and official-arm policies. Fixed saved JSON reports are reused; no reference inference, prediction replay, retrospective TEST or solver is run. Historical TEST is exposed, so version comparisons are descriptive. Missing saved historical TRAIN, notably R45A, is disclosed in metric_availability.csv and never reconstructed by inference.

stage2_summary.csv contains best/final TRAIN/VAL and last5 validation mean/population SD/min/max by task/family/ALL. Last5 are complete validation epochs36-40, not seed uncertainty. TRAIN exists only on the predeclared every5/final/selected epochs; no dense TRAIN or stage1 validation curve is invented.

## Actual Compute

| Arm | Stage | Successful Updates | Presentations | AMP Retries | Stage Wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 9600 | 7200000 | 4 | 2754.293 |
| A | 2 | 9600 | 7200000 | 4 | 2971.675 |
| B | 1 | 9600 | 7200000 | 1 | 2764.218 |
| B | 2 | 9600 | 7200000 | 5 | 2971.748 |

Actual pair: 38400 updates/28800000 presentations, 11506.549s summed arm wall. Planned eachstage9600/7.2M, eacharm19200/14.4M, pair38400/28.8M. Actual logs and terminal receipts are authoritative; planned counts are never substituted for interrupted execution. There are zero inherited pretrained selectors.

Real256/task, three distinct tasks/update, effective768; tail16/task/effective48. Both arms run40+40epochs. Official ScheduleFree/internal720-update stage warmup,3e-4 constant thereafter, native legal pools, FP32 mean task gradients/one clip1 and whole-triple AMP retries are unchanged. Stage transition receipts record terminal x, identical whole fresh head and fresh optimizer/scaler/counters. A projector and B score head are intentionally unused during stage1.

One selector forward per query and one completed solver at deployment; evaluation itself executes zero solvers. Reported TEST time is ordinary full-evaluation wall time including gather/transfer, metrics and prediction saving, not singleton latency. compute_summary.json retains measured steps, runtime, memory, counters and actual Slurm/tmux terminal receipts.

Numeric locked official TEST threshold reached: False. Whole-objective completion audit passed: False. A70% claim is permitted only when BOTH are true. Goal remains ACTIVE until parent closure; no severe-overfit, label-noise or accuracy-ceiling claim follows from this one-seed experiment.

## Assets and Limits

Selected checkpoints, complete prediction NPZ/raw costs/IDs, raw update logs, W&B and regenerable schedules stay local under current cleanup policy. Code, reports, configs, receipts, CSV curves, figures and review records can be curated into one later completed-version commit by parent. Nothing is force-added. Git alone does not contain all assets needed for inference.

The full18 native70% metric remains the user objective. R58 is fully deferred. The eight legacy B-task feasible-domain mismatches remain unresolved and disclosed; they are not quietly repaired or used to veto that native metric. Parent owns the final objective audit, reviews, memory and Git.
