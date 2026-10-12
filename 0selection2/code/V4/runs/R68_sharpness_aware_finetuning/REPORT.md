# R68 Locked Delivery

Official R68A, fine-tuning epoch28, globalepoch68: original18 TEST 8435/18000 = 46.861111% native Top1.

Checkpoint and official arm use maximum INTEGER original18 VAL native correct/18000, earliest epoch exacttie, arm tieA. Costs and TEST never select. Each checkpoint is one complete NativeUtilitySelector at ordinary unperturbed AdamW weights, loaded directly without optimizer transition, averaging, projector, head transplant or task-wise assembly.

This is one fixed-seed optimization contrast from the same R67B terminal stage1 SupCon trunk plus the entire identical fresh head670003. These are not fresh encoders. Both arms train all selector parameters for40 new epochs with native CE and legal FP32 symmetric RDrop. A uses ordinary AdamW; B uses global SAM rho.05 before clipping, restores the pre-pass RNG for identical dropout masks, copies original weights back exactly, clips the second mean gradient once and performs one AdamW/weight-decay step. SAM adds gradient compute: rows and updates match, FLOPs do not.

## Selected Metrics

| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R68A | train | 49.5844 | 82.4439 | 93.1717 | 1.065074 | 11.176339 | -1.079288 | 0.936865 |
| R68A | val | 48.0222 | 81.2833 | 91.9833 | 1.119215 | 11.174142 | -1.024455 | 0.974169 |
| R68A | test | 46.8611 | 80.4944 | 91.8444 | 1.133650 | 11.178087 | -0.960929 | 1.015239 |
| R68B | train | 49.6978 | 82.5678 | 93.3000 | 1.063684 | 11.176063 | -1.056603 | 0.962807 |
| R68B | val | 47.8667 | 81.4833 | 92.1222 | 1.113930 | 11.177486 | -0.987798 | 1.013203 |
| R68B | test | 46.6444 | 80.7333 | 91.8167 | 1.126804 | 11.176961 | -0.942605 | 1.042468 |
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

comparison.csv and selected_metrics.csv contain task/family/ALL TRAIN/VAL/TEST. ALL is the original equal18-task macro; MVRP has15 tasks. Original native winners, full legal pools and ascending-global-ID exact utility ties remain. Raw costs are FP64. vsSBS is100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Cross-task mean costs mix task units and are descriptive.

stage2_summary.csv reports best/final complete TRAIN/VAL and last5 validation mean/population SD/min/max by task/family/ALL. Last5 are epochs36-40, not seed uncertainty. matched_common_epochs.csv compares only actually available common TRAIN/VAL epochs; no interpolation or selection by those comparisons. Every VAL epoch is complete. TRAIN evaluation occurs every5/final and selected-best. R68 has no new stage1 curve; inherited R67 stage1 SupCon is not mislabeled CE or given an invented validation curve.

Historical R45A38/R60B39 retain their original cost-selected official policies. R64A/B,R65A/B,R66A/B,R67A/B retain original validation locks, parameter modes and official-arm rules. All references reuse fixed saved official reports, never inference, NPZ replay, retrospective TEST or solver runs. Historical TEST exposure makes version comparisons descriptive. Missing historical TRAIN, notably R45A, is disclosed in metric_availability.csv.

## Actual Compute

| Arm | New Updates | New Selected Rows | Successful View Rows | Attempted View Rows | AMP Retries | Wall s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 9600 | 7200000 | 14400000 | 14405632 | 5 | 3002.024 | 15.3113 |
| B | 9600 | 7200000 | 28800000 | 28807680 | 3 | 5677.557 | 15.3246 |

Inherited R67B stage1 counted ONCE: 9600 updates/7200000 rows, 2764.218s. Actual new pair: 19200 updates/14400000 selected rows. Inherited-once plus new pair: 28800 updates/21600000 rows. Planned new perarm9600/7.2M, pair19200/14.4M; A14.4M and B28.8M successful view rows. Actual measured logs and receipts are authoritative, including retries and attempted views.

The exact shared R67 global41-80 task/row/dropout slice is retained. Real256/task, three distinct tasks/update/effective768; tails16/task/effective48. Whole-triple AMP retries preserve rows/RNG, with counters advancing only after one successful base step. Fresh ordinary AdamW/scaler, external720-successful-update LR warmup to3e-4, native RDrop coefficient.35 warmup, one global norm clip1. No ScheduleFree optimizer/mode/averaging or early stop.

optimization_curves.csv, task_optimization.csv and update_curves.csv preserve first and perturbed native loss/CE/KL separately, LR, first/step gradients, clipping, perturbation norm, exact restoration, RNG replay, AMP scale/retries, rows/views, time/memory and one-step counters. These are training diagnostics, not an extra curvature experiment. compute_summary.json includes actual inherited identity, counts and terminal receipts. TEST time is ordinary full-evaluation walltime including transfer, metrics and NPZ saving, not singleton latency.

SAM exceeds matched AdamW on INTEGER validation count: False; SAM matches or exceeds it: False. SAM at least matches A but remains below fixed R60B TEST: False. Such a result is limited progress; one seed does not establish severe overfit, label noise or an accuracy ceiling.

Numeric official locked TEST threshold reached: False. Whole-objective audit passed: False. Supported70% claim: False. Goal remains ACTIVE pending parent whole-objective audit and closure; no70% promise follows from preparation or validation.

## Assets and Limits

Deployment is one selector forward and one completed solver per query; delivery evaluation runs zero solvers. Selected weights, NPZ/raw costs/IDs, raw update logs, W&B and regenerable caches remain LOCAL under user cleanup/ignore policy. Code, reports, configs, receipts, CSV curves, figures and reviews may later be curated once by parent. Nothing is force-added. Git alone does not contain all inference assets.

The eight legacy B-task feasible-domain mismatches remain unresolved and disclosed, not repaired or used as a native70% metric veto. R58 is fully deferred. Parent owns objective audit, reviewers, persistent state/memory, publication and Git.
