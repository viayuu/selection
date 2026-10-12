# R66 Locked Delivery

Official **R66A, epoch9**, frozen by original integer validation count before TEST: **8499/18000 = 47.216667% native TEST Top1**. Full18 native70% remains ACTIVE and unachieved.

Both arms inherit the same immutable R65B40 saved averaged-x base F and start one identically copied fresh member g at memberseed660002, experimentseed2. A optimizes standalone-g native CE/KL; B optimizes composite F+g native CE/KL. BOTH evaluate and deploy F+g_x with coefficient1. The ONLY new A/B intervention is whether member-training CE/KL uses the fixed F offset.

Official ScheduleFree trains member y and evaluates/saves member x; TEST loads the complete two-model saved-x bundle directly with no optimizer transition. Real256/task, three-task effective768, tail16/task/effective48, fixed40 epochs/9600 updates/7.2M presentations per member. Internal720-update warmup to3e-4 then constant; native CE plus existing RDrop, same plans/streams, one global clip1 and whole-triple retries. No controller or early stop.

## Validation Freeze

| Arm | Epoch | VAL Correct | Mode | Unique Bundle SHA256 |
| --- | ---: | ---: | --- | --- |
| A | 9 | 8778/18000 | averaged x | `1c8b97a758e08232a0c9538062f59a0b65d25b1cbe38fd10067e54bd0bf4573f` |
| B | 33 | 8760/18000 | averaged x | `de4eda23d3b4c0b1bc07ff89ed3d845b651f61ea6a437e8b7a44163f2bee6b7f` |

Each unique selected-epoch bundle is mechanically associated with its published best.pt; the lock retains both paths/hashes and their identical saved base/member states. Max integer native VAL count selects each checkpoint with earliest exact tie; official-arm tie chooses A. Cost, TEST and parameter modes do not choose weights. No per-task assembly.

B minus A validation: **-18 correct / -0.100000pp**. This is a descriptive one-stage, one-seed comparison. Standalone member optimization CE is not composite evaluation CE; max-validation performance does not establish heldout performance. No general boosting, label-noise or accuracy-ceiling conclusion follows.

## Composite Native Metrics

| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R66A | train | 48.7900 | 82.0128 | 92.8400 | 1.181554 | 11.176315 | -1.077692 | 0.942771 |
| R66A | val | 48.7667 | 81.8944 | 92.7000 | 1.206949 | 11.169383 | -1.059812 | 0.938885 |
| R66A | test | 47.2167 | 81.2111 | 92.1111 | 1.223669 | 11.172895 | -0.993691 | 0.982136 |
| R66B | train | 49.5006 | 82.7406 | 93.2089 | 1.063500 | 11.172509 | -1.103415 | 0.917486 |
| R66B | val | 48.6667 | 82.0389 | 92.4667 | 1.088236 | 11.169790 | -1.059714 | 0.942704 |
| R66B | test | 47.6056 | 81.7889 | 92.4333 | 1.095159 | 11.171490 | -1.008410 | 0.968539 |

ALL is the equal18-task macro; MVRP has15 tasks. Native Top1/2/3 use original native winners, raw FP64 costs drive cost/regret metrics, and exact utility ties use ascending global solver ID. vsSBS=100*(selector mean cost/SBS cost-1); actual regret is mean per-query relative excess over the legal-pool oracle. Mixed-task mean costs are descriptive across task units. All18 tasks, four families and ALL for TRAIN/VAL/TEST are in comparison.csv.

## Fixed Historical References

| Model | Epoch | VAL Top1 % | TEST Top1 % | VAL CE | TEST CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R66A | 9 | 48.7667 | 47.2167 | 1.206949 | 1.223669 |
| R66B | 33 | 48.6667 | 47.6056 | 1.088236 | 1.095159 |
| R45A | 38 | 49.1167 | 47.9111 | 1.089768 | 1.101119 |
| R60B | 39 | 49.3611 | 48.2444 | 1.092423 | 1.102036 |
| R64A | 40 | 48.6944 | 47.3778 | 1.094214 | 1.102747 |
| R64B | 40 | 48.7611 | 47.4111 | 1.094646 | 1.103343 |
| R65A | 38 | 47.7722 | 46.4722 | 1.148225 | 1.159455 |
| R65B | 40 | 48.7222 | 47.6389 | 1.090114 | 1.097159 |

R45A38 and R60B39 preserve historical official/cost-selected policies; R45A has no saved selected TRAIN evaluation. R64A/B and R65A/B reuse their original locked results, each historical official arm B. Paths, JSON pointers, hashes and fixed policies are in reference_provenance.json. Historical TEST is exposed, not pristine. Cross-version comparisons are descriptive; no reference inference, prediction replay or solver operation occurred.

## Validation Trajectory

| Arm | Best Top1 % | Final Top1 % | Last5 Mean % | Last5 Min % | Last5 Max % |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 48.7667 | 48.3889 | 48.3578 | 48.3389 | 48.3889 |
| B | 48.6667 | 48.5278 | 48.5600 | 48.5278 | 48.5889 |

validation_summary.csv reports all seven metrics/counts for best/final and last5 mean/population SD/min/max for ALL, families and every task; last5_validation.csv retains each epoch. These temporal summaries are not seed uncertainty. Curves retain complete saved evaluation histories and correctly labeled standalone-A/composite-B objective loss/CE/KL, LR, gradients, clipping, retries and memory.

## Compute and Deployment

| Arm | New Updates | Presentations | AMP Retries | Arm Wall s | Update Wall s | Peak GiB | Live TEST s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 9600 | 7200000 | 4 | 3006.986 | 2688.682 | 15.332 | 15.151 |
| B | 9600 | 7200000 | 4 | 3055.495 | 2708.011 | 15.332 | 14.642 |

New paired training totals19200 successful updates/14.4M presentations and6062.481s arm wall. Separately, shared inherited F was trained once earlier for9600 updates/7.2M presentations/2979.144s. These are marginal member costs plus an explicitly inherited base, not fully fresh training or total compute equal to one old model.

Owned training1465.94: COMPLETED0:0, GPU allocation step wall6109s. Owned TEST1465.95: COMPLETED0:0, GPU allocation step wall37s; paired TEST body34.496s. These are actual sole-GPU job/stage wall times including I/O/initialization where applicable, not hardware kernel-active time. Both tmux handles terminal. Only gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d in existing1465 was used.

Deployment rule: `FP32 eval; live frozen R65B40 saved x plus live R66 member averaged x; coefficient1; full legal pool; highest summed direct utility; exact ties by ascending global solver ID; one final solver`. Every current raw input goes through BOTH live selectors, FP32 eval batch128 with tails, with caching disabled. Ordinary complete evaluation times above include both forwards, transfers, metrics and all prediction saves. No separate latency/singleton benchmark was run. Two selector forwards choose one completed solver/query; this cost-record evaluation executed zero solvers. TRAIN/VAL base-cache times are not live two-selector deployment timings.

## Delivery Artifacts

final_test_lock.json, final_test_started.json, test_data_manifest.json, test_results_A/B.json, test_results.json and training/test_terminal_receipt.json retain the one-time frozen delivery evidence. compute_summary.json, analysis.json, analysis_receipt.json and delivery_receipt.json retain budgets, decisions and this one CPU analysis pass.
comparison.csv/comparison_deltas.csv/selected_metrics.csv contain full18/family/split metrics; learning/task/family/optimization/update_curves.csv and task_optimization.csv retain curves. method_metrics.csv/task_diagnostics.csv retain single-method baselines, exceeded/not-exceeded lists and selection distributions. curves/ contains evaluation/optimization panels and per-task Top1/cost/distribution plots.
Goal decisions use only official locked TEST native Top1 >=70%, with no cost or margin veto. Nightmare100/originalfull18/full legal pools/single completed solver remains the scope. R58 is fully deferred; the legacy eight B-task feasible-domain mismatch remains unresolved and disclosed.
The actual Astra/max primary review used the full pre-R66-TEST reviewer-memory snapshot and TRAIN/validation evidence only: performance4/not-ready, inspected integrity9/ready, next-plan8/ready PLAN ONLY. A fresh Astra/max repository-reading delivery review found no actionable defects (9/ready for delivery only). Raw responses are in reviews/; both assessments are same-family/provisional, not acceptance of the70% goal. The adopted next plan is R67 native-winner contrastive pretraining, with the loss-scale attribution clarified in its execution instructions. New R66 TEST did not select that design or change official A.

One completed-version Git commit contains code, reports, tabular curves, figures and review records. Existing repository cleanup rules are respected: weights, prediction arrays, update traces, W&B and regenerable schedules/caches remain local, not newly force-added. final_test_lock.json retains the exact local locked bundle identities. DELIVERY.md records the final entrypoints and this publication boundary; the launch-time README/source snapshot is retained unchanged.
