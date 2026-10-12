# R65 Locked Delivery

Official **R65B, epoch 40**, was frozen by original validation integer count before TEST. Locked official TEST: **8575/18000 = 47.6389%**. The full18 native70% goal remains ACTIVE and unachieved.

Fresh paired seed2; same initialization, original data, task/row and dropout streams. Both use the unchanged native selector/loss and arithmetic-mean task gradients, global clip1, real256/task (effective768; tail16/task), fixed40E/9600updates/7.2M presentations each. Common effective LR3e-4 after720 successful-update warmup; no controller or early stop. A is ordinary AdamW; B is official non-wrapper AdamWScheduleFree and deploys its already-saved averaged x weights.

## Validation-Only Freeze

| Arm | Selected Epoch | VAL Correct | Parameter Mode | SHA256 |
| --- | ---: | ---: | --- | --- |
| A | 38 | 8599/18000 | ordinary | `23f65ba8f1202ec6bd91fce41f9dfa12dbf5ce9d50eab19522c0cee7116ebd51` |
| B | 40 | 8770/18000 | averaged x | `d880cb2ed4605f410742b4c37945494323213168af95d02b22794d232b9e7ad8` |

Within-arm selection uses strict maximum original integer validation count with earliest exact tie; official-arm exact ties choose A. Cost, regret and TEST do not select weights, parameter mode, inference rule or official arm. Zero posterior per-task assembly/reselection.

B minus A selected validation: **+171 correct, +0.950000pp**. This is one seed and one fixed LR. Relative to R64, common LR/controller changes are additional differences; only the new A/B comparison identifies this optimizer-package contrast.

## Saved Aggregate Metrics

| Model | Split | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R65A | train | 50.5561 | 82.4217 | 93.1906 | 1.063218 | 11.179115 | -1.063344 | 0.956103 |
| R65A | val | 47.7722 | 81.1556 | 91.4667 | 1.148225 | 11.183102 | -0.965331 | 1.044606 |
| R65A | test | 46.4722 | 80.4722 | 91.2611 | 1.159455 | 11.183157 | -0.918209 | 1.070499 |
| R65B | train | 49.3822 | 82.6967 | 93.1867 | 1.066195 | 11.173045 | -1.094776 | 0.926387 |
| R65B | val | 48.7222 | 82.1500 | 92.4833 | 1.090114 | 11.169640 | -1.056037 | 0.945528 |
| R65B | test | 47.6389 | 81.7000 | 92.3944 | 1.097159 | 11.171845 | -0.999267 | 0.978280 |
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

Native Top1/2/3 are class accuracies. vsSBS is100*(selected mean cost/SBS cost-1), negative is lower cost. Actual regret is the saved mean per-query relative excess over the full legal-pool oracle (%). ALL is the unweighted18-task macro; MVRP has15 tasks. Mixed-task mean cost is descriptive across heterogeneous task units. Detailed18-task/family/split records are in CSV, not duplicated here.

## Historical Validation and TEST

| Model | Epoch | VAL Top1 % | TEST Top1 % | VAL CE | TEST CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R65A | 38 | 47.7722 | 46.4722 | 1.148225 | 1.159455 |
| R65B | 40 | 48.7222 | 47.6389 | 1.090114 | 1.097159 |
| R45A | 38 | 49.1167 | 47.9111 | 1.089768 | 1.101119 |
| R60B | 39 | 49.3611 | 48.2444 | 1.092423 | 1.102036 |
| R64A | 40 | 48.6944 | 47.3778 | 1.094214 | 1.102747 |
| R64B | 40 | 48.7611 | 47.4111 | 1.094646 | 1.103343 |

R45A epoch38 and R60B epoch39 retain their historical official checkpoints and policies (R60B was cost-selected). R45A has no saved selected TRAIN evaluation. R64A/B reuse their saved locked tests; R64 official was B. Historical tests are exposed, not pristine independent evidence. Cross-version differences are descriptive. No reference inference, solver or prediction replay was performed; paths, JSON pointers and SHA256 provenance are in reference_provenance.json.

## Final and Last-Five Validation

| Arm | Best Top1 % | Final Top1 % | Last5 Mean % | Last5 Min % | Last5 Max % |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 47.7722 | 47.3778 | 47.4300 | 47.1889 | 47.7722 |
| B | 48.7222 | 48.7222 | 48.6833 | 48.6333 | 48.7222 |

validation_summary.csv contains all seven metrics for best/final/last5 mean, population SD, minimum and maximum; last5_validation.csv has each epoch. These are temporal summaries, not independent-seed uncertainty.

## Actual Execution and Inference

| Arm | Updates | Presentations | AMP Retries | Arm Wall s | Update Wall s | Peak Allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 9600 | 7200000 | 4 | 2987.142 | 2679.756 | 15.319 |
| B | 9600 | 7200000 | 4 | 2979.144 | 2678.194 | 15.319 |

Training: tmux r65_train / Slurm1465.92, COMPLETED 0:0; elapsed6014s, tmux terminal confirmed. TEST: tmux r65_final_test / Slurm1465.93, COMPLETED 0:0; elapsed22s, tmux terminal confirmed. Both use only assigned gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d in existing allocation1465.

Inference: `FP32 eval; full legal pool; highest direct utility; exact ties by ascending global solver ID`. One FP32 model, original18 tasks, all1000 TEST rows/task and full original legal pools, batch128 with tails retained, then one completed solver/query. Evaluation scores existing cost records; no solvers were executed. B uses averaged x; no x/y/z choice. TEST evaluation wall is19.344s for both models, including existing stage work. No new singleton latency benchmark was authorized or run; saved per-split evaluation seconds are in compute_summary.json.

## Artifacts and Remaining Work

final_test_lock.json, final_test_started.json, test_results.json and training_terminal_receipt.json/test_terminal_receipt.json record the lock and actual terminal evidence. analysis.json, analysis_receipt.json, compute_summary.json and delivery_receipt.json record this one CPU analysis pass.
The one CPU analysis process exited0 (owned exec session57600 settled), wall52.42s, user51.28s, system0.55s; its body timer was50.648484s. It generated391 comparison rows and329 PNGs. Actual process timing is retained in analysis.log and the analysis/compute/delivery receipts.
comparison.csv/comparison_deltas.csv contain aggregate/family/full18 metrics; learning_curves.csv, task_curves.csv, family_curves.csv, optimization_curves.csv, update_curves.csv and task_optimization.csv retain curves. method_metrics.csv/task_diagnostics.csv retain saved single-method comparisons and selection distributions. curves/ includes loss/CE/KL/effectiveLR/gradnorm/clip/retry/memory panels, full18/family Top1 curves and per-task method/cost/distribution plots.
R58 remains fully deferred; the legacy eight B-task feasible-domain mismatch remains unresolved and disclosed. Nightmare/100round/full18/single-solver/native70% scope is preserved, with no cost or margin veto on that target. Evidence is limited to this seed, LR and recipe.
Actual GPT-6-Astra/max primary review rated performance4/notready, TRAIN/validation integrity9/ready and the next R66 design8/ready PLANONLY; new R65 TEST was excluded from planning. A fresh direct-artifact Astra/max delivery review found no actionable defects and rated delivery9/ready. Both are same-family/provisional, not acceptance of the70% goal. Raw responses/traces are retained under reviews/ and .aris/traces/R65_*. One whole-version Git commit follows this completed delivery; no replay or corrective experiment was needed.
