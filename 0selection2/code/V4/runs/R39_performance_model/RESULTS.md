# R39: Single-Seed Performance-Model Screening

## Scope

The user reduced the original three-seed budget to seed 2 while the first A/B pair was running. Both runs completed all 60 epochs and 16,200 successful updates, with 18 problems, batch 640, natural sampling and from-scratch initialization. Seeds 3 and 4 were cancelled before starting. No test data was read.

An earlier A/seed2 attempt was deliberately interrupted after 20 epochs and 5,400 updates to repair end-of-run W&B summary handling. Its model, loss, initialization, scales and schedule matched the definitive protocol. It is preserved under archived_attempts/logging_fix_restart/, excluded from reported comparisons, and was not a separate negative-result configuration. The definitive A run restarted from scratch; no partial weights were reused.

A uses the R34 winner-cost recipe and classification argmax. B uses independent classification/performance heads, 0.35 CE + 0.35 centered-cost MSE, and performance argmin. Both have the same encoder, solver features and local geometry. Best checkpoints are selected by validation macro vs_SBS, not Top1. This is a single-seed screening result, not a training-seed robustness claim.

## Main Results

All rows below are full validation results. The percentage columns average per-problem percentages; they are not ratios of the aggregate mean costs.

| Policy | Epoch | Top1 | Top2 | Top3 | Mean cost | vs_SBS (%) | Actual regret (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R34 fixed reference | 50 | 0.484667 | 0.817333 | 0.925556 | 11.170804 | -1.054385 | 0.940542 |
| R39A best, classification | 53 | 0.484278 | 0.819833 | 0.927444 | 11.170767 | -1.055374 | 0.945072 |
| R39B best, performance | 47 | 0.467111 | 0.804000 | 0.914833 | 11.172470 | -1.032047 | 0.969520 |
| R39B classification diagnostic, same checkpoint | 47 | 0.485389 | 0.819722 | 0.926444 | 11.169960 | -1.059563 | 0.935625 |

R34 is a historical reference, not the matched from-scratch control. Its original checkpoint selection rule differs. R39A is the matched control for assessing R39B. The B classification row is diagnostic only: it is not substituted for B's declared performance policy or blended with it.

| B minus A | Top1 (percentage points) | Mean cost | vs_SBS (percentage points) | Actual regret (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Best validation checkpoints | -1.7167 | +0.001702 | +0.0233 | +0.0244 |
| Final epoch 60 | -1.8167 | +0.001615 | +0.0217 | +0.0241 |
| Mean of last five epochs | -1.7278 | +0.001463 | +0.0210 | +0.0230 |

The performance-decision recipe did not beat the control. The same direction appears at the final epoch and over the last five epochs, not just at selected checkpoints. Neither policy approaches the earlier 80% Top1 target.

## What Was Learned

The performance head did learn instance-dependent cost information. At B's selected epoch, its macro MSE is 0.208502; a train-only fixed solver-mean predictor has macro MSE 0.643484. B improves this baseline on all 18 problems. This comparison fits the baseline exclusively on training labels, then evaluates it on validation labels.

However, accurate full-vector regression is not sufficient evidence of accurate minimum-cost selection. B's performance and classification decisions disagree on 28.46% of validation instances. The performance policy corrects 1,755 classification errors but changes 2,084 classification-correct instances into strict-winner errors: a net loss of 329 correct selections. Its mean cost is also 0.002510 higher than its own classification head's cost. These are separate policies evaluated on the same instances and raw costs.

Family-level behavior is not a uniform failure: B lowers TSP mean cost by 0.002533 despite losing 3.0 percentage points of TSP Top1. It also lowers VRPTW cost by 0.023601. Other changes, including OVRP, OVRPL and VRPLTW, offset those savings. The per-problem tables and arm distributions are in comparison.md; no conclusion here is based on forcing more solver picks.

A possible explanation is that full-vector MSE rewards prediction of larger differences involving poor candidates more than precise ordering among the strongest candidates. This remains a hypothesis, not an established implementation defect or a conclusion about label noise. The current experiment changes both B's supervision and its decision rule; it does not identify which training-loss term causes each difference.

## Curves and Integrity Checks

- Each run contains loss_curve.png, ce_curve.png, top1_curve.png, mean_cost_curve.png, vs_sbs_curve.png, performance_mse_curve.png, family_top1.png and family_vs_sbs.png. Training-mode CE and evaluation-mode validation CE are explicitly distinguished.
- The root contains comparison_top1.png, comparison_ce.png, comparison_mean_cost.png, comparison_vs_sbs.png and comparison_performance_mse.png.
- Each selected checkpoint has per-problem single-method Top1/cost comparisons and arm-distribution plots under method_plots/.
- All 2,160 saved problem/epoch validation records reproduce native winners, original FP64 costs, actual decisions and metrics. See prediction_replay.json.
- All four best/last checkpoints are bound to the declared run and selected/final history record, load strictly, and reproduce the saved logits, performance predictions and selections exactly. See checkpoint_replay.json.
- Paired initial weights, batches, per-task dropout streams and successful update counts all match. See paired_checks.json.
- The complete V4 test discovery passed all 74 tests, including guards against missing runs, mislabeled results and best/last checkpoint substitution. The retained execution log is verification_unit_tests.log; unit_tests.log preserves the earlier 49-test prelaunch run. A's performance head is not supervised and must not be interpreted as a trained alternative policy.
- W&B was recorded offline. Its local run directories are inside each experiment directory; no online dashboard is claimed.
- The fresh semantic audit is same-family/provisional. Its verifier findings were repaired and the deterministic checks rerun; the original reviewer verdict is retained rather than upgraded by the executor. See EXPERIMENT_AUDIT.md.

## Decision

Keep R34 as the established main baseline; R39A is essentially at the same validation level. Do not promote R39B's performance-argmin policy. Extra seeds of this unchanged recipe are not the next priority.

For future iterations, screen with seed 2 first. The small B classification-head gain is worth keeping as diagnostic evidence, but is not a convincing breakthrough. The next useful controlled question is whether cost supervision can improve ordering among the strongest candidates without sacrificing the classification policy. This report does not launch an additional experiment or alter Q/K/V, solver features, or loss weights after observing these results.

The launch snapshot is retained under source_launch/. The early standalone-report repair is documented in post_launch_reporting_fix.json. After both runs finished, only default-budget/reporting logic was updated for the user's single-seed request; see post_training_budget_change.json. The original queue was deliberately terminated after B's completed marker, not because training failed.
