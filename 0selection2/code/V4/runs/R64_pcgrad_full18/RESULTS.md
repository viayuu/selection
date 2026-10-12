# R64 Larger-Batch Locked Delivery

Recorded 2026-10-12T16:24:49.000+08:00. Both fresh arms completed normally. Official arm **B_pcgrad_seed2, epoch 40** was frozen using validation only before TEST. Its locked original full18 TEST native Top1 is **47.4111% (8534/18000)**. The full18, full legal pool, single-solver 70% goal is **not achieved** and remains active.

## Frozen Selection and Execution Receipts

| Arm | Epoch | Original VAL Correct | Original VAL Native Top1 % | TRAIN Correct | TEST Correct | TEST Native Top1 % |
| --- | --- | --- | --- | --- | --- | --- |
| A | 40 | 8765 / 18000 | 48.6944 | 87344 / 180000 | 8528 / 18000 | 47.3778 |
| B (official) | 40 | 8777 / 18000 | 48.7611 | 87508 / 180000 | 8534 / 18000 | 47.4111 |

Selection compares integer total native validation correct counts, retains the earlier epoch on an exact within-arm tie, and chooses A on an exact between-arm tie. B wins validation by 12 correct predictions (+0.066667pp). Cost, regret, pair diagnostics and TEST did not select the checkpoint or official arm. Each arm has one existing unique best.pt; both selected epochs also happen to be final epoch 40. No weights or per-task model combinations were assembled.

- A SHA256: `a51670b1dbbe4634e875dfd1d0710d904335942abbd9cfed176a98aa61865d45`.
- B SHA256: `b455e93db94d99409179e4a623147e90fad64ebd26fcfed46f14d7ef2f98d86e`.
- Freeze: `2026-10-12T16:09:30.935447+08:00`, after both result files and validation_candidates.json were published, under the parent's explicit mechanical-lock delegation in [R64_locked_delivery_executor.prompt.md](../../../../review-stage/r58-to70/R64_locked_delivery_executor.prompt.md).
- Inference: `FP32 eval; full legal pool; highest direct utility; exact ties by ascending global solver ID`. Original 18 tasks, 1000 TEST rows/task, full original legal pools, one solver per query, eval-mode FP32, evaluation batch 128.
- Training: owned `tmux r64_train / Slurm1465.90`, **COMPLETED 0:0**, 2026-10-12 14:28:07 to 16:07:16 CST, actual step elapsed **5949s**. The tmux handle was gone when observed at 16:08:27 CST.
- TEST: owned `tmux r64_final_test / Slurm1465.91`, **COMPLETED 0:0**, 2026-10-12 16:09:44 to 16:10:06 CST, actual step elapsed **22s**. The handle was gone when observed at 16:10:42 CST.
- Sole GPU: `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, gpu03/index1 RTX3090, existing allocation1465. The normal TEST-launch receipt records only that UUID, idle at 2MiB with no compute clients. Its legacy tmux_session field says r64_train; the actual TEST tmux handle was r64_final_test, as recorded here.
- Existing frozen-source/SHA checks accepted the lock before TEST. No source, README or snapshot was edited during this delivery. The exclusive final_test_started.json marker was created once; both models received one complete original TEST. No historical model was rerun.
- Existing CPU analysis executed **once, rc0**, completed artifact at `2026-10-12T16:14:42.498521+08:00`; it wrote analysis.json, four curve CSVs, selected_metrics.csv and **328 PNGs**. Analysis CPU process wall was not separately instrumented. No prediction or metric replay was used for this report.

Primary artifacts: [lock](final_test_lock.json), [TEST start](final_test_started.json), [TEST results](test_results.json), [TEST receipt](launch_receipt_20261012T080944762483Z.json), [analysis](analysis.json), [delivery receipt](delivery_receipt.json). Each arm retains 18 saved TEST NPZs, 18 selected TRAIN NPZs and 18 selected validation NPZs in its test_predictions, train_predictions/best and val_predictions/best directories.

## Paired Recipe and Batch Amendment

Fresh identical seed2 R45A dual-stream models use handcrafted34+ID, d128, four instance layers, two joint layers, four heads, FFN512, head hidden256 and dropout .1. Local geometry is fitted on original TRAIN. Both use direct encode_pairs+score_head utility logits with native CE plus the existing symmetric RDrop, ramping to .35 over three data epochs. AdamW remains LR1e-4, WD1e-4, betas(.9,.999), epsilon1e-8, with one final global clip1. There is no LR scaling.

Each epoch has 40 cycles. Every cycle shuffles all18 distinct tasks into six triples; each task supplies its next256 original rows, or its final16. Effective batch is normally768 and finally48. All three task gradients use the same parameter state followed by one successful optimizer step. This retains all180000 original TRAIN presentations/epoch and gives exactly240 successful updates/epoch. Both arms completed40 epochs, **9600 updates and 7.2M presentations each**. Warmup and RDrop ramp are720 updates. A's saved actual LR trace and40-epoch horizon were replayed by B.

B projects only the named shared dense prefixes: instance_encoder.layers, solver_encoder.feature_mlp, solver_encoder.norm, joint_layers, pool, decoder and score_head. Mutable task-gradient copies are projected against original other gradients in independently shuffled order using separate RNG6402, skipping zero denominators. A uses the arithmetic mean; nonprojected parameters use the ordinary mean in both arms. Gradients are unscaled FP32 before reduction. No per-task norm normalization or projection of weight decay is used. Nonfinite AMP gradients cause a whole-triple retry with RNG restoration and no successful update/sample advancement.

The user explicitly stopped the earlier batch42 run after A14 complete epochs plus partial15 (21383 updates); B never started and no TEST ran. It remains archived as [ABORTED batch42](../R64_pcgrad_full18_aborted_batch42_20261012T140928/ABORTED_BY_USER.json). Neither trained state nor results were reused. The old run is not a matched batch-effect control.

The adopted larger-batch amendment preserves maximum7.2M presentations but reduces the originally planned **57360 updates to9600 per arm**. Thus old/new comparisons cannot isolate batch effects or separate batch from the changed update budget. Next planning must consider measured throughput and matched update/sample budgets; filling memory is not an accuracy criterion.

Exactly one authorized TRAIN-only sizing pass used25.706883s, 20 discarded successful updates and19200 presentations. Selected256 had B throughput892.870queries/s and25.7971% physical headroom. Batch384 completed but had only2.9441% headroom;512 genuinely OOMed in both arms. All calibration state was discarded. Smaller-node first-formal throughput is not directly compared with this worst-node sizing throughput. See [batch decision](large_batch_decision.json), [calibration cost](calibration_cost.json) and [adopted amendment](adopted_large_batch_plan.response.md).

## Saved Native Metrics

Top1/2/3 below are native class accuracies, expressed in percent. CE and mean_cost retain their saved units. vsSBS is100*(selected mean cost/SBS cost-1); negative is lower cost. Actual regret is the saved mean per-query relative cost excess over the legal-pool oracle, in percent. ALL is the unweighted18-task macro; MVRP is the15 non-TSP/CVRP/ATSP tasks. Mixed-task mean costs remain a descriptive macro in heterogeneous task units.

| Model | Split | Native Top1 % | Top2 % | Top3 % | CE | Mean Cost | vs SBS % | Actual Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R64A | train | 48.5244 | 82.1294 | 92.7939 | 1.086541 | 11.178202 | -1.068215 | 0.950804 |
| R64A | val | 48.6944 | 82.0556 | 92.5778 | 1.094214 | 11.169958 | -1.057617 | 0.938619 |
| R64A | test | 47.3778 | 81.4500 | 92.2167 | 1.102747 | 11.173794 | -0.992692 | 0.981460 |
| R64B | train | 48.6156 | 82.1272 | 92.7806 | 1.085773 | 11.178306 | -1.069022 | 0.949743 |
| R64B | val | 48.7611 | 82.1111 | 92.4889 | 1.094646 | 11.170174 | -1.055205 | 0.940973 |
| R64B | test | 47.4111 | 81.2444 | 92.2500 | 1.103343 | 11.173766 | -0.995764 | 0.977711 |
| R45A | val | 49.1167 | 82.4444 | 92.7611 | 1.089768 | 11.168517 | -1.070041 | 0.926048 |
| R45A | test | 47.9111 | 81.7444 | 92.4333 | 1.101119 | 11.168366 | -1.032003 | 0.943221 |
| R60B | val | 49.3611 | 82.2389 | 92.7222 | 1.092423 | 11.168642 | -1.073544 | 0.922789 |
| R60B | test | 48.2444 | 81.7222 | 92.5111 | 1.102036 | 11.168404 | -1.035671 | 0.940486 |

[comparison.csv](comparison.csv) contains **230 rows** covering every18 task and ALL/TSP/CVRP/ATSP/MVRP for R64A/B TRAIN/validation/TEST, plus R45A/R60B validation/TEST, with all seven required metrics. Values are extracted from saved analysis and official reference metrics. No solver or prediction is recomputed.

### Full18 Native Top1

| Task | A TRAIN % | B TRAIN % | A VAL % | B VAL % | A TEST % | B TEST % |
| --- | --- | --- | --- | --- | --- | --- |
| TSP | 39.4200 | 39.5400 | 37.9000 | 37.6000 | 37.0000 | 36.7000 |
| CVRP | 45.4100 | 45.2300 | 48.0000 | 47.2000 | 44.4000 | 45.1000 |
| ATSP | 76.8300 | 76.3300 | 78.8000 | 78.3000 | 77.2000 | 76.3000 |
| OVRP | 37.4200 | 37.4500 | 34.9000 | 36.3000 | 35.5000 | 35.3000 |
| VRPB | 54.3800 | 54.3500 | 55.1000 | 55.5000 | 49.7000 | 49.6000 |
| VRPL | 33.0900 | 33.4300 | 30.8000 | 31.7000 | 31.0000 | 30.6000 |
| VRPTW | 46.4800 | 46.6200 | 45.9000 | 44.9000 | 44.6000 | 44.5000 |
| OVRPTW | 50.8900 | 51.1900 | 50.8000 | 50.8000 | 46.7000 | 46.4000 |
| OVRPB | 56.3300 | 56.8900 | 56.2000 | 57.1000 | 54.3000 | 55.8000 |
| OVRPL | 38.0200 | 38.0800 | 41.1000 | 41.3000 | 37.7000 | 37.2000 |
| VRPBL | 53.2400 | 53.3500 | 54.2000 | 54.9000 | 56.3000 | 55.9000 |
| VRPBTW | 44.7400 | 44.7100 | 44.6000 | 45.4000 | 44.1000 | 44.0000 |
| VRPLTW | 45.9400 | 45.9100 | 46.3000 | 46.1000 | 45.9000 | 46.6000 |
| OVRPBL | 56.0800 | 55.8100 | 56.7000 | 57.1000 | 56.6000 | 57.0000 |
| OVRPBTW | 48.9800 | 49.1700 | 51.6000 | 50.2000 | 51.6000 | 51.5000 |
| OVRPLTW | 50.6900 | 51.4800 | 49.7000 | 49.2000 | 48.1000 | 48.4000 |
| VRPBLTW | 45.5000 | 45.4000 | 43.5000 | 43.8000 | 43.5000 | 43.2000 |
| OVRPBLTW | 50.0000 | 50.1400 | 50.4000 | 50.3000 | 48.6000 | 49.3000 |

### Family Native Top1

| Scope | Split | A Native Top1 % | B Native Top1 % |
| --- | --- | --- | --- |
| ALL | train | 48.5244 | 48.6156 |
| ALL | val | 48.6944 | 48.7611 |
| ALL | test | 47.3778 | 47.4111 |
| TSP | train | 39.4200 | 39.5400 |
| TSP | val | 37.9000 | 37.6000 |
| TSP | test | 37.0000 | 36.7000 |
| CVRP | train | 45.4100 | 45.2300 |
| CVRP | val | 48.0000 | 47.2000 |
| CVRP | test | 44.4000 | 45.1000 |
| ATSP | train | 76.8300 | 76.3300 |
| ATSP | val | 78.8000 | 78.3000 |
| ATSP | test | 77.2000 | 76.3000 |
| MVRP | train | 47.4520 | 47.5987 |
| MVRP | val | 47.4533 | 47.6400 |
| MVRP | test | 46.2800 | 46.3533 |

### Historical Official References

R45A is the existing handcrafted model at epoch38; R60B is the existing cost-selected epoch39 model and predeclared incumbent. Their original checkpoints, historical inference and selection policies are retained. Historical tests are already exposed; they are not pristine independent evidence. Cross-version differences are descriptive and are not controlled mechanism or batch-effect estimates.

| Task | R45A VAL % | R45A TEST % | R60B VAL % | R60B TEST % |
| --- | --- | --- | --- | --- |
| TSP | 38.5000 | 35.9000 | 37.6000 | 36.7000 |
| CVRP | 47.9000 | 48.1000 | 49.3000 | 48.3000 |
| ATSP | 78.5000 | 77.5000 | 78.4000 | 77.6000 |
| OVRP | 38.0000 | 37.4000 | 36.6000 | 36.3000 |
| VRPB | 55.4000 | 51.7000 | 55.2000 | 51.9000 |
| VRPL | 33.5000 | 30.2000 | 34.3000 | 30.7000 |
| VRPTW | 47.0000 | 43.5000 | 46.7000 | 43.9000 |
| OVRPTW | 51.0000 | 49.1000 | 51.5000 | 49.1000 |
| OVRPB | 58.3000 | 55.1000 | 59.9000 | 57.9000 |
| OVRPL | 40.2000 | 37.0000 | 40.7000 | 37.7000 |
| VRPBL | 52.0000 | 54.7000 | 52.2000 | 55.3000 |
| VRPBTW | 44.4000 | 44.3000 | 42.7000 | 44.7000 |
| VRPLTW | 45.7000 | 44.5000 | 45.6000 | 45.4000 |
| OVRPBL | 58.8000 | 57.4000 | 59.2000 | 57.7000 |
| OVRPBTW | 50.0000 | 51.4000 | 50.8000 | 51.4000 |
| OVRPLTW | 50.8000 | 50.2000 | 52.6000 | 50.6000 |
| VRPBLTW | 44.0000 | 44.9000 | 42.8000 | 44.7000 |
| OVRPBLTW | 50.1000 | 49.5000 | 52.4000 | 48.5000 |

Reference paths, JSON pointers and source metric-file SHA256s are persisted in [reference_provenance.json](reference_provenance.json). R45A TEST is extracted from R45_solver_code_embeddings/test_results.json at /A; R60B TEST reuses R63_recurrent_workspaces/reference/R60B/test_eval.json. Validation is extracted from each existing result.json at /best/val. No R45A/R60B or older model was evaluated again.

### Differences

| Comparison | Split | Native Top1 Delta pp | CE Delta | Mean Cost Delta | Regret Delta pp |
| --- | --- | --- | --- | --- | --- |
| R64B minus R64A | train | 0.091111 | -0.000768 | 0.000104 | -0.001062 |
| R64B minus R64A | val | 0.066667 | 0.000432 | 0.000216 | 0.002354 |
| R64B minus R64A | test | 0.033333 | 0.000596 | -0.000028 | -0.003749 |
| R64A minus R45A | val | -0.422222 | 0.004446 | 0.001442 | 0.012571 |
| R64A minus R45A | test | -0.533333 | 0.001629 | 0.005428 | 0.038239 |
| R64B minus R45A | val | -0.355556 | 0.004878 | 0.001658 | 0.014925 |
| R64B minus R45A | test | -0.500000 | 0.002224 | 0.005400 | 0.034490 |
| R64A minus R60B | val | -0.666667 | 0.001791 | 0.001316 | 0.015830 |
| R64A minus R60B | test | -0.866667 | 0.000711 | 0.005390 | 0.040974 |
| R64B minus R60B | val | -0.600000 | 0.002223 | 0.001532 | 0.018184 |
| R64B minus R60B | test | -0.833333 | 0.001307 | 0.005362 | 0.037225 |

[comparison_deltas.csv](comparison_deltas.csv) also contains all five aggregate/family scopes. Positive Top1 deltas mean higher accuracy; CE, cost and regret deltas retain their ordinary directions. Official B is below R60B by0.600000pp on validation and0.833333pp on TEST, and below R45A by0.355556pp and0.500000pp respectively.

The predeclared B-over-A validation selection condition is met by12 correct predictions. The locked TEST difference is6 correct predictions (+0.033333pp). This is provisional single-seed evidence for this fixed recipe; no additional margin, cost or regret veto is introduced. It establishes neither a reliable larger effect nor harmful gradient interference, a70% ceiling or a label-noise explanation. TEST has not changed any checkpoint, rule, official arm or future mechanism selection.

## Final and Last-Five Validation

Both final/last epochs are40, with A8765 and B8777 native correct. The last-five epochs are36-40. Their mean native Top1 is A48.4133% and B48.3489%; last-five averages do not replace the frozen maximum-count checkpoint criterion. Full per-epoch native Top1/2/3, CE, mean cost, vsSBS and actual regret are saved in [last5_validation.csv](last5_validation.csv), and all40 epochs appear in learning_curves.csv/task_curves.csv/family_curves.csv.

A's normal selected-validation timing evaluation (first16/task singleton, remainder batched) records8767 correct, **+2** relative to the original epoch40 selection evaluation8765. B's corresponding count is8777, delta0. These observed evaluation-grouping differences are disclosed as saved; no additional numerical replay was run. Locks and reported selection validation metrics use the original epoch evaluations, and the supplemental timing evaluation never reselects.

Full eval-mode TRAIN, not online batches, was evaluated every5 epochs and at termination/selection. Selected TRAIN contains all180000 original rows. Training curves are shown only where full TRAIN evaluation exists.

## Actual Compute and Inference

| Arm | Epochs | Updates | Presentations | AMP Triple Retries | Arm Wall s | Epoch Loop s | Epoch Loop Queries/s | Peak Allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 40 | 9600 | 7200000 | 5 | 2943.004 | 2690.682 | 2675.902 | 15.324180 |
| B | 40 | 9600 | 7200000 | 4 | 2933.606 | 2682.552 | 2684.012 | 15.324180 |

Combined recorded formal-arm wall is5876.609646s (1.632392h), including ordinary per-arm evaluation/finalization. The training Slurm step5949s additionally includes preparation/calibration and launcher overhead; do not add the25.706883s calibration again to that step total. TEST wall inside the existing helper is19.745293s, within its22s Slurm step. Combined new train+TEST Slurm elapsed is5971s (1.658611h). The archived aborted batch42 execution remains a separate preserved cost record. No GPU-hour cap was imposed.

Epoch-loop throughput divides7.2M presentations by saved epoch-loop seconds, which include normal optimizer/checkpoint/retry work and exclude subsequent epoch validation/TRAIN evaluation. It is a measured workload average, not a direct comparison to worst-node sizing or the earlier first update.

| Arm | Original Conflicts / Comparable Pairs | Conflict % | Mean Reduced Gradient Norm | Mean Clip Coefficient | Mean Projection Change Norm | Mean Relative Projection Change |
| --- | --- | --- | --- | --- | --- | --- |
| A | 12582 / 28800 | 43.6875 | 1.281906 | 0.821298 | 0.000000 | 0.000000 |
| B | 12594 / 28800 | 43.7292 | 1.395831 | 0.794300 | 0.108290 | 0.112601 |

Original negative-dot pairs are12582/28800 for A and12594/28800 for B. B's mean projected-mean change is0.108290 norm and mean relative change0.112601. These are normal-training observations; original per-task full/shared gradient norms, independent projection orders and per-task change magnitudes remain in updates.jsonl/history.json. Negative alignment alone is not proof of harmful interference, and pair diagnostics are not the success criterion. Mean reduced norm is measured before the final global clip.

| Arm | Singleton Mean ms | Batched Queries/s | Normal Selected VAL Wall s | Locked TEST Eval s |
| --- | --- | --- | --- | --- |
| A | 6.146164 | 10402.988 | 3.624781 | 8.478108 |
| B | 5.942426 | 10570.764 | 3.536440 | 7.406531 |

These timings came once from each ordinary selected validation evaluation:288 singleton calls (16/task) and17712 batched queries across144 calls. Scope is cached input gather, forward and CPU output transfer, excluding disk decoding, geometry-cache preparation, checkpoint loading and solver execution. Eval mode, FP32, torch.no_grad. No standalone timing pass was added. [compute_summary.json](compute_summary.json) preserves these summaries and launch/wall records.

## Provenance and Remaining Scope

Original TRAIN/validation-only R64 mechanism planning was GPT-6-Astra/max: [closure and eight specifications](../R63_recurrent_workspaces/reviews/R63_closure_R64_plan.response.md), with .aris/traces/R63_closure_R64_plan.events.jsonl. The adopted larger-batch plan was GPT-6-Astra/max text/context planning; its permitted tools did not read local file contents or perform a direct local-code audit. Implementation, bounded sizing, launch and this delegated locked delivery were requested GPT-6.1-Sol/max execution, with .aris/traces/R64_implementation_launch.events.jsonl, R64_large_batch_execution.events.jsonl and R64_locked_delivery_execution.events.jsonl. No new reviewer gate or agent was added in this delivery.

R63 remains fully complete and pushed15363d0ec. Its locked A34, historical TEST and floating-tie history are unchanged; R64 uses the prospective integer-count rule. External legacy B semantics remain unresolved and R58 is deferred. Full18/fullpools/native winners/single-solver70% stays active. No R65/new experiment, solver changes, other-GPU/client/allocation/environment operation or Git add/commit/push occurred. Parent retains the final minimal Astra review/planning and one whole-version commit/push.
