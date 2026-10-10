# R55: Billed-edge cost-difference bottleneck

## Design and Boundaries
One scratch seed=2 model reuses the original four-layer, 128-dimensional instance Encoder. There is no graph-level binary head. A shared symmetric decoder predicts signed billed-count differences on every valid undirected edge, including depot edges. The only decision is the FP32 sum of predicted differences times input Euclidean distances; positive/zero selects MOEL, negative selects MTL.

D = A_MTL - A_MOEL. Counts retain multiplicity; a closed single-customer route charges the same undirected depot edge twice. Open-route directed returns to the depot are removed before merging directions. Self/padding edges are excluded. No true route, cost, Oracle, winner or true edge mask enters forward.

The objective is group-balanced edge MSE plus relative cost-difference MSE with the fixed 1% full-pool Oracle scale. Positive/negative/zero groups are averaged per instance and then equally combined among nonempty groups. There is no BCE, risk, successor CE, maximin, R-Drop, solver ID or post-sum learned classifier. Raw cost differences are not reported as calibrated classification logits.

[Koh et al., Concept Bottleneck Models, ICML 2020, sections 2-3](https://proceedings.mlr.press/v119/koh20a.html) motivates requiring final predictions to pass through supervised intermediate predictions rather than attaching an auxiliary task. Our signed route-count concepts and fixed distance readout are project-specific, not a replication of their medical/bird experiments.

## Data Conversion
Reused 300000 saved R54 training routes and 1920 fixed diagnostic validation routes. No new solver execution or annotation. Converted 150960 instances. The largest historical cost-difference reconstruction error is 1.30225836e-05; the largest purely algebraic identity error is 2.99413272e-14.
Original cost/winner files were not modified. Historical cost tolerance remains ATOL=5e-5, RTOL=2e-6 per method; difference tolerance is the sum of the two cost tolerances. Independent geometry uses FP64 sums after native FP32 coordinate conversion. Target counts are stored losslessly as int8 in the upper triangle.

Training uses the same 149976 non-tied queries (24 exact ties excluded), fresh shared Encoder initialization, complete shuffled coverage including tails, and interleaved task plan as R53/R54. Validation routes cover only a fixed 64-instance subset/task and are diagnostic-only; full18 validation selection uses original scalar costs and predicted signs. All normalizers are fit on training inputs only.

## Integrated Results
| Split | Model | Top1 % | Mean cost | Actual regret % |
|---|---|---:|---:|---:|
| val | R45A | 49.1167 | 11.168517 | 0.926048 |
| val | R53A | 49.2667 | 11.167375 | 0.922663 |
| val | R54 | 49.0667 | 11.167717 | 0.924308 |
| val | R55 | 48.9778 | 11.169596 | 0.939770 |
| test | R45A | 47.9111 | 11.168366 | 0.943221 |
| test | R53A | 47.7611 | 11.168996 | 0.949729 |
| test | R54 | 47.7389 | 11.168989 | 0.952255 |
| test | R55 | 46.8500 | 11.171375 | 0.977167 |

Best epoch 24; final epoch 24; 28440 successful updates. Best was the strict minimum integrated full18 validation actual regret. Last-five validation Top1 48.3544%, regret 0.961436%.
The fixed R45A Top2 trigger is unchanged. TSP/CVRP/ATSP decisions are untouched. Per-problem and MVRP macro results are in comparison.csv. All percentages preserve equal problem weights and native winner labels; raw costs are accumulated from original FP64 tables.

## Bottleneck Generalization
Initial diagnostic validation edge loss: 0.669096; all-zero prediction reference: 0.669095.
- train: pair sign accuracy 55.37%, pair regret 0.820900%; edge MSE 0.668656, zero-edge reference 0.668661; relative cost-difference MSE 10.554965, zero-delta reference 11.988443.
- val: pair sign accuracy 56.27%, pair regret 0.826681%; edge MSE 0.669090, zero-edge reference 0.669095; relative cost-difference MSE 10.973383, zero-delta reference 12.391591.
Saved diagnostic edge matrices reconstruct the final scalar outputs and grouped edge losses (bottleneck_replay.csv). Edge reconstruction alone is not evidence of better method choice. The predicted matrix is not constrained to a feasible route and is not a new solver.

## Corrections and Harms
- val vs R45A: corrected 1382, harmed 1407; both-wrong cheaper 165, more expensive 173; net macro regret change +0.013722 pp; 4/15 MVRP tasks have lower regret.
- val vs R53A: corrected 1588, harmed 1640; both-wrong cheaper 228, more expensive 230; net macro regret change +0.017108 pp; 4/15 MVRP tasks have lower regret.
- val vs R54: corrected 1454, harmed 1470; both-wrong cheaper 197, more expensive 209; net macro regret change +0.015462 pp; 5/15 MVRP tasks have lower regret.
- test vs R45A: corrected 1278, harmed 1469; both-wrong cheaper 178, more expensive 192; net macro regret change +0.033947 pp; 3/15 MVRP tasks have lower regret.
- test vs R53A: corrected 1505, harmed 1669; both-wrong cheaper 250, more expensive 228; net macro regret change +0.027439 pp; 3/15 MVRP tasks have lower regret.
- test vs R54: corrected 1350, harmed 1510; both-wrong cheaper 222, more expensive 210; net macro regret change +0.024912 pp; 2/15 MVRP tasks have lower regret.

## Inference Cost
All 15000 MVRP test-instance expert forwards (including Encoder and all-edge readout): 1.079 seconds. This includes ungated examples for reporting, not just deployed trigger calls.
Synchronized FP32 GPU benchmark, batch128/max nodes 57: R53A Encoder+binary head 3.448 ms; R55 Encoder+edge-sum 5.084 ms (+47.4%); isolated edge decoder 2.228 ms.
Warmup5/repeats30 on the same idle RTX3090. The common R45A ranking and CPU data loading are excluded, so these are expert/additional-selection costs, not complete route-solving or full selector system latency. No solver is executed at inference.

## Conclusion
- val vs R45A: Top1 -0.1389 pp; regret relative reduction -1.482%; predeclared advancement screen passed: False.
- test vs R45A: Top1 -1.0611 pp; regret relative reduction -3.599%; predeclared advancement screen passed: False.
This single-seed configuration did not achieve the predeclared selection improvement. Do not replace R45A based on intermediate edge metrics or small isolated differences; no additional weight, seed or architecture search was started.
This tests the full supervised-difference bottleneck, not the hypothesis that labels are random or that selection accuracy has a hard ceiling. A good edge loss does not ensure that signed errors cancel accurately enough to recover small true cost differences.

## Reproduction and Artifacts
`bash code/V4/run_v4_r55.sh all` with the authorized single idle GPU binding. Source and exact Slurm/tmux launch are in R55_README.md and execution.json. Code, configs, checks, curves, per-instance predictions and reports are versioned; large target caches, initialization, best/last weights and offline W&B logs remain local.
Validation-best SHA256 was locked before the one complete test inference. Saved full18 predictions reproduce all final metrics. Existing R45A/R53A/R54 checkpoints and results were preserved.

## Post-hoc Bottleneck Interpretation
This read-only diagnosis uses the locked checkpoint outputs on the existing fixed 960 validation route instances. It does not select a new checkpoint, change inference, or start another experiment.
Best epoch 24, final epoch 24: checkpoint saving uses any strict new validation minimum, while early stopping requires an improvement of at least 0.001 percentage points. A smaller strict minimum need not reset the plateau counter.
Mean absolute predicted billed-count difference: 9.32344277e-05; maximum absolute prediction: 0.00971249957. Predicted absolute counts at least 0.5: 0.000000% of valid edges; true nonzero edges: 2.4417%.
On true positive/negative difference edges, mean predictions are -0.000131404051/-0.000138666293. Group-balanced edge MSE is 0.669090350, versus 0.669095103 for all-zero predictions.
- train pair accuracy: 55.3666%; always-MTL reference: 54.6866%; model selects MOEL on 36.5678% of instances.
- val pair accuracy: 56.2703%; always-MTL reference: 54.4636%; model selects MOEL on 36.2476% of instances.
Every predicted difference on this diagnostic subset would round to zero as an integer count. The nonzero final scalar therefore comes from summing small continuous edge estimates, not from recovering the actual discrete route-count differences. The structural readout is connected, but this run did not learn a faithful difference-edge bottleneck. It does not establish whether accurate difference reconstruction would improve selection.
The fixed readout also uses observed input distances. This is a task-specific supervised structural bottleneck, not a strict concept-only replica of the cited paper. Loss-gradient scaling, decoder capacity and conditional predictability are not separated by this single configuration; no unique cause is established.
