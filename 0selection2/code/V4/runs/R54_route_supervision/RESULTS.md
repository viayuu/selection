# R54: Route-structure supervision for the pair specialist

## Design
The R53A four-layer, 128-dimensional instance encoder and unweighted BCE head are initialized from scratch. Two training-only directed successor heads supervise the native best MOEL and MTL routes. Every valid customer predicts another customer or END; self and padded destinations are excluded. Loss is BCE plus the mean of the two per-instance successor losses, divided by log(valid candidate count).

Inference skips both auxiliary heads and executes no solver. Frozen R45A supplies the full ranking; only its original Top2={MOEL,MTL} decisions can change. Validation checkpoint selection remains integrated full18 actual regret, never route accuracy.

[Joshi et al. (2019), section 4.1](https://arxiv.org/html/1906.01227v2) motivates solution-edge supervision. The directed VRP successor/END objective is our adaptation, not a reproduction of their undirected weighted edge classifier or TSP decoding.

## Annotation Cost and Boundaries
Stored training routes: 300000 solver-instance pairs (150000 original instances), plus 1920 routes on a fixed 64-instance validation subset per task. Replay elapsed 1022.3 seconds. Historical costs were preserved and checked, not replaced.
Existing R48 original training routes were reused after input/cost checks. R52 validation trajectories never entered training. Native argmax, augmentation=1, sample=1 and task-specific POMO start restrictions were retained; independent FP64 route-distance sums checked native costs. Native legal-action masks were checked during decoding. Missing historical CLI/source provenance remains as documented in route_plan.json.

Only one seed and one new run were used. R53A is a historical control, with matching common initialization/sample order but a validation-driven schedule that can differ. No claim of a strict same-run auxiliary-loss ablation is made.

## Integrated Results
| Split | Model | Top1 % | Mean cost | Actual regret % |
|---|---|---:|---:|---:|
| val | R45A | 49.1167 | 11.168517 | 0.926048 |
| val | R53A | 49.2667 | 11.167375 | 0.922663 |
| val | R54 | 49.0667 | 11.167717 | 0.924308 |
| test | R45A | 47.9111 | 11.168366 | 0.943221 |
| test | R53A | 47.7611 | 11.168996 | 0.949729 |
| test | R54 | 47.7389 | 11.168989 | 0.952255 |

## Structure Generalization
Structure validation is measured on 960 untouched validation instances, not on POMO starts treated as independent examples. These routes are diagnostics only, not additional classifier inputs or checkpoint criteria.
Random-initialization validation successor accuracy: 15.95%/0.00% (MOEL/MTL). Saved best-checkpoint successor predictions reproduce the reported validation accuracies.
- train: MOEL/MTL successor accuracy 42.12%/42.34%; normalized CE 0.41560/0.41304.
- val: MOEL/MTL successor accuracy 41.83%/42.24%; normalized CE 0.41772/0.41508.

## Corrections and Harms
- val vs R45A: corrected 1154, harmed 1163; both-wrong cheaper/more expensive 163/159; net regret change -0.001741 pp; regret improved on 5/15 MVRP tasks.
- val vs R53A: corrected 956, harmed 992; both-wrong cheaper/more expensive 121/111; net regret change +0.001645 pp; regret improved on 5/15 MVRP tasks.
- test vs R45A: corrected 1072, harmed 1103; both-wrong cheaper/more expensive 137/163; net regret change +0.009035 pp; regret improved on 7/15 MVRP tasks.
- test vs R53A: corrected 976, harmed 980; both-wrong cheaper/more expensive 121/111; net regret change +0.002527 pp; regret improved on 7/15 MVRP tasks.

## Conclusion
Best epoch 8; stopped at 15; successful updates 17775. Last5 validation: Top1 49.0300%, actual regret 0.930235%.
The prespecified system-level breakthrough screen is not met on both splits. Learning route connectivity alone has not demonstrated the required transferable solver-selection improvement. This does not establish a noise ceiling or prove that all structural supervision is ineffective.

Full task/family metrics, binary metrics and structural metrics are separate CSVs. Checkpoints, full route archives and W&B logs remain local; hashes and recipes are committed. Test used one locked validation-best checkpoint, without auxiliary heads or test-time solver execution.
