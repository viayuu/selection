# R56: Directed Node-Relation Encoder

## Protocol
One seed=2 specialist is trained from scratch on all 15 MVRP tasks, with ordinary unweighted BCE. The original mean/max pooling and binary head remain; the original four node-only layers and fixed distance bias are replaced by four d128/e32/four-head joint node/relation layers. Fixed R45A supplies the ranking; only original Top2={MOEL,MTL} decisions may change.

No solver reruns, new labels, R54 routes, R55 edge targets, source embeddings, retrieval, solver pretraining, behavior probe, regression, weighted BCE, risk or R-Drop are used. Original FP64 costs determine pair labels and evaluation. All non-ties are trained, including instances won by another full-pool solver.

## EGT Adoption and Limits
[EGT, KDD 2022, section3.2 equations3-7](https://arxiv.org/html/2108.03348v3) motivates persistent edge channels, clipped node QK interactions plus learned edge bias, post-softmax sigmoid gates and edge updates from the pre-softmax multihead interaction. Both streams have pre-normalization, residuals and FFNs. This is EGT-inspired, not a full reproduction.
Relations use the original ordered endpoint attributes (including coordinates, demand, depot, route limit, service and time windows), raw/mean-normalized distance, original constraints and real scale. Independent source/target projections allow E_ij != E_ji. All valid ordered pairs, self-pairs and the original three special tokens are retained; only padding is masked. No local-feasibility heuristic or label screens relations.
We use GELU/probability dropout0.1 rather than ELU/random attention masking; dynamic centrality scalers and SVD encodings are omitted. Terminal edge-only updates are computed for diagnosis but are not supervised by a node-only graph readout. Earlier edge residual updates and all four bias/gate projections do receive BCE gradients; no extra edge pooling or auxiliary head is added to hide this boundary.

## Integrated Results
| Split | Model | Top1 % | Mean cost | Actual regret % |
|---|---|---:|---:|---:|
| val | R45A | 49.1167 | 11.168517 | 0.926048 |
| val | R53A | 49.2667 | 11.167375 | 0.922663 |
| val | R54 | 49.0667 | 11.167717 | 0.924308 |
| val | R56 | 49.0111 | 11.168382 | 0.932960 |
| test | R45A | 47.9111 | 11.168366 | 0.943221 |
| test | R53A | 47.7611 | 11.168996 | 0.949729 |
| test | R54 | 47.7389 | 11.168989 | 0.952255 |
| test | R56 | 47.5556 | 11.169996 | 0.958129 |

R53A is a historical reference, not a newly rerun matched control. All percentages retain equal problem weighting; ALL raw mean-cost is not used to reconstruct macro ratios.

## Training and Selection
Best epoch 10; final epoch 18; 21330 successful updates. Last5 validation Top1 48.9667%, actual regret 0.940647%.
R53A batch128, AdamW LR1e-4/WD1e-4, dropout0.1, clip1, warmup3 and min15/max40 control are reused. All forty pre-generated epoch/task/index hashes match the R53 plan. Every successful epoch covers all non-tied training instances including tails. AMP overflow retries the same batch/RNG without consuming another planned update.
Checkpoint selection uses any strict new minimum full18 validation actual regret; LR/early stopping require improvement of at least0.001 percentage points. A tiny strict improvement need not reset the plateau counter. One best hash is locked before the one full test.

## Specialist Generalization
- train: pair accuracy 56.1674%, BCE 0.671062, two-method regret 0.810146%.
- val: pair accuracy 56.0569%, BCE 0.673975, two-method regret 0.822840%.
- test: pair accuracy 54.9203%, BCE 0.676982, two-method regret 0.828340%.

## Corrections and Harms
- val vs R45A: corrected 1195, harmed 1214; both-wrong cheaper/more expensive 178/180; net macro regret change +0.006912 pp; 7/15 MVRP tasks improve regret.
- test vs R45A: corrected 1117, harmed 1181; both-wrong cheaper/more expensive 157/204; net macro regret change +0.014908 pp; 4/15 MVRP tasks improve regret.

## Inference Cost
All15000 test expert forwards: 7.102 seconds. Synchronized same-batch FP32 benchmark, batch128/max nodes57: R53A 3.433ms, R56 30.820ms (+797.8%).
These include each expert Encoder/head, not the common frozen baseline ranking, data I/O or solver run. The whole test timing includes ungated examples for reporting; deployment only needs the expert on triggered inputs. One batch benchmark is not a full-node-size or full-system latency claim.

## Conclusion
- val vs R45A: Top1 -0.1056pp; regret relative reduction -0.746%; predeclared advancement screen: False.
- test vs R45A: Top1 -0.3556pp; regret relative reduction -1.581%; predeclared advancement screen: False.
This single configuration did not meet the joint validation/test meaningful-improvement screen. Do not claim success from evolving edges or training accuracy; no extra seeds/architectures were started.
R56 is not recommended as a replacement for R45A or the historical R53A specialist. At the selected checkpoint, expert train/validation accuracy and BCE are close, so this run does not reproduce R40's large fitting/generalization split; it instead fails to obtain a useful net selection gain with this joint relation encoder and training protocol. The roughly 9x same-batch specialist inference time makes the lack of benefit especially costly; this ratio must not be reported as full-system overhead.
This does not prove a hard accuracy ceiling, random labels or the absence of useful instance information.

## Artifacts
See protocol.json, precheck.json, runtime.json, sampling_plan.json, training_counts.csv, comparison.csv (ALL/MVRP/all18), binary_comparison.csv, corrections_and_harms.csv, learning_curves.csv/png, inference_latency.json, locked_checkpoint.json and per-instance predictions/{val,test}. Checkpoints, sampling plan tensors and offline W&B logs stay local. Historical models/data are unchanged.
