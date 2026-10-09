# R48: core two-solver diagnosis

Scope: OVRPTW, RELD_MOEL (class0) versus RELD_MTL (class1), seed2. Train/val only.
No main-model edit, no 18-task retraining, no test evaluation. Only the authorized idle RTX3090 was used.

## 1. Historical labels and independent route validation

Captured route rows: 1536; expected: 1536. Blocking anomalies: 0.
Maximum absolute historical-cost error on original inputs: 4.9610093810770195e-11.
Maximum absolute independent FP64 route-cost error versus solver scalar: 1.0660656251104683e-06.

The exact R41 two solver checkpoints, argmax decoding, all-customer POMO starts, no augmentation, sample_size=1 and minimum over starts are retained. All six solver input fields are checked after the historical FP32 conversion. Actual routes, winning start IDs and customer remappings are saved in solver_runs/.
The independent NumPy checker checks each customer exactly once, normalized capacity=1, speed=1, waiting/service/customer time windows, and open route distance. It does not charge customer-to-depot returns or treat the depot placeholder time window as a real deadline. Feasibility tolerance=1e-5; cost agreement uses atol=2e-5, rtol=2e-6. Both original-input and runtime-FP32-input FP64 costs are recorded.

| Split | Solver | Repeated instances | Exact scalar changes | Maximum absolute change |
|---|---|---:|---:|---:|
| train | RELD_MOEL | 64 | 0 | 0 |
| train | RELD_MTL | 64 | 0 | 0 |
| val | RELD_MOEL | 64 | 0 | 0 |
| val | RELD_MTL | 64 | 0 | 0 |

Checkpoint/path/config evidence applies to the pinned, locally available R41 weights; the unrecorded original historical checkpoint digest cannot be retroactively recovered. Matching reruns support the sampled rows, not a claim that every historical solver deployment has been audited.

## 2. Customer-node permutation

Four fixed customer permutations per R41 instance; depot stays fixed and demand/service/time-window fields move with coordinates. Routes are mapped back to original customer IDs before the same independent check.
Frozen R47 keeps the original full seven-method pool and train-only memory. Its two actual maximin scores are compared without rebuilding a smaller candidate pool. No solver labels enter its query inputs.

| Split | Instances | Permutations | Strict winner flips | Instances with flips | R47 binary changes | Max score change |
|---|---:|---:|---:|---:|---:|---:|
| train | 64 | 256 | 0 | 0 | 0 | 0.00583927333 |
| val | 64 | 256 | 0 | 0 | 0 | 0.00086170435 |

All permutation deltas use original_0 from the same fixed sample batch as their baseline. Differences from full-split inference are retained in separate *_from_full_split_reference columns; they also include batch-shape numerical effects and are not treated as isolated permutation effects.
Exact scalar differences and changes beyond the predeclared FP32 tolerance are separately saved in permutation_summary.csv. A strict winner flip caused by numerical closeness is not silently relabeled as a tie. Order-sensitive solver outcomes are recorded as behavior, not automatically classified as implementation bugs.

| Split | Solver | Cost changes beyond tolerance | Maximum absolute cost change |
|---|---|---:|---:|
| train | RELD_MOEL | 4 | 0.0651283264 |
| train | RELD_MTL | 0 | 0 |
| val | RELD_MOEL | 0 | 0 |
| val | RELD_MTL | 2 | 0.0930585861 |

Material order-sensitive outcomes (all independently feasible; no winner flips):

| Split | Instance index | Permutation | Solver | Original cost | Permuted cost | Cost change |
|---|---:|---|---|---:|---:|---:|
| train | 5367 | permutation_0 | RELD_MOEL | 9.407533646 | 9.342405319 | -0.065128326 |
| train | 5367 | permutation_1 | RELD_MOEL | 9.407533646 | 9.388444901 | -0.019088745 |
| train | 5367 | permutation_2 | RELD_MOEL | 9.407533646 | 9.454559326 | +0.047025681 |
| train | 5367 | permutation_3 | RELD_MOEL | 9.407533646 | 9.422286987 | +0.014753342 |
| val | 873 | permutation_0 | RELD_MTL | 15.212717056 | 15.119658470 | -0.093058586 |
| val | 873 | permutation_1 | RELD_MTL | 15.212717056 | 15.119658470 | -0.093058586 |

## 3. Clean binary learning

All original FP64 non-ties are used, even if neither RELD solver is the seven-pool winner. Exact ties are retained in predictions with label=-1 but excluded from fitting and binary metrics.
Regret here is relative to min(MOEL_cost, MTL_cost), not the seven-method Oracle. Binary accuracy is not comparable to the old seven-method Top1.

| Split | Total | Strict comparisons | Exact ties | MOEL wins | MTL wins |
|---|---:|---:|---:|---:|---:|
| train | 10000 | 9999 | 1 | 4919 | 5080 |
| val | 1000 | 1000 | 0 | 491 | 509 |

| Model | Split | N | Binary accuracy | CE | Selected cost | Two-method regret |
|---|---|---:|---:|---:|---:|---:|
| neural | train | 9999 | 56.496% | 0.672572 | 10.095732 | 0.752627% |
| neural | val | 1000 | 53.500% | 0.686491 | 10.043779 | 0.801388% |
| tree | train | 9999 | 93.489% | 0.451365 | 10.033220 | 0.110167% |
| tree | val | 1000 | 53.400% | 0.702489 | 10.048903 | 0.845969% |
| size_prior | train | 9999 | 51.095% | 0.692785 | 10.107660 | 0.905815% |
| size_prior | val | 1000 | 51.300% | 0.691715 | 10.051788 | 0.904807% |
| frozen_R47 | train | 9999 | 56.686% | 0.668400 | 10.093797 | 0.729391% |
| frozen_R47 | val | 1000 | 54.200% | 0.694688 | 10.040404 | 0.757015% |

Neural stop epoch=28; validation-CE-best epoch=18; successful updates=2212. All table entries for the neural model replay the same best checkpoint.
Neural protocol: scratch Encoder + binary head, CE only, batch128, fixed LR1e-4, AdamW WD1e-4, dropout0.1, FP32/TF32 off, at most60 epochs, full train/val eval every epoch, strict validation CE selection and ten nonimproving epochs to stop. No solver encoder, retrieval, pair/risk/R-Drop or augmentation.
Tree protocol: one fixed HistGradientBoosting configuration, 300 iterations and 15 leaves, no parameter sweep or validation selection. Input statistics are explicitly listed in config.json and r48_common.tree_features. Size-majority probabilities are train-fitted with add-one smoothing solely to make CE finite.
Frozen R47 is the locked epoch31 model (its checkpoint stores zero-based epoch=30). Its train memory was rebuilt with the original numeric settings; maximum summary error is exactly zero.

At the stopping epoch28, the neural model has train accuracy 60.286% / CE 0.659072, versus validation accuracy 53.400% / CE 0.691936. Later training continues to improve fitting without improving the predeclared validation-CE criterion. This endpoint is shown for diagnosis only; the comparison table uses epoch18, not a later accuracy-selected point.

## Findings and limits

1. No input/constraint mismatch, missing route, infeasible selected route or material cost-calculation error was found in this fixed audit. This is sample-level evidence, not proof about all labels.
2. Customer reorderings changed the strict two-solver winner in 0/512 trials. See per-instance costs and score changes rather than inferring a data error from a flip alone.
3. neural validation accuracy=53.500%; difference versus size prior=+2.200pp, versus frozen R47=-0.700pp; CE=0.686491, two-method regret=0.801388%. Accuracy, probability quality and selection cost must be read together.
4. tree validation accuracy=53.400%; difference versus size prior=+2.100pp, versus frozen R47=-0.800pp; CE=0.702489, two-method regret=0.845969%. Accuracy, probability quality and selection cost must be read together.
5. The independent tree fits 93.489% of training comparisons but only 53.400% of validation comparisons, a 40.089pp generalization gap. The explicit inputs support substantial training-set discrimination; that discrimination does not transfer well under this fixed tree recipe. This is not evidence that the neural classifier has been fully optimized, or that labels are random.
6. Neither clean model outperforms frozen R47 in validation accuracy or two-method regret. Removing seven-way competition, shared-task updates and auxiliary losses did not reveal a large validation gain. Those mechanisms are therefore not a sufficient explanation of this OVRPTW pair plateau under the tested settings.
7. Next work should prioritize transferable input-to-performance relationships and train/val coverage, not another decoder variant justified solely by the pooled 47% Top1. The present data does not establish a unique root cause: richer relational predictors, solver trajectory evidence, or a controlled sample-size study remain possible targeted probes. No new probe is trained in this run.
8. This diagnostic removes seven-way competition, shared-task updates and auxiliary losses. It cannot by itself prove random labels, a universal predictability ceiling, or that one architecture is optimal. The training/validation curves and an independently constructed tree provide separate evidence about the remaining signal.

## Execution and artifacts

Actual compute route: srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=10G; only GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 (RTX3090) is visible to each process.
```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh audit
R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh binary
R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh analysis
```

Required outputs: solver_audit.csv, permutation_audit.csv, binary_comparison.csv, learning_curves.png, val_predictions.csv, config.json. Additional train predictions, route JSONL, audit/permutation summaries, best/last neural checkpoints, tree pickle, exact sample/permutations and launch sources are retained.
The first precheck exposed a duplicate-key result-recording error, fixed before any accepted solver output. It was a diagnostic harness error, not a solver/data anomaly. Logs retain the failed launch.
Preparation also corrected the indexed original-cost CSV reader and restored the original TF32 settings for frozen R47 replay before starting the single fresh neural training run. No dataset, solver recipe or acceptance threshold was changed to make these checks pass.
Post-training reporting recomputes permutation deltas against the same sample-batch original_0 and adds the cost-change table. source_launch/ preserves the code at training launch; reporting changes do not alter neural weights, data, selection or predictions.
All saved binary prediction CSVs replay their metrics to numerical tolerance. W&B logs are offline; no API keys or external embedding requests were used.
