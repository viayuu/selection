# R51: cost-limited native solver behavior probes

## Main result

Predeclared continuation screen passed: **False**; predictive screen: **False**. Binary prediction is MOEL vs MTL only, not seven-method-pool Top1.

| Model | Selected update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |
|---|---:|---:|---:|---:|---:|
| R50B_static_reference | 200 | 58.41% | 0.665895 | 51.40% / 0.831422% | 56.90% / 0.709315% |
| A_static_seed2 | 200 | 58.41% | 0.665895 | 51.40% / 0.831422% | 56.90% / 0.709315% |
| B_behavior_seed2 | 200 | 58.99% | 0.663766 | 55.80% / 0.737330% | 54.30% / 0.737799% |
- internal, B minus A: accuracy +4.40pp; regret -0.094092pp; relative regret reduction +11.32%.
- original_val, B minus A: accuracy -2.60pp; regret +0.028484pp; relative regret reduction -4.02%.

This fixed behavior/readout configuration does not meet the predictive screen on both held-out sets. It does not establish random labels, an accuracy ceiling, or that all trajectory representations are ineffective. No probe-length, seed, pooling or classifier search was added.

## Locked protocol

Exact R49/R50 splits and4000x128 batch plan; all8000 training rows appear64 times. All21 evaluation points u0,200,...,4000 evaluate complete train/development/internal/original validation sets with eval() and tails retained. Strict minimum development CE alone locks one checkpoint per model; internal/original validation never select weights. No test read.
Both full original R48-pinned epoch5000 Train_ALL solvers are frozen/eval, FP32/TF32-off, greedy, no augmentation. Original FP64 full-solve cost labels are not changed. R50 pretrained static node caches are reused with SHA checks. The native six-field input is used, not selector8D input. Neither probing nor readout receives current cost/winner.
Both groups train fresh readouts with explicit identical common parameter values, seed2 and paired sample order. A is the exact R50 pretrained static readout architecture; B adds270->64->64 behavior MLP and graph-level concatenation. The shared385 columns of B first head layer and all other common layers copy A initialization; B extra columns are normally initialized. A/B are not parameter-count matched. Dropout call streams differ because B has an extra branch.
Fresh static A versus historical R50 pretrained B: maximum saved-logit difference across all four sets is 0.
Parameters: A 100482; B 130178. Fresh AdamW LR1e-3 WD1e-4, dropout0.1, gradient clip1, CE only;4000 updates, no scheduler/early stop.

## Behavior evidence

Two forced native moves (depot, each customer POMO start), then exactly10 network-selected greedy moves per solver. All customer starting nodes remain; no trajectory is completed or selected by cost while generating features. At steps1/5/10, each of9 fields is summarized across all POMO starts by mean/population std/q10/q50/q90:270 dimensions. Progress, pressure and accumulated cost are measured after the action; entropy/margin use that action's original pre-move distribution.
Features: served customer/demand fractions, depot returns, effective open distance, waiting, remaining capacity, feasible unserved fraction, normalized legal-action entropy and top1-top2 probability margin. Distance excludes every return-to-depot segment, includes the forced depot-to-start edge, and uses input mean off-diagonal distance. Waiting uses input mean travel time (native speed1). No final cost/oracle/gap normalization. Behavior standardization uses the8000 training rows only. Full definitions, means/std and row lists are saved.

## Actual full-pipeline timing

The selected solver actually resumes the live prefix environment and decoder K/V at native selected_count12; the prefix is not rerun. An uninterrupted decoding witness produces exactly identical routes and costs. Unselected solver probing and both encoders are charged. All POMO starts remain through completion.
Seed5102 locks12 training instances at min/median/max input size,4 each; batch1,3 repeats, synchronized wall time. Time includes CPU native collation/H2D, both encoders/probes, feature aggregation, selection, and chosen solver completion. Actual methods may differ between A/B; this is the cost of the deployed pipelines, not a matched-solver microbenchmark.

| Model | Both-encoder/probe feature ms | Selection ms | Chosen completion ms | Total ms |
|---|---:|---:|---:|---:|
| A_static_seed2 | 10.338 | 0.802 | 116.654 | 127.794 |
| B_behavior_seed2 | 51.275 | 0.903 | 100.679 | 152.857 |

B/A total latency ratio: **1.1961x**. Acceptable under predeclared <=1.25x engineering screen: **True**. The25% threshold is a practical screen chosen before training, not a universal application requirement. This is batch1 latency on one RTX3090, not a claim about batched throughput or other hardware. Timing full solves only provide runtime witnesses, never new labels used for feature fitting or training.

The timed completion uses only original native model/env steps after the probe. Independent accumulated-distance bookkeeping remains enabled in the route-continuation preflight, but is disabled during completion timing because no downstream feature uses it. This timing-only code change was made after training; cached features, weights, selected updates and saved predictions did not change.

## Corrections and harms

| Set | Corrected | Harmed | Net changed cost | Regret delta (pp) |
|---|---:|---:|---:|---:|
| internal | 180 | 136 | -0.0087246 | -0.094092 |
| original_val | 131 | 157 | +0.0023767 | +0.028484 |

Paired fixed-prediction bootstrap intervals are in paired_uncertainty.csv. They measure holdout sampling uncertainty, not training-seed uncertainty.

## Final-update observations

| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |
|---|---:|---:|---:|---:|
| A_static_seed2 | 99.56% / 0.016337 | 51.45% / 2.563613 | 52.70% / 2.487242 | 55.00% / 2.360072 |
| B_behavior_seed2 | 99.79% / 0.009836 | 55.46% / 2.627039 | 54.70% / 2.723896 | 50.90% / 2.841724 |

## Provenance and interpretation limits

The solver checkpoint hashes, source hashes and training-source limitations are retained in solver_provenance.json. Available ReLD training source uses fresh online generated instances, but exact historical instance overlap cannot be certified. No selector weights are used to initialize frozen solvers or readouts. External solver pretraining is not equal-data-budget training.
Reference: [Renau and Hart, On the Utility of Probing Trajectories for Algorithm-Selection,2024](https://arxiv.org/html/2401.12745v1). Section3.2 motivates short current-instance trajectories and combining solvers; section6 requires accounting for portfolio probing and discusses continuing the chosen run. Their experiments are continuous black-box optimization. The ten routing actions, native progress/constraint/probability features, POMO summaries and binary neural readout are this project's adaptation.

## Artifacts and reproduction

config.json, feature_definition.json, behavior_normalization.json, solver_provenance.json, preflight.json, comparison.csv, decision_changes.csv, paired_uncertainty.csv, learning_curves.png, internal_predictions.csv, val_predictions.csv, inference_timing.csv/json; per-group args, history, logs, predictions and selection. Large .pt static/behavior caches and best/last checkpoints remain local under existing repository ignore rules. W&B offline run files remain local. Execution command is in execution_command.md.
