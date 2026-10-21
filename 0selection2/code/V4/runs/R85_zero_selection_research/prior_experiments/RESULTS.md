# R80-R84: locked-model evidence for zero-selection research

Status: prior analysis complete; the broader optimization goal remains active.
Execution: CPU-only result analysis; no Astra result-review call. Previous goal turn: progress.

## Original 77,904-example release

| Model | TRAIN Top1 % | VAL Top1 % | TEST Top1 % | TEST mean_cost | TEST actual regret % | VAL zero groups / winners |
|---|---:|---:|---:|---:|---:|---:|
| R77B | 71.1767 | 70.8943 | 71.2252 | 11.345771171 | 0.394014 | 30 / 2,648 |
| R80A | 71.2071 | 70.9728 | 71.2266 | 11.346230197 | 0.397717 | 32 / 2,870 |
| R80B | 71.4382 | 70.9428 | 71.3151 | 11.345398139 | 0.391480 | 25 / 1,297 |
| R80C | 71.6722 | 70.9571 | 71.1753 | 11.346085863 | 0.397815 | 24 / 1,020 |
| R80D | 71.1681 | 70.8858 | 71.1767 | 11.345849405 | 0.393588 | 29 / 2,410 |
| R81B | 71.1216 | 70.8558 | 71.1510 | 11.345797728 | 0.393961 | 29 / 2,239 |
| R82B | 71.1452 | 70.9314 | 71.2780 | 11.345856131 | 0.393312 | 26 / 2,114 |
| R83 | 71.1309 | 70.8344 | 71.2067 | 11.345720350 | 0.393365 | 27 / 2,159 |

## NEW90k matched release

| Model | TRAIN Top1 % | VAL Top1 % | TEST Top1 % | TEST mean_cost | TEST actual regret % | VAL zero groups / winners |
|---|---:|---:|---:|---:|---:|---:|
| R84A | 71.2439 | 71.0852 | 71.2358 | 11.373843044 | 0.394847 | 29 / 2,807 |
| R84B | 71.2597 | 71.0000 | 71.2235 | 11.373993105 | 0.394592 | 25 / 1,471 |
| R84C | 71.3234 | 71.0235 | 71.2716 | 11.374011311 | 0.397181 | 26 / 1,528 |
| R84D | 71.3191 | 70.9543 | 71.2173 | 11.373922018 | 0.395155 | 23 / 1,067 |

## What changes the next action

1. R80-R83 belong to OLD77904 (70,114/3,895/3,895 per task); R84 belongs to NEW90k (81,000/4,500/4,500). Compare within each release. Raw costs across releases are not a controlled gain.
2. The current locked R84 models have TRAIN Top1 around 71.2-71.3%, close to VAL. Zero selection is already present in TRAIN. These models do not show the near-100% TRAIN vs 50% holdout behavior of earlier binary readouts.
3. R84A/B TRAIN have 26 zero-with-winner groups / 25,627 winners. Focal C/D leave 23 groups / 18,259 and 15,936 winners. Reducing zero groups does not establish correct recovery or system improvement.
4. All three R84 variants have lower locked VAL Top1 and higher mean_cost/actual regret than A. Focal alone has slightly higher TEST Top1 but also higher TEST cost/regret; TEST does not override the failed VAL acceptance gate.
5. R81 global AUC, R82 matched ranking, R83 source filtering and the R80 optimizer/loss changes do not provide a clear system-level breakthrough. Same-method ranking evidence must not be confused with becoming the correct cross-method argmax.
6. Next action: prioritize method-specific TRAIN winner discrimination and the gap between ranking/probability support and final argmax. A generic longer-training/overfitting narrative is not justified for these locked R84 models. Any new proposal must improve VAL Top1 and reduce missed winner mass without increasing cost.
7. All runs use seed 2 only; no multi-seed significance claim. Native/FP64 winner ties follow the saved label. False picks can have zero extra cost if tied. All18 metrics are equal task means; mixed raw mean_cost is descriptive.
8. TEST has informed earlier research. This retrospective analysis does not select new checkpoints, thresholds, temperatures or per-task model mixtures. No GPU inference or new training was used.

## Fixed target recovery

Fixed targets are baseline locked VAL methods with native winners and chosen share <=0.1%. This descriptive list never enters sampling/checkpoint selection.

| Model | VAL target correct / winners | Rescued / lost vs baseline | New false picks | TEST target correct / winners |
|---|---:|---:|---:|---:|
| R77B | 8 / 4358 | 0 / 0 | 0 | 11 / 4279 |
| R80A | 9 / 4358 | 5 / 4 | 3 | 11 / 4279 |
| R80B | 33 / 4358 | 25 / 0 | 23 | 36 / 4279 |
| R80C | 49 / 4358 | 42 / 1 | 43 | 43 / 4279 |
| R80D | 10 / 4358 | 4 / 2 | 2 | 15 / 4279 |
| R81B | 15 / 4358 | 8 / 1 | 8 | 17 / 4279 |
| R82B | 14 / 4358 | 7 / 1 | 9 | 19 / 4279 |
| R83 | 11 / 4358 | 3 / 0 | 5 | 13 / 4279 |
| R84A | 11 / 4625 | 0 / 0 | 0 | 13 / 4598 |
| R84B | 32 / 4625 | 21 / 0 | 22 | 31 / 4598 |
| R84C | 21 / 4625 | 10 / 0 | 17 | 24 / 4598 |
| R84D | 56 / 4625 | 46 / 1 | 56 | 52 / 4598 |

## TRAIN discrimination examples

| Model | Split | Problem / method | Winners | Selected | Correct | Recall % | Winner mean predicted rank |
|---|---|---|---:|---:|---:|---:|---:|
| R84A | TRAIN | TSP / DIFUSCO | 6806 | 0 | 0 | 0.0000 | 3.6422 |
| R84A | TRAIN | TSP / LEHD | 14279 | 15 | 7 | 0.0490 | 2.7815 |
| R84A | TRAIN | ATSP / MATPOENET | 6476 | 0 | 0 | 0.0000 | 2.2608 |
| R84A | VAL | TSP / DIFUSCO | 373 | 0 | 0 | 0.0000 | 3.6113 |
| R84A | VAL | TSP / LEHD | 792 | 0 | 0 | 0.0000 | 2.6768 |
| R84A | VAL | ATSP / MATPOENET | 373 | 0 | 0 | 0.0000 | 2.3003 |
| R84C | TRAIN | TSP / DIFUSCO | 6806 | 10 | 4 | 0.0588 | 3.6076 |
| R84C | TRAIN | TSP / LEHD | 14279 | 82 | 34 | 0.2381 | 2.7206 |
| R84C | TRAIN | ATSP / MATPOENET | 6476 | 0 | 0 | 0.0000 | 2.2997 |
| R84C | VAL | TSP / DIFUSCO | 373 | 0 | 0 | 0.0000 | 3.5710 |
| R84C | VAL | TSP / LEHD | 792 | 5 | 4 | 0.5051 | 2.6313 |
| R84C | VAL | ATSP / MATPOENET | 373 | 0 | 0 | 0.0000 | 2.3271 |

This shows method suppression already on TRAIN, not that all instance information is absent. AUC is only a mechanism metric; raw decision-logit AUC can depend on instance score offsets, so `method_probability_auc` is separately reported from temperature-1 softmax.
The zero-winner mass is not total missed-winner mass: after a single choice, an entire method leaves the strict-zero list. `total_missed_winner_count` in summary.csv measures all errors; fixed-target rescues measure actual improvement.

## Files and definitions

- `PRIOR_EXPERIMENTS.pdf`: concise authoritative report.
- `summary.csv`, `metrics_by_problem.csv`: locked ALL / every-task metrics with TRAIN-fitted SBS and Oracle.
- `zero_selection_methods.csv`, `method_diagnostics.csv`: actual counts from saved TRAIN/VAL/TEST predictions; probabilities are temperature-1 softmax of saved decision logits, not calibrated probabilities.
- `fixed_targets.csv`, `target_recovery*.csv`: fixed baseline target sets, recall, precision, real rescues, lost winners, new false picks and regret effects.
- `paired_changes*.csv`: same-instance corrections/harms and both-wrong cost changes; no uncontrolled cross-release pairing.
- `train_validation_evidence.csv`, `learning_curves.csv`: selected models and saved training trajectories. Missing predictions remain explicitly missing; absent artifacts never imply zero selections.
- `EVIDENCE_HANDOFF.json`, `analysis_data.json`, `protocol.json`: machine evidence and exact source/checkpoint provenance.

R80A is CE-only; it is not the original-objective control for R81/R82. R77B is a historical same-data reference. R80B uses batch512 and twice the updates to match sample exposures. R83 vs R80D isolates source-scope changes only to the extent recorded in their protocols. R84 A/B/C/D is the matched new-release comparison.
