# R87 Locked Result Handoff

Primary delivery: [ROUND_ANALYSIS.pdf](ROUND_ANALYSIS.pdf), two pages generated
from the existing report CSVs; no full-model figures duplicated.

CPU-only reading of existing report CSVs and VAL locks. No new inference,
fitting, checkpoint selection, provenance audit, or GPU work.
The research goal remains **active_unresolved**.

## Full18 Metrics

Frozen NEW90k data; native/FP64 winners; 4,500 instances/task/split.
ALL uses equal weights across 18 tasks. ALL mean_cost is descriptive across
different task units, not a pooled raw-cost objective. SBS was fitted on TRAIN.

| Split | Model | Top1 (%) | mean_cost | actual regret (%) |
| --- | --- | ---: | ---: | ---: |
| VAL | R84A | 71.085185 | 11.416611477 | 0.403494295 |
| VAL | R87A shared | 71.148148 | 11.416998736 | 0.406346461 |
| VAL | R87B private | 71.079012 | 11.416529026 | 0.402061483 |
| TEST | R84A | 71.235802 | 11.373843044 | 0.394846788 |
| TEST | R87A shared | 71.408642 | 11.373950438 | 0.394513450 |
| TEST | R87B private | 71.270370 | 11.373975362 | 0.394012261 |

| Split | Comparison | Top1 delta (pp) | mean_cost delta | regret delta (pp) |
| --- | --- | ---: | ---: | ---: |
| VAL | A - R84A | +0.062963 | +0.000387259 | +0.002852166 |
| VAL | B - R84A | -0.006173 | -0.000082451 | -0.001432812 |
| VAL | B - A | -0.069136 | -0.000469710 | -0.004284978 |
| TEST | A - R84A | +0.172840 | +0.000107394 | -0.000333337 |
| TEST | B - R84A | +0.034568 | +0.000132318 | -0.000834527 |
| TEST | B - A | -0.138272 | +0.000024923 | -0.000501189 |

Negative cost/regret deltas favor the first model. VAL B vs A corrects 1,075
and harms 1,131 instances (net -56); TEST corrects 1,034 and harms 1,146
(net -112). TEST does not override the predeclared VAL screen.

## Fixed 36 Zero/Near-zero Targets

The published R84A VAL list is unchanged: 4,625 VAL winners, 4,598 TEST
winners. R84A originally correctly selected 11/4,625 and 13/4,598.
Rescued/lost/new false picks below are same-instance changes against the
specified reference; precision/recall describe the resulting model's target picks.

| Split | Model / reference | Correct/support | Rescued | Lost | New false picks | Precision (%) | Recall (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | A / R84A | 24/4625 | 13 | 0 | 13 | 61.5385 | 0.5189 |
| VAL | B / R84A | 30/4625 | 19 | 0 | 26 | 50.0000 | 0.6486 |
| VAL | B / A | 30/4625 | 11 | 5 | 19 | 50.0000 | 0.6486 |
| TEST | A / R84A | 24/4598 | 14 | 3 | 14 | 51.0638 | 0.5220 |
| TEST | B / R84A | 32/4598 | 23 | 4 | 23 | 48.4848 | 0.6960 |
| TEST | B / A | 32/4598 | 12 | 4 | 17 | 48.4848 | 0.6960 |

This is a small amount of true recovery accompanied by new false picks, not
a substantial repair of missed methods. Fewer zero-selected groups alone
does not establish success. Unsupported denominators in the source per-method
tables remain undefined; no absent winner support is treated as zero recall.

## Selected-checkpoint TRAIN Discrimination

| Model | Selected update / epoch | TRAIN macro Top1 (%) | TRAIN zero groups / winner mass |
| --- | --- | ---: | ---: |
| R87A | 40320 / 28 | 71.351235 | 26 / 25627 |
| R87B | 36000 / 25 | 71.259396 | 25 / 18821 |

| TRAIN problem / method | True winners | A selected / correct | B selected / correct |
| --- | ---: | ---: | ---: |
| TSP / LEHD | 14279 | 9 / 5 | 1 / 0 |
| TSP / DIFUSCO | 6806 | 0 / 0 | 181 / 41 |
| ATSP / MATPOENET | 6476 | 0 / 0 | 0 / 0 |

B's TRAIN DIFUSCO recall is 0.6024%, precision 22.6519% (140 false picks).
Across the fixed targets, TRAIN A gets 480/83,279 correct (recall 0.5764%),
B gets 606/83,279 (0.7277%). These selected checkpoints still exhibit strong
decision suppression on TRAIN. The small TRAIN-VAL Top1 gaps do not support
describing this result simply as severe unseen-instance overfitting. Counts
do not identify a unique cause or show that instance information is absent.

## Gate and Evidence Boundaries

**FAIL:** `protocol_lock.json` requires B to improve unrounded VAL Top1 while
mean_cost and actual regret do not increase versus **both** matched A and
historical R84A. B lowers VAL cost/regret but loses Top1 against both. A also
does not jointly improve upon R84A: its VAL cost and regret increase.
Neither a better TEST Top1 nor a nonzero DIFUSCO selection count changes this.

A and B use independently VAL-selected unique checkpoints; no per-task model
assembly. A SHA256: `d3041a4292206bcd3ed4a2be6d71116bddd16f37afc033ed1f361c1fe309c8ab`.
B SHA256: `36e4685625906b790f042a7895fad3b57d8b9584afa90e04b988778a01657cf6`.
R84A is a same-data historical reference, not a newly matched R87 control.
TEST/LIB have participated in prior research and are not claimed as fresh blind tests.

Required locked metrics and selected-state TRAIN counts are present.
The declared shared/private parameterization is a protocol fact; these outcome
counts alone are not causal proof of a representation or deployment defect.

## Existing Sources

- `A_shared_residual/full_model_statistics/{metrics_by_problem.csv,paired_changes_vs_R84A.csv,target_recovery_summary.csv,train_metrics_by_problem.csv,method_selection_train_val_test.csv,zero_selection_summary.csv}`
- `B_private_residual/full_model_statistics/` contains the same named tables.
- `comparison_artifacts/{paired_changes_B_vs_A.csv,target_recovery_summary_B_vs_A.csv,protocol.json,COMPARISON.pdf}`
- `../R84_new90k_attention/A_baseline/full_model_statistics/metrics_by_problem.csv`
- `protocol_lock.json` (`empirical_screen`), and each arm's `final_test_lock.json`.

No R88 process or file was touched; no launch on the released GPUs.
