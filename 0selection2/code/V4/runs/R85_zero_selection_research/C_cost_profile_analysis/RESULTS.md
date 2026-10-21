# R85C: locked single-arm analysis

C: epoch 25, update 36000; SHA256 `fb9fd6437cb9459b476c241148fc8eebe494b5afc13639f1784a34a3777ec138`.
Historical R84A: update 37440. Same frozen NEW90k instances, candidate pools and native/FP64 winners.

## Full-system result

| Split | Model | Top1 (%) | mean_cost | actual regret (%) |
| --- | --- | ---: | ---: | ---: |
| VAL | R84A | 71.0852 | 11.416611477 | 0.403494 |
| VAL | R85C | 71.0086 | 11.416206518 | 0.397969 |
| TEST | R84A | 71.2358 | 11.373843044 | 0.394847 |
| TEST | R85C | 71.3111 | 11.373420994 | 0.389713 |

**Joint VAL gate: FAIL.** Top1 must improve while mean_cost and actual regret do not worsen. C improves cost/regret but not VAL Top1. A better TEST Top1 does not override this gate.

## Fixed zero/near-zero targets

36 groups were frozen from R84A VAL before this analysis (4,625 VAL winners). The per-arm strict-zero-only CSV has a smaller scope; this report consistently uses the full published list.

| Split | Model | True winners | Correct | Recall (%) | Picks | Precision (%) | Rescued | Lost | New false picks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | R84A | 4625 | 11 | 0.2378 | 17 | 64.7059 | 0 | 0 | 0 |
| VAL | R85C | 4625 | 22 | 0.4757 | 35 | 62.8571 | 11 | 0 | 10 |
| TEST | R84A | 4598 | 13 | 0.2827 | 25 | 52.0000 | 0 | 0 | 0 |
| TEST | R85C | 4598 | 24 | 0.5220 | 46 | 52.1739 | 14 | 3 | 13 |

Rescued/lost/new false picks compare the same instances with R84A. A new false pick means C newly selects a target on which that target is not the native winner. Undefined support or selection denominators remain undefined, never zero.

## Paired net changes

| Split | Corrected | Harmed | Both wrong: cheaper / dearer | Net correct | Regret delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| VAL | 1154 | 1216 | 655 / 526 | -62 | -0.005525 |
| TEST | 1193 | 1132 | 633 / 528 | 61 | -0.005134 |

`cost_change_categories.csv` partitions every instance, including both-wrong/equal-cost and both-correct. Each regret contribution uses the full problem population as denominator, then equal task weights; negative deltas favor C. Raw sums are retained per problem only.

## TRAIN: selected checkpoint, not final epoch

| Model | TRAIN Top1 (%) | Zero groups | Zero winner mass | Target correct / winners | Target recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R84A | 71.2439 | 26 | 25627 | 297 / 83279 | 0.3566 |
| R85C | 71.1817 | 27 | 39906 | 419 / 83279 | 0.5031 |

All TRAIN results already existed for the locked states. No GPU or new inference was used. TRAIN zero selection remains for well-supported methods, including TSP LEHD/DIFUSCO and ATSP MATPOENET. This is decision-level suppression already on TRAIN, not evidence that the only issue is unseen-instance overfitting. Counts do not identify its unique cause or establish a performance ceiling.

## Existing LIB results (C only)

| Dataset | n | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: | ---: |
| TSPLIB | 49 | 18.3673 | 8.215885069 | 0.938997 |
| CVRPLIB | 100 | 63.0000 | 67.274507074 | 0.591596 |

LIB rows quote the completed same-lock report only; neither LIB nor TEST selects a model here.

## Scope and next action

No missing requested TRAIN/VAL/TEST/LIB evidence. Full18 per-task VAL and TEST metrics, TRAIN selected-state counts, fixed-target rows, and category cost changes are in companion CSVs. TRAIN paired rescues/AUC are not newly computed; only saved selected-state decision counts are compared.

C does not meet the predeclared joint VAL gate and does not establish a substantial repair of missed methods. Wait for the other committed R85 arms before a research handoff. No follow-up experiment is authorized by this report.

Limits: one seed; R84A is a historical comparator, not a freshly matched R85A control; prior TEST/LIB have participated in research. ALL cost is a descriptive equal-task mean, not a pooled cross-unit objective. This analysis does not complete the active research goal.
