# R110 Whole-Version Comparison

A = A_tied_ce; B = B_private_ce. Common fresh original base and identical branch prototypes,
but NOT parameter/FLOP-matched and NOT identical complete state hashes.

TRAIN: one-forward unweighted raw-utility CE (weight 1). Comparison/risk/RDrop weights are 0.
Evaluation maximin CE is diagnostic, not the optimized raw-utility CE.

| Split | Arm | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | A | 71.616941 | 11.388930 | 11.400702 | 11.345073 | -0.198522 | 0.386376 | 0.395426 |
| TRAIN | B | 71.264472 | 11.389609 | 11.400702 | 11.345073 | -0.193596 | 0.391332 | 0.399642 |
| VAL | A | 71.103704 | 11.416779 | 11.427978 | 11.371692 | -0.192984 | 0.394679 | 0.405093 |
| VAL | B | 70.896296 | 11.417566 | 11.427978 | 11.371692 | -0.184524 | 0.403215 | 0.412389 |
| TEST | A | 71.307407 | 11.374707 | 11.385299 | 11.329932 | -0.186717 | 0.396557 | 0.406464 |
| TEST | B | 71.233333 | 11.374359 | 11.385299 | 11.329932 | -0.192514 | 0.390729 | 0.399449 |

## Frozen36 True Recovery

| Split | Winners | A recall (%) | B recall (%) | Rescued | Lost | New false picks | ALL cost delta | regret delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 83279 | 2.011311 | 1.036276 | 389 | 1201 | 629 | 0.000678993 | 0.004216 |
| VAL | 4625 | 1.945946 | 0.843243 | 17 | 68 | 36 | 0.000786992 | 0.007296 |
| TEST | 4598 | 1.522401 | 0.956938 | 21 | 47 | 41 | -0.000347507 | -0.007015 |

Native/FP64 ind; legal candidate pools; TRAIN-fitted SBS; task-equal ALL.
Fixed36 recovery is not selection frequency. Population cost/regret deltas are B minus A.
TEST has participated in development; no TEST/LIB checkpoint or inference-rule reselection.

[Whole-version PDF](MODEL_COMPARISON.pdf)

A_tied_ce: 2,305,033 parameters; update 43200; SHA256 1faa6a4190cd3cda9d00e6816b0d88f19e0730a6284190450f0872587ddf5ff0.
[Per-arm PDF](A_tied_ce/full_model_statistics/MODEL_STATISTICS.pdf)
B_private_ce: 10,220,553 parameters; update 21600; SHA256 34c0d7575e8ccad9cfa5a19849d2d8c404f63b2244dca688bc4bcd40f22cd59b.
[Per-arm PDF](B_private_ce/full_model_statistics/MODEL_STATISTICS.pdf)
