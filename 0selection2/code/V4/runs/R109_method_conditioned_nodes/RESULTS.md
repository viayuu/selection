# R109 Whole-Version Comparison

A = A_tied_late; B = B_private_late. Common fresh original base and identical branch prototypes,
but NOT parameter/FLOP-matched and NOT identical complete state hashes.

| Split | Arm | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | A | 71.481824 | 11.388574 | 11.400702 | 11.345073 | -0.203521 | 0.381317 | 0.389303 |
| TRAIN | B | 71.996091 | 11.387476 | 11.400702 | 11.345073 | -0.212836 | 0.371946 | 0.379300 |
| VAL | A | 71.102469 | 11.416634 | 11.427978 | 11.371692 | -0.192841 | 0.394804 | 0.404713 |
| VAL | B | 71.175309 | 11.417059 | 11.427978 | 11.371692 | -0.189295 | 0.398365 | 0.408363 |
| TEST | A | 71.385185 | 11.373796 | 11.385299 | 11.329932 | -0.197930 | 0.385240 | 0.394037 |
| TEST | B | 71.360494 | 11.373875 | 11.385299 | 11.329932 | -0.196958 | 0.386208 | 0.393914 |

## Frozen36 True Recovery

| Split | Winners | A recall (%) | B recall (%) | Rescued | Lost | New false picks | ALL cost delta | regret delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 83279 | 0.953422 | 2.829045 | 1708 | 146 | 1565 | -0.001098713 | -0.010003 |
| VAL | 4625 | 0.691892 | 1.837838 | 60 | 7 | 110 | 0.000425190 | 0.003650 |
| TEST | 4598 | 0.674206 | 1.761635 | 57 | 7 | 105 | 0.000079284 | -0.000123 |

Native/FP64 ind; legal candidate pools; TRAIN-fitted SBS; task-equal ALL.
Fixed36 recovery is not selection frequency. Population cost/regret deltas are B minus A.
TEST has participated in development; no TEST/LIB checkpoint or inference-rule reselection.

[Whole-version PDF](MODEL_COMPARISON.pdf)

A_tied_late: 2,305,033 parameters; update 37440; SHA256 2ba5345464b9b022974cc6bf91303a4728376e3b91b59948921bee8122127b96.
[Per-arm PDF](A_tied_late/full_model_statistics/MODEL_STATISTICS.pdf)
B_private_late: 10,220,553 parameters; update 37440; SHA256 93821e6c6dd794f5db08d32cc1defe7c4a2006365e6f864d36fe21f607bd0c84.
[Per-arm PDF](B_private_late/full_model_statistics/MODEL_STATISTICS.pdf)
