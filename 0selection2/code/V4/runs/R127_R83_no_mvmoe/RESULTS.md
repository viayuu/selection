# R127: R83 model-only source selector without MVMOE

One fresh seed2 full18 model. BQ/LEHD/MoSES_RF retained wherever originally legal.
Original new90k memberships; only MVMOE is additionally removed.

| Split | Top1 (%) | actual regret (%) | macro mean cost |
| --- | ---: | ---: | ---: |
| TRAIN | 81.2084 | 0.22100 | 11.38975324 |
| VAL | 81.1296 | 0.22518 | 11.41749489 |
| TEST | 81.2704 | 0.22283 | 11.37445282 |

R83 model-only source roles, original complete dual stream and original R77 loss.
SBS fitted on TRAIN after filtering; FP64 native winner recomputed in each legal pool.
ALL weights tasks equally; cross-problem raw mean cost is descriptive only.
Candidate deletion changes the Oracle and classification difficulty; Top1 alone is not a causal test of imbalance.
One VAL-selected checkpoint for TEST and LIB; no TEST/LIB retuning.
Primary: model_only_seed2/full_model_statistics/MODEL_STATISTICS.pdf.
Checkpoint SHA256: 389dcc8b1817e6c9852dc7eee052c4caec5da49eedda019e34269dae24b3a641
