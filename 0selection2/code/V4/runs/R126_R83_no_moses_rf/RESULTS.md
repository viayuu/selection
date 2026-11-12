# R126: R83 model-only source selector without MoSES_RF

One fresh seed2 full18 model. BQ and LEHD are retained. Original frozen new90k splits.

| Split | Top1 (%) | actual regret (%) | macro mean cost |
| --- | ---: | ---: | ---: |
| TRAIN | 73.6859 | 0.36857 | 11.38997339 |
| VAL | 73.3790 | 0.37982 | 11.41788088 |
| TEST | 73.6975 | 0.36774 | 11.37466641 |

Only encoder/decision MODEL code is embedded; inference/config/train code is excluded.
SBS identities are fitted on TRAIN after candidate filtering; native winner uses FP64 costs.
ALL weights problems equally; cross-problem raw mean cost is descriptive only.
Historical R83 used older data; R125 uses a different source scope and candidate pool.
Neither is a matched single-factor control. Removing a candidate also changes the Oracle.
The same VAL-selected checkpoint is used for TEST and both LIBs; no TEST/LIB retuning.
Primary: model_only_seed2/full_model_statistics/MODEL_STATISTICS.pdf.
Checkpoint SHA256: e6292042b06e20934908c335538eed82e2b82433e3ee097f17e1e1d10ab97293
