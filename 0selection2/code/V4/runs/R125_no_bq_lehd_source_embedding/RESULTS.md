# R125: classic source selector without BQ/LEHD

One fresh seed2 model, 18 tasks, original frozen new90k instance splits.
BQ and LEHD are absent from the forward candidate pools and all losses.

| Split | Top1 (%) | actual regret (%) | macro mean cost |
| --- | ---: | ---: | ---: |
| TRAIN | 73.3468 | 0.35972 | 11.38986811 |
| VAL | 73.1062 | 0.36887 | 11.41748291 |
| TEST | 73.2617 | 0.36026 | 11.37464114 |

TRAIN-fitted SBS and FP64 Oracle are recomputed on the reduced pool.
ALL weights tasks equally; cross-task raw mean cost is descriptive only.
Architecture reference: R80D; its older data/full-pool result is NOT a matched causal control.
Deleting candidates changes the task and Oracle. A Top1 increase alone does not prove family imbalance.
Primary results: code_seed2/full_model_statistics/MODEL_STATISTICS.pdf.
Distribution plots: code_seed2/full_model_statistics/selection_distributions.pdf.
The same VAL-selected checkpoint is used for TEST and both LIBs; no TEST retuning.
Checkpoint SHA256: 38c4b4c540a9e352294d2bc1439ea74137fa140d3d44e913b5b6bb52f3264bf0
