# R125/code_seed2

Single VAL-best checkpoint: update 40320

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 73.1062 | 11.41748291 | 0.3689 |
| TEST | 73.2617 | 11.37464114 | 0.3603 |

FP64 winner after deleting BQ/LEHD. SBS: full TRAIN only. ALL: equal-problem macro.
Fixed historical NSS logits, reduced-pool diagnostic only; NOT trained on the R125 reduced pool; no retraining or new inference.
Missing NSS entries: 
TEST informed earlier R&D; not fully blind. No TEST/LIB checkpoint or inference-rule selection.
Primary: MODEL_STATISTICS.pdf; selection_distributions.pdf; learning_curves.pdf.
