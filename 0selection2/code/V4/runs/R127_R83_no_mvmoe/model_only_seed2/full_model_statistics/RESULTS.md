# R127/model_only_seed2

Single VAL-best checkpoint: update 37440

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 81.1296 | 11.41749489 | 0.2252 |
| TEST | 81.2704 | 11.37445282 | 0.2228 |

FP64 winner after deleting MVMOE. SBS: full TRAIN only. ALL: equal-problem macro.
Fixed historical NSS scores restricted to the same legal pool; TSP unchanged8 / CVRP9; NOT a matched R127 training control; no retraining or new inference.
Missing NSS entries: 
TEST informed earlier R&D; not fully blind. No TEST/LIB checkpoint or inference-rule selection.
Primary: MODEL_STATISTICS.pdf; selection_distributions.pdf; learning_curves.pdf.

BQ/LEHD and originally legal MoSES_RF are retained; only MVMOE is removed.
B tasks K=2: Top2/Top3 both cover all candidates (min(k,K)); 100% is not extra ranking evidence.
