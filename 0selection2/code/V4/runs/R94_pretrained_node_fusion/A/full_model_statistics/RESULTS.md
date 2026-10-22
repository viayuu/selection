# R94/A

Unique VAL checkpoint: update 40320

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 70.998765 | 11.417197449 | 0.409521 |
| TEST | 71.290123 | 11.374086445 | 0.398131 |

SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.
NSS uses historical R79 weights, not a matched new90k training control.
No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.
PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.

A is the matched random-frozen ReLD128 donor control; B uses pinned pretrained weights in the same frozen donor. Both use identical fresh trainable selectors and early node-fusion projections before the original four inter-node layers. Full18 raw inputs; donor input coverage is16 routing tasks, not16-task pretraining. TSP/ATSP raw paths are unchanged. Original R77 CE=.35, comparison=.35, risk=.02, R-Drop=.35 and dropout=.1 are retained, not R93 pure CE. R89B is the main historical reference; R84A is secondary, neither newly matched. External donor pretraining compute/data are not budget-matched to scratch; provenance/overlap limits remain. R50 is negative evidence for pool-first readout, not proof that early trainable inter-node recomputation succeeds. No solver decoder, routes, cost-derived new labels, or restored ReLD MTL. Hardware differences must be disclosed; matching code/batches is not bitwise hardware matching. Development-exposed TEST; unique full18 VAL macro_vs_sbs lock, no TEST/LIB selection or per-task model assembly. Cached-feature selector latency is not full-system inference latency.

Timing: complete; see target_recovery_and_train.pdf.
