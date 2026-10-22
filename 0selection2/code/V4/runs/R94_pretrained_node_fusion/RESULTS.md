# R94: Frozen Donor Early Node Fusion

A is the matched random-frozen ReLD128 donor control; B uses pinned pretrained weights in the same frozen donor. Both use identical fresh trainable selectors and early node-fusion projections before the original four inter-node layers. Full18 raw inputs; donor input coverage is16 routing tasks, not16-task pretraining. TSP/ATSP raw paths are unchanged. Original R77 CE=.35, comparison=.35, risk=.02, R-Drop=.35 and dropout=.1 are retained, not R93 pure CE. R89B is the main historical reference; R84A is secondary, neither newly matched. External donor pretraining compute/data are not budget-matched to scratch; provenance/overlap limits remain. R50 is negative evidence for pool-first readout, not proof that early trainable inter-node recomputation succeeds. No solver decoder, routes, cost-derived new labels, or restored ReLD MTL. Hardware differences must be disclosed; matching code/batches is not bitwise hardware matching. Development-exposed TEST; unique full18 VAL macro_vs_sbs lock, no TEST/LIB selection or per-task model assembly. Cached-feature selector latency is not full-system inference latency.

| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R94/A | TRAIN | 71.245610 | 11.389364032 | 11.400702391 | 11.345073027 | -0.195764 | 0.389161 | 0.398072 |
| R94/A | VAL | 70.998765 | 11.417197449 | 11.427978473 | 11.371691902 | -0.188807 | 0.398882 | 0.409521 |
| R94/A | TEST | 71.290123 | 11.374086445 | 11.385299246 | 11.329932369 | -0.193783 | 0.389478 | 0.398131 |
| R94/B | TRAIN | 71.947874 | 11.386290286 | 11.400702391 | 11.345073027 | -0.223076 | 0.361663 | 0.367808 |
| R94/B | VAL | 71.428395 | 11.416278963 | 11.427978473 | 11.371691902 | -0.198659 | 0.388932 | 0.399079 |
| R94/B | TEST | 71.577778 | 11.372233605 | 11.385299246 | 11.329932369 | -0.211169 | 0.371939 | 0.379115 |
