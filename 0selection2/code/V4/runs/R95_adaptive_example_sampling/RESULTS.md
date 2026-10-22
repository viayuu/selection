# R95: Adaptive Example Sampling

R89B is the main historical same-data reference; R84A is secondary. Neither is a newly matched control. Both keep the original R88B execution role and original R77 loss/R-Drop. Both use the same natural rows for first7200 updates; then A uses uniform IID replacement, B uses an own-TRAIN adaptive proposal with .5 uniform mixture, refreshed every7200 updates. The entire row objective is corrected by 1/(N*q), arithmetic batch mean, no self-normalization. A uses3090_0 after R92A full delivery; B uses4090_0 after R92B full delivery. Common code/data/fresh initialization/update budget, not bitwise hardware-matched. No guaranteed variance reduction, causal claim, architectural novelty or root-cause proof. Development-exposed TEST; no TEST/LIB selection or per-task assembly. Deterministic eval final-affine gradient norm of CE+comparison+risk, including cancellation; eval R-Drop/KL is zero. Not full-parameter or stochastic-training gradient norm. Inverse correction is before global clipping/AdamW, not an unbiased optimizer-update claim.

Original R77 CE .35 / comparison .35 / risk .02 / R-Drop ramp .35; dropout .1 in both arms.
Original maximin inference (argmax order equivalent to raw utility), unchanged ties.

| Model | Split | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | --- | --- | --- | --- | --- |
| R95/A | VAL | 71.000000 | 11.416660494 | -0.192883 | 0.394762 | 0.404236 |
| R95/A | TEST | 71.249383 | 11.373720916 | -0.197992 | 0.385188 | 0.392933 |
| R95/B | VAL | 71.098765 | 11.416835655 | -0.192413 | 0.395220 | 0.405730 |
| R95/B | TEST | 71.254321 | 11.374067172 | -0.197046 | 0.386134 | 0.395211 |
