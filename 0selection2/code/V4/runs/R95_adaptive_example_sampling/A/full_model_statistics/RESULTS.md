# R95/A

Unique VAL checkpoint: update 36000

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 71.000000 | 11.416660494 | 0.404236 |
| TEST | 71.249383 | 11.373720916 | 0.392933 |

SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.
NSS uses historical R79 weights, not a matched new90k training control.
No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.
PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.

Original R77 CE .35 / comparison .35 / risk .02 / R-Drop ramp .35; dropout .1 in both arms.

R89B is the main historical same-data reference; R84A is secondary. Neither is a newly matched control. Both keep the original R88B execution role and original R77 loss/R-Drop. Both use the same natural rows for first7200 updates; then A uses uniform IID replacement, B uses an own-TRAIN adaptive proposal with .5 uniform mixture, refreshed every7200 updates. The entire row objective is corrected by 1/(N*q), arithmetic batch mean, no self-normalization. A uses3090_0 after R92A full delivery; B uses4090_0 after R92B full delivery. Common code/data/fresh initialization/update budget, not bitwise hardware-matched. No guaranteed variance reduction, causal claim, architectural novelty or root-cause proof. Development-exposed TEST; no TEST/LIB selection or per-task assembly. Deterministic eval final-affine gradient norm of CE+comparison+risk, including cancellation; eval R-Drop/KL is zero. Not full-parameter or stochastic-training gradient norm. Inverse correction is before global clipping/AdamW, not an unbiased optimizer-update claim.
