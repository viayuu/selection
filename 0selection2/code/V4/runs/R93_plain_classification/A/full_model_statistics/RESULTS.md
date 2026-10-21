# R93/A

Unique VAL checkpoint: update 57600

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 71.125926 | 11.417062443 | 0.407874 |
| TEST | 71.376543 | 11.374665445 | 0.402214 |

SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.
NSS uses historical R79 weights, not a matched new90k training control.
No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.
PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.

TRAIN CE: raw utility only; eval macro_ce: original maximin diagnostic.

R89B is the main historical same-data reference; R84A is secondary. Neither is a newly matched original-loss control. The A/B algorithmic difference is dropout (A=0.1, B=0); both use one-forward utility CE only, no R-Drop, WD=1e-4 retained. A uses released3090_1; B uses4090_1: common code/data/fresh initialization/update budget, not bitwise hardware-matched. Tiny A/B differences are not causal proof of dropout. Engineering control, not architectural novelty or root-cause proof. Development-exposed TEST; no TEST/LIB selection or per-task assembly.
