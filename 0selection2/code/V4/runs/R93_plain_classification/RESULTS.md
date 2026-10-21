# R93: Plain Classification Control

R89B is the main historical same-data reference; R84A is secondary. Neither is a newly matched original-loss control. The A/B algorithmic difference is dropout (A=0.1, B=0); both use one-forward utility CE only, no R-Drop, WD=1e-4 retained. A uses released3090_1; B uses4090_1: common code/data/fresh initialization/update budget, not bitwise hardware-matched. Tiny A/B differences are not causal proof of dropout. Engineering control, not architectural novelty or root-cause proof. Development-exposed TEST; no TEST/LIB selection or per-task assembly.

TRAIN: one-forward raw-utility CE. Eval macro_ce: original maximin diagnostic.
Original maximin inference (argmax order equivalent to raw utility), unchanged ties.

| Model | Split | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | --- | --- | --- | --- | --- |
| R93/A | VAL | 71.125926 | 11.417062443 | -0.191084 | 0.396535 | 0.407874 |
| R93/A | TEST | 71.376543 | 11.374665445 | -0.190578 | 0.392618 | 0.402214 |
| R93/B | VAL | 71.109877 | 11.417053061 | -0.190775 | 0.396862 | 0.407919 |
| R93/B | TEST | 71.306173 | 11.374635129 | -0.191137 | 0.392064 | 0.401964 |
