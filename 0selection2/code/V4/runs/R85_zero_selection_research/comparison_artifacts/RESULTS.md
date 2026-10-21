# R85 comparison

Goal remains active/unresolved. Next action: `research_handoff_needed`.

| Split | Model | Top1 (%) | mean_cost | actual regret (%) |
|---|---|---:|---:|---:|
| VAL | R84A | 71.085185 | 11.416611477 | 0.403494 |
| TEST | R84A | 71.235802 | 11.373843044 | 0.394847 |
| VAL | A_ovr_control | 70.967901 | 11.417059356 | 0.406901 |
| TEST | A_ovr_control | 71.232099 | 11.374097328 | 0.397280 |
| VAL | B_gradient_balance | 70.935802 | 11.416844932 | 0.405866 |
| TEST | B_gradient_balance | 71.211111 | 11.373750026 | 0.394872 |
| VAL | C_cost_profile | 71.008642 | 11.416206518 | 0.397969 |
| TEST | C_cost_profile | 71.311111 | 11.373420994 | 0.389713 |
| VAL | D_candidate_subset | 71.127160 | 11.416646199 | 0.403383 |
| TEST | D_candidate_subset | 71.437037 | 11.373489772 | 0.390487 |

VAL screens use unrounded Top1 up, cost/regret not worse versus R84A; B versus R85A is separately reported.
TEST belongs to the same previously locked checkpoint, never checkpoint/rule selection.
Targets reuse the published R84A locked-VAL zero/near-zero list (36 groups, 4625 winners), descriptively only.
Recall, precision, rescues, lost winners, new false picks and net regret are in the CSVs.
Native FP64 winners; equal18-task macros; TRAIN-fitted SBS. Prior TEST results have informed research.
A small pass does not complete the goal. No automatic retry, training, GPU inference or notification.
PDF: [COMPARISON.pdf](COMPARISON.pdf).
