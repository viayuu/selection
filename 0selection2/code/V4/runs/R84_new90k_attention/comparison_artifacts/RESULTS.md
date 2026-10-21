# R84 matched comparison

Goal: unresolved. Next action: `research_handoff_needed`.

Each arm uses one checkpoint selected only by its recorded full validation rule.
Acceptance requires unrounded VAL macro Top1 > A, mean_cost <= A and actual regret <= A.
TEST is reported separately for locked versions and does not select a checkpoint or rule.

| Split | Arm | Top1 (%) | mean_cost | Actual regret (%) |
| --- | --- | ---: | ---: | ---: |
| VAL | A_baseline | 71.085185 | 11.416611477 | 0.403494 |
| VAL | B_attention_direct | 71.000000 | 11.416697456 | 0.404999 |
| VAL | C_focal | 71.023457 | 11.417001853 | 0.408296 |
| VAL | D_attention_focal | 70.954321 | 11.417063866 | 0.408097 |
| TEST | A_baseline | 71.235802 | 11.373843044 | 0.394847 |
| TEST | B_attention_direct | 71.223457 | 11.373993105 | 0.394592 |
| TEST | C_focal | 71.271605 | 11.374011311 | 0.397181 |
| TEST | D_attention_focal | 71.217284 | 11.373922018 | 0.395155 |

Frozen targets: A locked VAL chosen share <=0.1% and native winners >0. This is a posthoc mechanism analysis, not a sampling change or success criterion.
Fewer zero-selection groups alone does not establish improvement.
All18 problems are equally weighted; mixed raw mean_cost is descriptive. SBS is fitted on TRAIN; oracle is each legal-pool FP64 minimum.
Previously reported TEST rows have participated in research; no fresh blind-test claim.
PDF: [COMPARISON.pdf](COMPARISON.pdf). CSVs include all methods, per-task corrections, harms, both-wrong cost changes, target rescues and new false picks.
A negative result requests a research handoff. No new training is launched automatically.
