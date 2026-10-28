# Astra Scientific Reassessment

Requested: 2026-10-28 13:23 CST. The autonomous experiment goal is PAUSED.
Existing R119 learning and final evaluation pipelines remain untouched. This
request does not authorize a new experiment or a new training arm.

The user requests a detailed, broad scientific reassessment after the last
Astra review, including the actual implementations and available results.
Priorities remain full18 Top1 and genuine recovery of missed winning methods.
Investigate major architecture, objective, supervision, training and data
paradigms; do not recycle a small head, bias, coefficient or local expert.

## Authorship

- Scientific research, interpretation and recommendations: explicitly configured
  `gpt-6-astra`, `reasoning_effort=max`.
- Parent Sol: pause bookkeeping, preservation of live experiments and mechanical
  PDF production only. No Sol-authored scientific substitute.
- Read code to understand scientific mechanisms, not to perform a code review.
- Use verified primary papers and distinguish direct evidence from analogies.

## Evidence Scope

Start with the two 2026-10-25 comprehensive and major-paradigm reviews, the
2026-10-26 followup and the three 2026-10-27 direction/decision/execution-role
reports. The last published followup is
`execution_conditions_and_symmetry_20261027`, commit
`6403a6a3478e9cde741500be2c817a774190b2e8`.

Inspect R118 native captures, the R119 native annotation/pretraining/model
implementation and completed pretraining diagnostics. R119 selector training is
still running: do not report final TEST or LIB results that do not exist.

The saved-only completed-model error-budget reference is under
`code/V4/runs/R119_native_state_transfer/error_budget_reference/`. It covers
R94B and R117/B, not unfinished R119 predictions. Respect its separate
cross-task directed-pair, task-by-pair, fixed36 and strict-zero definitions.

The reference snapshot supplied to Astra at 13:27 CST has R119 A at17,280 and
B at17,352 of28,440 updates, both selecting update17,064 by the locked VAL rule.
A VAL Top1=71.216049%, actual regret=0.407684%;
B VAL Top1=71.181481%, actual regret=0.414612%.
These are provisional observations, not final improvement claims. Freeze and
date the research snapshot rather than waiting for the running experiments.

Keep the frozen new90k data, legal candidate pools and R58 deferral explicit.
TEST has participated in earlier research and is not a new blind evaluation.
All future GPU work is restricted to the same two authorized RTX3090s, but this
research request itself performs no GPU work.

## Deliverables

Within this directory only: `REPORT.md`, `evidence.md` or `evidence.csv`,
`sources.csv`, `directions.csv`, and `HANDOFF.md`.

The report should update prior hypotheses, explain the evidence and its limits,
compare genuinely different major paradigms, and give a justified preferred
scientific direction plus a substantially different fallback if appropriate.
For each direction, make the information flow, objective, distinction from
already-tested work, new information/computation required, and falsifying
outcomes concrete. Do not force a ready direction if none is supported, but
avoid another unbounded sequence of small checks.

Parent will render the unchanged scientific text as downloadable PDFs. This
request does not resume the goal, launch experiments or modify any live source.
