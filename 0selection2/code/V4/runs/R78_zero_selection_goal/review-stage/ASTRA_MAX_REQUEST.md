# R78 Astra/max Research-Only Review Request

Requested model: gpt-6-astra. Requested reasoning effort: max.
This is a research review, NOT a code review or an experiment.

Read ONLY these local research documents:

- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/GOAL.md
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/RESEARCH.md
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/review-stage/RESEARCH_REVIEW.md

Do not follow their local report links, inspect project source, inspect raw data,
read TEST reports, access GPU, execute tests/training/inference/solvers, inspect
Git diffs, edit files, touch STATUS.json, or use application goal tools. The old
application goal must remain paused and unmodified. Do not propose resuming it.
You may browse only the linked primary papers to check research claims.
If a shell read is needed, initialize conda then activate easynco first.

Evaluate the mathematical assumptions, causal identifiability, validation
protocol, objective alignment and feasibility of the ONE mainline proposal.
The first proposal deliberately fits candidate intercepts on TRAIN scores from
a frozen base, at logical batch1024, then assesses identity versus this one
candidate on VAL. No coefficient sweep, TEST-derived arm subset, forced
frequency matching, or automatic second experiment is authorized.

Be critical without manufacturing defects. In particular examine:

1. The argmax/sign invariance for pure-difference maximin, and whether this
   supports the location and need for additive correction.
2. Whether natural-frequency intercept fitting distinguishes a decision bias
   from missing ranking signal; state exactly what remains non-identifiable.
3. The fixed regularizer `mean CE + ||b||^2/(2N)` and TRAIN-only convergence;
   whether the actual research protocol is sufficiently specified without
   turning a paper review into an optimizer/code audit.
4. The one-candidate macro Top1/cost gate, taskwise harm, adaptive VAL reuse,
   in-sample TRAIN scores, and prior TEST exposure.
5. Whether the conditional R78-B risk deletion is a defensible controlled
   diagnostic or should be narrowed/deferred further. It is not the mainline.
6. Original-paper attribution/assumption mismatches. Balanced error is not the
   native-frequency objective; inference-time cost matrices cannot be assumed.
7. All-task scope, identical TEST/LIB weights and rule, and continuing the R78
   research loop without claiming that this document completes the goal.

Return concise findings ordered by severity, exact proposed research-text
corrections, and one recommended next action. No code suggestions, code review,
commands, experiments executed by you, or implementation certification.
Say explicitly whether the mainline is ready as a mechanism screen, subject
to stated limitations. Record `review_independence: same-family` and
`acceptance_status: provisional` for any positive research assessment.
