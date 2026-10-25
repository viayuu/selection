# Comprehensive Research Request

Date: 2026-10-25 (Asia/Shanghai).

The user requests a fresh, comprehensive, detailed, deep Astra literature
investigation and research-direction synthesis after many selector experiments.
This is a research deliverable, not authorization to stop healthy jobs, change
their protocols, or start additional training during this investigation.

## Roles and scope

- Literature, scientific hypotheses, alternatives, and final idea synthesis:
  `gpt-6-astra`, reasoning effort `max`.
- Mechanical evidence inventory, current-status reading, PDF production, and
  any later executable experiment planning: `gpt-6.1-sol`, effort `max`.
- Astra may read existing implementation to understand what mechanism was
  tested. It is not assigned engineering code review or result certification.
- No solver reruns, new labels, raw-data changes, new split, or GPU benchmark.
- No modification of running R109/R111 sources, controllers, or processes.
- The native goal is currently paused; this research request is not silently
  treated as resuming autonomous experiment launches.

## Scientific questions

1. Why do adequately supported winner methods still receive zero/near-zero
   argmax selections, sometimes even on TRAIN?
2. Why does full18 Top1 remain near 71% across materially different mechanisms?
3. Which hypotheses are supported, weakened, untested, or not identifiable from
   the current experiments? Avoid an assumed common bug, Bayes ceiling, or
   blanket overfitting explanation.
4. Which genuinely different interventions have the best evidence-to-effort
   ratio? What precisely changes the learned information or decision path?
5. What should stop, what should continue, and what observation would reverse
   each recommendation? Do not substitute more papers for a falsifiable idea.

## Evidence entry points

- `AGENTS.md` and `code/V4/runs/zero_selection_research_state.json`.
- `docs/research/experiment_mechanism_index.md` (R60-R79).
- `docs/research/experiment_mechanism_index_R80_R102.md` (historical snapshot).
- `docs/research/deep_direction_review_20261023/` (R35-R104 history).
- `docs/research/deep_direction_review_v2_20261023/` (quantitative evidence,
  literature, objective/representation analyses).
- `docs/research/post_r108_learning_mechanisms_20261024.md`.
- `docs/research/post_r110_broad_mechanism_20261025.md`.
- Actual `code/V4/runs/R*/RESULTS.md`, plans, protocols, comparisons and model
  code. Read actual relevant implementations, not just version titles.

## Non-negotiable interpretation boundaries

- Distinguish legacy18, NEW70114, and current NEW90k; candidate changes explain
  part of the Top1 jump, and cross-era comparisons are not causal ablations.
- Current NEW90k is 81,000/4,500/4,500 per task over 18 tasks, seed2,
  logical batch1024. No current-query labels/cost/winner in model forward.
- Historical TRAIN context is allowed with query/base-instance exclusion.
- R47 was a FULL18 learned behavior-memory experiment. It is not an OVRPTW
  two-method study. Plain learned TRAIN memory is not an untouched idea.
- R69 already trained a complete task-private bank; R63 had candidate-specific
  node states; R71 modeled a joint performance distribution. Identify material
  differences, rather than claiming those broad mechanisms are new.
- R109 and R111 are ongoing, not negative final results. R110 has complete
  locked evaluations and is published (commit 370e0b49a6783036ba0899fdbc995f3d2230a27f).
- TEST/LIB have influenced the research history and are not pristine blind data.
  Recommendations must not tune a current version's checkpoint on TEST.
- Fewer zero-selection groups alone is not success. Count genuine winner
  rescues, new false picks, regret/cost changes and full18 outcomes together.
- R58 scenario repair remains deferred. Do not reopen full-label generation or
  use unresolved legacy semantics as a universal explanation of current errors.

## Deliverables

Three independent, non-overlapping Astra research lenses, then an Astra
cross-examination/synthesis. Chinese report with concise executive conclusions,
traceable experiment coverage, verified primary papers, concrete mathematical
ideas, nearest prior experiment and crucial difference, disconfirming evidence,
ranked next directions and explicit negative-result stop conditions. Primary
delivery is a self-contained PDF, with Markdown/CSV and model-call trace as
supporting artifacts. Claims remain provisional research judgments, not an
independent result audit.
