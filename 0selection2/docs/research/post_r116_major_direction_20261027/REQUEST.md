# Research-only follow-up after R115/R116

User goal remains ACTIVE: improve whole18 single-solver selection and genuinely
recover zero/near-zero selected methods without sacrificing cost. Do not claim
that another negative version completes the research goal.

Requested model: gpt-6-astra, reasoning_effort=max. Astra owns ONLY primary-paper
research and next scientific ideas. Sol owns experiments, numerical inspection,
protocol/budget, code, execution, monitoring and publication. No substitution.

Read EVIDENCE.md as Sol-curated completed evidence, not a request to audit or
re-verify numbers. Understand the earlier research and experiments, including:

- docs/research/deep_direction_review_20261023
- docs/research/deep_direction_review_v2_20261023
- docs/research/comprehensive_direction_review_20261025
- docs/research/major_paradigm_review_20261025
- docs/research/post_major_paradigm_results_20261026
- code/V4/runs/R109_method_conditioned_nodes/RESULTS.md
- code/V4/runs/R112_region_interfaces/RESULTS.md
- code/V4/runs/R113_executable_process/RESULTS.md
- code/V4/runs/R114_paired_responses/preparation/finish/tensor_boundary_combined_20261026/RESULTS.md
- code/V4/runs/R115_cross_base_response/RESULTS.md
- code/V4/runs/R116_subset_elimination/RESULTS.md
- code/V4/runs/R116_subset_elimination/target_status.json and target_recovery_summary.csv

Use prior reports to trace earlier R47/R48/R49/R50/R51/R53/R54/R55/R56,
R67/R71/R77/R78/R81/R82/R84/R88/R89/R91/R94/R97/R100/R104/R106.
Read relevant implementation only to understand the learned object and knowledge
coverage, not as a correctness reviewer. Do not repeat a giant integrity audit.

Main questions:

1. Why have genuinely different training/representation/decision mechanisms
   still failed to localize support for methods such as TSP LEHD/DIFUSCO?
   Distinguish evidence-supported findings from conjecture and legitimate
   multiclass Bayes/argmax behavior. Do not call all zero selections bugs.
2. Which major, meaningfully untested change in representation, training
   knowledge or computation is most justified NOW? Do not merely add a head,
   coefficient, bias, quota, threshold, larger MLP or another local ReLD expert.
3. What is the strongest counterargument to that scientific hypothesis, and
   which single decisive matched comparison would distinguish it? Explain the
   causal/knowledge difference, not a large ablation matrix.

Important factual boundary: R94B uses one frozen pretrained ReLD encoder for
CVRP/the15 MVRP tasks; it has NO donor features on TSP/ATSP. R50 used two ReLD
encoders for a narrow binary task. Do not say every candidate's own pretrained
instance computation has already been transferred, or confuse fixed source-code
embeddings with weights applied to the current instance. This observation does
not itself prove that broader transfer will help; inspect real coverage before
making such a scientific recommendation.

Research broadly. At most three distinct major directions; rank one primary.
Actually read relevant primary papers/sections, not only title lists or generic
benchmarks. Prefer at least six materially informative primary sources beyond
simply repeating the last report. State paper task/assumptions and what does NOT
transfer to solver selection. Include quantitative/qualitative evidence from
our past versions that would weaken each proposed direction. Do not assert an
accuracy ceiling, random labels, or an easy hidden bug without evidence.

No actual experiment planning, seed grids, batch/lr schedules, exact launch
budgets, code edits, GPU work, solver calls, API training, commits or changes to
the active goal. Sol will design the finite experiment from the scientific
recommendation. You may assess feasibility with two24GB RTX3090s, but do not
use them. New TRAIN solver/process annotations are permitted in a subsequent
Sol experiment; do not acquire them during research.

Constraints for scientific proposals:

- Keep the frozen90k scenario, whole18 scope and legal pools unless an explicit
  future new-version rationale is given; do not silently restore deleted solvers.
- Single final selected solver, not running two complete solvers as an Oracle.
- Query labels/cost/winner/routes cannot enter inference. Historical TRAIN
  labels may be context only with current/base-relative query exclusion.
- Validation chooses one whole-model checkpoint/inference rule; TEST/LIB already
  participated in past R&D, are not pristine blind data, and never reselect it.
- R58 semantic repair remains deferred; do not restart that work as a new gate.
- R115 failed only a small, high-dimensional fixed-statistics linear pilot. It
  did not refute all neural response functions; no automatic label expansion.
- R116 improved ranking NLL modestly but did NOT improve fixed36 recall over
  the historical best. Do not continue adding ranking heads simply because the
  scientific story is attractive.

Write ONLY these UTF-8/LF files in this directory:

1. REPORT.md: Chinese, questions/findings/primary direction/strong objections;
   concrete and deep, preferably8-12 PDF pages rather than a sprawling inventory.
2. sources.csv: title,year,primary_url,actually_read_sections,relevance,limits.
3. directions.csv: priority,direction,learned_object,knowledge_change,
   closest_failed_precedent,why_not_equivalent,decisive_comparison,main_risk.

Do not generate PDF code; Sol will render and publish the completed research.
