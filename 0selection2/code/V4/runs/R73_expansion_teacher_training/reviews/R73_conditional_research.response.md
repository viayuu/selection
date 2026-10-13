**PREPARATORY: no plan adopted; R73 final validation remains pending.** I examined four primary papers and the permitted sources. Neither family currently justifies automatic adoption: one has limited identifiable supervision; the other changes inference information substantially but lacks established full18 feasibility.

`review_independence=same-family; acceptance_status=provisional`

The completed evidence does not support a universal bottleneck. [R57](/public/home/shiys/0selection2/code/V4/runs/R57_task_signal_audit/RESULTS.md:25) establishes useful instance signal beyond problem/size priors. R69 demonstrates substantial additional TRAIN fitting without generalization; R71 shows weak native fitting with a small generalization gap. [R49’s](/public/home/shiys/0selection2/code/V4/runs/R49_existing_data_diagnosis/RESULTS.md:3) sample-count result is one binary task, not a full18 scaling law. The current encoder already consumes coordinates, constraints and directed matrix information; no newly demonstrated missing-input defect emerged from this reading.

**1. Full-pool learning from partially observed outcomes**

The substantive hypothesis is to learn the native winner jointly from complete original labels and **sets of still-possible winners** on additional inputs, allowing uncertainty instead of manufacturing complete labels.

For deployment-compatible observed costs, define
\(S(x)=P\setminus\{j:\exists i\text{ observed with }c_i<c_j\}\).
The native winner must remain in \(S\). Unobserved candidates remain possible; exact ties need not exclude either candidate. These sets constrain training only, never prune the inference pool.

There is relevant measured information, but its limits matter:

- R60 records 1.35M additional inputs with two measured costs across 15 MVRP tasks.
- [R62’s preparation](/public/home/shiys/0selection2/code/V4/runs/R62_portfolio_performance_pretraining/data_cache/preparation.json) records 1.62M inputs and 11,402,307 observed cells. Its [inventory](/public/home/shiys/0selection2/code/V4/runs/R62_portfolio_performance_pretraining/data_cache/inventory.json) records small-sample replay agreement for **39/128 columns**, versus historical differences for 89.
- Those 39 include R60’s 30 columns plus nine others: TSP BQ/LEHD; CVRP BQ/LEHD/ReLD; ATSP ICAM; and RouteFinder on OVRP/OVRPL/VRPL. Agreement covers only four or eight examples. It neither certifies complete equivalence nor makes the remaining columns universally invalid.

The strongest supporting paper is [Lv et al., ICML 2020](https://proceedings.mlr.press/v119/lv20a/lv20a.pdf): Section4, Equation10 progressively redistributes label responsibility within the candidate set. Section5 reports gains on synthetic partial-label image/UCI datasets and real partial-label datasets. However, Section3’s consistency result requires deterministic labels and **ambiguity degree below one**, meaning every false candidate has some chance of exclusion.

That assumption fails for the R60 pair alone: a strict comparison leaves six of seven candidates possible, and the five unobserved candidates are always retained. Original fully labelled rows can anchor learning, but the extra pair data alone cannot identify their relative winner probabilities. Likewise, [Ishida et al., ICML 2019](https://proceedings.mlr.press/v97/ishida19a/ishida19a.pdf), Equation5 and Theorem1, require uniformly generated complementary labels for their unbiased-risk result; Section4 constructs experimental labels accordingly. Our fixed, outcome-dependent solver comparisons do not follow that mechanism.

**Difference and counterevidence.** This would directly constrain full-pool winner learning, unlike R60’s pair BCE or R62’s auxiliary comparison pretraining. It does not repeat R39/R43/R71’s continuous-cost prediction, ranking or density estimation. Nevertheless, merely replacing R60’s BCE with a partial-label loss would remain a small variant. R60’s modest gain and R62’s negative scaling result are strong counterevidence to expecting much from repackaging those observations. R73 already tests broader inputs with teacher decisions, so another pseudo-label variant is not independently justified now.

**Minimum meaningful screen, conditional only:** a full18 comparison using the same accepted observations, backbone and exposure, contrasting joint partial-label identification with matched comparison supervision. All original native labels and inference candidates remain. No native validation gain rejects the package; lower ambiguity loss does not rescue it. Computation should remain close to existing selector training, with small label-set storage and unchanged single-selector inference.

The unresolved question is useful *native-compatible* constraint coverage beyond the pair. At most one future small TRAIN-only row query could inspect aligned, co-observed costs and survivor-set sizes for those nine additional columns, retaining deployment provenance. That would characterize coverage, not certify historical equivalence. **I did not execute it.**

**2. Adaptive acquisition of bounded solver-state information**

This family changes what the selector can observe at query time: it chooses whether and where to spend a strictly limited amount of incomplete computation, then predicts one winner from the entire legal pool.

[Janisch et al., AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/download/4287/4165), “Problem definition,” Equations1–2, formulate actions that either acquire a feature or terminate with a class prediction. Figure4/Table1 evaluate eight classification datasets with 8–784 features. This supports adaptive information acquisition, but their assigned feature costs and tabular observations do not establish useful routing probes. Their error-plus-cost objective also cannot replace our native Top1 target.

[Renau and Hart, January 2024](https://arxiv.org/html/2401.12745v1), Sections3–4 and6, provide algorithm-selection evidence for short trajectories. The conditions are narrow: three continuous optimizers, 24 BBOB functions, and validation on held-out instances of functions also represented in training. Labels use performance after 100,000 evaluations. Their fitness observations evaluate complete candidate points; under our boundary, complete routing solutions cannot quietly become probe features.

**The substantive change** would be a learned acquisition-and-stop policy over incomplete construction states, with a declared information budget. All candidates remain selectable. No complete tour, final cost or current-query label enters the selector. Nonselected computations must stop well before completion; selected execution must preserve its original deployment recipe. Probe computation, model loading and final solver execution all count toward latency.

**Strong local counterevidence:** [R51](/public/home/shiys/0selection2/code/V4/runs/R51_solver_behavior_probe/RESULTS.md) already tested real solver behavior, not merely an embedding. Its fixed ten-step, all-start, pooled two-solver probe reduced original binary validation accuracy from 56.9% to 54.3%, with measured pipeline latency 1.1961 times the static comparator. Simply lengthening that prefix or changing pooling is unjustified.

What remains untested is full18 native classification with **state-dependent acquisition**, rather than R51’s fixed probe, R47’s historical-neighbor retrieval, R50’s static solver encoding, or R53–56’s pair specialists and predicted route structure. R60–73 primarily change learning from existing inputs; they do not establish the value of this additional execution information.

**Minimum meaningful screen, conditional only:** fixed versus adaptive acquisition under the same bounded probe menu and maximum work, both predicting original full-pool winners. Compare native validation accuracy and complete latency, retaining the static incumbent as a reference. No adaptive gain rejects the allocation hypothesis; a runtime reduction alone does not advance70%.

The small acquisition policy is unlikely to dominate memory. Native decoder states, all-start execution and heterogeneous solver interfaces are the uncertain costs. Sequential execution may fit the3090, but full18 prefix availability, recipe-preserving continuation and useful latency are unestablished. R51’s measurements cannot be extrapolated. **This family is more distinct mechanistically, not more strongly demonstrated.**

**Conditional implication:** neither another pair-loss substitution nor another fixed pooled prefix is justified by these papers. Completed R73 TRAIN/validation can inform the next decision, but cannot establish missing deployment equivalence or full18 probe feasibility.

**Memory update**

Preparatory research examined four primary papers on partial/complementary-label learning and adaptive information acquisition. R62 metadata records39 small-sample replay-agreeing columns, including nine beyond R60’s30; this does not certify additional native full-pool labels. Fixed-pair exclusions leave persistent ambiguity, so standard partial-label consistency assumptions do not apply directly. Adaptive bounded solver-state acquisition remains distinct from fixed-input learning, but R51’s negative binary probe and unestablished full18 interfaces are material counterevidence. Neither family is adopted or launch-ready. R73 outcomes were excluded and final validation remains pending. Historical TEST exposure is disclosed and was not recommendation evidence. R58 and legacy B discrepancies remain deferred; the original full18 single-completed-solver70% goal is unchanged.

Only read-only source/metadata and primary-paper research occurred. No R73 learning or delivery outcomes, arrays, weights, inference, GPU/job queries, tests, edits, Git operations or agents were used.
