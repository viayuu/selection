# Post-R102: Relative Competence Is Not Performance Reconstruction

Date: 2026-10-22. Scope: PAPERS AND IDEAS ONLY. Research-lit source selection: `web`.

Requested model: `gpt-6-astra`, `reasoning_effort=max`. Independent backend identity and reasoning-effort attestation are unavailable. This memo does not claim an authenticated model switch. No subagents were used.

## 1. Decision and Evidence Boundary

**DECLINE to recommend an additional mechanism now.** The only candidate examined in depth is **performance-derived latent instance-solver interaction factors, inferred from permitted instance inputs at cold start**. There is genuine algorithm-selection evidence for this family, not merely generic representation-learning evidence. However, the literature does not establish a sufficiently different mechanism relative to the supplied history, or the missing input-to-relative-competence bridge in this setting. Waiting for the existing work is preferable to relabeling a cost head or performance-pretraining objective.

This is a bounded literature judgment, not a claim that low-rank interactions cannot help, that current models are optimal, or that no other mechanisms exist.

The following project facts are supplied by the user/Sol, not independently inspected or recomputed:

| Supplied fact | What it does and does not establish here |
| --- | --- |
| Natural all18 TRAIN and VAL Top1 are both about 71%, with a small gap; some methods with nontrivial positive winner mass are never selected. | A motivation to seek conditional discrimination. Not proof of a Bayes ceiling, label randomness, an optimization diagnosis, or a need for uniform method usage. |
| R94 frozen pretrained native node fusion: TEST Top1 71.5778%, actual regret 0.379115%; fixed36 true correct 117/4598 versus R89B 45/4598; 86 new rescues, all MVRP. | Limited evidence that some input-derived transferred knowledge can help. The subgroup counts are not whole-TEST accuracy; new rescues are not net benefit or evidence of controlled false picks. They do not establish transfer to TSP/ATSP. |
| R97/R98/R100 natural full81k historical-TRAIN context mostly produces overlapping target rescues, with no joint gain. | A negative prior against another repackaging of performance-profile or binary-pair context. Not an impossibility result for all historical supervision. |
| R99A/B row/Sinkhorn and R101A/B random/pretrained node-hint workers are healthy; R102A/B native TSP/ATSP feature coverage is queue-ready. | These are supplied pending statuses, not empirical answers. Nothing here requests changes to those workers or their queue. |

FullNEW90k splits and legal pools remain the conceptual setting. This memo neither changes them nor proposes R58, solver calls, route/cost regeneration, current-query outcomes, or new contexts. Historical TRAIN outcomes are permissible supervision, but cannot silently become the current query's input. Recovery of actual winners, false picks and their costs remain distinct from selection frequency.

## 2. The Only Candidate Considered

The idea is to learn a shared coordinate system from historical instance-by-solver performance, then predict a new instance's coordinates from its permitted input. Solvers have different loadings on those coordinates, so the same structural property can favor one solver over another.

For one problem and its fixed legal pool, a conceptual decomposition is

\[
c_a(x)=b(x)+\beta_a+z(x)^\top v_a+\varepsilon_a(x).
\]

Here `b(x)` is common difficulty, `beta_a` is a method main effect, and the interaction `z(x)^T v_a` is the candidate competence signal. This is a hypothesis about structure, not an estimated decomposition of project data or a new causal interpretation. Without identifying conventions the terms are not unique, and latent axes are not automatically interpretable solver skills.

The relevant comparison is

\[
c_a(x)-c_b(x)
=(\beta_a-\beta_b)+z(x)^\top(v_a-v_b)
+\varepsilon_a(x)-\varepsilon_b(x).
\]

Common difficulty cancels; the interaction can change which solver is preferable across instances. Method main effects need not disappear: removing them from the final comparison would change the decision problem, not merely expose hidden competence. Nothing in this decomposition requires uniform selection.

The attractive hypothesis is **shared relative-performance structure**, rather than separately relearning each comparison. Its unproved part is that a compact structure exists, is recoverable from allowed inputs, and preserves the distinctions that determine the winner. Reconstructing historical rows using their outcomes does not establish any of those properties for a new query.

## 3. Three Primary Full-Text Sources

Only the following three papers supply the substantive external evidence. Reading covered the specified body sections, equations and application descriptions, not just search snippets. They are peer-reviewed papers; none establishes an improvement for this project.

| Paper and full-text location | Actual supporting evidence | Limitation for the present question |
| --- | --- | --- |
| David Stern, Horst Samulowitz, Ralf Herbrich, Thore Graepel, Luca Pulina and Armando Tacchella. **Collaborative Expert Portfolio Management**. AAAI 2010, 179-184. [Publisher PDF](https://cdn.aaai.org/ojs/7561/7561-13-11091-1-2-20201228.pdf): "Expert Portfolio Management Model," p.180; "Application and Evaluation," pp.181-184, especially Tables 2 and 5. [DOI](https://doi.org/10.1609/aaai.v24i1.7561). | Matchbox maps task and solver descriptions into a shared latent space, uses their inner product plus biases, and learns from performance feedback. Table 2 separately reports batch-trained QBF selection without online adaptation, trying one selected solver per test problem. This is direct evidence that the family can support instance-specific solver choice. | The feedback is runtime-oriented, with timeout/slow/fast modeling and a utility-based decision. Other results include online adaptation or scheduling and cannot be transferred to input-only selection. No evidence addresses native routing winner recovery together with false-pick cost. |
| Pini Zilber and Boaz Nadler. **Inductive Matrix Completion: No Bad Local Minima and a Fast Algorithm**. ICML 2022, PMLR 162:27671-27692. [Publisher PDF](https://proceedings.mlr.press/v162/zilber22a/zilber22a.pdf): Sections 2-3, equation (1), Theorem 3.1; Section 5.2, Theorems 5.1 and 5.4. [Official metadata](https://proceedings.mlr.press/v162/zilber22a.html). | The recovery results assume suitable known side-information spaces, incoherence and sufficiently many uniformly sampled entries. The noiseless landscape result and bounded-error recovery result make explicit when bilinear structure is learnable rather than merely descriptive. | The true matrix's row/column spaces must be covered by the supplied features, up to the controlled error allowed by the relevant result. The theorem does not discover an adequate graph representation, guarantee native winner signs, or directly cover this project's entirely held-out query rows. |
| Chengrun Yang, Yuji Akimoto, Dae Won Kim and Madeleine Udell. **Oboe: Collaborative Filtering for AutoML Model Selection**. KDD 2019. [Author-hosted PDF](https://web.stanford.edu/~udell/doc/yang2019_oboe.pdf): Section 3.1 and Figure 2; Sections 5.1-5.2, including "Cold-start." [DOI](https://doi.org/10.1145/3292500.3330909). | Historical model-error matrices admit useful low-rank approximations in the reported AutoML setting. New-task coordinates are inferred by fitting selected models on that new dataset and observing their errors. The study evaluates selection regret as well as reconstruction-related quantities. | Those new-task observations are exactly the information source unavailable here. Removing them and replacing them with an input-only prediction is a missing mechanism, not a consequence of Oboe's results. AutoML error differences and routing's native winner labels are also different targets. |

**The strongest positive precedent is Matchbox's batch setting, not Oboe's probes.** Its auction results progress from matching the best individual solver in one dimension to higher utility with richer factors. This supports conditional solver matching, not rare routing winner recovery. [Application sections and Tables 2/5](https://cdn.aaai.org/ojs/7561/7561-13-11091-1-2-20201228.pdf).

The matrix-completion theorem and the applied papers form a useful conceptual connection, **not a combined theorem for Matchbox, Oboe, or a neural selector**. Entirely unseen rows, learned graph features and near-tied solver costs introduce assumptions not discharged by the matrix-recovery results.

Retrieval limitation: ALORS was located, but its publisher/HAL full text was not successfully retrieved. Its abstract was not promoted to section-level evidence. The conclusions above do not rely on that inaccessible text. No Zotero import, bibliography synchronization or local PDF-library search was performed under `sources:web`.

## 4. Why This Is Not Yet a Genuinely Different Direction

The following overlap assessment uses only the authorized research memos and the mechanisms supplied in the prompt. It is not an implementation audit, and absence of an exact name in those memos is not proof that a mechanism was never tried.

| Existing family | Overlap that prevents a novelty claim |
| --- | --- |
| R62 source-separated portfolio-performance pretraining; existing cost heads and R71 joint cost-difference modeling | Historical solver performance already supervises instance representations or performance structure. Compressing the same targets into factors may alter inductive bias, but does not introduce a new source of competence information. R62's historical setting is not a matched negative result for the proposed factorization. |
| Existing method-conditioned representations, method-private residuals, pair specialists and ranking objectives | A fixed-pool bilinear score `z(x)^T v_a + beta_a` is algebraically a shared representation with a method-specific linear readout. Without a demonstrated additional structural restriction or information source, "competence embedding" can merely rename a familiar head. This algebra does not assert the exact architecture of any unread implementation. |
| R97/R98/R100 performance-profile and pairwise historical-TRAIN context | Changing contextual inference into a learned latent coordinate system is a real statistical change, but it reuses the same outcome information. The supplied overlap of rescues is not erased by changing the storage of that information from context to parameters. |
| R99/R101/R102 input transformation and pretrained knowledge routes | Those pending routes concern how permitted input information becomes usable. Factorizing output supervision does not answer their unresolved questions or supply evidence of independent gains. |

The overlap sources are [experiment_mechanism_index.md](experiment_mechanism_index.md), [post_r97_major_directions.md](post_r97_major_directions.md), [post_r98_transfer_without_online_donor.md](post_r98_transfer_without_online_donor.md), [post_r100_broad_direction.md](post_r100_broad_direction.md), and [pretrained_representation_coverage_tsp_atsp.md](pretrained_representation_coverage_tsp_atsp.md). Their code/result links were not followed, and their old statuses do not supersede the present user-supplied facts. R63 already covers candidate-state GRU recurrence; no recursive-network rebranding is proposed here.

**Strongest counterargument: the useful signal may be exactly what compression discards, and the permitted query input may not predict it.** A low reconstruction error can be dominated by common difficulty, persistent method advantages or large frequent losses. Even after common effects are separated, rare small advantages need not align with leading interaction factors. The current task already has historical outcome supervision; its unresolved problem is not simply filling a sparse ratings table.

A purely mathematical illustration, not project evidence, makes the distinction explicit:

\[
c_1(x)=d(x)+\delta s(x),\qquad
c_2(x)=d(x)-\delta s(x),\qquad s(x)\in\{-1,+1\},\ \delta>0.
\]

Predicting both costs as `d(x)` can have arbitrarily small cost-reconstruction error as `delta` decreases, yet conveys none of the sign that distinguishes the winners. The factorization can exist even if the allowed representation makes `s(x)` difficult to learn. This does not assert randomness or impossibility for the original input. It explains why successful reconstruction, latent clustering and a nicer performance geometry cannot stand in for winner recovery.

## 5. The Exact Unresolved Bridge

The missing claim is not "solver performance has latent structure." It is:

> From the permitted raw input of an unseen instance, historical TRAIN supervision can teach a representation that predicts the instance-dependent signs and consequential magnitudes of solver cost differences, including the regions where presently unselected methods really win, without substituting method popularity or current-query outcomes.

This bridge contains three distinct claims: performance interactions are compressible in a decision-preserving way; their coordinates are predictable from allowed inputs without query outcomes; and their errors preserve the relevant winner boundaries rather than merely lowering average reconstruction error. The three papers do not jointly establish these claims for the supplied setting. Expected-cost prediction also need not optimize winner probability, so naming the target "relative competence" does not remove the Top1/cost distinction.

**Final judgment: retain the conceptual question, decline a new recommendation.** The most defensible next intellectual focus is decision-relevant predictability of solver contrasts, not marginal usage or another named latent architecture. Current R99/R101/R102 work may inform that question, but its outcomes are not presumed here. Waiting does not imply that those results will automatically validate this candidate.

This memo adds no experiment arms, budget, thresholds, engineering gates, implementation, result review or launch instructions. Only this document was written. No project code, result files, datasets, jobs, models or checkpoints were inspected; no GPU commands, package changes, notifications or Git actions were used.
