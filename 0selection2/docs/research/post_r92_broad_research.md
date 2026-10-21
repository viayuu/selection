# Post-R92 Broad-Mechanism Research: Explicit Decline

Date: 2026-10-22. Focused literature audit, not an exhaustive survey.

**Recommendation: do not prioritize a new training mechanism from this screen.** SPO+/Fenchel-Young (FY) do not provide the requested broad departure here; algorithm-execution representation learning is different but insufficiently justified with the authorized supervision. This is not a claim that improvement is impossible.

Scope: project facts come exclusively from the supplied Sol summary; no project code/results were inspected. No experiments, budgets, GPU work, commits, or delegation. Zotero was unavailable; primary full texts were checked directly. Requested-model limitation: this session cannot verify execution as `gpt-6-astra`; this is a session-authored note, not an attributed Astra review.

## Evidence Boundary

NEW90k has 81k/4.5k/4.5k instances per task and all legal candidates' scalar costs. Supplied R89B TEST: Top1 71.4198%, regret 0.38965%; the 36 fixed zero/near-zero targets have true-winner recall below 1%. Supplied R92 regression: Top1 69.5716%, regret 0.424277%; Sol judged its joint screen unsuccessful. The classifier is not final and coarsening runs are pending. These facts do not eliminate all regression or representation approaches. LEGACY18 findings are not controlled NEW90k comparisons. NSS's similar zeros and negative R91 results do not establish a cause.

## Verified Primary Sources

| Source | Full-text location and relevant finding |
|---|---|
| Elmachtoub & Grigas, **Smart "Predict, then Optimize"**, Management Science 68(1), 2022. [Author preprint, v5](https://arxiv.org/pdf/1710.08005) | Definition 3; Proposition 3; Assumption 1/Theorem 1. SPO+ bounds decision regret; consistency is conditional, not universal. |
| Blondel, Martins & Niculae, **Learning with Fenchel-Young Losses**, JMLR 21(35), 2020. [Full text](https://jmlr.org/papers/volume21/19-021/19-021.pdf) | Proposition 1; Definition 2/Table 1; Section 3.4. Unifies CE, hinge and cost-augmented losses; symmetric regularization preserves score ordering. |
| Liu & Grigas, **Risk Bounds and Calibration for a Smart Predict-then-Optimize Method**, NeurIPS 2021. [Full text](https://proceedings.neurips.cc/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf) | Section 3, Theorem 3.1, Eq. (4). Regret calibration uses restricted conditional cost distributions, including symmetry and a Gaussian density lower bound. |
| Tewari & Bartlett, **On the Consistency of Multiclass Classification Methods**, JMLR 8, 2007. [Full text](https://www.jmlr.org/papers/volume8/tewari07a/tewari07a.pdf) | Section 5.1, pp. 1017-1018. The Crammer-Singer multiclass hinge is not universally classification-consistent. |
| Velickovic et al., **The CLRS Algorithmic Reasoning Benchmark**, ICML 2022. [Full text](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf) | Sections 3.1-3.2 and 4.1, Eqs. (1)-(3). Algorithm input/output supervision, optional intermediate hints, and repeated encode-process-decode computation. |

## Mathematical Screen

The following are **our specializations**, not reported selector results. Within one task's legal pool A, assume a unique cost winner y and positive c_y for the derivation; this does not redefine existing winner/tie conventions. Define g_j=(c_j-c_y)/c_y.

**SPO+ collapses to a cost-augmented multiclass hinge.** Selecting one method has feasible set conv{e_j:j in A}, not a routing polytope. Definition 3 gives

\[
L_{\rm SPO+}(\hat c,c)
=\max_{j\in A}\{c_j-c_y+2(\hat c_y-\hat c_j)\}
=\max_{j\in A}\{c_j-c_y+s_j-s_y\},\quad s=-2\hat c.
\]

The score subgradient can be e_{j*}-e_y, where j* maximizes the bracket. Gradient allocation changes; end-to-end training can learn different features, but the representational family and available supervision are unchanged. There is no hidden route-optimization signal. Replacing c by g targets normalized instance regret, not a ratio of mean costs. Small gaps induce small margins despite unit Top1 penalties. [SPO+, Definition 3/Proposition 3](https://arxiv.org/pdf/1710.08005)

**FY is a construction, not automatically a new mechanism.**

\[
L_\Omega(s,e_y)=\Omega^*(s)+\Omega(e_y)-s_y.
\]

Negative Shannon entropy on the simplex gives exactly logsumexp(s)-s_y: ordinary CE. Cost augmentation with d_y=0 gives logsumexp(s+d)-s_y; an unregularized simplex gives max_j(s_j+d_j)-s_y, recovering the hinge above. Augmentation uses supervision only during training; deployment cannot use unknown d. Symmetric FY links, including sparsemax, preserve fixed-score ordering rather than create missing argmax regions. Retraining can change scores, but that is not a recovery guarantee. [FY, Table 1/Section 3.4/Proposition 1](https://jmlr.org/papers/volume21/19-021/19-021.pdf)

Convexity in scores is not convexity in neural parameters. SPO consistency concerns decision regret, not native-winner Top1. Since min_j(g_j)=0, normalized gap vectors lie on a union of lower-dimensional faces; the cited full-space density assumptions do not apply directly. An affine-coordinate/normalization analysis would be needed, not an automatic consistency claim. Using unit mismatch costs alone recovers the Crammer-Singer hinge, which is not universally classification-consistent. This does not prove failure on these data. [Calibration, Section 3](https://proceedings.neurips.cc/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf); [Tewari-Bartlett, Section 5.1](https://www.jmlr.org/papers/volume8/tewari07a/tewari07a.pdf)

## Why Joint Improvement And Zero Recovery Remain Unproven

For representation Z and p_j(Z)=P(Y=j|Z), the Bayes decisions differ:

\[
a^*_{01}=\arg\max_j p_j(Z),\qquad
a^*_{\rm regret}=\arg\min_j\mathbb E[g_j\mid Z].
\]

Our counterexample: two instance types share Z. With probabilities 0.6/0.4, their costs are respectively (1,1.001)/(1.01,1). Always selecting A gives Top1 60%, mean cost 1.004; selecting B gives 40%, 1.0006. Better representation could separate both types and improve both metrics. This is neither an empirical noise diagnosis nor a ceiling claim.

A combined decision cost d_j=1[j!=y]+lambda*g_j has conditional risk 1-p_j(Z)+lambda*E[g_j|Z]. One objective therefore still encodes a tradeoff; its surrogate need not inherit its Bayes rule. Simultaneous improvement remains possible if current predictions are suboptimal, not guaranteed by scalarization.

Also E[p_j(Z)]=P(Y=j) does not imply a positive-probability region where j is maximal: constant probabilities (0.50,0.34,0.16) have perfectly matched frequencies but zero argmax selections for the latter classes. Current data need not follow this example. Coverage quotas cannot substitute for conditional discrimination.

## Anti-Repeat Boundary And Decision

| Candidate | Boundary against supplied history |
|---|---|
| Shannon FY | Exact CE re-expression; not new after direct utility CE/R91. |
| SPO+/cost-augmented FY | Not proven identical to R82 ranking or R91 from summaries. Still the same scalar-score cost/ranking family, not a new representation mechanism. |
| Replace the entire objective bundle | Different from merely changing R91's link. Nevertheless, harmful objective conflict is unestablished; an untried ablation is not automatically a broad recommendation. |
| Algorithm-execution representations | Stepwise dynamics differ from R88 start-role descriptors/R83 code embeddings. CLRS does not establish cost-only selector transfer; its algorithm targets/hints are not established as authorized here. Hints are optional, not universally required. |

Reducing the last option to generic structure reconstruction, partial-cost pretraining, or neighborhood contrastive learning returns toward R54/55, R61/R62, or previously tried contrastive mechanisms. CLRS's execution evidence does not bridge that gap. [CLRS, Sections 3.2/4.1](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf)

**Decline now.** The unresolved research need is a defensible way to extract solver-relative, instance-conditional distinctions from authorized inputs beyond prior representation learning. None of these sources supplies that bridge. Pending R92 components are not negative evidence, and neither another loss name nor blind architecture growth merits a new run on this record.
