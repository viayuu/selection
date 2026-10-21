# R92: Two Structural Alternatives

2026-10-22. Paper research and ideas only; no experimental plan. All project facts below come from the supplied Sol summaries, including the R89/R91 and older-negative updates; no project code/results were inspected. Backend model/effort is not independently verifiable here; no additional model was invoked.

**Recommendation:** prioritize a learned multiresolution instance backbone; retain a standalone tree selector as the second, lower-confidence alternative. Neither is a new loss coefficient, class quota, solver gate, or winner-label hierarchy.

**Fixed scope:** original raw instances, NEW90k 81k/4.5k/4.5k per task, frozen 18-task legal pools, unchanged winner labels and cost matrix. Costs/winners are supervision only. Both directions select a legal method without solver execution, routes, new data, relabeling, or retrieval at inference. Existing neural seed/batch conventions are not redesigned here.

## 1. Recommended: Learn Graphs of Groups Before Global Encoding

**Verified mechanism.** DiffPool learns separate node embeddings and row-softmax node-to-cluster assignments, aggregates both embeddings and connectivity, then applies another GNN to the resulting cluster graph. This is repeated before final graph readout, not merely learned-query pooling. Its equations (3)-(6) specify the computational change. [Ying et al., NeurIPS 2018, sections 3.2-3.3](https://papers.nips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)

**Concrete project adaptation, not a paper result:**

- Start from raw node attributes, existing condition/presence inputs, and directed distance-derived relation channels. Distinguish input travel distances `D` from the supervised solver-cost matrix `C`; never feed `C` into the backbone.
- At each resolution compute `Z = GNN_embed(E,X)` and `S = row_softmax(GNN_assign(E,X))`. Form `X_next = S^T Z` and, separately for each relation channel, `E_next[r] = S^T E[r] S`.
- Carry cluster masses and raw depot/type information. Process the new weighted, directed cluster graph with further message passing and repeat coarsening. These relations are latent aggregate evidence, not valid routes or feasibility certificates.
- Only after intermediate cluster-graph computation produce the instance representation for the existing method-conditioned scorer. Keep the supplied separable utility-training interface; assignment learning receives its gradients through this backbone.

**Why this could help:** different instances can share local radius/CV/anisotropy summaries yet differ in how local structures connect. Group-to-group propagation could expose those arrangements and constraint interactions before global compression. This is a conditional hypothesis about useful information, not a diagnosis that current representations have lost it.

**Anti-repeat boundary:** existing k4/8/16 geometry plus mean/max does not construct another graph. R31 queries/PMA/solver-set attention change readout; this changes intermediate computation. R56 edge paths and R48-55 pair/route encodings are nearby negative evidence, but the supplied descriptions do not establish repeated learned coarsening followed by cluster-graph propagation. This is not R53-style winner/pair-trigger hierarchy: groups contain instance nodes, never candidate classes. Repeating only attention pooling would not implement the recommendation.

**Risks and evidence limits.** DiffPool's original input formulation excludes edge features; directed multichannel pooling is an adaptation. Its clusters need not be spatially connected. Section 3.3 reports difficulty learning assignments from classification alone and adds link-reconstruction and entropy objectives. Our unchanged-loss proposal therefore does **not** inherit full-method evidence; NOLP removes link reconstruction, not all auxiliary regularization. [Original formulation and caveats](https://papers.nips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf) For ATSP, copying the symmetric `SS^T` reconstruction target would conflict with directed relations. Assignment collapse, erased rare-winner detail, cluster-size effects and dense quadratic relation storage remain real risks.

**Why not prioritize the screened pair-tensor alternative:** PPGN explicitly composes pair states through a shared intermediate node, with quadratic storage and cubic dense computation. Its expressiveness theorem is not evidence of better solver selection or superiority to this particular attention backbone. Given existing edge/pair negatives, hierarchy provides the clearer distinct computation to recommend now; PPGN is not a third proposal. [Maron et al., NeurIPS 2019, section 6](https://arxiv.org/pdf/1905.11136)

## 2. Independent Alternative: Replace the Selector With Boosted Trees

**Mechanism:** a task-conditioned multiclass tree ensemble on explicit, instance-derived structural features, replacing the neural encoder/interactions/scorer rather than supplying another head. Each method score is `F_m(x,t) = sum_b tree_b,m(phi(x),t)`. Tree paths learn feature conjunctions and piecewise decision regions; predictions remain natural argmax within the frozen legal pool.

Use existing geometry summaries alongside explicit raw-distance/direction and active-constraint summaries; no new nearest-neighbor statistics are the claimed innovation. Task identity/presence flags are ordinary inputs, not target encodings. Original winner labels support ordinary multiclass log-loss without class balancing, pseudo-labels, or per-method thresholds. This is a separate learning formulation, not a claim that stock tree software supports the current custom utility objective.

**Primary support and limits:** CatBoost describes additive tree fitting and ordered residual estimation using preceding examples, avoiding self-target contamination; sklearn boosting must not be called that ordered algorithm. [Prokhorenkova et al., NeurIPS 2018, sections 2, 4-5](https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) Grinsztajn et al. provide evidence for tree inductive biases on irregular tabular functions, but their main benchmark uses approximately 10k samples and balanced binary classification, unlike this natural multiclass setting. Their balancing is not imported here. [NeurIPS 2022, sections 3.2 and 5](https://papers.nips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)

**Anti-repeat and practical boundary:** R48's single-OVRPTW tree failure is directly relevant negative evidence, not erased by expanding to 18 tasks. Whole-selector tree partitioning still differs from appending hand features to a neural network, OVR gradient reweighting, private heads, or changing CE's probability link. Trees cannot recover information absent from their features and may overfit small winner regions. The supplied sklearn installation offers a potential histogram-boosting route; its exact installed version was not checked. CatBoost installation is not a research prerequisite. [Official multiclass estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Joint-Goal Boundary

Matching class marginals with zero argmax selections does not prove random labels or an accuracy ceiling. Conversely, increasing a method's selection count is not true-winner recovery. Both ideas require useful instance-conditional separation to exist. Higher natural Top1 can still raise mean cost through fewer but more expensive mistakes; none of these four papers establishes the requested joint improvement. No R89/R91 result is reinterpreted as achieving it. The recommendation is structural and conditional, not a promised gain or an instruction to launch work.

