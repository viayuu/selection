`review_independence=same-family; acceptance_status=provisional; PLAN/SCOUT ONLY`

**My conditional preference is a full18 comparison of shared versus task-specific complete selector parameters, if completed R68 does not advance the validation incumbent. No R69 is adopted.** I checked five primary papers. R68 outcomes and TEST artifacts were excluded; historical TEST exposure in memory remains disclosed.

R67 supports retaining its useful contrastive initialization: B gained105 validation decisions across13 tasks, but remained13 decisions below R60B. Its late TRAIN/validation divergence makes additional capacity risky, while providing no proof that shared capacity is sufficient. I compared these three directions.

1. **Change which tasks share parameters.**

   [Standley et al., Sections4–5](https://proceedings.mlr.press/v119/standley20a/standley20a.pdf) show that task grouping benefits depend on network capacity and training-data volume. [Fifty et al., Section4.1, Eq.1](https://arxiv.org/pdf/2109.04617) measure affinity through the relative loss change after another task’s update:
   `Z(i→j) = 1 − L_j(θ − η∇L_i)/L_j(θ)`.
   Their grouping method differs from modifying gradients while retaining one shared model.

   **Project adaptation:** compare one shared full18 selector against a single task-indexed bundle containing18 complete selectors. Initialize every copy from the same R67B terminal pretraining trunk and identical fresh head. Each task retains its entire legal pool. Known problem identity activates one selector, which chooses one completed solver. Select the entire bundle at one common epoch using aggregate validation correct count.

   This removes cross-task parameter sharing throughout fine-tuning. R64 PCGrad retained shared parameters; R53 only modified a gated solver pair. R49–56’s diagnostics, frozen readouts and route-supervised specialists do not establish this full18 contrast.

   **Counter-evidence and control:** R64’s negligible PCGrad gain weakens an interference explanation; R67’s overfitting warns that reducing each model’s training population to10,000 examples could hurt. Shared versus untied models with identical per-task exposure is the minimum control. It changes total parameter capacity, so a gain would not isolate gradient interference. Failure to improve full18 validation would reject this bounded unsharing recipe.

   **Feasibility:** sequential task forwards preserve the calibrated batch256 activation shape. The local single-selector checkpoint is7.46MB;18 parameter copies are modest relative to activation memory, although optimizer storage and runtime must be disclosed. There is one active selector forward per query.

2. **Train on interpolated hidden examples while preserving ordinary native supervision.**

   [Manifold Mixup, Section2, Eq.1](https://proceedings.mlr.press/v97/verma19a/verma19a.pdf) jointly interpolates hidden representations and one-hot targets:
   `h̃ = λh_i + (1−λ)h_j`, `ỹ = λy_i + (1−λ)y_j`.
   [RegMixup, Eq.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/5ddcfaad1cb72ce6f1a365e8f1ecf791-Paper-Conference.pdf) retains clean-example CE and adds interpolation CE instead of replacing ordinary supervision.

   **Project adaptation:** interpolate aligned candidate-conditioned representations between TRAIN examples from the same task and legal pool, at one fixed location before the utility head. Keep original CE/RDrop. These are auxiliary latent targets, not asserted winners for newly constructed routing instances. No coordinate transformation or solver-label invariance assumption is needed.

   This supplies predictions between training representations. RDrop compares dropout views of one example; R67 clusters examples by winner without imposing these interpolated predictions. It also differs from R54/R55 reconstruction and R61/R62 pretraining targets.

   **Counter-evidence and control:** winner boundaries may be discontinuous, so linear interpolation could erase useful distinctions. R67’s modest benefit and R66’s weak fitting make stronger smoothing uncertain. Compare an extra unmixed head-loss pass against the same computation using mixed features and targets, with one fixed mixing recipe. Better calibration or CE without better native validation Top1 would not support the selection hypothesis.

   **Feasibility:** reuse backbone features, adding small mixing and head computations; inference remains unchanged. This is a plausible secondary direction, with stronger overlap with R68’s generalization motivation.

3. **Predict a joint cost distribution and choose the most probable native winner.**

   [Salinas et al., Sections3–4, Eq.5](https://proceedings.neurips.cc/paper_files/paper/2019/file/0b105cf1504c4e241fcc6d519ea962fb-Paper.pdf) combine likelihood training, marginal transformations and covariance `Σ = D + VVᵀ`. Their evidence concerns forecasting.

   **Project adaptation:** use the existing complete TRAIN cost vectors to learn a conditional joint distribution, then select
   `argmax_m P(C_m = min_j C_j | x)`.
   Joint samples must be mapped back to original cost units before comparing candidates. Sampling predicted costs executes no solvers.

   This differs from [R39’s point estimates](/public/home/shiys/0selection2/code/V4/dual_stream.py:110), R43 pairwise margins and R62 comparison pretraining: dependence between candidate costs affects the probability of being the minimum. It adds no observations.

   **Counter-evidence and control:** previous cost-supervised approaches have not established a broad gain. Likelihood may emphasize absolute-cost variation while missing winner boundaries; learned variance does not establish label noise. Compare diagonal versus fixed rank-two covariance with identical backbone, marginal treatment and sampling rule. Better likelihood without better native validation Top1 would refute the proposed benefit.

   **Feasibility:** small legal pools make covariance operations and fixed-budget sampling inexpensive relative to the backbone. Inference adds distribution sampling to one selector forward. This remains more speculative than task unsharing.

For the preferred conditional direction,40 full-data epochs would preserve7.2M selected-row presentations per arm. Under the existing triple schedule, that is9,600 logical updates; each untied task model receives1,600 active updates and400,000 rows. Those differences must be reported alongside18-fold parameter storage. One globally selected bundle prevents task-wise checkpoint assembly; no task-group search is proposed.

Adoption must await completed R68 TRAIN/validation. Any subsequent experiment retains original18 tasks, full pools, native winners, validation-only checkpoint/official-arm selection, and one frozen full18 TEST per model or bundle. These papers and project results provide **no evidence that70% is attainable**, and establish no accuracy ceiling.

**Memory update:** Conditional scout checked five primary papers covering task grouping, latent-example interpolation and joint cost-distribution learning. Full-model task unsharing is the preferred conditional next comparison if completed R68 does not advance full18 validation; it differs from R64 gradient projection and prior gated pair specialists. Its main risks are reduced per-model data and increased parameter capacity. Latent interpolation is distinct from RDrop/SupCon but may oversmooth winner boundaries; joint cost distributions remain assumption-heavy. No R69 is adopted. R68 outcomes and TEST evidence did not guide this scout; historical TEST exposure is disclosed. R58 and legacy eight-B-task semantics remain deferred, and the full18 single-solver70% goal remains active and unachieved.