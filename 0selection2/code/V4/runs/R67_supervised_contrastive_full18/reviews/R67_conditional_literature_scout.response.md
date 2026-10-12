`review_independence=same-family; acceptance_status=provisional; PLAN/SCOUT ONLY`

**My conditional preference is one SAM-versus-AdamW experiment if completed R67 fails its matched validation comparison. This is not R68 adoption.** I checked five primary papers and relevant local implementation paths. R67 partial outcomes and TEST evidence did not inform this recommendation; historical TEST exposure in memory remains disclosed.

R66 supplies little support for adding further correction stages: corrective B reached48.666667% validation versus independent-sum A’s48.766667%, with little additional TRAIN fitting. R65 provides a more specific optimization signal: ordinary AdamW developed late TRAIN/validation divergence, whereas schedule-free stabilized predictions without beating R60B’s49.361111%. Neither observation establishes sharpness as the cause.

1. **Solver-pretrained encoder transfer: feasible, but substantially overlaps R59.**

   [ReLD, Sections3–4.2](https://arxiv.org/html/2503.00753v1) improves contextual decoding through residual context injection and a nonlinear query transformation. Its solver results do not establish that its encoder distinguishes competing solvers. [RouteFinder, Sections4.2–4.4](https://arxiv.org/html/2406.15007v4) offers a different upstream representation: global attributes enter the encoder, with mixed-variant RL training and reward normalization. Transferring that representation into a selector would be a project adaptation.

   The important local correction is that [R59 already fine-tuned the native encoder end to end]( /public/home/shiys/0selection2/code/V4/r59_transfer.py:139), against the same randomly initialized architecture. Its final A/B budgets differed, so it does not reject every transfer recipe, but “unfreeze R50” is not an untried mechanism. The compatible ReLD-MTL checkpoint still exists. RouteFinder’s50/100-node checkpoints also exist with a local loader; their existence does not establish selector compatibility.

   A defensible new transfer comparison would require the same RouteFinder-based architecture in both arms, pretrained versus random encoder, identical downstream initialization and exposure, preserving all input fields and the existing TSP/ATSP paths. Failure to improve full18 validation over the random control would refute that bounded transfer hypothesis. This is feasible in principle, but batch256 memory is not established for the changed backbone, and upstream data budgets/provenance remain additional limitations. **Lower priority.**

2. **SAM: the strongest conditional choice, with a fitting caveat.**

   [Foret et al., Section2, Algorithm1](https://arxiv.org/html/2010.01412v3) optimize loss in a parameter neighborhood: compute a gradient, perturb weights in its normalized direction, then update the original weights using the perturbed gradient. This differs from PCGrad’s task-gradient projections and schedule-free parameter averaging. AppendixC.3 examines a fixed radius0.05.

   The [transformer study, Section4 and Table1](https://arxiv.org/html/2106.01548v3), finds improved image-classification generalization with SAM, including **higher ordinary training loss**. That supports testing neighborhood robustness; it does not support promising better TRAIN fitting here. Applying SAM to this selector’s native CE/RDrop and task-averaged gradient is our adaptation.

   Matched control: ordinary AdamW versus AdamW with one fixed SAM perturbation, sharing architecture, initialization, examples and successful-update budget. The hypothesis fails this bounded comparison if SAM does not improve validation Top1. Recovering a weak AdamW baseline while remaining below R60B would be limited progress. Sequential gradient passes should preserve realbatch256 capacity, with approximately twice B’s gradient computation and unchanged inference cost.

3. **Joint performance-distribution learning: distinct, but more speculative.**

   [Salinas et al., Sections3–4, Equation5](https://proceedings.neurips.cc/paper_files/paper/2019/file/0b105cf1504c4e241fcc6d519ea962fb-Paper.pdf) learn a joint predictive distribution using marginal transformations, likelihood training and low-rank-plus-diagonal covariance. Their evidence concerns multivariate forecasting.

   The project adaptation would learn the **joint distribution of the existing complete solver-cost vector**, then select the candidate with greatest predicted probability of being the native minimum. This differs from [R39’s point performance estimates]( /public/home/shiys/0selection2/code/V4/dual_stream.py:110), pairwise comparisons and R67’s cross-example winner clustering. It adds no observations; it changes supervision and the decision rule.

   Matched control: diagonal versus rank-two correlated predictive distributions, with identical backbone, marginal treatment, training exposure and fixed sampling-based winner rule. Better likelihood without better native validation Top1 would not support the selection hypothesis. Small candidate pools make this plausible at batch256, with one encoder forward plus inexpensive distribution sampling at inference. However, covariance misspecification and density fitting unrelated to winner boundaries are substantial risks. Predictive variance would not demonstrate label noise. **Second priority, not an adopted backup experiment.**

The conditional SAM experiment should remain narrow:

- Use fresh paired R64 `NativeUtilitySelector` models, seed2, original full18 inputs/pools/native winners and unchanged CE/RDrop. Reuse R65A’s AdamW settings: LR3e-4 after720 successful-update warmup, then constant; fixed40 epochs,9,600 updates and7.2M selected-row presentations per arm.
- Keep real256/task, three tasks/update, effective768 and existing tails. B uses fixed L2 radius0.05 from the **unscaled FP32 mean gradient across the entire three-task update**. Recompute on the same rows and dropout masks at perturbed weights, restore weights, then apply the second gradient with the existing final clipping. Both share initial weights and data order.
- Disclose unequal computation: A has two dropout forwards per selected row; B has four. Budgets match updates and selected examples, not FLOPs. Sequential passes require little additional persistent memory. No new sizing or curvature-measurement campaign is proposed.
- Select each model by integer full18 validation-correct count, earliest exact tie; freeze the official arm by validation, tieA. Keep the full-pool utility rule and one completed solver. Only then permit one frozen full18 TEST per model. Cost remains disclosure.

This tests a different training objective with established external evidence and modest implementation scope. It could still worsen this weak-fitting regime. No result from the papers justifies a70% projection, and completed R67 evidence must precede any adoption.

Memory update, proposed without editing files: Conditional scout checked five primary papers. R59 already implemented trainable solver-encoder transfer; repeating unfreezing is not a new mechanism. SAM is the preferred conditional next contrast because neighborhood optimization remains untested and R65A showed actual late fitting/generalization divergence, while R66’s weak fitting limits confidence. Joint cost-distribution learning is distinct but carries stronger modeling assumptions. R67 remains unresolved, no R68 plan is adopted, and no R67 outcomes or TEST feedback guided this scout. R58 and legacy eight-B-task semantics remain deferred; the full18 single-solver70% goal remains active.