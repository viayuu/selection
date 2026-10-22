# Unbiased Per-Example Importance Sampling: Literature Screen

Scope: primary-paper research only. Project facts are supplied, not audited: full18, 81k TRAIN/problem, 2-10 legal methods, additive native-winner CE/compare/risk/RDrop row losses, batch 1024, approximately 71% TRAIN accuracy. Prior EQLv2 used positive/negative OVR-BCE class weights with natural shuffling, not adaptive example sampling.

## Mechanism And Exact Boundary

**Yes: distinct sampling dynamics, unchanged objective.** Class weighting/EQLv2, focal modulation, and methodwise gradient rescaling generally change expected loss contributions or gradient direction. Corrected example sampling instead changes exposure and gradient covariance, not class penalties or the legal-method pool.

For a natural uniform row objective within a problem, retain existing component coefficients and inter-problem averaging:
\[
F(\theta)=\frac1N\sum_i\mathbb E_\xi\ell_i(\theta;\xi),\qquad
\widehat g_t=\frac1B\sum_{b=1}^B
 \frac{\nabla_\theta\ell_{I_b}(\theta_t;\xi_b)}{Nq_t(I_b)}.
\]
Conditional on pre-draw history \(H_t\), draw \(I_b\) independently with replacement from \(q_t\), then independent original-law training randomness \(\xi_b\). With current gradients and valid differentiation under expectation,
\[
\mathbb E[\widehat g_t\mid H_t]
=\sum_i q_t(i)\frac{\mathbb E_\xi\nabla\ell_i}{Nq_t(i)}
=\nabla F(\theta_t).
\]
This extends the finite-sum identity in [Zhao-Zhang, Sec. 3.1](https://proceedings.mlr.press/v37/zhaoa15.pdf#page=2). Nonuniform fixed target masses require \(a_i/q_t(i)\), not \(1/(Nq_t(i))\).

Require full support and a positive probability floor \(q_t(i)\ge q_{\min}>0\); no value is prescribed. Use the actual draw-time probability, detach it from differentiation, and correct the **entire** row loss. Average over \(B\), not summed weights: self-normalization, clipped weights, or gradient clipping generally destroy the stated unbiasedness. Flooring the proposal and correcting its actual probabilities is different from clipping inverse weights.

Because \(q_t(i)/(Nq_t(i))=1/N\), mistakes receive **no greater expected penalty**. More frequent exposure is canceled by smaller per-occurrence weight. TRAIN labels/costs may inform loss and sampling; neither reference labels nor sampling metadata belong in model-forward inputs.

## Three Verified Primary Full Texts

1. Katharopoulos and Fleuret, **Not All Samples Are Created Equal: Deep Learning with Importance Sampling**, ICML 2018. [Full text, Secs. 3.2-3.3, 4.1; Eq. 20](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf#page=3). Their proxy is the final-preactivation loss-gradient norm times a common network bound, not scalar loss. Bound tightness depends on upstream activations/Jacobians; minimizing a loose bound need not reduce actual variance. Their experiments show loss-based sampling can increase variance. Scoring requires forward computation; their method accounts for that overhead. A CE-only residual is not a certified proxy for this project's complete mixed-loss gradient. Their 1,024-to-128 resampling ablation does not establish gains at update batch 1,024.

2. Zhao and Zhang, **Stochastic Optimization with Importance Sampling for Regularized Loss Minimization**, ICML 2015. [Full text, Sec. 3.1, Eqs. 4-6](https://proceedings.mlr.press/v37/zhaoa15.pdf#page=3). Gradient-norm sampling minimizes estimator variance; Lipschitz/smoothness bounds offer cheaper proxies. Convex convergence guarantees do not establish deep-selector underfitting causes.

3. Alain, Lamb, Sankar, Courville and Bengio, **Variance Reduction in SGD by Distributed Importance Sampling**, arXiv:1511.06481v7, 2016 revision. [Full text, Theorem 1, Secs. 3.3, 4.2-4.3, Appendix B.3](https://arxiv.org/pdf/1511.06481v7). Their efficient exact-norm calculation covers fully connected layers, not arbitrary parameter-sharing networks. Stale probabilities preserve unbiasedness with fresh gradients and matching correction, but can increase variance and produce rare large updates. Their distributed scoring cost is not a free benefit.

## Implications And Recommendation

The following are mathematical implications, not project findings. For IID draws with finite second moments, writing \(g_i=\nabla\ell_i(\theta;\xi)\),
\[
\operatorname{tr}\operatorname{Cov}(\widehat g)
=\frac1B\left[\frac1{N^2}\sum_i\frac{\mathbb E_\xi\|g_i\|^2}{q_i}
-\|\nabla F\|^2\right].
\]
Thus the unconstrained optimum is \(q_i^*\propto\sqrt{\mathbb E_\xi\|g_i\|^2}\), reducing to gradient norm without stochastic masks. Scalar difficulty or winner frequency need not track it. The relevant gradient is the sum of the existing loss-component gradients, including their cancellations.

Refresh trades stale-proxy quality against computation, including dropout/RDrop passes. Excluding stale rows violates the full-support condition. Fresh training masks preserve the original two-pass RDrop law; mask-dependent score reuse needs a joint-selection justification. Replacement duplicates are legitimate but reduce distinct-row coverage: retain multiplicity, with independent masks. Silent deduplication breaks this estimator; unequal-probability without-replacement sampling requires inclusion-probability corrections. Batch-coupled forward behavior also needs separate justification.

Finite-time optimization can change through noise and exposure order despite identical expected gradients. AdamW's second-moment state and nonlinear normalization mean unbiased gradients imply neither the same expected update as uniform sampling nor an identical trajectory.

Batch 1024 weakens an automatic variance-reduction rationale: absolute variance savings scale as \(1/B\), although relative savings need not vanish. Natural reshuffling is not IID across steps. No cited paper proves the supplied 71% fit or 16%-winner/zero-pick case is variance-limited.

**CONDITIONAL ADVANCE, conceptual only:** credible only with a cheap, reliable complete-loss gradient proxy whose benefit survives refresh cost and the large-batch baseline. **Biggest failure mode:** proxy mismatch/staleness purchases expensive, possibly higher-variance updates rather than useful learning. There is no promised zero-selection recovery; genuine winner recall must accompany false-pick and cost/regret evidence, not uniform method choices. No protocol or hyperparameters are selected.
