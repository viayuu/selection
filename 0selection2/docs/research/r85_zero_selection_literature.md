# R85: recovering informative winner regions without frequency forcing

Research finalized and frozen: 2026-10-20. Scope: primary-paper research and improvement ideas only.

**Freeze instruction:** no further paper search before GPU launch. C is the adopted nonwinner-KL mechanism. D remains a project hypothesis based on existing exact cost rankings, not an externally demonstrated improvement. Norm-classifier literature is deferred until subsequent measurements motivate reopening it. This research handoff is closed; no further background review is scheduled.

## Final technical confirmation: EQLv2

**Status: FINAL / FROZEN for the current research request.** The primary mechanism recommendation is **C: nonwinner cost-landscape KL with hard native-winner supervision retained**, not full-profile KL. **A: natural OVR / B: EQL-OVR** remain the user's specified matched gradient-diagnostic pair, with posterior distortion and the absence of demonstrated negative-gradient imbalance explicitly acknowledged. **D: actual candidate-subset training** is a separate project exploration, not a published performance claim. Norm geometry is deferred. These are distinct mechanisms, not instructions to implement or combine everything at once. No broad additional paper search, experiment planning, or project-result verification is required for this research handoff.

**Verified by the same native Astra worker against both author and OpenMMLab implementations:**

The official constructor defaults are **`gamma=12`, `mu=0.8`, `alpha=4.0`**. The pinned author implementation was also re-read during finalization.

\[
f(r)=\sigma(12(r-0.8)),\qquad w^-=f(r),\qquad
w^+=1+4(1-f(r)).
\]

For binary winner target \(y_{ij}\), sigmoid output \(b_{ij}\), and current weight \(w_{ij}=y_{ij}w_j^++(1-y_{ij})w_j^-\), the code accumulates
\[
G_j^+\leftarrow G_j^++\sum_i y_{ij}|b_{ij}-y_{ij}|w_{ij},\qquad
G_j^-\leftarrow G_j^-+\sum_i(1-y_{ij})|b_{ij}-y_{ij}|w_{ij},
\quad r_j\leftarrow\frac{G_j^+}{G_j^-+10^{-10}}.
\]

- These are cumulative **weighted absolute logit-gradient** sums, not EMA, class counts, parameter-gradient norms, or unweighted BCE statistics. The code collects the analytical gradients before batch-loss averaging and the outer loss multiplier.
- Each forward obtains weights from the previous ratio, computes the weighted BCE, then updates detached statistics for the next forward. Collection occurs before backward; it is not a backward hook.
- Both implementations initialize the two accumulators to zero but the ratio separately to **100**, making the first-forward weights neutral. Subsequent ratios use the denominator epsilon above, with no explicit zero/zero-to-one rule.
- Consequently, `zero/zero -> 1` is a project adaptation, not the official initialization. At `r=1`, negative and positive weights are about `0.9168` and `1.3327`, not `(1,1)`. At `r=0`, they are about `0.0000677` and `4.9997`: suppression can be very strong immediately after the neutral initial forward if the observed ratio is small.
- Per-problem/per-method accumulators are also an adaptation. Original detection code has a separate background/objectness channel, excluded with `[:-1]` from these statistics; that convention does not designate any portfolio solver as background. Only legal participating candidates have meaningful positive/negative comparisons in the proposed selection setting.

Primary sources: [Tan et al., EQLv2, CVPR 2021, Section 3](https://arxiv.org/html/2012.08548v2); [author code, commit 907fad905b0cac6372f7ec4035f48ef3d0801098](https://github.com/tztztztztz/eqlv2/blob/907fad905b0cac6372f7ec4035f48ef3d0801098/mmdet/models/losses/eqlv2.py); [OpenMMLab code, commit cfd5d3a985b0249de009b67d04f37263e11cdf3d](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/losses/eqlv2_loss.py). The relevant code locations are initialization, `forward`, `collect_grad`, and `get_weight`. Remote commit IDs were obtained during verification; no local project implementation was inspected.

**Evidence boundary for natural OVR versus EQL-OVR.** The formula-level distinction is established. Natural OVR changes output coupling; EQL additionally changes positive/negative gradient weights. Neither by itself creates independent per-method nonlinear parameters. EQL's detection results do not prove better natural-frequency solver Top1, learnability of native near-tie winners, fewer false selections, or non-worsening cost. Its aggressive negative suppression is a plausible training intervention, not evidence that negative suppression is the diagnosed cause here. The weighted-posterior calculation in Section 4 explains the natural-distribution risk. No implementation, experiment, or result approval is implied by this source verification.

**Latest supplied context, not independently verified:** R80 CE-only still had zero choices after jointly removing the R77 comparison/risk/KL terms. The user reports small effects from batch/lr/source-code variants. R84 zero/near-zero groups had 4625 VAL winners across 36 groups, with 11 original correct selections versus 32/21/56 in the supplied B/C/D variants, and additional false picks offsetting recovery. The latest supplied TRAIN facts are TSP_LEHD 14279 winners, 15 selections, 7 correct; DIFUSCO 6806 winners and zero selections; ATSP MATPOENET 6476 winners and zero selections; TRAIN macro 71.24 versus VAL 71.09. Focal gives LEHD 34 correct out of 14279 and MATPOENET still zero. These facts motivate a training-discrimination focus rather than assuming ordinary train/validation overfitting; they do not isolate the causal mechanism. No result files were read or metrics recomputed.

**Disposition of the alternatives.** The user has now explicitly adopted the nonwinner-only formulation for C; this is not a silent replacement of the earlier full-profile \(q_m\propto\exp(-r_m/0.01)\) proposal. Full-profile KL alongside hard CE remains a mathematical comparison in Section 3, not the selected C mechanism. D retains the original full-pool objective and adds legal-subset CE with the **actual subset entering candidate attention**, not posthoc masking. Its auxiliary winners are derived only from existing costs. The canonical costs and full-pool labels remain unchanged; no solver runs, new cost measurements, or geometric label augmentation are proposed. D is specified in Section 5 as a project hypothesis without a foreign-paper performance endorsement.

## Final research synthesis

**Recommendation:** retain C as the principal literature-derived improvement idea, A/B as the explicitly acknowledged gradient diagnostic, and D as a distinct exploratory change of comparison contexts. Norm geometry and structured pretraining remain background directions, not additional current recommendations. Cost non-worsening is a desired constraint; none of the cited papers guarantees it here. Solver-selection gains and learnability are not established by this review.

**Provenance and limits.** The requested research-lit skill was read. All project descriptions below are supplied assumptions. No project source, results, logs, datasets, or prior reports were inspected. The requested report did not exist when initially checked. Sources are primary papers and author/project repositories; no local PDFs were read. No experiments, implementation, protocol changes, solver changes, canonical relabeling, or wiki edits were performed. The parent launched and resumed the same native Codex worker with `gpt-6-astra`, `model_reasoning_effort="max"`; both CLI runtime headers explicitly reported `model: gpt-6-astra`, `provider: OpenAI`, and `reasoning effort: max`, session `01a154ca-4ba3-7292-9dba-e1c6720c01d3`. This is launcher-observed runtime provenance, not model self-attestation. No alternate research model was substituted. The parent transcribed the worker's verified findings and finalized the document's scope according to the user's subsequent decisions; no new broad literature review or review loop was launched for finalization.

## 1. Decision and assumed setting

The early preferred candidate was a cost-derived listwise auxiliary objective with hard-winner supervision retained. The final recommendation C keeps the refinement supported by the distillation decomposition: learn the conditional distribution among nonwinners separately from the winner probability. The A/B diagnostic and D exploration have different evidentiary roles and are not a performance ranking.

| Role | Mechanism | Distinct information or geometry | Main limitation |
|---|---|---|---|
| **C: recommended mechanism** | **Hard native-winner CE plus cost-derived nonwinner KL** | Learn finite relative performance differences among competitors, with representation gradients and no direct auxiliary winner-logit gradient | Omits the common winner-to-rest gap; can reinforce an always-second method or conflict through shared parameters |
| **A/B: specified diagnostic** | **Natural OVR versus EQL-weighted OVR** | Both use sigmoid outputs; B additionally intervenes on cumulative positive/negative gradient weights | No established harmful gradient imbalance; B changes the statistical target and has no natural-Top1 or cost guarantee |
| **D: project exploration** | **Original full-pool loss plus actual-subset auxiliary training** | Change the candidate-attention context using existing costs | Adds no new cost/order information; subset success need not improve full-pool decisions; no external performance claim |

These are three distinct mechanism families. A/B's matched status is supplied by the user, not verified against any implementation. Independent per-method nonlinear parameters are a separate architectural choice, not an automatic consequence of either OVR or EQL. Norm geometry is deferred rather than assigned another current slot.

Assumed setting: 18 tasks; natural winner imbalance; new data with 81k TRAIN and 4500 VAL per problem, plus TEST; the supplied candidate pool excluding single ReLD and B. The encoder has four width-128 layers, solver ID and 34 solver features, two bidirectional cross-attention layers, graph-query attention, and MLP utilities followed by `score_difference` and maximin logits. The supplied loss is 0.35 CE + 0.35 within-instance pairwise logistic over unequal-cost pairs with Top3 emphasis + 0.02 expected(relative gap / 0.01), plus ramped RDrop. LEHD's roughly 16% wins, roughly 15% mean probability and zero argmax selections, DIFUSCO's roughly 8% wins and zero selections, and TRAIN zero-picks are assumptions, not independently verified findings.

The supplied failures remain informative: fixed posthoc bias/calibration, frozen cost head, ensemble, supervised contrastive learning, code embeddings, global AUC / per-method conditional-regret ranking, and R84's added focal term plus direct QK without MLP. None is recommended again under another name.

## 2. What zero selections establish, and what they do not

Let \(J_t(x)\) be the legal candidate pool for task \(t\), \(y\) its existing winner label, and
\[
\eta_j(x)=\Pr(y=j\mid x,t,J_t(x)).
\]
Natural Top1 is maximized by \(\arg\max_j\eta_j(x)\). Positive marginal prevalence does **not** imply a nonempty decision region.

**Counterexample, derived here.** Suppose the available representation carries no winner information and the conditional probabilities are always
\[
(0.45,0.31,0.16,0.08).
\]
An exactly calibrated Bayes classifier selects only the first method. The third and fourth methods have 16% and 8% true wins and perfectly matching mean probabilities, yet zero optimal argmax selections. Forcing either into the predictions reduces natural accuracy. This is a statistical counterexample, not a claim that the project's instances have no useful information. With deterministic costs, an insufficient representation can create the same effective ambiguity.

Consequently, matching \(\mathbb E[p_j]\) to prevalence is only a marginal statement. Recovering correct selections requires instance-dependent evidence that changes which candidate leads. TRAIN zero-picks make fitting limitations plausible, but do not prove an implementation bug or identify a particular architectural cause.

Natural accuracy weights class recalls by their natural prevalences; balanced accuracy weights class recalls equally. They differ even within a task. Equal weighting across the 18 problems does not imply equal weighting across solvers inside each problem. Likewise,
\[
\arg\max_j\Pr(y=j\mid x)\quad\text{and}\quad
\arg\min_j\mathbb E[c_j\mid x]
\]
need not coincide. More correct winners can coexist with worse mean cost if the remaining mistakes become sufficiently expensive. All mechanisms below must be understood under that unresolved tradeoff, without pooling unrelated tasks' raw cost units.

## 3. C: hard native winner plus nonwinner cost-landscape KL

### Evidence and transfer

Cao, Qin, Liu, Tsai and Li's **ListNet** (ICML 2007) uses a probability distribution over a query's items and a listwise cross-entropy, with the exponential top-one model in Sections 4-5. This supplies the relevant within-instance normalization, not evidence about solver selection. [Primary paper, Sections 4-5](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf).

Zhao et al.'s **Decoupled Knowledge Distillation** (CVPR 2022) separates target-vs-rest knowledge from the conditional distribution among nontarget classes. Section 3.1 decomposes ordinary KL into a binary term and a conditional term weighted by the teacher's nontarget mass. Its image-classification evidence supports studying the latter independently; it does not establish that measured solver costs are an equivalent teacher. [Paper, Section 3](https://arxiv.org/html/2203.08679v2); [official implementation](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py).

### Mathematical recipe

For one instance, let \(c_* = \min_{j\in J}c_j\). Use the project's existing dimensionless relative gaps \(r_j\); for positive costs the familiar expression is \((c_j-c_*)/c_*\). The following does not redefine zero/negative-cost conventions or the existing winner/tie labels. Define a cost-derived distribution
\[
q_j=\frac{\exp(-r_j/\tau)}{\sum_{k\in J}\exp(-r_k/\tau)},\qquad \tau>0.
\]
Here \(\tau\) specifies cost resolution, not class-frequency balancing. There is no target marginal frequency for any method.

For model logits \(z\), define \(p=\operatorname{softmax}(z)\). Conditional on the supplied winner \(y\), form
\[
q_j^- = \frac{\exp(-r_j/\tau)}{\sum_{k\in J\setminus\{y\}}\exp(-r_k/\tau)},\quad
p_j^- = \frac{\exp(z_j)}{\sum_{k\in J\setminus\{y\}}\exp(z_k)},\quad j\ne y.
\]
The preferred new term is
\[
L_{\mathrm{land}}=D_{\mathrm{KL}}(q^-\Vert p^-),
\qquad L_{\mathrm{winner+land}}=-\log p_y+\lambda L_{\mathrm{land}}.
\]
The existing hard label remains the winner target. The cost vector supplies auxiliary supervision; it is not used as an inference input. The intended learning is end-to-end into the representation, without a frozen cost predictor or additional solver. The displayed expression isolates the mechanism, rather than prescribing a complete replacement for all existing loss terms. For fewer than three legal methods, the nonwinner term is empty or identically zero.

**Why this is more specific than generic soft labels.** Ordinary full-list KL decomposes exactly as
\[
D_{\mathrm{KL}}(q\Vert p)
=D_{\mathrm{KL}}\big((q_y,1-q_y)\Vert(p_y,1-p_y)\big)
 +(1-q_y)D_{\mathrm{KL}}(q^-\Vert p^-).
\]
The first part asks the classifier to reproduce a cost-derived winner probability, which is not generally the probability of being the true winner. The proposal keeps hard-winner CE for that role and transfers the conditional landscape separately. At the logit level,
\[
\frac{\partial L_{\mathrm{land}}}{\partial z_y}=0,\qquad
\frac{\partial L_{\mathrm{land}}}{\partial z_j}=p_j^- - q_j^-\ (j\ne y).
\]
This removes a direct auxiliary gradient on the winning logit. It does **not** isolate shared-parameter gradients or guarantee that the winner's margin improves.

### Difference from existing supervision

Pairwise sign logistic supervision has the form
\[
\ell_{jk}=\log(1+\exp(-(z_j-z_k)))\quad\text{when }c_j<c_k.
\]
It encodes ordering, not a finite target cost ratio. Cost landscapes \((0,\epsilon,M)\) and \((0,M,M+\epsilon)\) have the same winner and ordered-pair signs, but qualitatively different near-best competitors. Top3 emphasis does not recover the missing magnitudes. The new conditional target implies
\[
\log(q_j^-/q_k^-) = -(r_j-r_k)/\tau,
\]
so matching it asks for finite, cost-dependent relative logit differences.

The expected-gap term already contains cost magnitudes and must not be described as cost-blind. Its role differs:
\[
L_{\mathrm{gap}}=\sum_jp_jr_j,\qquad
\frac{\partial L_{\mathrm{gap}}}{\partial z_j}
=p_j\left(r_j-\sum_kp_kr_k\right).
\]
It encourages low expected cost without specifying the whole competitor distribution. A nearly suppressed candidate receives little gradient through its factor \(p_j\); the listwise derivative retains \(-q_j^-\) when the candidate is genuinely competitive. This is a difference in gradient and target structure, not new access to cost data. In particular, a supplied mean probability near 15% is not evidence of probability saturation.

### Limits and novelty judgment

This is a **partial** transfer of the full cost landscape: \(q^-\) distinguishes nonwinner cost differences but is invariant to adding the same gap to every nonwinner. It therefore loses the common winner-to-rest separation. The existing expected-gap term contains that information; this memo does not claim that CE plus \(L_{\mathrm{land}}\) alone reconstructs every gap.

The earlier full-profile proposal was **hard CE plus full-list KL**, not replacement of CE. For a fully identified individual instance, its unconstrained optimum is \(p^*=(e_y+\lambda q)/(1+\lambda)\). If the native label \(y\) is also the unique minimum under the costs used to construct \(q\), that target retains its winner. Native near-tie conventions and cost precision cannot silently be assumed identical. Nor does individual-instance ordering settle what a restricted or ambiguous representation learns.

For two methods at an ambiguous representation, let A win 51% of cases by relative gap \(\epsilon\), while B wins 49% by gap \(M\). As \(\epsilon/\tau\to0\) and \(M/\tau\to\infty\), the mean soft target for A approaches \(0.51/2=0.255\), although A is the natural-accuracy Bayes winner. With hard CE retained, the unrestricted conditional optimum is \((\eta+\lambda\mathbb E[q\mid x])/(1+\lambda)\); its argmax can change depending on the auxiliary weight. This is not a claim that full KL necessarily harms accuracy. It explains the preference for C's conditional nonwinner transfer while retaining the shared-parameter and learnability caveats.

Recovery remains conditional on learnable structure: if a zero-picked method is often near-best in identifiable regions, conditional landscape supervision could organize those regions so its actual wins become distinguishable. If it is irreducibly second-best everywhere in conditional probability, this may only produce a better runner-up. With ambiguous labels/costs or restricted features, even CE plus the nonwinner term need not retain CE's population decision rule. Excessive emphasis on loser differences can harm Top1; close-cost distinctions can reflect noise. The remaining pairwise loss may also prefer ever-larger margins where this term prefers finite ones.

Novelty against the supplied routes is the **within-instance, cost-magnitude target with representation gradients**, not a global AUC objective, a per-method ranking across different instances, focal emphasis, a frozen cost head, an ensemble, or generic contrastive separation. It is an adaptation of established listwise/KD ideas, not a claim of literature novelty. C remains the principal recommendation because it adds a specific supervisory structure absent from pairwise signs while making the accuracy conflict explicit. Existing cost-sensitive terms already contain weak cost signals, and the supplied CE-only failure prevents attributing the entire phenomenon to those terms.

## 4. A/B: natural OVR versus EQL-OVR gradient diagnostic

The user specifies A/B as a controlled diagnostic, not a claim that EQL restores a balanced or natural-consistent target. This memo confirms the official mechanism and its limits; it does not verify matching, set an experiment protocol, or approve an implementation. Negative-gradient imbalance has not been established by the supplied class frequencies or TRAIN zero-choice counts.

Tan et al.'s **EQL v2** (CVPR 2021), Section 3.1, treats classification as sigmoid subtasks and Section 3.2 changes their positive/negative weights. The independent-output factorization is useful here; its reweighting is a separate decision. Its empirical target is detection AP, not natural solver Top1 or cost. [Paper](https://arxiv.org/html/2012.08548v2); [author repository](https://github.com/tztztztztz/eqlv2); [project loss implementation](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/eqlv2_loss.py).

Use unconstrained, instance-conditioned binary logits \(a_j\), with \(b_j=\sigma(a_j)\), and the existing one-hot winner label:
\[
L_{\mathrm{OVR}}
=\sum_{j\in J}\left[-\mathbf1[y=j]\log b_j
 -(1-\mathbf1[y=j])\log(1-b_j)\right].
\]
Positive and negative examples receive their natural weights. Inference is \(\arg\max_j a_j\); a threshold of 0.5 is inappropriate for choosing exactly one method because every method's winner probability may be below 0.5. This is one jointly learned selector, not an ensemble.

**Standalone statistical target, derived here.** Conditional risk for method \(j\) is
\[
R_j(a)=\eta_j\log(1+e^{-a})+(1-\eta_j)\log(1+e^a),
\quad R_j'(a)=\sigma(a)-\eta_j.
\]
Thus \(b_j^*=\eta_j\) and \(a_j^*=\log(\eta_j/(1-\eta_j))\). The argmax is the natural-accuracy Bayes decision, with no uniform-frequency objective. This is a population statement for the standalone proper loss with sufficient expressivity, not a finite-network guarantee or a theorem about the current composite loss.

**Optimization distinction.** Softmax CE has logit Hessian \(\operatorname{diag}(p)-pp^T\). Independent BCE has diagonal logit Hessian \(\operatorname{diag}(b_j(1-b_j))\). Removing cross-logit curvature changes output competition, but the encoder remains shared. It cannot eliminate shared-feature interference, natural negative examples, or a genuine Bayes class with no winning region.

**Output factorization is not parameter independence.** A shared scorer \(a_j=f_\theta(h(x),e_j)\) and a per-method bank \(a_j=f_{\theta_j}(h(x),e_j)\) can both use either of the above output losses. The latter changes parameter sharing and finite model capacity; sigmoid BCE alone does not perform that change. Its shared encoder can still receive conflicting gradients. A bank is not the supplied R53 two-method encoder, but no solver-selection gain for that bank has been established by the cited detection papers. It must not be silently attributed to EQL or treated as part of a loss-only interpretation of A/B.

Absolute logit semantics matter. BCE is not invariant to a common logit shift, whereas softmax is. Do not apply this recipe blindly to scores whose common offset has been fixed by a maximin transform. Likewise, naively combining BCE on \(a\) with ordinary softmax CE on the same \(a\) asks for different optimal scales: log-odds versus log-probabilities. The recipe is a different classification parameterization, not an extra arbitrary penalty on existing logits. This mathematical qualification is not a diagnosis of the uninspected implementation.

**Why this differs from the failures.** Focal loss changes example weighting; OVR changes the probability factorization and logit coupling. Global/per-method AUC compares examples for a class and does not identify comparable posterior levels across classes; proper unweighted OVR does in the unrestricted population setting. A/B introduces no new supervision. R84's failure does not directly settle this mechanism, but neither does it imply that CE's normalization was the cause. B's additional selective positive/negative reweighting is a genuine gradient intervention, not evidence that its effect will be beneficial.

### B changes weighting, not the evidence boundary

EQL v2 uses accumulated positive/negative gradient information to reduce negative weights and increase positive weights. Seesaw uses the cumulative count ratio \((N_j/N_y)^p\) to mitigate certain negative-class penalties and a prediction-ratio compensation factor to penalize false positives. Its official code implements both components. [Wang et al., CVPR 2021, Section 3.1](https://arxiv.org/html/2008.10032v3); [official project implementation](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/seesaw_loss.py).

**Target-distribution objection.** With fixed positive/negative weights \(A,B>0\), the weighted binary optimum is
\[
b^*=\frac{A\eta}{A\eta+B(1-\eta)},
\qquad \operatorname{logit}(b^*)=\operatorname{logit}(\eta)+\log(A/B).
\]
Class-dependent weight ratios can change cross-class ordering. This fixed-weight calculation is not an exact solution of adaptive EQL/Seesaw; it demonstrates why their natural-posterior consistency cannot be assumed. Compensation against false positives is useful, but is not a proof of cost non-worsening.

Nor does prevalence alone prove harmful gradient imbalance. For calibrated constant BCE output \(b=\pi\), expected positive and negative gradient magnitudes are both \(\pi(1-\pi)\), despite many more negative examples when \(\pi\) is small. Aggregate equality can coexist with poor discrimination. Accordingly, no equalization-to-one target, inverse-frequency loss, class-uniform sampler, or selection quota is recommended from the supplied frequencies alone.

## 5. D: actual candidate-subset contexts, project exploration

This mechanism is retained as the user's project exploration. No foreign-paper benchmark result or generalization theorem is claimed for it. Let \(J\) be the original legal pool, \(S\subseteq J\) a sampled legal subset, and \(y_S\) the subset winner derived from already available costs and their applicable ordering/tie conventions. Define
\[
L_D(x)=L_{\mathrm{original}}(x,J)
+\lambda\,\mathbb E_S\left[-\log
\frac{\exp z_{y_S}(x,S)}{\sum_{j\in S}\exp z_j(x,S)}\right].
\]
The original full-pool objective remains intact. The notation \(z_j(x,S)\) requires feeding the **actual subset** through candidate attention and recomputing its representations; applying a mask only to full-pool logits is not this proposal. The expression describes the requested mechanism, not a sampling schedule or numerical loss-weight prescription.

For strictly ordered costs, all subset winners are already determined by the full pool's complete pairwise ordering. D therefore supplies no new measured performance or ordering information beyond existing costs/all-pairs supervision. Its proposed distinction is the context and optimization problem seen by the network: methods become winners of auxiliary comparison sets, and candidate attention sees different competing sets. Neither that extra training role nor success on smaller pools guarantees a nonempty correct decision region on the original pool.

The measured cost of a fixed solver on a fixed instance does not change merely because other candidates are hidden. Unhelpful context dependence and a mismatch between subset and full-pool behavior remain risks. Keeping the original objective is not a mathematical no-regression guarantee. Native full-pool labels are retained, unresolved ties are not invented away, and no new solver run or cost measurement is required. D is separate from C and A/B; this memo does not recommend combining all branches into one model.

## 6. Deferred: normalized final discriminative geometry

Kang et al.'s **Decoupling Representation and Classifier for Long-Tailed Recognition** (ICLR 2020) links classifier weight norms to class frequency and studies cRT, tau-normalization, and cosine classifiers. Its Section 5.1 describes balanced evaluation distributions, a crucial mismatch with the present natural-frequency target. The relevant evidence is the geometry, not an instruction to copy class-balanced cRT or posthoc tau-normalization. [Paper, Sections 4-5 and Appendix B.7](https://arxiv.org/html/1910.09217v2); [official code](https://github.com/facebookresearch/classifier-balancing).

Seesaw also studies normalized linear activation independently of its reweighting: Table 2 reports LVIS AP 23.7 to 24.7 for that component alone. This supports a trainable normalized classifier as a distinct component, without transferring the AP gain to solver accuracy. [Seesaw, Sections 3.2 and 4.3](https://arxiv.org/html/2008.10032v3).

**Recipe.** Retain a nonlinear, solver-conditioned interaction vector \(v_j(x)\) and solver vector \(w_j\) derived from ID/features, then score
\[
z_j(x)=s\left\langle\frac{v_j(x)}{\|v_j(x)\|_2},
                         \frac{w_j}{\|w_j\|_2}\right\rangle,\quad s>0.
\]
Zero-norm handling is necessary. The shared scale and vector directions are learned with natural winner supervision; this is normalization during learning at the final discriminative readout. Normalizing a scalar utility would merely retain its sign and is not this proposal. No equal-angle prototype target or prescribed class selection frequency is implied.

**Geometric explanation, conditional on a simpler classifier.** For a bias-free linear classifier on a common feature \(h\), if \(w_j\) lies in the convex hull of the other classifier vectors, then
\[
w_j^Th\le\max_{k\ne j}w_k^Th\quad\text{for every }h.
\]
It has no strict winning region. Distinct unit vectors avoid this particular obstruction: at \(h=w_j\), the own-class inner product is 1 and all others are smaller. However, the encoder may never reach that region, directions may coincide, and solver-conditioned features do not obey the common-feature proof. This demonstrates a possible geometric failure mode; it does not establish one in the supplied architecture. Natural priors can be represented by how often inputs occupy regions, so nonempty abstract regions do not impose balanced frequencies.

Fang, He, Long and Su's **Layer-Peeled Model** (PNAS 2021) adds evidence that constrained feature/classifier optimization can collapse minority directions. Theorem 5 proves an asymptotic result as imbalance tends to infinity; finite-threshold collapse is numerically supported, not a universal finite-imbalance theorem. Its two-class-size-group and norm-constraint setting cannot diagnose LEHD or DIFUSCO from 16%/8% prevalence. [Primary article, Minority Collapse and Theorem 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC8639364/); [author code](https://github.com/HornHehhf/LPM).

**Overlap with R84 must remain explicit.** Direct QK and a cosine classifier both use bilinear similarity. Cosine differs only if normalization, the final scoring location, shared scale, or retained nonlinear interaction actually differs. If R84 already normalized the final Q/K vectors with equivalent scoring, this candidate loses that claimed distinction. If R84 removed an MLP while using unnormalized attention scores, it does not isolate final angular geometry. The supplied description does not resolve this. Geometry is therefore deferred for reconsideration only if later evidence motivates it, not claimed as entirely new or added to the current mechanisms.

Norms can carry legitimate confidence or solver-specific performance information; removing them can reduce capacity. This route also cannot repair indistinguishable input features or guarantee cost. It is neither fixed posthoc calibration nor code embedding, and it does not reuse the supplied supervised-contrastive objective.

**No inferred maximin bottleneck.** If, and only if, pairwise scores are exact potential differences \(d_{jk}=u_j-u_k\) and the minimum includes all legal opponents including self, then
\[
\min_k d_{jk}=u_j-\max_k u_k.
\]
Both softmax and argmax are unchanged by this common shift. Excluding self still preserves the argmax for exact potential differences, although the softmax values need not be identical. Learned pair-specific comparisons, masks, or other transformations can invalidate the identities. Source inspection was prohibited, so none is asserted to describe the current implementation.

## 7. Data geometry and pretraining: considered, not promoted

**Optimization symmetry is insufficient.** Sym-NCO proves rotational problem symmetry under its Euclidean reward assumptions and trains neural solvers with symmetry-aware objectives. It does not prove that each fixed heuristic's realized cost is invariant. [Kim, Park and Park, NeurIPS 2022, Section 2.2 and Theorem 2.1](https://papers.nips.cc/paper_files/paper/2022/file/0cddb777d3441326544e21b67f41bdc8-Paper-Conference.pdf); [official code](https://github.com/alstn12088/Sym-NCO).

Let a solver return \(S_j(x;\omega)\), with realized cost \(c_j(x;\omega)=f_x(S_j(x;\omega))\). A valid problem transformation may satisfy
\[
f_{gx}(g\pi)=a_g f_x(\pi)+b_g,\qquad a_g>0,
\]
for every feasible solution. That preserves optimal solution ordering. To reuse the original empirical solver costs as targets, one also needs a relationship such as
\[
S_j(gx;\omega')=gS_j(x;\omega)
\]
under the actual solver's randomness and operating conditions, or an appropriate distributional version. The first equation does not imply the second.

As a concrete counterexample, a heuristic that visits nodes in input order can change tour length after node permutation, although the TSP optimum is unchanged. A geometry-aware heuristic could retain its tour length. Their empirical winner can therefore change. Rotation can alter a coordinate-sensitive neural policy; scaling can alter rounding, stopping tolerances, or resource constraints. With fixed time limits, equal mathematical problems need not produce equal realized heuristic performance. An affine cost transformation preserves ordering but preserves the usual relative gaps only in special cases, such as common positive scaling with no additive shift.

EGNN provides an architecture equivariant to translations, rotations, reflections, and permutations, not arbitrary scaling or invariance of heuristic runtime/quality. A mandatory invariant selector could discard precisely the coordinate/order sensitivity needed to predict actual solvers. [Satorras, Hoogeboom and Welling, ICML 2021](https://proceedings.mlr.press/v139/satorras21a.html); [official code](https://github.com/vgsatorras/egnn). No task-independent geometric augmentation is justified across all 18 unspecified tasks from these results.

GraphMAE offers genuinely noncontrastive masked feature reconstruction, rather than the supplied failed supervised-contrastive route. But feature reconstruction is not evidence of better prediction of small solver-performance differences. With the supplied substantial labeled data, this bounded review found no task-specific basis to rank structured pretraining above direct performance-landscape supervision. GraphMAE is therefore not a fourth recommendation. [Hou et al., KDD 2022](https://arxiv.org/abs/2205.10803); [official code](https://github.com/THUDM/GraphMAE).

## 8. Primary-source evidence map

All entries below are published papers, not unreviewed preprints; arXiv links provide author manuscripts. Repository status and source content were checked during this bounded review. Code links are provenance references, not claims that the proposed solver-selection adaptations already exist.

| Paper and venue | Verified content relevant here | Author/project code and transfer limit |
|---|---|---|
| Cao, Qin, Liu, Tsai, Li, **Learning to Rank: From Pairwise Approach to Listwise Approach**, ICML 2007 | Sections 4-5 define top-one probabilities and listwise cross-entropy | [Author institution paper page](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/). No original author implementation verified in this bounded search; no third-party repository presented as official |
| Zhao et al., **Decoupled Knowledge Distillation**, CVPR 2022 | Target/rest KL decomposition and separately weighted non-target knowledge | [Official DKD code](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py). Neural-teacher evidence; cost-teacher transfer is the memo's hypothesis |
| Tan et al., **Equalization Loss v2**, CVPR 2021 | Independent sigmoid tasks, gradient-dependent positive/negative weights | [Author repository](https://github.com/tztztztztz/eqlv2). Detection AP gains do not establish natural Top1 consistency |
| Wang et al., **Seesaw Loss**, CVPR 2021 | Count-based mitigation, false-positive compensation, normalized activation | [Paper-linked project code](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/seesaw_loss.py). Natural posterior and cost guarantees are absent |
| Kang et al., **Decoupling Representation and Classifier**, ICLR 2020 | Norm/frequency relationship, cRT, tau-normalization, cosine comparison; balanced evaluation | [Official repository](https://github.com/facebookresearch/classifier-balancing). Geometry evidence, not an endorsement of paused calibration or class balancing |
| Fang, He, Long, Su, **Exploring Deep Neural Networks via Layer-Peeled Model: Minority Collapse in Imbalanced Training**, PNAS 2021 | Constrained-model collapse; asymptotic theorem and numerical finite-threshold findings | [Author repository](https://github.com/HornHehhf/LPM). Does not diagnose this selector |
| Kim, Park, Park, **Sym-NCO**, NeurIPS 2022 | Problem/solution symmetries and symmetry-aware solver learning | [Official repository](https://github.com/alstn12088/Sym-NCO). Solver training evidence, not fixed-portfolio label invariance |
| Satorras, Hoogeboom, Welling, **E(n) Equivariant Graph Neural Networks**, ICML 2021 | Euclidean and permutation equivariance | [Official repository](https://github.com/vgsatorras/egnn). No general scale or solver-performance invariance |
| Hou et al., **GraphMAE**, KDD 2022 | Masked feature reconstruction with scaled cosine error | [Official repository](https://github.com/THUDM/GraphMAE). No verified advantage for this solver-selection setting |

The justified research direction is to improve conditional discrimination and the information carried by performance supervision. Nonzero selection counts, balanced class recalls, and matching marginal frequencies are not substitutes for higher natural Top1 with acceptable cost.
