**R78-A is ready as an exploratory mechanism screen**, subject to the already-required TRAIN-only convergence declaration and the interpretation limits below. It cannot establish the cause of zero selection or confirm a generalizable performance improvement.

`review_independence: same-family`  
`acceptance_status: provisional`

1. **High: success and failure have narrower meanings than the result-to-claim table suggests.**  
   In [RESEARCH.md §8](/public/home/shiys/0selection2/runs/R78_zero_selection_goal/RESEARCH.md:331), “correctable low-capacity decision bias” should be explicitly operational. Even constant, uninformative scores can pass the gate by switching from an inferior constant policy to a better one. Conversely, failure of the converged, regularized CE optimum does not establish that every offset would fail the Top1/cost gate: those are different objectives. Neither outcome identifies label imbalance, harmful risk training, missing information in the original features, or the Bayes-optimal policy. In-sample TRAIN scores further confound score correction with backbone generalization.

   Proposed replacement: “A passing candidate demonstrates a beneficial fixed-offset decision change on the assessed VAL. It does not identify the cause of the original decisions or establish recovered instance discrimination. A rejected candidate rejects this prescribed TRAIN-CE fit under this gate, not the existence of useful offsets or learnable information. Claims of instance discrimination require comparison with the already-specified TRAIN-selected constant and conditional references.”

2. **High: the macro gate supports exploratory acceptance, not taskwise safety or independent confirmation.**  
   The [gate](/public/home/shiys/0selection2/runs/R78_zero_selection_goal/RESEARCH.md:131) is coherent and already acknowledges these limitations. Nevertheless, substantial losses on one task can be offset elsewhere, and zero empirical cost tolerance is not statistical non-inferiority. One new comparison does not erase any earlier VAL use for backbone selection or research design. Paired intervals also do not repair adaptive selection. The prior TEST disclosure is handled appropriately; freezing another evaluation cannot restore freshness.

   Proposed addition: “Promotion means passing the empirical fixed-18-task macro gate. Report every task’s paired changes, including adverse effects, and paired sampling uncertainty with its resampling unit specified beforehand. Record prior VAL use; uncertainty intervals do not account for the full adaptive development history. Preserve the exact inherited cost-vs-TRAIN-SBS definition and aggregation. Future VAL-guided revisions remain exploratory.”

3. **Medium: R78-B is defensible as a later controlled diagnostic, but its trigger is too permissive.**  
   The [current trigger](/public/home/shiys/0selection2/runs/R78_zero_selection_goal/RESEARCH.md:214) includes CE correction improving winner prediction while causing severe cost losses. That can reflect a legitimate accuracy-versus-cost conflict, without implicating harmful risk-term interference. The matched initialization, updates, schedule, and retained losses make deletion interpretable, but only as the effect of that deletion under the tested training and checkpoint-selection procedure. One seed cannot establish a general causal explanation for zero picks.

   Proposed replacement: “A Top1/cost trade-off alone does not trigger R78-B. Keep B deferred pending a separately stated, TRAIN/VAL-supported hypothesis of detrimental objective interaction. If subsequently selected, B estimates the effect of deleting this risk term under the matched protocol; it does not isolate all cost sensitivity or establish why historical zero selections arose.” Also specify retaining the frozen reference if neither arm has an eligible checkpoint.

4. **Low: the invariance proof is correct; the inference about correction location needs qualification.**  
   For a unique utility winner \(k\), \(z_k=u_k-u_{(2)}>0\), whereas \(z_i=u_i-u_k<0\). Positive candidate-specific scaling therefore preserves the winner; tied maxima remain tied at zero under a consistent tie rule. This rules out the final pure-difference maximin transformation changing the greedy winner for fixed utilities. It does not rule out effects of training through that transformation. Nor does it establish post-maximin placement as uniquely necessary.

   Replace the concluding sentence in [§3](/public/home/shiys/0selection2/runs/R78_zero_selection_goal/RESEARCH.md:81) with: “Within positive diagonal affine transformations of these unshifted maximin scores, changing the winner requires an additive component. Post-maximin placement is the chosen intervention, not a consequence of the invariance proof.” Pairwise margins receive constant shifts; multiclass margins need not, because the strongest competitor can change.

5. **Low: the fitting objective is sufficiently specified for research review, with a meaningful convergence prerequisite.**  
   The [penalty](/public/home/shiys/0selection2/runs/R78_zero_selection_goal/RESEARCH.md:109) has the stated summed-CE interpretation and ensures a unique finite optimum. At convergence, the constrained stationarity condition gives
   \[
   \overline p_j-\widehat\pi_j=-b_j/N.
   \]
   Thus even mean probability matching is approximate under shrinkage, and it never implies matching selection frequencies. The low-dimensional convex fit is a feasible screen; batch1024 alone does not establish convergence.

   Proposed addition: “The unit Gaussian is a fixed prior in the frozen score units, not an empirically validated optimal penalty. Declare the numerical convergence criterion and budget before fitting; budget exhaustion without convergence makes the fit inconclusive.” The existing delegation of those numerical details is adequate; an optimizer audit is unnecessary here.

6. **Low: the important primary-paper qualifications are substantially correct.**  
   [Guo](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) and [Kull](https://proceedings.neurips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf) support calibration families, but do not validate this reuse of backbone TRAIN scores or this fixed prior. [Menon](https://arxiv.org/pdf/2007.07314) supports the stated adjustment signs and balanced-error interpretation; arbitrary maximin margins are not automatically posterior logits. [Ren](https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) explicitly assumes the relevant distribution mismatch. [Hepburn](https://proceedings.mlr.press/v94/hepburn18a/hepburn18a.pdf) assumes example-dependent cost matrices are supplied at decision time.

   Suggested addition: “These references motivate modeling choices; their guarantees do not transfer to the R78 mixed objective or macro-cost acceptance gate.” Menon was checked through the listed arXiv paper because the author-hosted PDF was unavailable.

**Recommended next action:** have the execution owner finalize the R78-A declaration with these interpretation corrections and supply the single natural-frequency, logical-batch1024 TRAIN fit versus identity on all 18 VAL tasks. Keep B deferred. Continue the research review on that evidence; this document does not complete R78. Any formal delivery must preserve the same locked weights, calibration, candidate definitions, and decision rule across TEST and LIB, with prior TEST exposure disclosed and no LIB-label fitting.

Only the request, its three permitted local documents, and the cited primary papers were reviewed. No files were edited or experiments executed.