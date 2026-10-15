Proceed with one bounded, matched TRAIN/VAL trial of the proposed replacement. It is a defensible hypothesis, sufficiently different from frozen-offset fitting, but the evidence does not establish that it will improve full-B. This is **same-family provisional research criticism, not independent acceptance**.

The strongest objections are:

1. **The available signal does not establish beneficial switching.** The calibration bins show association with winning, but do not identify conditional cost advantages over competing actions. Zero/near-zero regret contributions describe losses under current decisions; they are not recoverable improvement budgets. The squared-offset failure neither validates softplus nor disproves instance-dependent cost learning. This proposal changes how existing cost supervision is used; it adds no information.

2. **The sign and centering are sound, with an important qualification.** Write \(w_j=r_j/a_t\). For \(a_t>0\),
   \[
   \frac{\partial L_{\mathrm{mono}}}{\partial z_k}
   =\frac1K\left[w_k\sigma(h_k)-\frac1K\sum_jw_j\sigma(h_j)\right].
   \]
   Thus a zero-regret candidate receives an upward score update under direct score gradient descent whenever the summed term is positive. Centering removes the common-shift escape, and softplus avoids the squared surrogate’s negative-score derivative reversal. However, centering couples candidates: it does **not** make every nonwinner’s score decrease. For unrestricted scores, the term alone favors conditional expected-relative-regret ordering; that does not establish consistency or improvement for the constrained architecture and mixed objective.

3. **The replacement targets neither all three metrics nor exactly unscaled macro regret.** Relative regret already divides by each instance’s oracle cost. “No per-example normalization” should therefore mean “no additional per-example normalization.” Raw mean cost weights errors differently, and native Top1 also distinguishes equal-cost labels. Moreover, \(1/a_t\) changes task tradeoffs within the shared model despite equal task sampling. The VAL gates are necessary because the training objective cannot guarantee their joint satisfaction.

4. **Scale remains a substantive uncertainty.** TRAIN mean normalization controls average cost weight, not weight tails or parameter-gradient balance against CE, compare, and R-Drop. Retaining `.02` is a legitimate preregistered choice, but does not preserve the old block’s influence. Consequently, this evaluates the entire risk-block replacement, including normalization. There is no supplied evidence that the old block saturates or that softplus remedies the limiting failure.

5. **A short continuation can establish only a local result.** Optimizer reset changes the training trajectory; a matched reset control is essential. Twelve checkpoint opportunities on an already-used VAL set introduce selection optimism. One seed cannot establish robustness, and zero-tolerance numerical gates are acceptance rules, not statistical evidence. Beating B without outperforming the matched control would not support a replacement-specific benefit.

Use this single all18 mainline and protocol:

1. **Freeze the intervention.** Start both branches from B ep28/u34776 weights. Keep full18 TRAIN, natural frequencies, equal task sampling, logical batch1024, seed2, architecture, masks, native scoring, greedy inference, CE/compare/R-Drop weights, and compare mixture unchanged. Treatment replaces only the `.02` risk block with the stated centered softplus term. Compute each \(a_t\) once from TRAIN using the declared averaging convention; keep it fixed. Use exactly zero replacement loss when \(a_t=0\). No coefficient scan, task exceptions, or coverage objective.

2. **Use two matched reset branches.** Reset both optimizers identically and use the same batch order and RNG initialization. Preserve the same inherited R-Drop warmup state and logical update position. Freeze the exact shared late learning-rate sequence before execution. Run the original-objective control and replacement for the declared **14,904 successful updates**, with the same twelve inherited epoch endpoints. An unrecoverable run failure ends this trial rather than initiating recovery searches. Do not substitute an unmatched uninterrupted B trajectory.

3. **Predeclare selection and the decision.** Evaluate complete all18 VAL at those endpoints. Independently select each branch’s maximum native macro Top1 among checkpoints satisfying both B-relative cost gates, using unrounded values and zero tolerance; break ties by lower regret, then earlier update. Include B as fallback:
   - raw macro mean cost \(\le 11.39555834354421\);
   - actual macro regret \(\le 0.40132674544775454\%\);
   - promotion requires native macro Top1 \(>70.89430894308944\%\).

   To adopt the replacement, additionally require its selected Top1 to exceed the selected control’s Top1. Report its cost and regret differences against control explicitly; passing B’s gates does not imply Pareto dominance over control. Otherwise retain B and close this route.

4. **Report the complete outcome and lock once.** Report all18 task metrics and changed-decision gains and harms, including cost changes on switches to incorrect labels. Keep zero/near-zero counts descriptive and preserve native-label treatment of exact cost ties. Freeze one selected checkpoint and rule before one complete TEST; LIB uses those same weights and rule. No further tuning follows TEST.

This trial is justified by a coherent surrogate and a distinct trainable intervention. Its value remains an empirical question, with a finite stopping rule.