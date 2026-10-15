Round001 remains **numerically inconclusive, with identity retained**. VAL Top1 improved by 0.142633 percentage points, but actual regret increased by 0.002676 points and raw macro cost increased by 0.000101003. TRAIN shows the same aggregate trade-off. Only one task passed the declared convergence rule, so the exact CE optimum remains unresolved. The regret interval crossing zero does not establish non-inferiority or override the empirical gate. Reused VAL also prevents independent confirmation. [Evidence]( /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round001/RESEARCH_HANDOFF.md)

The changed decisions justify investigating cost magnitudes. ATSP corrected 160 VAL decisions and broke 125, yet increased regret; OVRPL increased regret on both splits. These aggregates do not establish that a few extreme losses dominate or implicate the backbone’s `.02` risk term. Unchanged zero-pick counts conceal membership changes. LEHD’s marginal probability agreement and mean margins establish neither conditional calibration nor score separability. [Decision changes](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round001/changed_decisions_val.csv)

Native Top1 counts winning events; expected regret weights their consequences. Raw mean cost and per-instance relative regret also differ: TSP’s VAL regret improved slightly while raw cost worsened. These distinctions must survive aggregation and acceptance.

The squared surrogate has a material limitation: when a costly action’s score is below −1, minimizing its squared loss can **raise that score**. Centering removes translation dependence, but the margin of one makes inherited score units consequential. Cost normalization does not remove this dependence. The primary paper supports the centered cost-sensitive construction and unrestricted calibration, not success within frozen offsets or under the joint gate. [Avila Pires et al., Section 2 and Appendix C.3](https://proceedings.mlr.press/v28/avilapires13.pdf)

**Recommend one next mainline: a single fixed-scale, cost-sensitive squared-offset screen across all 18 tasks.** Leave CE completion unresolved. This screen directly introduces observed cost magnitudes within the same decision class and permits an exact numerical audit. Its value does not depend on assuming that completing CE would fail. The requested **gpt-6-astra/max review is same-family; this positive recommendation is provisional**.

Freeze the following declaration before fitting.

For each task, retain R76 epoch18, its candidate pool, all natural-frequency TRAIN rows, and \(K-1\) intercept degrees of freedom. Assume finite scores/costs and strictly positive row-minimum costs; invalid inputs make the screen inconclusive.

\[
m_i=\min_j C_{ij},\quad r_{ij}=(C_{ij}-m_i)/m_i,\quad
h_{ij}=z_{ij}-K^{-1}\sum_\ell z_{i\ell},
\qquad c=\frac1{NK}\sum_{ij}r_{ij},\quad w_{ij}=r_{ij}/c.
\]
\[
\min_{\mathbf1^\top b=0}J(b)
=\frac1{NK}\sum_{ij}w_{ij}(1+h_{ij}+b_j)^2
+\frac{\|b\|_2^2}{2N}.
\]

Set \(c=1\) when all regrets are zero. Preserve every positive regret without clipping or additional row normalization. Fix the score multiplier and prior precision at one. The TRAIN-derived \(c\) also fixes the data-loss/prior trade-off; it is not subsequently tuned. Predict \(\arg\max_j(h_{ij}+b_j)\), resolving ties by ascending global solver ID.

Use CPU FP64, zero initialization, and seed2 independently per task. Define projected minibatch SVRG by

\[
g_i(b)=\frac2K w_i\odot(1+h_i+b)+b/N,\quad
P=I-\mathbf1\mathbf1^\top/K,\quad
L=\max_{ij}(2w_{ij}/K+1/N).
\]
\[
b\leftarrow P\!\left[b-\frac1{8L}
\left(g_B(b)-g_B(\tilde b)+\bar g(\tilde b)\right)\right],
\qquad \bar g(\tilde b)=N^{-1}\sum_i g_i(\tilde b).
\]

Each update draws **exactly 1,024 rows uniformly with replacement**; \(g_B\) averages those draws. Set the snapshot \(\tilde b=b\) initially and every 1,000 updates, computing its full-TRAIN gradient. Explicitly distinguish those snapshot passes from minibatch parameter updates. This specifies actual batch1024 optimization, including its full-gradient correction, rather than merely 1024-row I/O.

Cap each task at 30,000 updates. Every 1,000 updates, require projected gradient infinity norm \(\le10^{-5}\) and absolute objective change \(\le10^{-7}\), at two successive checks. Additionally audit the constrained quadratic optimum:

\[
d_j=\frac2K\overline{w_j}+\frac1N,\quad
a_j=\frac2K\overline{w_j(1+h_j)},\quad
\lambda=-\frac{\sum_j a_j/d_j}{\sum_j1/d_j},
\qquad b_j^*=-(a_j+\lambda)/d_j.
\]

Require \(\|b-b^*\|_\infty\le10^{-6}\) at both qualifying checks; report the objective gap \(\frac12\sum_jd_j(b_j-b_j^*)^2\). The analytic optimum audits TRAIN optimization only. It neither initializes nor replaces the minibatch-trained candidate. Stop at the first qualifying pair. Any task exhausting its budget makes the entire screen inconclusive; retain identity without extending the budget or changing thresholds.

After all 18 fits pass, lock the complete candidate and evaluate VAL once. Average instances within each task, then weight tasks equally:

\[
A=\frac1{18}\sum_t\overline{\mathbf1[\hat j=y_{\rm native}]},\qquad
M=\frac1{18}\sum_t\overline{C_{\hat j}},\qquad
R=\frac{100}{18}\sum_t\overline{r_{\hat j}}.
\]

Freeze the **executed round001 gate**: candidate-minus-identity \(\Delta A>0,\ \Delta M\le0,\ \Delta R\le0\), using unrounded values and zero tolerance. VS-TRAIN-SBS remains reporting-only, using each TRAIN-selected solver’s costs on the evaluated split. Record that the original proposal also gated VS-SBS; preserve that distinction without rewriting round001. Raw macro cost remains a scale-dependent criterion. [Executed protocol](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round001/protocol.json)

Report every task, method-selection counts, corrected/broken decisions, signed regret changes, selected-regret quantiles/maxima, and upper-tail contributions. Include descriptive paired uncertainty. These diagnostics cannot select taskwise offsets, impose coverage, or soften the gate.

Passing would support this particular cost-supervised offset rule under the frozen empirical gate. A converged failure would reject this objective at its declared score scale, not all cost-aware offsets or learnability. Numerical failure would leave that question unresolved. A positive scalar temperature applied to final scores preserves argmax; a prior sweep changes shrinkage without introducing cost-magnitude supervision. Neither addresses this hypothesis as directly.