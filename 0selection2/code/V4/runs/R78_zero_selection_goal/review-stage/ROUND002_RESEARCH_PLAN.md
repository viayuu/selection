# R78 Round002: One Cost-Sensitive Offset Screen

Post-execution research disposition: **stop further optimizer work** after the
strongly negative round002 TRAIN result. The original protocol below is retained
as history, not a relaunch instruction. The latest user permits one TRAIN-only
metric readout of the known analytical reference, with no fitting or VAL/TEST.
Current handoff: [POST_ROUND002_DECISION.md](POST_ROUND002_DECISION.md).

Date: 2026-10-15 (Asia/Shanghai).
Canonical directory: `/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal`.
Research goal: active. Application goals: unmodified; old broad goal remains paused.
Research owner writes documents only. Sol/max owns implementation, code review,
and execution. This plan does not claim that round002 has started or finished.

Actual Astra/max research review: [response](ROUND002_ASTRA_MAX_RESPONSE.md).
Review independence: same-family. Acceptance: provisional research recommendation.

## 1. Observed Signal And Non-Claims

The sole evidence is round001 TRAIN/VAL reports. TEST remains outside design.

| Split / metric | Identity | Round001 candidate | Candidate minus identity |
|---|---:|---:|---:|
| TRAIN Top1 % | 69.369646 | 69.425982 | +0.056337 pp |
| TRAIN CE | 0.752703080 | 0.743902973 | -0.008800107 |
| TRAIN actual regret % | 0.460650092 | 0.462689801 | +0.002039709 pp |
| TRAIN macro mean cost | 11.373004840 | 11.373099773 | +0.000094932 |
| TRAIN VS-TRAIN-SBS % | -0.143845417 | -0.139222429 | +0.004622988 pp |
| VAL Top1 % | 69.236913 | 69.379546 | +0.142633 pp |
| VAL CE | 0.755865163 | 0.746502800 | -0.009362362 |
| VAL actual regret % | 0.464503404 | 0.467179029 | +0.002675625 pp |
| VAL macro mean cost | 11.402474922 | 11.402575925 | +0.000101003 |
| VAL VS-TRAIN-SBS % | -0.142069159 | -0.137179498 | +0.004889661 pp |

The exact macro values and original deltas in the JSON reports take precedence
over this rounded table. Identity remains selected. The inconclusive TRAIN fit
status and the observed VAL gate failure are two separate facts.

Only OVRPBTW satisfied the two-successive-check convergence rule; all 18 tasks
reached the 30,000-update budget. Thus round001 did not establish the fully
converged CE optimum or disprove useful offsets. The training objective improved
on both splits, so "calibration did nothing" is also inaccurate.

| Task | VAL changed decisions | Repaired / broken Top1 | VAL regret change, pp |
|---|---:|---:|---:|
| TSP | 82 | 26 / 18 | -0.000039601 |
| CVRP | 565 | 105 / 66 | -0.066841570 |
| ATSP | 302 | 160 / 125 | +0.039645257 |
| OVRP | 170 | 80 / 51 | -0.008499009 |
| VRPL | 481 | 160 / 132 | -0.008666724 |
| OVRPL | 755 | 209 / 248 | +0.092562896 |
| Other 12 tasks | 0 | 0 / 0 | 0 |

TSP's raw mean cost increased despite its very small regret improvement.
ATSP gains 35 correct decisions but increases regret. OVRPL increases regret
on both TRAIN and VAL. These are observations about event counts versus cost
magnitudes, not proof that a few catastrophic outliers dominate.

The zero-pick/positive-winner counts remain TRAIN55 and VAL54, but membership
changes: CVRP/BQ VAL selections are 0 -> 32 and CVRP/ELG 281 -> 0. Do not use
the unchanged total as evidence that the fitted rule never changed decisions.

TSP/LEHD's TRAIN mean probability moved to 0.169449 versus winner frequency
0.169453 while still receiving no selections. This illustrates the distinction
between marginal probability correction and greedy actions; it does not prove
conditional calibration or insufficient instance information.

## 2. Single Research Question

Can using observed TRAIN cost magnitudes, rather than winner CE, produce an
acceptable decision rule within the SAME frozen-score intercept family?

Recommend this one new objective screen. Do not concurrently finish the CE fit,
fit vector slopes, scan priors/temperatures, add conditional features, or retrain
the backbone. CE completion stays an explicitly unresolved question. Choosing
this screen is a research-priority decision, not a conclusion that CE completion
would fail. The old R78-B backbone risk-term deletion is still deferred.

The chosen objective is a fixed-scale squared cost-sensitive surrogate adapted
from the existing P7 reference, [Avila Pires et al., ICML2013, Section2 and
Appendix C.3](https://proceedings.mlr.press/v28/avilapires13.pdf). Its unrestricted
calibration result does not guarantee success for frozen offsets, finite TRAIN,
a fixed prior, or a joint Top1/cost gate. No additional papers are introduced.

## 3. Exact Fitting Objective

For each task separately, retain the R76 epoch18 score cache and the existing
fixed candidate pool. Let N be its TRAIN rows and K its candidates. Use the
existing finite costs, native targets and candidate identities; this screen
does not change feasibility definitions or repair solver data.

\[
 m_i=\min_j C_{ij},\qquad r_{ij}=(C_{ij}-m_i)/m_i,
 \qquad h_{ij}=z_{ij}-K^{-1}\sum_\ell z_{i\ell}.
\]
\[
 c=\frac1{NK}\sum_{i,j}r_{ij},\qquad w_{ij}=r_{ij}/c,
 \qquad
 J(b)=\frac1{NK}\sum_{i,j}w_{ij}(1+h_{ij}+b_j)^2
       +\frac{\|b\|_2^2}{2N},\qquad \mathbf1^\top b=0.
\]

If all TRAIN regrets are zero, set c=1; w remains zero and the fixed prior
selects b=0. Nonfinite inputs or nonpositive row-minimum costs make this declared
screen inconclusive; do not silently drop rows or change candidates.

All other settings are fixed: score multiplier1, prior precision1, margin1,
no clipping of positive regrets, no per-row cost normalization, and no
candidate-frequency weighting. The task-wide c is fitted on TRAIN exactly once.
It changes the data-loss/prior scale and is part of the declared method, not a
coefficient to tune. Natural sampling frequencies remain unchanged.

The deployment rule is `argmax_j(h_j+b_j)`, equivalently `argmax_j(z_j+b_j)`.
Ties use ascending global solver ID. There are still only K-1 trainable
parameters per task. Row-centering preserves identity decisions and fixes the
score gauge; it does not introduce a new feature or new ranking capacity.
Inference requires scores and the locked b only, never true solver costs.

Important limitation: `(1+s)^2` can increase a costly action's score when
s < -1. It is not a monotone penalty on that score. The inherited score units
matter; cost normalization does not remove this dependence. Do not introduce
a temperature or alternate margin if this particular screen fails.
These are decision scores, not certified winner-probability logits or predicted
cost values. NLL/ECE, if reported descriptively, cannot establish this objective's
success; the unchanged native Top1/cost gate is authoritative.

## 4. Finite CPU Fitting Protocol

Use CPU FP64, zero initialization, and seed2 independently for every task.
No new backbone forward, GPU, solver, TEST data, or LIB fitting is required.
The execution owner may implement the following declared mathematical optimizer;
this research document does not inspect or certify its code.

Use projected minibatch SVRG with

\[
 g_i(b)=\frac2K w_i\odot(1+h_i+b)+b/N,\quad
 P=I-\mathbf1\mathbf1^\top/K,\quad
 L=\max_{i,j}(2w_{ij}/K+1/N),\quad \eta=1/(8L).
\]
\[
 b\leftarrow P\left[b-\eta\{g_B(b)-g_B(\widetilde b)
                          +\overline g(\widetilde b)\}\right],
 \qquad \overline g(\widetilde b)=N^{-1}\sum_i g_i(\widetilde b).
\]

Each parameter update draws exactly **1024 TRAIN rows uniformly with
replacement**; g_B averages those draws. No gradient accumulation enlarges that
draw. Set the snapshot to the current b initially and every1,000 updates; use
its full-TRAIN gradient as the explicit variance-reduction correction.
Report the full-TRAIN snapshot passes separately from optimizer updates.
This is a specified minibatch1024 algorithm with full-gradient snapshots, not
a full-batch fitted solution disguised as 1024-row I/O. No learning-rate search.

Cap each task at **30,000 successful optimizer updates**. Every1,000 updates,
check full-TRAIN projected gradient infinity norm <=1e-5 and absolute objective
change <=1e-7. Require both at two consecutive checks; the initial check has no
preceding objective and cannot count as convergence. Add the following exact
quadratic audit at both qualifying checks:

\[
 d_j=\frac2K\overline{w_j}+1/N,\qquad
 a_j=\frac2K\overline{w_j(1+h_j)},\qquad
 \lambda=-\frac{\sum_j a_j/d_j}{\sum_j1/d_j},\qquad
 b_j^*=-(a_j+\lambda)/d_j.
\]
\[
 \|b-b^*\|_\infty\le10^{-6},\qquad
 J(b)-J(b^*)=\tfrac12\sum_jd_j(b_j-b_j^*)^2.
\]

The analytical b* is solely a TRAIN numerical reference. It does not initialize,
replace, or become a second deployed candidate. Report the objective gap and
L for each task. Stop at the first pair of qualifying checks. The finite budget
is a cap, not a claim that convergence is guaranteed for every TRAIN cost tail.

If any task exhausts the budget or fails the stated input assumptions, retain
identity, report the round as inconclusive, and do not inspect VAL for that new
candidate. Do not extend the budget, relax tolerances, or switch the optimizer
within this round. The persistent research goal remains active for the next
evidence-based decision; a numerical limit is not a claim of missing information.

## 5. Locked Gate And All-Task Scope

When all 18 fits qualify, lock the complete candidate before evaluating VAL
once. Compare against **original R76 identity**, not the rejected round001 rule.
Use all existing VAL rows; average instances within task, then the 18 tasks
equally. The exact inherited evaluator definitions are fixed:

\[
 A=\frac1{18}\sum_t\overline{\mathbf1[\widehat j=y_{native}]},\qquad
 M=\frac1{18}\sum_t\overline{C_{\widehat j}},\qquad
 R=\frac{100}{18}\sum_t\overline{r_{\widehat j}}.
\]

Preserve the **executed round001 gate** without loosening any cost tolerance:
`delta_A > 0`, `delta_M <= 0`, `delta_R <= 0`, unrounded, candidate minus
identity. Identity remains selected if any check fails. No taskwise rule mixing,
per-task acceptance flags, or rule fitting to VAL/TEST.

The original research draft named VS-TRAIN-SBS as a cost gate. The executed
round001 protocol instead treats it as reporting-only and gates raw macro M.
This distinction is recorded rather than rewritten. Round001 failed all these
cost measures, so its rejection is unaffected. Round002 preserves the executed
gate; VS-SBS remains reporting-only, computed with each TRAIN-selected SBS's
costs on the current evaluated split. Raw M is scale-dependent across tasks;
do not present it as a unit-invariant cross-problem cost measure.

Every task participates: TSP, CVRP, ATSP, OVRP, VRPB, VRPL, VRPTW, OVRPTW,
OVRPB, OVRPL, VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW, OVRPLTW, VRPBLTW,
OVRPBLTW. Neither the adverse-task list nor the TEST zero-winner list determines
fitting, coefficients, capacity, candidate inclusion, or task selection.

Report every task's Top1, existing Top2/3 definitions, mean cost, regret, VS-SBS,
fixed-method comparisons, selected counts, corrected/broken decisions, and
signed regret gains/losses. Include p95/max and a fixed upper-decile regret
contribution summary to distinguish broad small losses from a costly tail;
do not assert tail concentration before observing it. Retain fitting curves
and the exact-optimum numerical audit. Reuse the already declared descriptive
paired uncertainty protocol; it does not adjust for adaptive VAL reuse.

## 6. Conclusions Allowed After This Screen

- A passing rule supports this particular cost-supervised offset design on the
  assessed VAL under the fixed gate, not a general causal explanation or a claim
  that every task improved. Compare TRAIN-selected constant/conditional references
  before claiming recovered instance discrimination.
- A converged failure rejects this fixed-scale squared-offset candidate under
  this gate, not all cost-sensitive calibration, all offsets, or learnability.
- Numerical failure leaves even that candidate-optimum question unresolved.
- Fewer zero picks alone is never success. No method quota is introduced.
- The .02 backbone risk term is unchanged; its causal effect is not tested here.

Only a passing and fully locked official rule may subsequently receive the
single formal full TEST evaluation, followed by LIB with identical backbone and
calibration weights. TEST/LIB labels never select the rule. Prior historical
TEST exposure remains disclosed; no independent-confirmation claim follows.

## Report Sources

- [Round001 handoff](../round001/RESEARCH_HANDOFF.md).
- [TRAIN comparison](../round001/train_comparison.json).
- [VAL comparison](../round001/val_comparison.json).
- [Executed protocol](../round001/protocol.json).
- [VAL decision](../round001/val_decision.json).
- [TRAIN changed decisions](../round001/changed_decisions_train.csv).
- [VAL changed decisions](../round001/changed_decisions_val.csv).
- [TRAIN methods](../round001/methods_train.csv) and [VAL methods](../round001/methods_val.csv).

No source code, code-review record, raw arrays, TEST output, or STATUS.json was
read to construct this recommendation. No experiment was executed by this agent.
