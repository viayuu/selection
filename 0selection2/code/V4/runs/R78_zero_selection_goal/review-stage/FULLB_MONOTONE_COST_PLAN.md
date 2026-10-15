# Round003 Full-B Mainline: One Monotone Instance-Cost Replacement

Date: 2026-10-15. Research specification, not an execution record.
Research owner: papers/ideas only. Implementation and all execution: Sol/max.
Persistent R78 state: active. app_goal_unmodified: true. STATUS.json untouched.

## Decision First

Use **R77 full-B ep28/u34776** as the new reference. Propose one all18 paired
short-continuation trial: original objective control versus replacement of
only its cost-risk block by the centered, cost-weighted softplus below.
Keep CE, compare, R-Drop, features, architecture and inference unchanged.
No R76 calibration, learnable intercept/vector scaling, frequency quotas,
method-specific bonuses, ReLD-only subproblem, new teacher, or weight sweep.

Latest user authorization: preparation is complete and two RTX3090 GPUs are
allocated for simultaneous Control/NewRisk, one independent branch per GPU.
Each uses logical batch1024, seed2, fresh AdamW and the same B ep28/u34776
weights. This supersedes the prior hardware wait; the scientific protocol below
is unchanged. The user reports NSS restored to single RTX4090 gpu05 after a
rejected DP attempt with0 updates. No GPU probing/allocation, code review or
job start by this researcher. Await the matched whole18 TRAIN/VAL feedback;
no more branches, precision checks or research-review loops are requested.

One actual gpt-6-astra/max review supports a bounded trial, with same-family
provisional status only. It does not predict performance or certify code.
[Request](R77B_MAINLINE_ASTRA_MAX_REQUEST.md),
[verbatim review](R77B_MAINLINE_ASTRA_MAX_RESPONSE.md).

## Why This Test, And What Is Still Unknown

Full-B has VAL Top1 70.8943%, actual regret0.4013267%, and30 zero-with-winner
pairs. Zero pairs account for13.1847% of its regret; including near-zero pairs
accounts for24.1975%. Most remaining cost loss is outside those sets. Thus the
training rule covers every valid candidate and every task, not selected zeros.

Existing TRAIN/VAL calibration bins show some score-associated winner signal:
for example, TSP/LEHD's TRAIN observed wins rise from2.68% in p[0,.05) to30.08%
in p[.25,.30), with analogous VAL3.16%/33.17%. ATSP and several MVRP methods
also show association or underestimation; others, such as VRPL/MTPOMO, do not
have an underestimated marginal prior. Neither near-zero picks nor approximate
calibration establishes that more picks would improve expected cost.

The final native-name handoff covers all89 task-candidate entries. Its TRAIN/VAL
zero sets intersect in23 pairs and have32 in their union, not a list of methods
to promote. TSP/LEHD's VAL winner-conditioned mean p=.20623 with0 picks is not
zero probability and is not a measured true posterior. A positive posterior can
remain below competitors everywhere, and losses on other instances can justify
conservative cost decisions. Its36.2325% within-TSP raw costloss share is not a
whole18 improvement budget. No LEHD-specific loss, threshold or sampling change.

The intervention changes **the use of existing TRAIN cost labels**, not the
information supplied to the model. It is a supervised objective hypothesis,
not a finding that old risk gradients saturate, the .02 coefficient is wrong,
or the features are sufficient/insufficient. No additional diagnostic pipeline
or NSS completion is required before this one finite research comparison.

Unlike the rejected fixed-offset screen, the same existing network is allowed
to change instance-dependent scores. Frozen-score CPU updates cannot evaluate
this hypothesis. The cost is two short matched continuations, not a new full
R77 baseline or a collection of hyperparameter trials. This agent runs neither.

## Exact Treatment

Use the existing valid candidate set J_t(x), K=|J_t(x)|, existing maximin
logits z_j(x), and the unchanged native inference/tie rule. Define TRAIN targets
from the existing FP64 method costs:

\[
C_*(x)=\min_{j\in J_t(x)} C_j(x),\quad
r_j(x)=\frac{C_j(x)-C_*(x)}{C_*(x)},
\]
\[
a_t=\frac1{N_t}\sum_{x\in\mathrm{TRAIN}_t}
             \frac1{|J_t(x)|}\sum_{j\in J_t(x)}r_j(x),\qquad
h_j(x)=z_j(x)-\frac1K\sum_{k\in J_t(x)}z_k(x),
\]
\[
L_{\rm mono}(x)=\frac1K\sum_{j\in J_t(x)}
                  \frac{r_j(x)}{a_t}\log(1+e^{h_j(x)}).
\]

Compute a_t once on TRAIN, and freeze it. If a_t=0, define this replacement
block as exactly zero for that task. Relative regret already normalizes each
instance by C_*; do **not** add another per-instance sum/max/std normalization.
Do not fit a_t, a temperature, clipping threshold, margin or method weight on
VAL. Cost ties have equal r, while the retained native CE still uses the fixed
native winner label. Invalid methods remain excluded by the existing mask.

The intervention is

\[
L_{\rm treatment}=L_{\rm R77B}-.02L_{\rm old\ scaled\ risk}
                                      +.02L_{\rm mono}.
\]

Under the supplied R43 description this leaves .35 CE, .35 compare with its
.5 true3/.25 pred3/.25 all mixture, and .35 R-Drop unchanged. Preserve their
actual current reductions, dropout averaging and warmup convention. Sol owns
the implementation interpretation; no research agent reads or audits its code.
No new prediction head or change to the maximin map is proposed. TRAIN costs
are supervision only; greedy inference needs no true costs or extra solver run.

Centering removes a common-score-shift escape. The scalar softplus is monotone
and does not have the squared penalty's derivative reversal below -1. However,
centering and shared network parameters couple actions: a nonwinner score can
still increase. There is no monotone improvement guarantee for the decisions.

The primary theoretical motivation is the centered cost-sensitive surrogate
family in [Avila Pires et al., ICML2013, Section2 and Table1](https://proceedings.mlr.press/v28/avilapires13.pdf).
Its standalone population result does not certify this mixed neural objective,
restricted optimization path, or a simultaneous improvement in Top1, raw cost
and relative regret. Fixed task scaling 1/a_t changes task tradeoffs within a
shared network even with equal task sampling. Normalizing average weight does
not bound tails or match the old gradient magnitude. Keeping .02 is a fixed
choice, not proof of equal influence. This tests the **whole risk-block
replacement**, including its normalization, not pure softplus curvature.

## Finite Matched Protocol

1. Start control and treatment from the same B ep28/u34776 weights. Keep all18
   full TRAIN sets,70114 rows/task, natural class frequencies, the same task
   sampler, batch1024, seed2, architecture and data/feature preprocessing.
2. Reset both optimizers identically. Reinitialize both RNG/batch-order states
   identically at seed2. Keep scheduler logical position at u34776 and preserve
   the inherited R-Drop warmup position. Sol records the exact shared late LR
   sequence and optimizer reset once before launch; no extra tuning is implied.
   Do not substitute old uninterrupted B for this reset control.
3. Run exactly14,904 successful updates per branch, the inherited12-epoch
   continuation span to logical u49680. Control keeps the original risk block;
   treatment changes only the block defined above. No convergence-tolerance
   extension, second scale, weight scan, extra epochs or methodwise exception.
4. Evaluate complete all18 VAL at only three predeclared endpoints: +4968,
   +9936, +14904 updates, corresponding to logical u39744/u44712/u49680
   (continuation epochs4/8/12). The original locked B is the zero-update fallback.
   Training logs may continue normally; other checkpoints are not candidates.
5. A genuine unrecoverable execution failure is reported by Sol as such; it is
   not a negative scientific result or an invitation to a new optimizer search.
   R78's persistent research goal remains active regardless of this trial.

The reviewer considered all12 inherited epoch endpoints. The final research
specification deliberately uses just4/8/12 for both branches, before new results,
to limit evaluation work and adaptive selection opportunities. This narrowing
is recorded, not claimed to have undergone a second review. No further review
loop is requested. Existing B's selected checkpoint already used this VAL;
three new endpoints do not make VAL independent.

## Fixed Selection And Decision

For each branch, include B as fallback and use the same rule: among complete
all18 checkpoints satisfying BOTH cost gates below, maximize native macro
Top1; tie-break by lower actual regret, then earlier continuation update.

- Raw macro mean cost <= **11.39555834354421**.
- Actual macro relative regret <= **0.40132674544775454%**.
- A new rule must have native macro Top1 > **70.89430894308944%** to replace B.

Use unrounded metrics and the inherited zero tolerance. No taskwise patchwork,
rounding allowance, forced-frequency pass, or TEST-selected exception.

For a replacement-specific positive conclusion, the selected treatment must
also exceed the selected matched control's native macro Top1. Always report
its mean-cost/regret differences against control: passing B's cost gates does
not establish Pareto dominance over control. If it fails, retain B for this
mainline and close this finite replacement trial without another coefficient
attempt. If the control alone improves, record a continuation/reset finding,
not a softplus finding; do not silently auto-promote a different research rule.

Passing is exploratory single-seed evidence under this protocol, not statistical
non-inferiority, robustness across seeds, or a guarantee for every task.

## Minimal Result Handoff

Reuse Sol's existing TRAIN/VAL reporting. Return macro native Top1, raw mean
cost and actual regret for B/control/treatment, plus all18 task rows. Include
the selected checkpoint under the fixed rule and the existing loss-block logs.
No numerical-optimum certification or new code-audit loop is a prerequisite.

On changed decisions, report positive and negative cost changes and native
Top1 fixes/breaks, including wrong-to-wrong switches. Keep exact cost ties and
native labels distinct. Report zero/near-zero identities/counts descriptively,
never as the acceptance condition. A cost decrease despite unchanged zeros is
meaningful; fewer zeros with worse costs is not a repair.

This research handoff is not permission to inspect TEST. Only after the owner
freezes one whole18 checkpoint/rule for a formal release is complete TEST
evaluated once; LIB uses the identical weights/rule. Subsequent idea design
still uses TRAIN/VAL only. No NSS labels, TEST, or LIB labels enter this trial.
