# Round004 Proposal: Frozen-B Conditional Relative-Cost Head

Research specification only; not an execution record or job submission.
Owner: research/papers/ideas. Code, checks and execution: Sol/max only.
R78 remains active. Old app goal and resource-owned STATUS.json are unchanged.

**Historical plan, finite route closed negative.** The fixed24840-update fit
completed and the actual fitted candidate failed all three VAL gates. B is
retained. Do not relaunch, extend, change coefficients or propose a fifth similar
variant. See [final result](../round004/RESULTS.md) and RESEARCH.md Section17.
The next research action is the existing NSS locked TRAIN/VAL handoff, not a
new implementation or solver run. The original specification below is retained
for provenance and is not a current execution instruction.

## One Recommendation

Keep R77 full-B ep28/u34776 frozen. Fit **one small conditional-relative-cost
head per task** from its already saved TRAIN score vectors and candidate costs.
Assemble the18 heads into one complete rule and compare its single fixed
endpoint with unchanged B on VAL. CPU preferred; no new backbone/solver pass
or use of the ongoing NSS results. No second backbone branch is necessary.

This is not a continuation of round003, a retuned risk coefficient, or another
constant-offset fit. It learns cost targets using cross-candidate score patterns.
There is no method-frequency target, cold-method bonus, taskwise VAL choice,
extra input role, temperature search, width/rank search, or multi-seed run.

One actual gpt-6-astra/max review supports this bounded test, same-family and
provisional only: [request](ROUND003_FINAL_ASTRA_MAX_REQUEST.md),
[verbatim response](ROUND003_FINAL_ASTRA_MAX_RESPONSE.md).
No performance gain, code correctness or independent acceptance is asserted.

## Why It Is A Different Hypothesis

Round003 failed all six scheduled VAL gates despite lower CE. Both fitted arms
also reduced first-VAL zero-pair coverage without improving actual cost. More
CE reduction or another coefficient is not supported by those observations.

The proposed head instead directly predicts candidate relative-cost magnitudes
conditional on the frozen score vector. Every TRAIN row supervises every valid
candidate, including methods that rarely win. An existing probability vector
can be useful as features without being a calibrated winner posterior. This
does not add information beyond B, and no report proves that a profitable
conditional switching region exists. It is a low-cost empirical hypothesis.

Do not conflate two uses of a square: round002 penalized a fixed score margin
with cost weights, r_j*(1+fixed_score_j+b_j)^2. Here the target itself is the
observed relative cost, and the prediction depends on the whole score vector.
Neither a squared fitting loss nor a smaller fitting error guarantees better
native decisions. The existing three-part VAL gate remains decisive.

## Inputs And Fixed Model

Use only the already locked B's TRAIN/VAL score/cost caches, with the same
task pools, candidate order, valid masks, FP64 cost values and native labels.
The research agent does not read arrays or checkpoints; Sol uses the existing
artifacts and implementation conventions, without a new audit pipeline.

For task t with K_t candidates, let p_t(x) be the valid masked softmax of the
frozen B logits at fixed temperature1. It is a feature vector, not an assertion
that p_t equals the true conditional posterior. No other feature is introduced.
For TRAIN labels define

\[
C_*(x)=\min_{j\in J_t}C_j(x),\qquad
r_j(x)=\frac{C_j(x)-C_*(x)}{C_*(x)},\qquad
a_t=\frac1{N_t}\sum_{x\in\mathrm{TRAIN}_t}\frac1{K_t}\sum_j r_j(x).
\]

Freeze the task constant a_t from TRAIN. Relative regret already uses each
instance's C_*; do not add per-instance sum/max/variance normalization. If
a_t=0, retain B for that TRAIN-defined degenerate task. This is not a VAL
exception. Inactive candidates stay masked and cannot be selected.

For a_t>0, initialize D_t=0 with D_t in R^(K_t x K_t), and define

\[
g_t(p)=\boldsymbol{1}-p+D_t p.
\]

The reported18 pools contain89 candidate entries and sum K_t^2=513 fitted
parameters. There is no extra intercept vector or hidden layer. Constant
effects are still representable because p sums to1; do not claim to exclude
all biases. The distinction is that pairwise decisions can depend on the
whole p(x), rather than adding the same fixed offset to each original logit.

## Exact TRAIN Objective

For each task fit the same fixed objective:

\[
L_t(D_t)=\frac1{N_t K_t}\sum_{x,j}
 \left[g_t(p_t(x))_j-\frac{r_j(x)}{a_t}\right]^2
 +\frac{\lVert D_t\rVert_F^2}{2N_t}.
\]

The unit-prior penalty and its normalization are fixed before fitting. No
penalty/cost-scale sweep, class reweighting, frequency balancing or new .02
tradeoff coefficient. Tasks have separate heads and equal reporting weight;
there is no shared-backbone loss balancing in this fit. Actual current losses
and weights inside frozen B are not changed or re-evaluated by this researcher.

The conditional-mean justification is the standard squared-error identity:

\[
E[(q-r_j/a_t)^2\mid p]
 =\bigl(q-E[r_j\mid p]/a_t\bigr)^2
   +\operatorname{Var}(r_j/a_t\mid p).
\]

With finite second moments and an unrestricted regressor, minimizing it yields
the conditional relative-cost mean, whose minimum is the corresponding Bayes
action. Our linear family, regularization, finite sample and finite optimizer
do not inherit that unrestricted guarantee. It is not assumed that cost depends
only on the native winner, nor that p is a calibrated posterior.

Expected raw cost additionally involves E[C_* r_j | p], while Top1 rewards the
most probable native winner. Even perfect relative-cost prediction can trade
against those objectives. A task's positive a_t does not resolve that difference.
Own-TRAIN fitting is in-sample; a small B aggregate TRAIN/VAL gap does not prove
stable conditional cost relationships. MSE can be dominated by large residuals
or improve irrelevant cost levels without correcting the selected minimum.

## Fixed Budget And Optimizer

Use seed2 and actual logical minibatches1024. For every nondegenerate task:

- Exactly1380 successful optimizer updates, about20 TRAIN passes at70114 rows.
- Cycle seeded natural TRAIN permutations and fill batches across cycle
  boundaries, so each update contains1024 examples. No class-balanced sampler.
- Fresh AdamW: betas=(0.9,0.999), eps=1e-8, weight_decay=0 because the penalty
  is already explicit in L_t. D_t starts at zero.
- One fixed cosine LR schedule with no warmup: eta_s=0.01*(1+cos(pi*s/1380))/2
  for s=0,...,1379. The schedule reaches zero at the budget boundary.
- Frozen scores only: no backbone gradients, R-Drop passes, solver runs,
  additional input features, full-gradient snapshots or optimum certificates.

Maximum total:24840 tiny-head updates across18 tasks,513 trainable parameters.
There is no wall-clock performance guarantee or GPU requirement. Resource
allocation belongs to Sol; this researcher neither queries nor reserves GPUs.
Use existing TRAIN loss logging for a concise fit trace, not as a checkpoint
selection or convergence-extension rule. The Adam endpoint is not claimed to
be the exact minimizer of this strictly convex fitting objective.

Do not evaluate a menu of head checkpoints on VAL. Freeze all18 matrices at
the declared endpoint before the one candidate VAL readout. No averaging,
alternate optimizer, LR rescue, coefficient retry or second training budget.

## Inference And B Fallback

Choose argmin over valid g_t(p)_j. Resolve EXACT predicted-cost ties by the
original B logit ranking, then the original global solver-ID order. No tuned
epsilon, switch margin or confidence threshold. At D=0 this gives B's original
greedy rule, including probability-quantization ties resolved with its logits.

Do not clip negative cost predictions: clipping changes the declared ranking
rule and is not needed for this empirical decision model. Scores g are not
winner probabilities; do not present softmax(-g) CE as calibrated winner CE.
B's original probabilities and CE stay unchanged by this decision head.
Only the learned matrices and frozen-B scores are needed at inference, not
the true instance costs, C_* or any solver outputs produced on demand.

## One VAL Gate And Stop

Reuse B's existing unrounded baseline record, masks, native labels and metric
definitions. Displayed reference values are Top1 approximately70.894308943%,
raw mean cost11.395558343544 and actual regret0.401326745%; these rounded
displays do not reset the zero-tolerance thresholds.

Promote only if the single complete18-task candidate simultaneously has:

- Native macro Top1 strictly greater than B.
- Raw macro mean cost no greater than B.
- Actual macro relative regret no greater than B.

CE, fitted MSE, zero counts, winner coverage and individual-task improvement
cannot replace this gate. No best-task mixture, post-VAL fallback for only
unfavorable tasks, or new threshold. Frozen B is the sole control and default.

A completed gate failure closes this finite head trial and retains B; do not
scan penalties, features, temperature, optimizer settings or rank afterwards.
An unstable/unfinished fit is an incomplete attempt and also retains B, not
proof of an information limit. A pass is single-seed exploratory evidence on
the repeatedly used VAL, not statistical non-inferiority or all-task safety.
Every outcome leaves the ongoing R78 research goal active pending an agreed
research conclusion or later supplied evidence, not an automatic new trial.

For the report, distinguish frozen B, the actually fitted head, and any final
B fallback in BOTH TRAIN/VAL tables. Reuse the existing all18 native metrics
and method counts; show changed-decision Top1 fixes/breaks and positive/negative
cost changes, including wrong-to-wrong switches. This avoids interpreting
fallback TRAIN metrics as if they describe a fitted predictor. No precision-
certification loop or new code audit is requested by the research plan.

No TEST/LIB is read to select or revise this rule. Only after the owner freezes
one unique whole18 VAL candidate for formal release is complete TEST run once;
LIB uses identical B/head weights, masks and rule. Formal release is a separate
decision, not automatic permission from this research document.

## Primary Evidence And Limits

[Avila Pires et al., ICML2013, Section2](https://proceedings.mlr.press/v28/avilapires13.pdf)
defines the conditional action-cost target. Its centered surrogate-risk theorem
is not a theorem for our restricted squared-error head. The conditional-mean
identity above is the separate regression justification used here.

[Kull et al., NeurIPS2019, Section3](https://proceedings.neurips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf)
shows how cross-score dependence can alter decision boundaries and discusses
capacity control. Its log-probability/softmax calibrator, log-loss target and
heldout tuning differ from this TRAIN cost-regression model. We do not import
its calibration claims, empirical gains or regularization grid.

No bibliography expansion, TSP-role architecture, R58 revival, solver rerun,
or NSS teacher is part of this proposal. The current reports support a limited
mechanism test, not a claim that these513 parameters contain a useful solution.
