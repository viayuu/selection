# R78 A/B Equal-Weight Screen: Bounded Research Decision

Date: 2026-10-15. Research only; same-family/provisional review requested.
Use gpt-6-astra, reasoning effort max. One review, no follow-up loop.

## Boundaries

Read this request and, only if useful, these two already-authorized reports:

- `/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/review-stage/R77_AB_VAL_COMPLEMENTARITY.md`
- `/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/review-stage/R77_AB_VAL_COMPLEMENTARITY.csv`

Do not follow their source-code/cache/manifest links. No project code, raw
arrays, checkpoints, TEST/LIB, root result files, training, inference, solver,
GPU, tests, audits, Git, goal tools, file edits, subagents or nested reviewers.
For any permitted shell read, activate easynco using the conda shell hook first.
No new literature search is needed; verified primary-source evidence is below.

## Evidence

User asks for a single yes/no decision: is one cheap CPU fixed equal-probability
average screen worth doing, or should we only await NSS? The user does not
require a positive proposal. All implementation/execution/checks belong to Sol.

R77 A ep13/u16146 and B ep28/u34776 are already VAL-best locked, same initial
weights/architecture/seed2, different TRAIN sets and optimization trajectories.
Do not claim independent errors, fresh label information or a pure data-size
causal effect. A's full-data relationship does not make p_A a known function
of p_B. Four R78 interventions failed; the p_B-only relative-cost-head route
is closed. That closure is not proof that a second fitted predictor is useless.

Sol reports aligned VAL source indices, method names/global IDs, native winner,
original costs and node counts: 18 tasks x3895 =70110 instances. No ensemble
rule was evaluated and no TEST was read. Counts:

- Different top1 choices:5399 (7.7007559549279705%).
- Both correct47652; A-only correct1565; B-only correct2052.
- Both wrong18841: same choice17059; different1782.
- In both-wrong cases, A cheaper821, B cheaper961, equal cost17059.
- A native Top1 70.19968620738838%; B70.89430894308943%.
- A raw macro mean11.397558589696828; B11.395558343544208.
- A actual macro regret0.4186125882061135%; B0.40132674544775454%.
- Disagreements concentrate in CVRP/OVRP/VRPL/OVRPL. No taskwise switches allowed.

Native-label correctness need not imply a strict cost advantage in tied costs.
The report contains no confidence-margin diagnostics or realized ensemble cost.
Do not require an additional statistics pipeline before this bounded decision.

## Verified Original Papers

Breiman (1996), Bagging Predictors, Machine Learning24:123-140,
https://link.springer.com/content/pdf/10.1007/BF00058655.pdf .
Full text discusses training-set perturbation and learner instability; bagging
can help but can also hurt. R77 unequal-size fits are not its bootstrap design.
Squared-error averaging results do not guarantee beating B in Top1/routing cost.

Lakshminarayanan, Pritzel and Blundell (2017), Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles, section2.4,
https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf .
They average predicted class probabilities uniformly, train with different
random initializations/data shuffles, and report bagging hurt their experiments.
R77 is not a reproduction of that training recipe; no new ensemble training
is proposed. Calibration/NLL improvement is not a decision-cost guarantee.

## Proposed Single Decision To Criticize

Recommend exactly one cheap, deterministic CPU cache-only screen, not adoption
or a prediction of gain. There is observed but limited complementary signal;
the fixed-rule readout can resolve the empirical question without fitting any
new parameters. B is stronger overall, so equal averaging may still fail.

Pin the sole rule before any ensemble metric is observed:

`p_E(j|x) = 0.5*p_A(j|x) + 0.5*p_B(j|x)`;
choose argmax across the task's original valid candidate set, tie by ascending
global method ID. Each p is its locked artifact's original masked softmax
distribution at T=1, derived from existing cached native final logits; use each
member's own frozen preprocessing, no score normalization/recalibration.
No logit mean, rank mean, vote, weight/temperature scan, taskwise/method-specific
switch, extra checkpoint, trained gate/head or forced method frequency.

All18/70110 VAL rows, one complete candidate. CPU chunks1024; this is not a
training batch and has zero optimizer updates. Existing seed2 artifacts fixed;
no new seed, TRAIN full inference, GPU/model/solver invocation or API call by
the execution worker. No evaluation is performed by this research reviewer.

Gate against the existing unrounded B reference, unchanged zero tolerance:
native macro Top1 strictly increases AND raw macro mean cost does not increase
AND actual macro relative regret does not increase. CE and zero counts are
descriptive only. Readout actual ensemble metrics and per-task costs, separate
from B fallback; no new precision/convergence/uncertainty-check loop.

If any condition fails, close this finite A/B equal-weight route, retain B,
await NSS. No rescue weights, temperatures, logit variant or selected subtasks.
If it passes, it is exploratory VAL eligibility only; no automatic TEST release.
Lock the complete artifact before one formal TEST: both checkpoint identities,
each member's frozen preprocessing, original candidate mapping/masks, native
probability definition/T=1, weights0.5/0.5 and global-ID tie rule. LIB must use
the same complete bundle. Never consult already-completed R77 TEST.

Known limitations: same reused VAL selected both members and motivates this
screen, so not independent confirmation. The inference artifact needs both
models, roughly twice the single-model network arithmetic and additional
weight storage, not a guaranteed2x wall-clock latency. Cached CPU screening
does not price deployment overhead; routing cost gate excludes that overhead.

Mathematical boundaries: averaging probabilities preserves a shared top1 under
the common tie convention in exact arithmetic, but may choose a third method
on disagreement rows. A member-picking oracle is not the deployed rule or a
strict upper bound for all soft-ensemble outcomes. Jensen gives CE_E <= mean
member CE, not CE_E <= CE_B and not Top1/cost improvement.

## Required Answer

In at most650 Chinese characters, give GO-ONE-SCREEN or NO-GO with strongest
reason, any material flaw in the fixed rule/gate, and explicit claim limits.
Do not create alternatives, audits, precision checks or future experiments.
R78 remains active; old paused app goal and STATUS.json remain untouched.
