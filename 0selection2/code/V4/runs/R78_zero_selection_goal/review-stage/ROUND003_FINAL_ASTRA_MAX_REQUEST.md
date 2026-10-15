# Round003 Final Result And One Possible Next Route

Research-only review request, gpt-6-astra / max, same-family/provisional.
User wants a final interpretation and at most ONE justified low-cost next
route, not another coefficient/temperature/bias scan. Reject a new run if the
evidence is insufficient. Persistent goal remains active; no app-goal operations.

## Hard Read/Action Boundary

No source code, tests, execution, model/solver inference, GPU, raw arrays,
checkpoint reads, Git, TEST/LIB, resource/status files or project edits. No
subagents or nested review. Return at most1,100 words. One bounded review only.
This request is self-contained. If essential, the only additional local data
reads allowed are these exact research reports:

- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003/FINAL_RESULTS.md
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003/selected_all18_train_val.csv
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003/method_selection_train_val.csv
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003/first_val/summary.json
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003/first_val/native_method_counts.csv
- /public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/train_val_selection_diagnosis/RESEARCH_HANDOFF.md

Do not follow checkpoint/source links. Sol owns every code check and experiment.
If using a shell for reading, activate easynco via conda shell hook first.
The caller saves your final response; do not edit files yourself.

## Completed Trial

Frozen reference B: full18,70114 TRAIN/task,3895 VAL/task, seed2, originally
locked ep28/u34776. The matched continuation used two independent RTX3090s,
same B weights, fresh AdamW, logical batch1024, identical inherited late LR,
14904 successful updates/arm; only+4968/+9936/+14904 VAL candidates.

Control retained the original R43 objective. Treatment replaced only .02 scaled
relative risk with .02 mean_j[(r_j/a_t) softplus(z_j-mean(z))], where r is
TRAIN FP64 relative regret and a_t its fixed task mean. CE .35, compare .35
(.5true3/.25pred3/.25all), R-Drop .35 and other settings stayed unchanged.

| Rule/update | VAL Top1 % | Raw macro mean cost | Actual macro regret % | CE |
|---|---:|---:|---:|---:|
| B/0 |70.894308943|11.395558343544|0.401326745|0.713641610|
| Control/4968 |70.921409214|11.395716839270|0.405973048|0.705434735|
| Control/9936 |70.850092711|11.395744982192|0.408499816|0.708710366|
| Control/14904 |70.814434460|11.395855343252|0.410925983|0.708355988|
| NewRisk/4968 |70.897161603|11.395848830432|0.407708383|0.700871229|
| NewRisk/9936 |70.832976751|11.395886356713|0.410676066|0.703012548|
| NewRisk/14904 |70.828697761|11.395991601929|0.413110693|0.702879354|

All6 checkpoints failed the fixed gate: Top1 strictly above B, mean cost and
actual regret both <=B, unrounded/zero tolerance. Both selected branches
therefore fall back to B. Outcome: finite_replacement_trial_closed_negative.
No TEST release, no coefficient retry. NewRisk's costs/regret are worse than
matched control at all3 points; its final Top1 is slightly higher than control
but below B. Thus it is not correct to claim every treatment metric is worse
than control everywhere. CE improvement is not decision improvement.

IMPORTANT: selected_all18_train_val.csv and method_selection_train_val.csv
contain the SELECTED rules. All3 labels (B/control/treatment) are B fallback,
so their TRAIN/VAL rows coincide. These are not the trained endpoint TRAIN
metrics. Do not infer identical training trajectories, a fitted TRAIN gain,
or fitted train/val overfitting from these files. No extra endpoint inference
or precision-audit request is warranted for the practical stop decision.

First-VAL descriptive snapshot: zero-with-winner pairs30(B)->27 in both arms;
zero-covered winners2648->1988 of70110. The3 removed zero pairs are
OVRPB/MTPOMO, OVRPB/MVMOE, VRPBL/MTPOMO, with221+234+205=660 true winners.
Their Control picks are1,3,1; NewRisk picks1,3,2. Coverage reclassification by
5/6 picks does NOT mean660 correct selections recovered. Their correctness
is not reported in the allowed count table. Costs deteriorated regardless.
TSP/LEHD remains0 picks/641 winners; mean p on its winners .20623(B),
.20354(Control),.20583(NewRisk), ranks2.8066,2.7551,2.7504. No zero-probability
collapse or proof that it should be top-ranked follows.

## Prior Evidence And Limits

R76 fixed intercept CE screen slightly increased native Top1 but harmed costs.
Its fixed-scale cost-weighted squared-offset successor was catastrophically
bad on TRAIN across18 tasks. Those are closed, not baselines to repeat.
R77 optimization/full-data improvements reduced old VAL zero winner coverage
20.3281% to3.7769%, improving native Top1/regret together. Do not blame batch
alone, a solver name, or one ReLD binary task.

B actual TRAIN Top1/regret71.1767%/0.393012%; VAL70.8943%/0.401327%.
Across18 tasks the candidate counts are8,10,5, followed by seven6-candidate
and eight3-candidate tasks:89 task-candidate entries, sum K_t^2=513.
B has some predictive scores; the reports do NOT establish a region where a
particular alternative improves native accuracy and both cost notions. Winner-
conditioned means do not estimate a true conditional posterior, and posterior
winner choice need not minimize conditional expected action cost.

Owner-supplied TSP input fact: only coordinates, other padded channels zero,
no explicit physical-start role marker. R57 saw node0 endpoint roles in BQ/
LEHD source, but did NOT bind historical deployment/cost columns to that rule.
Do not justify a new role architecture, solver rerun, R58 revival or code audit
from that fact. NSS still has4-5h of ongoing work, no matched new evidence to use.

## Preferred Candidate For Your Criticism

Not another backbone loss swap. **Freeze B; fit one small, score-conditioned
relative-cost predictor per task using already cached TRAIN scores and costs.**
This directly targets cost magnitudes, not CE or class-frequency matching.
No new backbone/solver pass or geometry features. One whole18 learned rule,
one frozen-B control, no full retraining, no mandatory GPU. All89 candidates
stay available, including classes with few/no native winners.

For task t, p=softmax(valid frozen B logits at fixed temperature1). Let
r_j=(C_j-C_*)/C_* and a_t=TRAIN mean of the within-row candidate mean of r.
No additional instance-wise cost normalization. For a_t>0, train D_t of size
K_t by K_t, starting at zero, with

  g_t(p) = 1 - p + D_t p,
  L_t = (1/(N_t K_t)) sum_x,j (g_t(p(x))_j-r_j(x)/a_t)^2
        + ||D_t||_F^2/(2 N_t).

One fixed unit-prior penalty, not a regularization sweep. At D=0 the rule is
B's ranking. Use original B logits then global solver ID to resolve EXACT
predicted-cost ties, so p quantization ties do not invent an initialization
change. No epsilon-based switch margin. Inference chooses argmin g; no
clipping of predictions or probability interpretation. If a_t=0 retain B for
that TRAIN-defined degenerate task. No VAL-selected task switches.

The513 parameters mix the whole probability vector. Pairwise ordering depends
on p(x), unlike a fixed offset added to the original logits. The family still
contains constant effects (p sums to1); do not claim all biases are excluded.
It is a conditional cost regression hypothesis, not Dirichlet calibration or
another r*(1+fixed_score+b)^2 surrogate. It cannot create absent information.
MSE population reasoning is E[(q-r)^2|p]=(q-E[r|p])^2+Var(r|p), assuming finite
second moments. Linear approximation, regularization and finite optimization
do NOT inherit an unrestricted consistency or all-metric improvement guarantee.

Proposed fixed CPU fit: seed2, actual minibatches1024,1380 successful updates
per task (24840 tiny-head updates total, roughly20 TRAIN passes), cycling
seeded TRAIN permutations to fill every batch. AdamW with weight_decay0
(penalty already explicit), betas(.9,.999), eps1e-8; one cosine LR from1e-2
to0 over that budget. No backbone gradient/R-Drop, no full-gradient snapshots,
closed-form certification, strict numerical-optimum tolerance, tuning or retry.
At the end freeze all18 matrices; one VAL candidate vs B, not best-of-epochs.
If unstable, report this finite trial negative/incomplete as appropriate and
stop; no LR/feature/penalty sweep disguised as a rescue.

Keep B's existing VAL gate and unrounded baseline record, not the rounded
table above: native macro Top1 strictly greater, raw macro mean cost and
actual macro regret nonincreased. All tasks reported, no taskwise patchwork.
Reject on any gate failure and retain B. On success describe single-seed
exploratory evidence only; lock exactly one rule before a formal TEST once,
LIB same weights/rule. Do not report softmax(-g) CE as calibrated winner CE.
B's probabilities/CE are unchanged by a frozen decision head. No validation
search for thresholds, nonlinear width, rank, features or shrinkage strength.

## Primary-Source Boundary

Revisited existing primary papers, not a new survey:
- Pires/Ghavamzadeh/Szepesvari ICML2013 Sec2:
  https://proceedings.mlr.press/v28/avilapires13.pdf
  Supports conditional expected action cost as the decision target. Its
  centered-surrogate theorem is NOT a theorem for this squared-error regression.
- Kull et al. NeurIPS2019 Sec3 and fitting section:
  https://proceedings.neurips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf
  Cross-score maps can express instance-dependent decision changes; fitted
  log-loss probabilities and heldout regularization differ from our TRAIN
  cost regression. Do not import their empirical calibration claims or CV grid.
Base-model own-TRAIN fitting is in-sample, not an independent calibration set.
The repeatedly used VAL has adaptive history; one new comparison does not
remove that history. No statistical safety guarantee is claimed.

## Decision Requested

Confirm or correct the round003 conclusion. Then approve ONE bounded next
route with necessary corrections, or reject new training for now. Is this cost
head a genuinely defensible low-cost different mechanism, or just another
unsupported proxy? Critique the conditional-cost assumptions, support/capacity,
fitting budget, native Top1/cost tradeoff and fixed stopping rule. You may reject
the favored candidate; if you do, give at most ONE better-supported next action
and say what evidence is missing, not a menu or a new audit pipeline. No code.
