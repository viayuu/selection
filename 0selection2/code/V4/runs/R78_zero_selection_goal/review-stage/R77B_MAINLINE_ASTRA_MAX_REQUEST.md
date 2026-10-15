# Final Full-B Residual Research: One Bounded Astra/max Review

Date: 2026-10-15. Request: gpt-6-astra, reasoning effort max.
review_independence: same-family. acceptance_status: pending/provisional.

## Hard Scope

Research criticism only. No source-code reading/review, tests, experiments,
model inference, arrays, GPU, training, optimizer runs, Git, goal operations,
or edits to project files. Do not spawn/nest another reviewer. No root R77
RESULTS/status/TEST files. The complete necessary context is below. You may
read only this request and the named TRAIN/VAL report folder if needed:

`/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/train_val_selection_diagnosis/`

Restrict reads there to RESEARCH_HANDOFF.md and CSV research reports. Do not
follow source paths to predictions or checkpoints. Only the existing primary
papers linked below may be revisited; do not expand the bibliography.
Return at most 1,000 words, one decision and one mainline, not a menu or sweep.
The final response will be saved by the caller. Do not edit files yourself.

## User's Current Decision

Old app goal paused and unmodified. R78 persistent research goal active.
Sol/max owns all execution/code work. Full-B is now explicitly the new baseline.
Full18 TRAIN-only fitting, logical batch1024, seed2. Natural class frequencies,
equal task weighting, no per-method coverage target, no TEST-guided rule.
Keep native macro Top1 strictly up, raw macro mean cost and actual macro regret
nonincreased, using unrounded values and zero tolerance, before any promotion.
All18 rule, no taskwise patchwork. Formal release locks one VAL checkpoint/rule
before one complete TEST; LIB uses the same weights/rule. No repeated audits.

R76 frozen-offset CE screen moved VAL Top1 +0.142633pp but worsened regret and
mean cost. Cost-weighted fixed-scale squared-offset screen was severely bad on
TRAIN: Top1 69.37% to40.85%, regret0.46065% to2.82622%, all18 harmed, including
14 tasks meeting strict convergence. Stop these routes. One already-delegated
closed-reference TRAIN comparison is closure only, not a reason to wait here.
This does not establish missing information or disprove instance-cost learning.

## Final R77 Evidence (TRAIN/VAL Only)

All18 tasks: TSP,CVRP,ATSP,OVRP,VRPB,VRPL,VRPTW,OVRPTW,OVRPB,OVRPL,
VRPBL,VRPBTW,VRPLTW,OVRPBL,OVRPBTW,OVRPLTW,VRPBLTW,OVRPBLTW.

- B: 70,114 TRAIN/task,3,895 VAL/task; selected ep28/u34776. TRAIN Top1
  71.17670270%, CE0.704273, regret0.393011700%, mean cost11.365977444.
- B VAL Top1=70.89430894308944%, CE0.713642, regret=0.40132674544775454%,
  raw macro mean cost=11.39555834354421. Zero-with-winner pairs30, winner
  coverage3.7769%; TRAIN25/1.7795%. Near-zero pairs6 VAL/13 TRAIN, descriptive
  threshold 0<selection<=0.1%, never an optimization target.
- A 10k selected ep13/u16146: VAL70.1997%, regret0.4186126%.
- User-provided common end u49680: A/B VAL69.4808/70.8415%; TRAIN-VAL CE
  gap0.2952/0.0100. Full data generalizes better under paired protocol. Selected
  checkpoints differ. R76-to-R77 changed batch/LR/update regime jointly.
- R76 VAL54 zero pairs covered20.3281% of winners; A32/4.9337%, B30/3.7769%.
  No need to explain historical collapse by semantics or an ReLD binary task.
- NSS still incomplete, VALID TSP0.44674%, CVRP0.71150% user supplied. Sol will
  confirm denominator/instances/candidate oracle. No NSS-based target or new
  feature is available for this proposal; no cross-system performance claim.

B all18 VAL (Top1%,actual relative regret%):

|Task|Top1|Regret|
|---|---:|---:|
|TSP|45.750963|0.410813|
|CVRP|46.315789|0.689987|
|ATSP|77.483954|0.369275|
|OVRP|41.206675|0.631665|
|VRPB|80.949936|0.375689|
|VRPL|37.766367|0.397483|
|VRPTW|80.128370|0.302225|
|OVRPTW|83.722721|0.250521|
|OVRPB|88.318357|0.261906|
|OVRPL|39.589217|0.669945|
|VRPBL|78.998716|0.436541|
|VRPBTW|78.716303|0.456611|
|VRPLTW|78.536585|0.321750|
|OVRPBL|89.268293|0.267276|
|OVRPBTW|83.748395|0.314843|
|OVRPLTW|83.106547|0.284609|
|VRPBLTW|79.229782|0.440844|
|OVRPBLTW|83.260591|0.341898|

## Residual Cost And Signal

From existing B CSVs, equal-task arithmetic, not raw-cost pooling:

- VAL zero-with-winner rows contribute0.052913606pp of0.401326745% regret
  (13.18467%); zero+near-zero0.097111222pp (24.19755%). These are loss
  attributions under the current decisions, not attainable gains from flipping.
- TRAIN corresponding contributions0.019298668pp/4.91046% and
  0.103699050pp/26.38574%. Different zero/near-zero set membership across splits
  matters; do not interpret the zero share difference as a distribution shift.
- VAL examples: TSP/LEHD641 wins,0 picks,0.137041 task regret pp; ATSP/MATPOENET
  318 wins,0 picks,0.122532pp; VRPBL/MTPOMO205 wins,0 picks,0.152939pp;
  OVRPTW/MVMOE594 wins,3 picks,0.21582pp; OVRPLTW/MVMOE621 wins,1 pick,0.23909pp.
  The latter two are near-zero, not zero. None is a targeted exception list.
- TRAIN score-bin evidence, existing fixed width0.05 bins: TSP/LEHD p0-.05
  n9549/winner2.68%; p.25-.30 n10889/winner30.08%. VAL corresponding
  n538/3.16%, n627/33.17%. Global mean p15.414% vs winner16.945% TRAIN.
  Signal exists but winner probability remains well below1; rival/cost ordering
  cannot be deduced from this one-dimensional calibration table.
- ATSP/MATPOENET TRAIN p0-.05 n28791/winner1.07%; p.30-.35 n410/winner55.37%.
  VAL corresponding n1602/0.94%, n23/65.22% (small upper bin). TRAIN global
  p5.920% vs winner7.957%; mean p on wins10.926%, on nonwins5.487%.
- OVRPTW/MVMOE TRAIN p.20-.25 n10623/winner26.10%; VAL n588/26.87%.
  Global p12.209% vs winner15.361% TRAIN; mean p on wins20.427%, nonwins10.718%.
- VRPL/MTPOMO near marginal calibration: TRAIN mean p3.695% vs winner3.300%,
  yet0 picks. Not all cold methods have underestimated priors.
- Conditional-on-winner means are not P(Y=j|X). Calibration bins include both
  winning and losing examples but do not provide conditional action cost on
  losses. No proof of a cost-beneficial switching region or irreducible error.
- Native winners are FP64 cost minima in these reports; exact cost ties remain
  native-label Top1 distinctions. Retain that distinction in evaluation.

## Proposed Single Mainline For Criticism

One **monotone instance-cost supervision replacement**, updating the same
full-B architecture, not another frozen offset/vector/temperature fit.
Keep the stated R43 CE/compare/R-Drop blocks and their weights. Replace only
the .02 scaled-relative-risk block, NOT add a fourth new supervised objective.
No claim that the old risk implementation is buggy or measured to saturate.

For task t, valid K candidates, FP64 TRAIN cost C, r_j=(C_j-min C)/min C.
Let a_t=mean over all TRAIN rows and valid candidates of r_j (fixed TRAIN-only,
if all costs tie use zero replacement loss). No per-example cost normalization.
Let h_j=z_j-mean_valid(z), with z the existing maximin scores. Define

  L_mono(x) = (1/K) sum_j (r_j/a_t) softplus(h_j),
  L_new = .35 CE + .35 compare + .02 L_mono + .35 RDrop.

The compare true3/pred3/all mix stays .5/.25/.25. Preserve R-Drop warmup state.
Native scoring, masks, and greedy argmax inference stay unchanged; no cost
labels needed at inference, no new head/intercept, no arbitrary margin/temperature.
Centering prevents free common shifts. Unlike fixed squared (1+h)^2, softplus
has no negative-score reversal. Updating instance-dependent scores is a
genuinely different hypothesis from increasing offsets or retrying their weights.

Target: learn from bad as well as good cost regions, in every task and for
every valid method. Existing scores contain some winner signal; this is NOT a
proof that a new loss will unlock cost-beneficial decisions. Retain .02 as the
predeclared coefficient, not as a claim of equal gradient scale to old risk.
Replacing its scaling too (if old scaling differs) estimates a risk-block
replacement effect, not pure softplus curvature. No follow-on coefficient scan.

Possible minimal matched execution: from B ep28 weights, two short branches,
original objective control vs this single replacement, both reset optimizer/RNG
identically at seed2, both batch1024 and full TRAIN, same R77 late LR sequence
for14,904 successful updates (u34776 to49680;12 full-B epochs). Starting from
the same weights avoids relying on unreported saved optimizer state. This is
one intervention with one necessary control, not a hyperparameter sweep. If
the full continuation state is already available, Sol may reuse the truly
matching existing B continuation instead; record that fact, no code review here.
Never compare an optimizer-reset treatment with old uninterrupted B as if
matched. Freeze exact shared schedule before launch. No new data/features.

Gate and selection: baseline B included as fallback; evaluate whole18 at
predeclared inherited epoch endpoints, choose max VAL native Top1 among rules
with mean cost<=B and actual regret<=B; ties by lower regret then earlier update.
Treatment must improve native Top1 strictly. Use identical checkpoint selection
for control; do not claim objective-specific improvement without its comparison.
Report changed-decision gains AND harms, all18 costs, zero/near-zero counts
descriptively. No TEST until the one chosen rule is frozen. No guaranteed win.

## Existing Primary Sources (No Expanded Search)

Pires/Ghavamzadeh/Szepesvari ICML2013, Section2 Eq1-3 and Table1:
https://proceedings.mlr.press/v28/avilapires13.pdf
Uses centered score vectors and cost-weighted phi; softplus is an included
surrogate. Population/unrestricted-function assumptions do not prove a finite
mixed-loss neural training improvement, simultaneous Top1/mean-cost/regret
Pareto improvement, or frozen-offset consistency. Do not repeat that mistake.

Guo et al. ICML2017:
https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
Kull et al. NeurIPS2019:
https://proceedings.neurips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf
Calibration evidence is weaker than a cost-optimal conditional decision.

## Requested Criticism

Is the proposed single intervention defensible given these reports, or still
too speculative/repetitive? Be explicit about what the data do and do not show.
Check score-centering/signs, scale, all18 cost-vs-Top1 trade-off, and the short
matched-control design. Prefer one finite practical test with honest limits
over more diagnostic pipelines, strict numerical tolerances, or paper lists.
If you reject it, give ONE better-supported all18 mainline and concrete minimal
protocol, not a menu. State why it is better given existing evidence. Do not
automatically resurrect risk deletion, R76 calibration, or frequency balancing.
