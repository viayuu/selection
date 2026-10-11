# R62: source-separated portfolio-performance pretraining

Date: 2026-10-11. Performance round 4/100. Locked production design.
Plan readiness: 8/10, ready; same-family/provisional, PLAN ONLY.
This is not implementation verification, performance acceptance or a 70% claim.
One mechanism, one nested-data control, no search grid or further review gate.

## Completed R61 and fixed incumbent

R61 completed rc0 at 19:00:56 CST; parent analysis ran once. Its RESULTS.md
confirms both arms had 21,120 stage-1 and 45,504 stage-2 successful updates.
Both fine-tuned for 32 epochs. Strict validation-cost-selected outcomes:

| Arm | Best epoch | Native macro Top1 | Actual macro regret % |
|---|---:|---:|---:|
| R61 A | 24 | 0.4881111111111111 | 0.9249750876386753 |
| R61 B | 32 | 0.49077777777777776 | 0.9290347063357645 |
| R60 B incumbent | 39 | 0.4936111111111112 | 0.9227890652257744 |

R61 B gained 0.2667pp Top1 over A, but lost 0.2833pp versus the incumbent
and had worse regret than both. Its locked advancement screen was false.
This activates the already drafted R62 direction, not a claim that behavioral
learning cannot work or that selector accuracy has reached a ceiling.

Incumbent stays R60 `B_expanded_seed2`, validation-cost-best epoch39:
macro_top1 0.4936111111111112; macro_actual_regret_pct 0.9227890652257744;
macro_vs_sbs_pct -1.0735441392087839. It is a reference, not initialization.
Also report historical R45A: 49.11666667% Top1 / 0.9260483583% regret.

## Paths and single hypothesis

- ROOT: `code/V4/runs/R62_portfolio_performance_pretraining` from the repo root,
  or `runs/R62_portfolio_performance_pretraining` relative to `code/V4`.
- A: `A_subset_seed2`; B: `B_full_seed2`.
- Each arm uses `pretrain/` for stage 1; the arm root holds stage-2 checkpoints,
  histories, result.json and validation predictions. Parent writes RESULTS.md.

Pretrain the complete original R45A instance-solver interaction representation
on real saved comparisons across the auxiliary deployment portfolio, then reset
the auxiliary utility readout and fine-tune on original full18/fullpool labels.
The hypothesis is that broad comparative algorithm-response supervision across
independent inputs transfers better than two-method route imitation or local
pair supervision. No new architecture, static features, solver execution,
teacher pseudo-labels, route head, retrieval or candidate gate is introduced.

## Existing assets and source boundary

Auxiliary root: `/public/home/shiys/训练数据_每类10万`.
Existing sources are `标签运行进度.csv`, `标签配方与校准记录.csv`,
`标签配方与校准说明.md`, published tables and already saved cost shards.
The candidate's recorded snapshot had 101/128 columns complete at 90k additions
and 27 incomplete. All 15 MVRP tasks had 6--7 complete columns each; TSP, CVRP
and ATSP also had saved observations. This is not a claim of 128 complete
columns, complete joint rows, or current queue completion.

Use only new TRAIN inputs outside the original 10k prefix, with actual input,
row and method identities and corresponding saved observed cost cells. During
ordinary CPU cache conversion freeze the then-available usable observations
for BOTH arms; ignore subsequent arrivals and do not wait for queues. Record
actual per-task rows, observed cells and comparison-bearing coverage as normal
conversion output. Unreadable, failed or missing observations stay masked;
do not fill them, solve replacements or infer their ranks. Existing source and
needs-investigation limitations remain disclosed, not silently certified away.

Expansion is an explicitly DIFFERENT auxiliary deployment source. The existing
39 historical sample matches / 89 differences do not certify equivalence.
RF/TW/backhaul constraints, sampling and inference budgets may differ. Neither
matching small calibration samples nor unrecovered historical recipes establish
all-instance equality. These observations are usable transfer knowledge, not
replacement legacy costs, winners or a repaired common feasible domain.

Keep stage-1 auxiliary labels separate from stage-2 legacy labels. Use actual
recorded auxiliary profiles for any recipe-dependent descriptors; unknown fields
must not be guessed or described using an incompatible historical recipe.
Use the same auxiliary encoding in both arms, without adding new feature fields.
Original-TRAIN geometry statistics are shared throughout. Preserve existing
input overlap/VRPLTW repair limitations without a new provenance campaign.

Full18 tasks, 22 solver identities and 128 legal deployments are retained.
All task-legal candidate tokens remain in each forward, including candidates
whose auxiliary costs are missing. Masks restrict supervision, not inference
pools. Query costs, ranks, winners, routes and row identities are never model
inputs. No validation/test answers, diagnostic routes or test data are read.
R58 remains deferred; the legacy B semantic discrepancy remains unresolved.

## Necessary comparison and common initialization

A uses a fixed actual-size-stratified 10k NEW auxiliary queries per task, nested
inside B's new-input bank and selected without cost values/winners. Include
every actual-size stratum needed by B's schedule. Use existing source/availability
metadata in constructing the subset and report its observed-cell coverage.
B uses all available new inputs, up to 90k per task, with their actual masks.
Both use the SAME expansion source, deployment definitions and observation
snapshot: this is not old-label-recipe A versus changed-recipe B.

Both start from the identical fresh seed2 original R45A ScoreDifferenceSelector,
using `r45_experiment.parameters('A')`. Sampling RNG is separate from parameter
initialization. Preserve the existing instance/solver encoders and joint blocks,
raw/local geometry and all task/constraint fields. No R59 native weights or
R60/R61 checkpoint initialization. All parameters are trainable in both stages.

B retains its natural input distribution. A samples its fixed subset with
replacement to match B's task and actual-size vector for each stage-1 batch,
including tails/padding shapes. Match forwards and successful optimizer-step
budgets, not just epochs. Missing-cell masks may give different numbers of
observed comparisons; report these rather than claiming identical labeled
targets or inventing them. There is no third newly trained baseline.

## Stage 1: fixed portfolio-performance exposure

Two complete shuffled traversals of the 90k new-input positions per task,
18 tasks, batch128 including 16-row tails: 12,672 steps/pass and 25,344
successful steps/arm. This counts input exposure, not complete fullpool labels.
Use the actual ordinary-conversion bank; no requirement to finish missing costs.

For each query, form every unordered pair of observed finite FP64 costs with
exactly unequal values. Apply BCEWithLogits to `u_i - u_j`, target `1[c_i<c_j]`.
Average over the query's observed nontied pairs, then over the batch. Exact
cost ties and rows with fewer than two comparable observations contribute zero
comparison loss. Log comparison-bearing presentations separately from total
scheduled steps. No sampled full winner, oracle normalization, auxiliary CE,
risk target, route loss, hard-pair mining, auxiliary R-Drop or loss-weight scan.
One complete-pool forward and one optimizer update per scheduled batch.

AdamW LR1e-4, WD1e-4, dropout .1, clip1. Linear warmup for the first 12,672
steps to 1e-4, then cosine decay for 12,672 steps to 1e-5. Same existing AMP
behavior in both arms. Retries repeat the same batch/RNG without advancing
the schedule. Transfer the FINAL checkpoint at step25,344; no stage-1 validation
selection, early stopping, extra passes or post-outcome rescue.

## Stage 2: untouched original selection task

Transfer the instance encoder, solver encoder and trainable joint interaction
blocks from each arm's final stage-1 state. Reset ONLY the auxiliary utility
`score_head` to identical fresh original initial values in both arms, to relearn
legacy-source score calibration; no head-design change. Reset optimizer/scaler
and common fine-tuning RNG. All parameters remain trainable. Final architecture
and single-solver score-difference/maximin inference stay original R45A.

Use ONLY original 10k fullpool TRAIN/task and original validation. No auxiliary
performance/route branch remains. Keep exact R43 two-dropout-forward objective:
each forward `.35*CE + .35*comparison + .02*risk`, original focus weights,
risk scaling/clamp and predicted-top3 construction; average the two and add
original symmetric R-Drop ramped to .35 over three epochs. One optimizer update.

Seed2, original paired_schedule, batch128/tails/full coverage; identical original
batches in A and B. AdamW LR1e-4, WD1e-4, dropout .1, clip1, warmup3. Reuse
original CostController: effective improvement .001 percentage point, LR
patience3/factor .5, min15/max40 epochs, stop patience8. A defines the COMMON
stage-2 LR trace and horizon E. B follows exactly E, without independent
stopping/extension. Both have 1,422*E successful stage-2 updates.

Each independently selects STRICT minimum original full18 validation
macro_vs_sbs_pct; ties retain the earlier epoch, not best-Top1 selection.
Use original solver metadata/labels/native winners and stable global-ID ties.
Stage/cursors/RNG follow existing resume conventions; stage-1 counts and
optimizer state must not leak into stage2. No multiple-solver minimum at inference.

## Locked decision

Advance selected B only if ALL hold, using unrounded validation metrics:
- Full18 native macro Top1 >= 0.5236111111111112 (R60B +3pp).
- Full18 native macro Top1 >= selected A +0.02 (A +2pp).
- Full18 actual macro regret <= selected A AND <= 0.9227890652257744%.

Report complete curves and all18/family metrics: native Top1, actual regret,
macro_vs_sbs_pct and raw mean cost, with R60B/R45A references. Auxiliary pair
accuracy/loss is diagnostic only. NO TEST ACCESS, even if this screen passes.

B-over-A supports increased independent auxiliary coverage under this recipe,
not equivalence of deployment definitions. Both improving equally does not
establish a data-scaling benefit. Better auxiliary fit without legacy selection
gain is a transfer failure, not a reason to change losses/heads/seeds. Preserve
partial gains when the substantial-gain screen fails. One seed is a direction
screen, not replication, a universal negative result or evidence of 70% Top1.

## Material risks and minimal execution workflow

R54--56 supplied contrary evidence for structural/relational mechanisms; R59
native transfer failed. R60 gave only +.2444pp versus R45A, and R61's expanded
route transfer also failed its screen. Broad direct performance supervision is
the specific new hypothesis, not evidence that generic pretraining will work.
Changed constraints/budgets/stochastic outcomes can cause negative transfer;
easy dominated pairs or task/size priors can absorb the learning signal. Resetting
the utility readout cannot guarantee removal of those effects.

Sol handles normal CPU conversion and implementation. At most ONE necessary
loader-boundary check before immediate formal training: the actual loader must
associate new TRAIN inputs with the correct task/row/candidate cost cells and
masks, exclude the old prefix and validation/test, and preserve pair direction.
Conversion coverage reporting is ordinary loading, not a separate audit. No new
reviews, tests/replays, GPU preflights, timing pilots or provenance campaign are
required by this plan. Report a real unresolvable input association problem;
do not silently reconstruct costs, change tasks/pools or add a rescue branch.

Run A pretrain -> A finetune -> B pretrain -> B finetune. Use easynco, tmux,
existing offline W&B/resume patterns and ONLY assigned available RTX3090 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` on gpu03. No queue/asset/process
changes, installs, solver runs or new annotations. Planning estimate 3--6 GPUh
for both arms/stages plus CPU cache conversion, not measured runtime or a limit;
normal training records actual throughput. No GPU-hour cutoff. Parent analysis
once after both arms complete; ONE commit/push for the completed whole version,
no intermediary commit. This planning action starts no implementation or GPU job.
