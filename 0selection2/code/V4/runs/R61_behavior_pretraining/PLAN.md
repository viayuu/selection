# R61: dense saved-route behavior pretraining

Date: 2026-10-11. Performance round 3/100. Locked production design.
Plan readiness: 8/10, ready; same-family/provisional, not performance acceptance.
One mechanism and one matched control. Sol implements/executes; this planner
performs no implementation, tests, solver/GPU runs or commit.

## R60 outcome and incumbent

R60 is COMPLETE, both arms 40 epochs / 56,880 successful updates. Validation-cost
best A epoch32: native Top1 48.75555556%, actual regret 0.9311753497%.
Best B epoch39: 49.36111111% / 0.9227890652%. B gains 0.6056pp versus A
and 0.2444pp versus R45A, with slightly better macro regret. Preserve this
partial gain; the substantial-gain screen was false, not a learning ceiling.
Raw mean cost and individual tasks did not uniformly improve. No test access.

LOCKED incumbent: R60 B_expanded_seed2, validation-cost-best epoch39, recorded in
`code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2/result.json`:
- macro_top1 = 0.4936111111111112.
- macro_actual_regret_pct = 0.9227890652257744.
- macro_vs_sbs_pct = -1.0735441392087839.

This is a reference, NOT R61 initialization. Also report historical R45A
49.11666667% / 0.9260483583%. The full18 single-solver 70% goal is unachieved.

## Locked paths and mechanism

- ROOT: `code/V4/runs/R61_behavior_pretraining` from the repository root,
  or `runs/R61_behavior_pretraining` relative to `code/V4`.
- A directory: `A_replay_seed2`; B directory: `B_expanded_seed2`.
- Each arm's `pretrain/` holds stage-1 artifacts; its arm root holds stage-2
  best/last checkpoints, history, result and validation predictions.

Pretrain the original R45A instance encoder on TWO saved ReLD solution
structures per TRAIN instance, then transfer it into unrestricted full18/fullpool
learning. A uses oldTRAIN route replay; B uses independent expanded TRAIN routes.
Dense observed customer-level behavior supplies more supervision than R60's
single comparison bit. This is representation learning, not a pair expert,
candidate gate, external native-weight transfer or another attention/head tweak.

R54 learned approximately 42% held-out successors without improving selection:
this is the main contrary evidence. R61 changes independent coverage, separates
pretraining from selection, and fine-tunes the full pool instead of gating R45's
top two. That change may still fail. No route module, solver probing, predicted
edge-cost sum or additional solver execution exists at deployment.

## Existing data and minimal worker resolution

LEGACY scenario: unchanged original18 tasks, 22 solver identities, 128 legal
deployments, original10k fullpool TRAIN/task and original validation/native
winners. R58 stays deferred; B feasible-domain discrepancy stays unresolved.

Stage 1 covers all15 MVRP tasks. A sources the existing R54 saved TRAIN routes
on original10k/task: 300k recorded method routes. Its diagnostic validation
routes are EXCLUDED. Cost ties do not remove valid structural examples.
B uses existing new90k/task inputs and corresponding MOEL/MTL tours, excluding
the old10k prefix: expected1.35M queries / 2.7M method routes.

Existing30-column progress/calibration reports and the producer's saved `tours`
field support availability, but DO NOT prove every required route body is present
and correctly paired. During ordinary target-cache conversion the data worker
minimally resolves actual route presence, input/method identities, counts and
convertible successor targets. Use actual node counts and recorded identities,
including the existing VRPLTW repair pairing; never infer metadata from index.
Report actual resolved counts. No separate audit, solver replay, route-cost
reconstruction, historical-source certification or new annotation job.

Missing/unusable required route bodies are a real data blocker: report the
task/method/count. Do not silently shrink the roster, substitute a different
deployment or generate replacements. Preserve queues/assets. Existing4/8-sample
calibration, source-provenance and pre-repair data-only no-overlap limitations
remain disclosed. No other historically mismatching expanded columns are used.

## Shared model and initialization

Reuse `r45_experiment.parameters('A')` and original handcrafted-solver
ScoreDifferenceSelector architecture/input adapters. Preserve raw/local geometry,
service, route-limit and task fields. Geometry statistics are fitted ONLY on
original TRAIN and shared throughout. No native R59 encoder, external weights,
new static features, geometry augmentation, retrieval or architecture search.

Stage-1 encoder and training-only route heads start identically from scratch,
seed2. Keep sampling RNG separate from initialization; only encoder/route heads
are optimized in this stage. For stage2, create identical fresh full R45A models
and replace ONLY instance_encoder states with the corresponding final pretrained
states. All other initial values match; all parameters are trainable. Discard
route heads and stage-1 optimizer/scaler states; initialize fresh common
stage-2 optimizer/RNG settings. Preserve original-TRAIN normalization.

## Stage 1: fixed dense behavior pretraining

- B makes TWO complete shuffled traversals of new90k/task, batch128 including
  16-row tails: 10,560 updates/pass, 21,120 successful updates and 2.7M query
  presentations per arm. No adaptive extension or stage-1 early stopping.
- A samples oldTRAIN with replacement, matching B's per-batch task and ACTUAL
  customer-count vector, tails and padding shapes. Equal instances, method
  targets and successful updates. B retains its natural distribution; no
  cost/winner/route-difficulty balancing.
- Reuse R54's directed per-customer successor/END target convention and decoder
  math: self/padded destinations excluded, per-instance successor CE divided
  by log(valid candidate count), then mean the two methods' normalized losses.
  Supervise archived native best tours in original node order. Do not canonicalize
  routes, reassign physical roles, reverse them or assume permutation invariance.
- ONE shared encoder forward supplies both training-only heads. Routes, targets
  and identities never enter encoder inputs. No winner BCE, scalar cost/risk
  loss, edge-difference regression, auxiliary R-Drop or invented fullpool winner.
- AdamW LR1e-4, WD1e-4, dropout.1, clip1. Linear warmup over the first10,560
  updates to1e-4, then cosine decay over the remaining10,560 to1e-5. Same schedule
  and existing AMP behavior in both arms. No LR/loss scan.
- Transfer the FINAL encoder at update21,120. No validation-route access or
  stage-1 checkpoint selection. AMP retries repeat the same batch/RNG without
  advancing the data cursor or successful-update count.

## Stage 2: original full18/fullpool learning

Use only original10k/task fullpool labels on every original task/candidate.
No expanded-pair branch or structural auxiliary loss remains. Preserve original
score-difference/maximin inference and stable global-ID tie breaking, selecting
ONE solver. No query answers, routes or identities enter a forward pass.

Reuse EXACT R43 two-dropout-forward objective: each forward uses
`.35*CE + .35*comparison + .02*risk`, with original comparison focus weights,
risk scaling/clamp and predicted-top3 construction. Average the two losses and
add original symmetric R-Drop, ramping to.35 over three epochs. ONE optimizer
update per original batch, not a second auxiliary update.

Seed2, identical original paired_schedule, batch128 with tails/full coverage.
AdamW LR1e-4, WD1e-4, dropout.1, clip1, warmup3. Original CostController:
effective improvement.001 percentage point, LR patience3/factor.5, min15/max40
epochs, stop patience8. Run A first; its validation controller defines the
COMMON stage-2 LR trace/horizon E for BOTH arms. B follows through exactly E
epochs, without independent stopping or post-result extension. Each arm selects
its own STRICT minimum original full18 validation macro_vs_sbs_pct checkpoint;
equal values keep the earlier epoch. Both have1,422*E successful stage-2 updates.

Stage/epoch/cursors/RNG must resume consistently. Do not carry stage-1 optimizer
steps into stage-2 counts; AMP retries cannot consume extra scheduled examples.

## Locked decision and interpretation

Use each arm's validation-COST-selected checkpoint, never best-Top1 epochs.
Advance B only if ALL hold, using unrounded metrics:
- Full18 native macro Top1 >= 0.5236111111111112 (incumbent+3pp).
- Full18 native macro Top1 >= selected A+0.02 (A+2pp).
- Full18 actual macro regret <= BOTH selected A and0.9227890652257744%.

Report R60B incumbent and R45A, complete validation curves, native Top1, actual
regret, macro_vs_sbs_pct, raw mean cost and family/all18 task metrics. Structural
loss and training fit are diagnostics, not replacement success criteria.
B-over-A supports the value of additional independent behavioral examples under
this recipe. Both improving without B-over-A does not establish data-scaling
benefit. Better successor learning without fullpool gain rejects this transfer
benefit under the recipe, not all structural learning or instance signal.
One seed is a direction screen, not replication. Preserve partial gains even
on screen failure; never declare a70% ceiling or achievement.

NO TEST ACCESS/EVALUATION, even if the screen passes. No task/pool/label changes,
new annotations, solver runs, R58 operations, coefficient search or rescue branch.

## Execution scope and cost

Sol handles conversion and runner implementation. Reuse existing R54 target
conversion and compatible R60 input caches. Necessary data resolution occurs
during conversion, with ordinary first-update observations during execution;
no separate validation/replay campaign, timing pilot or planner-review gate.

Use easynco, tmux, existing offline W&B and epoch-atomic resume. Only available
assigned RTX3090 UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` on gpu03.
No other GPU, foreign-process termination, environment install or queue changes.
Execution order: A pretrain -> A finetune -> B pretrain -> B finetune.
Planning estimate6--10GPUh total plus unmeasured CPU cache preparation, not a
timing result or cutoff. Dense successor prediction is parallel, not route
decoding. Record ordinary first-epoch throughput and actual runtimes. No GPU-hour
limit. One commit/push for the completed whole R61 version, no intermediary
commit; parent separately owns the completed R60 commit.

## Memory update

R60 complete: small Top1/regret gain, advancement false, no test/ceiling claim.
Incumbent locked to R60B epoch39. R61 fixes dense saved-route pretraining,
old replay versus independent new TRAIN, equal stage budgets/shared initialization,
then original full18/fullpool transfer under A-controlled common fine-tune LR/E.
Actual route-body availability remains for minimal data-worker resolution,
without solver regeneration/audits. R54's absent selection gain remains the
main scientific risk. Open: implementation/execution and performance. Plan8/ready
only, same-family/provisional; R58 deferred, B unresolved,70% unachieved.
