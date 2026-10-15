# Sol Next Experiment Decision

Date: 2026-10-15. Author/decision owner: Solmax, 6.1Sol max.
Status: CONDITIONAL GO for ONE bounded mechanism trial; NO EXECUTION NOW.
Finish NSS delivery first, then require explicit owner/resource approval.
Astra supplied literature/ideas only, not this protocol, budget, gate or review.

## Decision And Evidence Boundary

There is one sufficiently different, implementable hypothesis worth a finite
trial: train the existing instance representation so its cross-instance
geometry reflects the complete relative candidate-cost profile. Keep B's
architecture, ordinary routing objective and single-network inference.

This is not another bias, temperature, frozen-score head, ensemble, solver
quota, ReLD-only repair, new input role or solver/data acquisition. Rounds001-005
stay closed. Round003 already changed trainable features through a different
within-instance loss; it is NOT evidence that merely unfreezing B is novel.
The new distinction is explicit supervision of relations BETWEEN instances.

R77B substantially reduced the original cold-choice scope. All five R78 trials
failed acceptable net-gain gates, so the prior probability of benefit is low.
Neither cold winner support nor a small TRAIN/VAL aggregate gap proves that
remaining advantages are observable, or diagnoses gradient starvation. This
proposal is an inductive constraint on existing labels, not additional
information or an established explanation of those failures.

Evidence used: PAPER_IDEAS_ONLY.md; the R77B/R78 results already supplied in
this thread; and narrow interface reads in pairwise_selector.py,
dual_stream.py, pairwise_objective.py and r76_experiment.py. No old statistics
were recomputed, no NSS comparison file or data was inspected, and no tests,
model forwards, resource probes or external research calls were run here.

## Why The Minimal Path Exists

- DualStreamSelector.encode_state already produces pooled h, after the joint
  instance/solver layers; encode_pairs repeats that h in the first embedding
  block of each candidate's features.
- A new R78-local selector/objective wrapper can expose h while reproducing
  ScoreDifferenceSelector's existing utility, margins, masks and maximin logits.
  No new trainable projection, prediction head or inference branch is needed.
- Current compare/risk terms compare candidates within one instance. R-Drop
  compares two dropout views of that instance. They do not impose the proposed
  cross-instance target-distance ordering.
- Existing R77 data/sampler and R76 RNG capture/replay provide the required
  inputs. The extra implementation is a profile loss and logical-batch gradient
  cache, not a rewrite of the backbone, R77, R79 or their running launchers.

## Exactly One Method

For each task t, use its full fixed legal candidate pool J_t and original
positive FP64 TRAIN costs C_ij. Construct TRAIN-only targets in FP64:

    C*_i = min_j C_ij
    u_ij = log(C_ij / C*_i)
    q_ij = u_ij - mean_{j in J_t}(u_ij)
    d_ik = mean_{j in J_t} abs(q_ij - q_kj)

This preserves within-instance preference order and relative log gaps, removes
common instance scale, and compresses extremely costly candidates. It does
not guarantee decision-relevant neighborhoods: bad candidates may still
dominate distance, and narrow winner boundaries may still be smoothed away.
No learned normalization, clipping, target search or frequency balancing.

Use pairs only within the SAME task and SAME fixed candidate pool. Never
compare unlike tasks or impute missing candidate costs. The native TRAIN
path must provide its declared fixed pool; an incompatible artifact is an
implementation/precondition failure, not permission to silently change it.

Let e_i = normalize_L2(h_i), using the usual fixed 1e-12 zero-norm guard.
For each existing dropout view independently, use the following explicitly
specified Rank-N-Contrast-inspired loss, not a claim of verbatim paper code
or transfer of its theorem:

    s_ik = dot(e_i, e_k)                 # fixed training temperature1
    L_profile = mean_{i != j} [
        logsumexp_{k != i, d_ik >= d_ij}(s_ik) - s_ij
    ]

N=1024 is the complete logical task batch, not the physical micro-batch.
Exclude self-pairs; exact target-distance ties share the full >= denominator,
with no tuned epsilon. Implement by sorting target distances and gathering
tie-group-end log-cumulative-sums: O(N^2 log N), not N^3 denominators or
N*N*embedding_dim tensors. Feature/loss arithmetic is FP32; costs, q and
target-distance ordering originate in FP64 and never enter model.forward.

    L_new = L_R77_base + 0.10 * (L_profile(view1)+L_profile(view2))/2

0.10 is the sole new tradeoff coefficient, fixed now without a norm-matching
pilot or scan; it is a design choice, not an estimated optimum. Keep original
CE/compare/risk, their coefficients/focus rules, and R-Drop0.35 unchanged.
No CNC proxy groups, spectral-decoupling term, new augmentations, teacher,
memory bank, output calibration or new supervision branch is added.

## Logical1024 And The Real Implementation Risk

The present micro-batch accumulator cannot obtain this loss by computing
separate within-micro contrastive terms. Retaining all encoder graphs would
also defeat the established memory cap. Use exact feature-gradient caching:

1. With parameters unchanged, run the two ordinary TRAIN dropout forwards
   over all micro chunks without encoder graphs. Save each chunk's RNG state
   and both h vectors; preserve the final RNG state.
2. On the complete1024 detached feature leaves, calculate the symmetric
   profile objective and gradients for BOTH anchor and key contributions.
   Average the two dropout-view losses and include0.10 here.
3. Replay each chunk with its saved RNG. Backpropagate the original base loss
   weighted by chunk_rows/1024, plus sum(h_replay * cached_feature_gradient).
   The cached auxiliary gradient ALREADY includes global averaging: do not
   multiply that dot-product term by chunk_rows/1024 again.
4. Unscale once, clip global gradient once at1, and take ONE optimizer step.
   Restore the post-first-pass RNG state, so logical RNG progression remains
   the base two-view progression. Overflow replays the same whole batch with
   unchanged weights/sampler, using the inherited finite retry bound.

No stop-gradient keys, partial contexts or memory-bank approximation are
authorized as a fallback. If exact replay cannot be supported, stop this
implementation rather than silently substitute a different objective.

Only three necessary future checks: a tiny profile/tie/mask example; tiny
full-batch versus cached/replayed gradient equivalence including dropout; and
one real finite budget-update1. No gradient-stationarity/optimum certificate,
18-task code audit, broad profiling or separate discarded optimizer smoke.
These checks have NOT been executed by this planning task.

## Fixed Data, Initialization And Budget

- All18 native tasks /89 candidate entries. Existing full-B TRAIN70114 rows
  per task and VAL3895 per task,70110 VAL total. No new data or solver calls.
- Start ONLY from locked B ep28/u34776, checkpoint SHA256
  31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b.
  Preserve B's original TRAIN-derived preprocessing and solver representations.
  This is a new run initialized from B, not an old-goal/R74/R58 resume.
- Train the existing backbone/joint layers/pool/score head. No frozen-score
  input bottleneck and no new inference parameters. Raw instance inputs are
  needed for training; cached logits alone cannot implement this mechanism.
- Exactly14904 successful continuation updates:828 per task,1024 examples per
  update, seed2, natural per-task continuous permutations filling boundaries.
  Reuse round003 control's task order and fresh sampler initialization.
- Fresh AdamW betas(.9,.999), eps1e-8, weight_decay1e-4 and fresh GradScaler,
  matching that control. Reuse its fixed round003/late_lr.npy sequence,
  first2.4543907940693327e-5 and last1e-6; no warmup/restart/extension.
- One single-GPU branch only, no DP or new control training. Initial physical
  micro cap TSP/CVRP352, others1024. Global mean/clip/step semantics remain1024.
- Operational hard ceiling:4 GPU-hours INCLUDING the budget-update1 check,
  fitting and one final full TRAIN/VAL readout. Planning estimate on the
  previously used3090 class: roughly2-3.5 GPU-hours, not a new benchmark or
  guarantee. Feature caching adds forwards and label-distance/sort work.
  A wall/numerical/resource stop before completion is INCOMPLETE, not a
  negative scientific conclusion, and does not authorize extra budget.
- No resource reservation or launch now. NSS delivery has priority. Only
  after explicit owner assignment may execution use one approved UUID inside
  its existing exact Slurm allocation; never SSH-launch GPU work.

## One Checkpoint, Validation And Formal Release

Save/lock ONLY the unique complete18 final update14904 candidate, its SHA,
preprocessing, pool/mask definitions and unchanged native maximin/global-ID
inference BEFORE candidate VAL metrics are opened. No intermediate candidate
VAL, best-epoch menu, taskwise selection, coefficient change or seed search.
One final complete TRAIN and VAL readout is in the above wall budget.

Use unchanged unrounded locked B records, not rounded displays. The inherited
full18 equal-task macro gate is conjunctive and has zero tolerance:

    candidate native Top1 > B native Top1
    candidate FP64 raw mean cost <= B raw mean cost
    candidate actual relative regret <= B actual relative regret

Reference B VAL: Top1 fraction0.7089430894308943,
mean cost11.395558343544208, actual regret0.40132674544775454%.
CE, profile loss, macroVS, cold-method counts or one-task gains are descriptive,
not replacements for this gate. Keep native labels, all pools, row alignment
and original cost definitions. Report actual new candidate separately from B
and whole-version B fallback, all18 per-task metrics/native method counts,
corrected/harmed/wrong-to-wrong choices and signed FP64 cost changes.

Reuse the ALREADY completed round003 control at its actual final+14904 VAL
endpoint as a descriptive matched-update comparator, NOT its selected B
fallback. Its known Top1 fraction0.7081443446013407, mean cost11.395855343252101
and regret0.41092598253845325% are already worse than B. No recomputation,
retraining, new checkpoint choice or missing control-TRAIN metrics is implied.

Any completed gate failure closes THIS fixed profile-loss trial and retains
whole B. Do not rescue it with another profile, temperature, feature layer,
loss weight, optimizer, duration or task subset. A pass locks one exploratory
candidate and returns its complete release manifest to the owner. It does
NOT authorize TEST/LIB. Only a later explicit approval allows one complete
TEST evaluation of that locked version; no TEST-based revision or second
attempt. LIB, if separately approved, uses identical weights/rule/preprocessing.

## Limits And Stop/Go Ownership

One seed and repeatedly reused VAL do not establish generalization, per-task
safety, statistical non-inferiority or universal observability of rare solver
advantages. Improvement over B alone confounds extra training with the added
loss. Historical control helps constrain that explanation but is not a fresh
randomized comparison: protocol/kernel/code drift and prior VAL exposure
remain limitations. No causal feature-starvation or paper-level mechanism
claim is approved by this finite screen. Negative outcomes do not prove an
information limit or exact optimization failure.

Decision today: the mechanism and implementation boundary are concrete enough
for ONE later owner-approved trial, not an automatic next experiment. Await
NSS locked TRAIN/VAL and its normal delivery; Sol/Feynman inspect that handoff.
The owner can cancel this entire trial if it is unnecessary after that delivery,
not retune this recipe using NSS TEST/LIB. No further Astra planning, budget,
gate-setting or result review is required or requested.

Execution status: NOT STARTED. No code or training artifact was changed here.
R78 file workflow stays active/pending; native and legacy goals are untouched.
