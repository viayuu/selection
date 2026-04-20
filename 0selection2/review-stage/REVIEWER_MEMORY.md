# Reviewer Memory

_Persistent across rounds. Loaded at the start of each round._

## Round 1 — Score: 7.2/10 (from prior session, oracle-pro)

- **Verdict**: Almost — "basic approach is sound but three concrete gaps to address before re-assessment"
- **Three prescribed fixes**:
  1. **SBS gate calibration** — confidence-gate the selector's pick against SBS so low-confidence cases collapse to SBS. Reason: selector occasionally ties or slightly under-performs SBS when uncertain; the calibrated gate closes the gap from +0.16% to near 0.
  2. **Combined cost loss** — weighted sum of CE (regret-soft) + expected-regret + switch-hinge. Reason: regret-soft CE alone under-penalises when selector chooses a dominated solver far from the oracle; the switch-hinge explicitly rewards picking the true winner *over* SBS whenever the instance-level VBS-SBS gap exceeds a threshold (2e-3 rel).
  3. **Per-problem per-solver learned bias + oversampling** — add a `problem_solver_bias ∈ R^{P×M}` table (init 0) and oversample the 10 under-performing problems (CVRP×4, VRPB×3, OVRPB×3, OVRP×2, rest ×1.5). Reason: the head-MLP shares weights across all 18 problems; some problem-solver-priors are hard to learn from pooled gradients; a tiny bias table + balanced sampling gives each problem a dedicated knob.
- **Suspicions flagged**:
  - Selector might be learning problem-ID priors rather than instance structure (already falsified by R2 MetaOnly baseline = +75.23% vs R1 +0.16%).
  - 2-epoch pilot may not generalize — oracle wants multi-seed + extended training.
  - ATSP pool has only 3 arms; risk that selector collapses to SBS there. This is OK if macro beats SBS elsewhere, but should be reported honestly.
- **Unresolved going into Round 2**:
  - Does the combined loss actually help vs regret-soft alone? (isolation run needed)
  - Does oversampling hurt under-sampled problems or help them? (check per-problem deltas)
  - Are all 观测指标.md metrics computed? (top1/2/3, per-method win/lose lists, arm distribution, loss curve, and the 3 bar-chart figures)
  - Gate calibration alone brought macro from +0.160% to +0.014% on the 2-epoch pilot; need to confirm this stacks with the other fixes.

## Round 2 — Score: 6.4/10 (oracle-pro browser)

- **Verdict**: Not ready.
- **Main recommendation**: center the next round on **A = gated inference as the actual method**, with conservative SBS fallback and held-out calibration.
- **Validated fixes landed**:
  - `meta-only` evaluation now zeroes instance features during both train and eval/analyze.
  - `analyze.py` now correctly consumes `gate_eval.json -> gammas`.
  - fractional oversampling weights are preserved in `train.py`.
  - `train.py` has an experimental `--schedule interleaved` mode for cross-problem mixing.
- **Corrected metrics after those fixes**:
  - R1 ungated: val `+0.159%`, test `+0.228%` vs SBS.
  - R1 gated: val `+0.014%`, test `+0.052%` vs SBS.
  - R2 meta-only: val `+7.861%` vs SBS, so instance features still matter materially.
- **Critical weaknesses carried forward**:
  1. Raw selector still does not beat SBS; it mainly survives by abstaining back to SBS.
  2. CVRP / VRPB / OVRPB remain the main empirical sink.
  3. Combined-loss + bias + oversampling currently looks like a dead end unless a bug is found.
  4. Longer training hurts, so checkpoint selection / training stability is unresolved.
  5. Gating is promising, but needs proper calibration split, switch diagnostics, and bootstrap CIs.
- **Minimum fixes requested by reviewer**:
  1. Reuse `R1_seed0`; evaluate gated inference rigorously on held-out calibration/test splits with switch-rate / switch-precision / switch-benefit diagnostics.
  2. Add conservative per-problem fallback-to-SBS rule when calibrated switching is not positive.
  3. Try cheap cost-aware gate scores before any new training.
  4. Pause R3 objective tuning until the failure mode is diagnosed by lightweight logging/analysis.
  5. If training more, replicate only the simple winning recipe over 3 seeds and report CI.
- **Operational caveat**:
  - Launching two interleaved jobs in parallel triggered a container OOM kill, so interleaved training remains unvalidated from an empirical-results standpoint.

## Round 3 — Pending external score

- **Implemented exactly the reviewer-requested cheap path**:
  - no new selector training;
  - held-out calibration from `val`;
  - final evaluation on `test`;
  - per-problem score-family search over `margin`, `prob_gap`, `margin_over_entropy`;
  - conservative fallback-to-SBS when calibration improvement does not exceed one bootstrap SE.
- **New best deployment result**:
  - ungated test `macro_vs_sbs = +0.228%`
  - calibrated/fallback gated test `macro_vs_sbs = +0.011%`
  - bootstrap 95% CI on gated test `macro_vs_sbs = [-0.004%, +0.026%]`
- **Interpretation update**:
  - This is no longer just “gate helps on val”; it is now a held-out test result consistent with the safe-selector framing.
  - The strongest story is now explicitly: **SBS by default, learned switching only where calibrated benefit is supported**.

## Round 4 — Score: 7.1/10 (oracle-pro browser)

- **Verdict**: Almost.
- **Main conclusion**: the paper is now coherent if and only if it centers on **gated SBS-fallback** as the main deployed method.
- **What changed this round**:
  - added 2 more seeds for the simple recipe
  - aggregated 3-seed raw and gated test results
  - generated full best-seed test analysis for the gated policy
- **Updated empirical picture**:
  - raw multi-seed test `macro_vs_sbs = +0.105%` mean
  - gated multi-seed test `macro_vs_sbs = +0.005%` mean
  - best-seed gated test `macro_vs_sbs = +0.0003%`, `match = 18/18`
- **Reviewer-approved framing**:
  - raw selector becomes an ablation / failure mode
  - gated SBS-fallback becomes the main method
  - contribution should be phrased as **safe selective switching** rather than macro improvement
- **Reviewer-requested final cheap fix before “READY”**:
  1. tighten the enable rule one more notch (e.g. `2 bootstrap SE`)
  2. add a compact switch-diagnostics table
- **Explicitly deprioritized**:
  - no more combined loss
  - no more oversampling
  - no longer training longer
  - no interleaved schedule tuning
  - no larger raw models

## Round 5 — Score: 7.5/10 (oracle-pro browser)

- **Verdict**: Ready, but only as a modest conservative paper.
- **What changed this round**:
  - strict gate = `2 bootstrap SE` + calibration precision floor
  - 3-seed strict-gate aggregate
  - compact switch-diagnostics table
- **Updated empirical picture**:
  - strict-gated 3-seed test `macro_vs_sbs = +0.001%`
  - seed-bootstrap CI `[+0.000%, +0.003%]`
  - enabled problems per seed ≈ `1.0`
  - mean switch precision on enabled problems ≈ `0.348`
- **Reviewer-approved interpretation**:
  - this is now `READY` if framed as a **strict gated SBS-fallback** paper
  - the contribution is safety / calibrated abstention, not raw performance gain
- **Minimum non-compute fixes still worth doing**:
  1. add a per-problem enable/fallback table
  2. make the “mostly SBS” behavior explicit in the narrative
  3. demote raw selector to an ablation / failure mode
  4. add a limitation paragraph about 3 seeds and conservative scope
- **Explicitly not worth doing**:
  - any new selector training
  - relaxing the gate to inflate the result
  - more score-family tuning
  - more architecture work

## Round 6 — Score: 7.6/10 (oracle-pro browser)

- **Verdict**: Almost.
- The conservative strict SBS-fallback story is the right paper.
- The main blocker was a paper-facing trust issue in the strict per-problem table.
- Minimum fix:
  - separate enabled seed count from enabled seed ids
  - define positive `vs_sbs` as degradation relative to SBS
  - state clearly that the deployed policy is mostly SBS and abstains by design
- No new training, larger models, longer schedules, or relaxed gates are worth doing.

## Round 7 — Score: 7.6/10 (oracle-pro browser)

- **Verdict**: Almost.
- The macro story is now coherent, but the switch-diagnostic denominator was still ambiguous.
- Minimum fix:
  - label seed-averaged diagnostics explicitly, or recompute truly enabled-only diagnostics
  - rename `selective-switch` to `calibration-enabled`
  - replace `SBS-safe` with near-SBS-parity strict fallback wording
- Remaining work is claim hygiene and table-caption consistency, not compute.

## Round 8 — Effect-Focused Experimentation

- User redirected the loop away from paper framing and toward **actual raw-selector performance**.
- New main line:
  - `soft_sbs_risk + interleaved + sbs_bias_init=0.5`
  - quick 3-seed held-out tests all reached near-SBS parity with `match = 18/18`
- Strong negative result:
  - full-data `sequential` training is unstable and can catastrophically blow up `TSP`
- Tried and did **not** beat the new near-parity line:
  - richer pooling / size features
  - problem FiLM + larger capacity
  - challenger-only threshold calibration
  - switch-hinge on top of `soft_sbs_risk`
  - lower risk weight / lower SBS bias / no problem-solver bias
  - explicit deterministic global stats alone
- Current interpretation:
  - objective + schedule mattered much more than extra capacity
  - the low-cost search space now mostly converges to **near-SBS parity**
  - the next serious direction should be better switch-worthiness signal / representation, not more small loss nudges in the same regime

## Round 9 — Two-Stage Improvement Line

- New strongest raw-selector recipe:
  - Stage 1: `soft + interleaved + global_stats`
  - Stage 2: initialize from Stage 1, then run `soft_risk` with:
    - `risk_weight = 1.0`
    - `soft_ce_weight = 1.0`
    - `sbs_bias_init = 0.0`
    - `global_stats = true`
    - `schedule = interleaved`
- Seed 0:
  - val `macro_top1 = 0.5187`
  - val `macro_vs_sbs = -0.086%`
  - test `macro_top1 = 0.5190`
  - test `macro_top2 = 0.8724`
  - test `macro_top3 = 0.9831`
  - test `macro_vs_sbs = -0.070%`
  - test `macro_vbs_gap_closed = +2.59%`
  - main gain concentrated on `ATSP = -1.351%` and `TSP = -0.0765%`
- Seed 1 partial replication:
  - Stage-2 val `macro_top1 = 0.5183`
  - Stage-2 val `macro_vs_sbs = -0.073%`
- Seed 2:
  - queued and running in `tmux` on `gpu1`
- Important engineering fix:
  - [analyze.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/analyze.py) and [gate.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/gate.py) now reconstruct `global_stats` / `rich_pool` / `problem_film` / `size_feature` correctly from checkpoint args
- Updated interpretation:
  - the previous near-parity ceiling was not absolute
  - the missing ingredient was a **discriminative warm start plus balanced risk fine-tuning**
  - this is now the best raw-selector direction to continue, ahead of more gate-only tuning

## Round 10 — Screening Against The New Champion

- Multi-seed held-out confirmation for the current best line:
  - `macro_top1 = 0.5221 ± 0.0022`
  - `macro_vs_sbs = -0.081% ± 0.012%`
  - `macro_vbs_gap_closed = +3.60% ± 0.94%`
- New negative result:
  - 8-fold coordinate augmentation (`--coord-augment 8`) did not beat the current best line
  - stage-2 val with augmentation: `macro_top1 = 0.5159`, `macro_vs_sbs = -0.068%`
- New negative result:
  - lightweight constraint-expert residual head did not beat the current best line
  - stage-2 val with experts: `macro_top1 = 0.5157`, `macro_vs_sbs = -0.052%`
- Important engineering additions:
  - train-time 8-fold coordinate augmentation in [data.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/data.py)
  - `--constraint-experts` path in [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py)
- Current active mainline:
  - full-data continuation of the best seed-2 two-stage recipe
  - `R8_softstats_cont_full_seed2` -> `R8_init_softstats_soft_risk_cont_full_seed2`
  - monitor log: [selector_monitor_r8.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r8.log)
- Already queued next candidate:
  - `R9_hier_softstats_pretrain_seed2` -> `R9_init_hier_softstats_soft_risk_bal_seed2`
  - this tests NSS-style hierarchical pooling on top of the current best two-stage training recipe
- Further queued candidates:
  - `R10_rank_hier_softstats_pretrain_seed2` -> `R10_init_rank_hier_softstats_soft_risk_bal_seed2`
    - this tests NSS-style ranking supervision on top of hierarchical pooling
  - `R11_problemhead_softstats_pretrain_seed2` -> `R11_init_problemhead_softstats_soft_risk_bal_seed2`
    - this tests a shared backbone plus per-problem residual scorer, optimized for raw top1 gain
- Operational note:
  - `num_workers = 8` caused the first full-data launch to be OOM-killed because the trainer instantiates one loader per problem
  - `num_workers = 2` was still not robust enough under the container memory cgroup
  - current stable mainline launch uses `num_workers = 0` and a short `tmux` session command
- Updated interpretation:
  - current best recipe is robust enough that new ideas must clear a stronger bar
  - quick literature-inspired add-ons are easy to underperform against the new champion
  - the right next compute spend is longer training on the confirmed winner, not immediate promotion of weaker variants

## Round 11 — Codex Review Pivot + New Mainline

- Codex reviewer summary:
  - score `5/10`
  - the current raw-selector line is real progress, but still too SBS-safe
  - the highest-ROI next work is:
    - `gap-regression + ranking`
    - manual problem-structured features
    - real problem-conditioned adapters
- Fresh strongest single-seed checkpoint:
  - [R8 stage1 test](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R8_softstats_cont_full_seed2/analysis_test/analysis_test.json)
  - `macro_top1 = 0.5276`
  - `macro_vs_sbs = -0.118%`
  - `macro_vbs_gap_closed = +6.76%`
- Important failure:
  - `R8_init_softstats_soft_risk_cont_full_seed2` died by memory-cgroup OOM around `ep0 step4100`
- New active mainline:
  - `R12_gaprank_manual_adapter_seed2`
  - then `R12_init_gaprank_manual_adapter_soft_risk_seed2`
  - current monitor: [selector_monitor_r12.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r12.log)
- Engineering interpretation:
  - keep the better `R8` stage-1 checkpoint as the warm-start anchor
  - stop treating further `soft_risk` polishing as the main route
  - push the supervision target closer to `mean cost`

Additional short-term outcomes worth remembering:

- `TTA-8` on `R8` is a negative result:
  - [analysis_test_tta8](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R8_softstats_cont_full_seed2/analysis_test_tta8/analysis_test.json)
  - `macro_top1 = 0.5274`
  - `macro_vs_sbs = -0.117%`
  - slightly worse than the non-TTA `R8` result
- `R12` stage-1 epoch0 is healthy but not a new best:
  - [eval_epoch0.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R12_gaprank_manual_adapter_seed2/eval_epoch0.json)
  - `macro_top1 = 0.5188`
  - `macro_vs_sbs = -0.081%`
- `R13` is already queued as the next follow-up:
  - `gap_rank + manual_features + problem_adapter + hard_case curriculum`

OOM / operational lessons:

- `R12` stage-1 promising result at `eval_epoch0` did not complete; it was later killed by container memcg OOM
- the first `R12` stage-2 rescue also died by memcg OOM around `step450`
- removing `--wandb` is important here because even failed uploads spawn helper processes that waste memory
- current safer rescue for `R12` stage-2 uses:
  - `batch_per_problem = 8`
  - `num_workers = 0`
  - no `wandb`
  - monitor: [selector_monitor_r12_stage2.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r12_stage2.log)
- future queued scripts are hardened accordingly:
  - `R13` and `R14` no longer use `wandb`
  - `R13` and `R14` use `grouped_interleaved`

Training-system root cause now understood:

- the repeated OOMs were caused by CPU-side container memory, not GPU VRAM
- old training kept too many datasets/loaders resident simultaneously
- fixed in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py) by:
  - on-demand val loading
  - label-only support-stat construction
  - `grouped_interleaved` training with configurable group size
- current proof point:
  - `R12` stage-2 rescue with `grouped_interleaved(group_size=4)` is stable past the previous kill region
  - observed RSS dropped to about `3.1-3.5 GB`

Latest architectural pivot:

- The current bottleneck is no longer “the shared backbone cannot rank at all”; it is that many oracle-winning long-tail solvers never enter the final competition set.
- New remedy now implemented in code:
  - `support_head` inside [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py)
  - shortlist supervision in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py)
  - shortlist-aware inference and coverage metrics in [analyze.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/analyze.py)
- Intended effect:
  - first decide which solvers deserve to enter a shortlist
  - then rank only within that shortlist using the main scorer
  - this directly targets the persistent zero-pick / hidden-winner issue instead of only regularizing a single masked softmax
- Next queued long run uses this new path:
  - [run_r14_solveraware_support_seed2.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/run_r14_solveraware_support_seed2.sh)
  - plus watchdog-based handoff via [launch_r14_after_r13.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/launch_r14_after_r13.sh)

## Post-literature bridge findings

- `R16_multigen_support_sanity_seed2`:
  - val improved slightly (`macro_vs_sbs = -0.130%`)
  - but test did **not** beat `R12`
  - critical failure: shortlist degenerated to nearly all-ones
  - `macro_support_top1_recall = 1.0`, `macro_support_arm_coverage = 1.0`
- `R17_multigen_budget_support_sanity_seed2`:
  - budget penalty made shortlist sparse
  - but over-corrected and hurt both top1 and mean cost
  - test `macro_vs_sbs = +0.044%`
- `R18_margin_sanity_seed2`:
  - winner-vs-competitor margin loss is stable
  - but essentially ties `R12`, not a breakthrough
- strongest current empirical clue:
  - `macro_top3` is already around `0.986-0.987`
  - so the next useful fix probably needs stronger *top-1 discrimination among near-top competitors*, not another blunt shortlist trick
