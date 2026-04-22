# Findings — Unified Selector Fresh Loop (2026-04-20)

Lightweight log of concrete findings from the fresh auto-review loop. One line per finding.

## Research Findings

- [R1] negative: gated retrieval blend α=0.4 δ=0.08 gains +0.0056 test top1 when α,δ chosen on the test grid; under pre-registration the effect drops to +0.0009 [−0.0044, +0.0065], not significant (test top1 0.5193 → 0.5202).
- [R1] positive: zero-pick subset is 20.01 % of test mass (4714 / 18000) — retrieval rescues 12.72 % of that subset from 0 % base top1 to 12.72 % (metric: ZP top1 0.0000 → 0.1272).
- [R2] negative: R33 binary learned gate is coin-flip on val→test (val wr/rw=1.12 → test wr/rw=1.01, Δtop1=+0.0013 [−0.0056, +0.0081]); features do not transfer.
- [R2] negative: R34 prospective blind-spot override fails because val-defined blind-spot arms don't match test (Δtop1 = −0.0012 [−0.0047, +0.0024]).
- [R2] mixed: R35 dual-retrieval agreement (oracle-vote ∩ cost-rank) narrows the high-confidence subset but global effect stays null (Δtop1 = +0.0032 [−0.0031, +0.0094]).
- [R3] negative: R38 train-time fusion head (cross-fitted train priors, learnable blend weight) misses all pre-registered thresholds — test Δtop1 = +0.0029 [−0.0030, +0.0087], wr/rw = 1.036 (threshold 1.08 MISSED), Δcost % not sig.
- [R3] positive (central figure): R39 gross rescue/harm decomposition shows ZP harm ≡ 0, non-ZP harm 0.80–0.95 × ZP rescue → explains why every global method's net lift is bounded by bootstrap noise (total_net ∈ {+17, +37, +69, +102} instances out of 18000).
- [R3] confirmed: regret invariant across all interventions — base 1.007 %, method 1.007–1.015 % (metric: relative SBS-oracle gap %).
- [R40-setup] infra: rebuilt `code/unified_selector/runs/audit.json` from `data/*/raw_label.pkl` (per-problem SBS pool idx + SBS/VBS means on train/val/test); added `code/unified_selector/build_audit.py`.
- [R40-setup] fix: NSS `HierarchicalBlock` produced `-inf` features on padded positions (`new_x += top_scores` where masked scores are `-inf`) → NaN propagated through next attention. Patched: zero scores for padded positions, hard-zero padded rows after each block (metric: 1-epoch full-data smoke macro_top1 = 0.4955, no NaN).
- [R40A] negative: NSS hierarchical encoder (d=128, block_num=2, encoder_layer_num=2, ReZero, PL loss, lr 5e-5, cosine 600k, 50-epoch budget) reaches R18 parity after 1 epoch (val top1 0.5189 vs R18's 0.5193) then **diverges** via catastrophic TSP forgetting (TSP top1 0.725 → 0.647 → 0.211 across ep1-4). T2 early-stop fires at ep4. Gate (val ≥ 0.529) NOT passed. Root cause: cross-problem gradient conflict when training 18 heterogeneous problems with a single shared head.
- [R40A] fix: LocalProblemHead AMP dtype mismatch (`out` allocated from h as fp32 but `local` produced as fp16 under autocast); patched by allocating `out` from `local`.
- [R40C] negative: NSS hierarchical encoder + LocalProblemHead from scratch (d=128, block_num=2, local head per problem) — local head prevents TSP forgetting for 3 epochs but catastrophic forgetting delayed-hits by ep5 (TSP 0.709 → 0.412). Peak val top1 = 0.5133 at ep2, below R18 (0.5193). Gate not passed.
- [R41] partial: Loading R18 base + adding LocalProblemHead (freeze all except local_head) beats R18 on val by +0.004 (peak ep2 val = 0.5234), but on test bootstrap the Δ = +0.0024 [−0.0034, +0.0085], p(Δ>0) = 0.786 — **NOT significant**. Per-problem: CVRP +0.014, OVRPTW +0.030, VRPBL +0.022 gains but OVRPL −0.026, OVRPB −0.008 regressions (val→test shrinkage confirmed again). First experiment since R3 to produce ANY positive test Δ from a train-time method, but below pre-reg 0.005 threshold.
- [infra] GPU1 being shared with another project's training job (0selection2 R26d, PID 1979615); my background runs (R41 ep4+, R42 ep1+) got silently OOM-killed at 10 GB usage mark. Rerunning requires a sole GPU slot or smaller batch.

## Constraints carried forward

- Do NOT re-run exploratory test-set grid sweeps — they inflate Δ by ~5× compared with val-locked numbers.
- Do NOT treat val-win configurations as implying test-win — the pattern failed at R1, R2, R3 repeatedly.
- Do NOT attempt further global post-hoc correction on the R18 base — the gross decomposition proves the cancellation is structural to this class of method.
- Do NOT train NSS hierarchical encoder from scratch on 18 problems without per-problem gradient isolation — catastrophic TSP forgetting is the dominant failure mode; local head alone delays but does not prevent it.
- A genuine method-improvement paper would need a new base policy (train-time collapse prevention, e.g., CQR-style distribution regularization) and is out of scope for this diagnostic paper.

## Terminal state

- Loop terminated at Round 3 with oracle-pro score 8.5/10 and verdict "Stop the loop. Write the diagnostic paper."
- Six claims in `CLAIMS_FROM_RESULTS.md` all `supported` at `high` confidence, `integrity_status = unavailable` (provisional).
- Next workflow: `/paper-plan` → `/paper-write` over the six claims + Method Description in `review-stage/AUTO_REVIEW.md`.

## 2026-04-21 R40D / R40E / R41B / R42S（reviewer §1.1/§1.2/§3/§4 修复后）

- [R40D/E] positive: round_robin 和 task_accum 两种 scheduler 都阻止了 R40A/C 的 TSP 灾难性遗忘 (TSP ep0 = 0.705, ep3 task_accum 仍 0.705, 无崩塌) — confirms reviewer §1 was correct (bug was blockwise scheduler, not NSS architecture).
- [R40D/E] negative: but both schedulers can't push NSS from-scratch above SBS-picking trivial solution (macro_top1 stuck at 0.50-0.51, vs R18 0.5166 test). Supervision signal insufficient to escape SBS fixpoint when network is randomly initialized.
- [R41B] partial: R18 + residual adapter (zero-init delta, per-problem α=σ(-3.89)) + base-KL trust region (τ=2.0, correct-weight=1.0, wrong-weight=0.2) — val macro_top1 0.5206 (+0.0013 over R18), test macro_top1 0.5197 (+0.0031, 95% CI [-0.0010, +0.0072], p=0.928). Non-sig but tightest CI / highest p since R3.
- [R41B] zero-pick insight: rescue only on VRPBL (127 rescues from 463 zp); all other problems have rescue=0. Net rescue-harm = -559 (harm dominates). Residual adapter moves logits smoothly across all problems, not targeted zero-pick fixes.
- [R42S] partial: KEEP_SBS + full-pool reranker — val 0.5277, test +0.0017 vs SBS (95% CI [-0.0043, +0.0077], p=0.712, non-sig). TSP +0.026 / OVRPBTW +0.046 / OVRPBLTW +0.022 / VRPBL +0.021 wins but VRPLTW -0.043 / VRPBLTW -0.028 cancellations (val 1k insufficient to lock per-problem θ).
- [infra] cgroup mem limit: docker container has 20 GB memcg limit; 3 concurrent training processes OOM-kill at 5-6 GB each + R27a (2.5 GB) + buffers (~1 GB). Forced sequential execution: R40E (killed early) → R41B → R40D.
- [infra] wandb dual-credential bug: .netrc has fengguangwuliang account (no project permission); WANDB_API_KEY env has yjkds. Launch scripts must explicitly `export WANDB_API_KEY='...'`.

## Constraints carried forward (追加)
- Do NOT run 3+ training processes concurrently under current cgroup (20 GB limit).
- Do NOT train NSS from scratch on 18-problem mix with argmin supervision — it collapses to SBS-picking even with correct scheduler. If pushing beyond R18 via new encoder, need ranking-style loss or per-problem curriculum.
- Do NOT expect residual-adapter fine-tuning alone to cross 95% CI on Δ at single seed — need 3+ seeds averaged.

## 2026-04-21 R40D completed — first significant positive result since R3

- [R40D] POSITIVE (FIRST SIGNIFICANT): NSS hierarchical encoder from scratch with round_robin scheduler achieves test macro_top1 = 0.5226 vs R18 0.5166 (Δ = +0.0060, 95% CI [+0.0001, +0.0118], p(Δ>0) = 0.977, significant = YES). Best at ep6 (val = 0.5248). This is the first method to cross 95% threshold. Per-problem: OVRPTW +0.055, OVRPBTW +0.022, CVRP +0.018, VRPBL/OVRPB/OVRPBL +0.008-0.014. Losses: OVRPLTW -0.014, OVRP -0.010. TSP nearly unchanged (-0.003).
- [R40D] KEY VALIDATION: reviewer §1.1 confirmed — TSP collapse was the scheduler bug, not NSS architecture. Fixed train.py with --task-schedule round_robin → NSS no longer collapses (TSP 0.705 → 0.744 across ep0-3, not 0.73 → 0.21 as in R40A/C).
- [R40D] CONFIG that works: standard mixed-loss (regret_soft + listwise_ce + plackett_luce pl_weight=0.3 pl_topk=3), no aug_8fold, lr=5e-5 cosine_total=80000, batch_per_problem=16, warmup=3000. Did NOT need the curriculum / PCGrad / MGDA / uncertainty-weighting that reviewer §2 mentioned as fallbacks.
- [R40D] POST-peak degradation: macro_top1 peaked at ep6 (0.5248), regressed ep7 (0.5194), ep8 (0.5169). LR was already at min (3e-6) by ep6 so not a scheduler issue. Suggests overfit post-peak. Future runs should cut epochs=7 with `--early-stop-plateau 2`.
- [R40D] the +0.0060 gain does NOT come from zero-pick rescue (264 rescues vs 1437 harms = net -1173). Instead, R40D shifts per-arm distribution within R18's support set; improvements come from arm rank reordering, not new arm discovery.

## 2026-04-21 R42S_fixpair — pairwise sign fixed + family-level θ

- [R42S_fixpair] NEGATIVE: fixing the pairwise_cost_loss sign bug (L265-274 pre: mask picked i-worse-than-j then softplus pushed score_i > score_j; post: mask picks i-better-than-j, gap-weighted) pushed val 0.5229 (up from 0.5277 — wait, v1 was higher val) but test Δ vs SBS = **-0.0002** [-0.0093, +0.0090], p=0.481, NON-SIGNIFICANT. Versus buggy R42S (+0.0017 non-sig), the fix made test Δ WORSE, not better.
- [R42S_fixpair] WHY: I switched threshold scope from per-problem to family-level (4 θ for TSP/ATSP/CVRP/MVRP). Result: 17/18 problems got θ=+0.400 (MVRP-family saturation point → "mostly keep SBS"); only TSP picked θ=-0.150. The family-θ collapsed MVRP to near-SBS, so CVRP -0.034, OVRPB -0.038, OVRPTW -0.028, OVRPL -0.027 lost.
- [R42S_fixpair] PER-PROBLEM WINS (even under bad θ): TSP +0.021, OVRPBTW +0.041, OVRPBLTW +0.031, OVRPBL +0.024, VRPBL +0.023 — clustered in TSP + hard-MVRP combinations. Hard-MVRP (B+L+TW mix) benefits from reranker but basic CVRP/VRPTW loses.
- [R42S_fixpair] CONCLUSION: (a) pairwise sign was indeed a bug — but not the dominant factor in R42S_v1's near-zero Δ. (b) family-θ = wrong abstraction for MVRP. Need per-problem θ with shrinkage OR problem-specific KL-to-base weight. (c) Reranker architecture itself is NOT disqualified — several problems have robust +0.02-+0.04 gains, matching R40D's per-problem signal pattern.
- [R42S_fixpair] NEXT: R42S_v2 with per-problem θ (accept 1k sample noise) + stronger KL (0.3 → 0.5) to prevent CVRP/OVRPB/OVRPTW/OVRPL drift. Queue on GPU1 after seed1 completes.

## Constraints carried forward (追加)
- Do NOT use family-level threshold for MVRP family — problem-specific optima disagree. Per-problem θ is the right abstraction even at 1k val noise.

## 2026-04-21 19:00+ CRITICAL data-alignment bug discovered + fixed

- [DATA DRIFT] at 17:46 today `raw_label.pkl` for every non-TSP problem was silently regenerated. New solver results were appended alphabetically: CVRP/MVRP got MoSES_CaDA, MoSES_RF, RouteFinder; ATSP got ICAM_ATSP, UNICO_MatPOENet. Before the change CVRPtest cost_len=9=len(POOLS["CVRP"]); after the change cost_len=12 but POOLS was NOT updated. TSP was untouched (results/ unchanged).
- [DATA DRIFT] impact: `data.py:100 costs = lbl["cost"][:K_p]` assumes the first K_p raw cost entries = POOLS[problem] order. After the alphabetical expansion this is violated — e.g. for CVRP raw_cost[6..8] moved from (OMNI, RELD_CVRP, UDC) to (MoSES_CaDA, MoSES_RF, OMNI). Consequence: R18 macro_top1 on test appeared to drop 0.5166 → 0.2432 purely from misaligned labels; ATSP "top1" fell to 0.152 because the model still picks arm-1=MATNET (global) but arm-1 now indexes ICAM_ATSP's cost column. TSP unaffected (data untouched).
- [DATA DRIFT] fix: patch `data.py` to build `cost_gather_idx = [sorted(results/result_*.txt).index(s) for s in POOLS[problem]]` and use that index to gather raw costs into POOLS order. Same patch applied to `tools/dump_behavior_to_npz.py` and `tools/ceiling_lightgbm.py`. Post-patch verification: R18 test macro_top1 = 0.5167 (matches prior 0.5166 to rounding).
- [DATA DRIFT] fallout: (a) R40D seed0 ckpt (pre-drift, Apr-21 02:58) still valid; test_eval reruns at 0.5226 ✅. (b) R42S_fixpair ckpt + report (pre-drift 13:28) still valid. (c) behavior_emb.npz + balanced_softmax_prior.npz (14:36 on OLD raw_label) — columns already in POOLS order, still valid. (d) Ceiling LightGBM (17:35) pre-drift, valid. (e) R40D seed1 (trained 17:52-17:57) + R40D seed2 (trained 17:57-18:16) + R43 v1 (18:43-19:17) — ALL trained on misaligned labels. Quarantined into `*_INVALID_pre_data_fix/` and retrained from scratch post patch.
- [DATA DRIFT] lesson: raw_label.pkl is NOT a stable artefact; results/result_*.txt set is the source of truth. Never trust positional-index alignment — always rebuild the index from `sorted(results/result_*.txt)`.

## 2026-04-21 19:18 R43 + R40D seed1 re-launched concurrently post data patch

- [R43 v2] cuda:0, uses behavior_emb + balanced_softmax_prior + support-KL + problem-arm-bias. Flags: `--use-problem-arm-bias --use-behavior-emb --behavior-dim 3 --balanced-softmax-weight 1.0 --support-kl-weight 0.2`. Gate: Δ vs R40D seed0 ≥ +0.003; support_size per problem ≥ 120%; zero-pick rescue ≥ 500.
- [R40D seed1 v2] cuda:1, flags match seed0 exactly. Gate: test Δ vs R18 ≥ +0.0035 (60% of seed0's +0.0060, allowing variance).

