# Experiment Plan — Claim-Driven, 58 GPU-h Core

**Compute**: 2× RTX 3090 (aggregate GPU-h). Labels pre-cached. One full selector seed ≈ 5 GPU-h; MetaOnly ≈ 1.5 GPU-h; ~2 GPU-h contingency.

## A. Claim × Evidence × Run × Decision-Gate Matrix

| Claim | Run(s) | Primary metric | Decision gate | GPU-h |
|-------|--------|----------------|---------------|------:|
| **C1** Unified selector matches/beats SBS on macro mean cost (18 problems) | **R0** (eval audit) + **R1-Full** (3 seeds) | `ΔSBS = 100·mean_p(cost_model_p/cost_SBS_p − 1)`; VBS-gap closed; per-problem wins | **Pass**: `ΔSBS ≤ 0.0%` with 95% CI ≤ `+0.15%`, no more than 3 problems worse by >1%. **Pivot**: `0.0 < ΔSBS ≤ 0.3%` → frame as "matches". **Fail**: `ΔSBS > 0.3%` or broad negative transfer. | 15 |
| **C2** Gains from instance features, not problem-id / mask priors | **R2-MetaOnly** (2 seeds; zero-out `x`, keep ID + mask + bits + coord-dist) | `ΔMeta = cost_MetaOnly − cost_Full` / SBS; per-problem win count | **Pass**: Full beats MetaOnly by ≥ 0.5% macro on ≥ 12/18 problems. **Pivot**: 0.2–0.5% macro → "instance features help". **Fail**: < 0.2% or MetaOnly wins. | 3 |
| **C3** Regret-soft listwise > CE / pairwise / gap-regression | **R3-CE**, **R3-Pairwise**, **R3-GapReg** (1 seed each, matched to R1 seed 0) | Macro normalized cost; VBS-gap closed; regret@top1 | **Pass**: regret-soft best against all three, margin ≥ 0.3%. **Pivot**: best but <0.3% → "more stable/competitive". **Fail**: an alternative clearly better. | 15 |
| **C4** MVRP compositional zero-shot via factorized scoring | **R4-ZS-Fact** (2 seeds, held-out bitvectors `{0011, 0101, 1010, 1111}`) + **R4-ZS-Flat** (1 seed, non-factorized baseline) | Held-out macro normalized cost vs SBS and vs Flat; per-combo cost; seen→held-out degradation | **Pass**: Fact beats Flat by ≥ 0.7% held-out macro or ≥ 10pp more VBS-gap, no combo worse by >1%. **Pivot**: positive but smaller → "promising". **Fail**: Flat ≥ Fact or any combo collapses >2%. | 15 |
| **C5** BS-mask lifts tail without hurting head | **R5-NoBS** (2 seeds, same as R1 minus BS-mask) | Tail (bottom 6 by `N_eff = exp(H(mean p*(·\|problem)))`) / head (top 6) macro cost | **Pass**: tail ↑ ≥ 0.5% or ≥ 5pp VBS-gap closed AND head harm ≤ 0.15%. **Pivot**: tail 0.2–0.5% or head harm 0.15–0.3%. **Fail**: tail < 0.2% or head harm > 0.3%. | 10 |

## B. Run Graph (with blocks annotations)

1. **R0 — Evaluation & split audit** · 0 GPU-h. Compute SBS, VBS, per-problem normalization, mask sanity, zero-shot bitvector manifest, paired-bootstrap scripts. **Blocks R1–R5**.
2. **R1-Full seed 0** · 5 GPU-h. Validates stability, masking, cost normalization, checkpoint selection. **Blocks R2/R3/R4/R5 at scale**; template config.
3. **R1-Full seeds 1–2** · 10 GPU-h. Completes C1 headline.
4. **R2-MetaOnly seeds 0–1** · 3 GPU-h. **Blocks C2**.
5. **R5-NoBS seeds 0–1** · 10 GPU-h. **Blocks C5**.
6. **R4-ZS-Fact seeds 0–1 + R4-ZS-Flat seed 0** · 15 GPU-h. Held-out bitvectors fixed a priori; no held-out val used for checkpointing. **Blocks C4**.
7. **R3 loss ablations (CE, Pairwise, GapReg), 1 seed each** · 15 GPU-h. **Blocks C3**.

**Subtotal: 58 GPU-h. Contingency: ~2 GPU-h.**

## C. Seeding Protocol

- **In-domain core (R1, R2, R5)**: fixed train/val/test splits and cached labels. R1 = 3 seeds (headline variance); R2 + R5 = 2 seeds each, paired to R1 seeds 0–1. Variance matters most for R1 (main claim) and R5 (tail/head tradeoff with noisy tail estimates).
- **Loss ablations (R3)**: 1 seed each, matched to R1 seed 0, same update budget, same checkpoint rule, same batch size, same BS-mask setting. If regret-soft does not clearly beat all on this seed → mark C3 pivot/fail; do not rescue with selective reruns.
- **Zero-shot (R4)**: 2 seeds for factorized, 1 seed for flat. Held-out bitvectors fixed before training; checkpoint only on seen-combo val. Report macro-over-combos AND per-combo so one easy combo can't hide collapse.

## Minimum Core vs Ablation Extras

**Core (58 GPU-h, ships all 5 claims)**: R0, R1×3, R2×2, R5×2, R4-Fact×2, R4-Flat×1, R3-{CE,Pairwise,GapReg}×1.

**Ablation extras (not in core)**: second seeds for R3 losses; second R4-Flat seed; problem-ID-dropout sweep (0 / 0.25 / 0.5 / 1.0); ECE calibration; arm-distribution entropy/KL; ATSP-specific encoder diagnostics; runtime-aware VBS-gap.

**If budget cut to ~43 GPU-h**: drop C3 from the main narrative and keep the loss comparison as exploratory rather than canonical.
