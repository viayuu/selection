# Research Review — Refined Idea + Implementation Plan

**Reviewer**: GPT-5.4 Pro via Oracle MCP (browser), session `selector-impl-review-1`.
**Date**: 2026-04-18.
**Scope**: one-round consolidation over the three prior reviews.

---

## 1. Main method — 10-line forward pass

Global: `M` = total solver vocabulary; `B` batch; `d` width; `P=18` problems; `K` = constraint-bit dim; `D` = coord-distribution code dim.

Batch inputs: instance `x_b`, problem id `p_b`, constraint bits `v_b∈{0,1}^K`, coord-dist code `q_b∈{0,1}^D`, availability mask `A∈{0,1}^{B×M}`, cost vector `C∈R^{B×M}`.

1. `T_b = Adapter_mod(x_b) ∈ R^{n_b×d}` — shared coord adapter for TSP/CVRP/MVRP, shared matrix adapter for ATSP.
2. `g_b = Pool(Encoder_θ(T_b)) ∈ R^d` — shared encoder, **LayerNorm/GraphNorm only**, never BatchNorm.
3. `h_b = LN(W_g g_b + E_prob[p_b] + W_v v_b + W_q q_b) ∈ R^d` — problem-id embedding is the only problem-specific learned table.
4. `e_s = E_sol[s] ∈ R^d`; pair feature `z_{b,s} = [h_b, e_s, h_b⊙e_s, |h_b−e_s|] ∈ R^{4d}`.
5. Main logit: `ℓ^main_{b,s} = MLP_head(z_{b,s}) ∈ R^{B×M}`.
6. MVRP add-on (applied only when `p_b ∈ MVRP`): `ℓ_{b,s} = ℓ^main_{b,s} + MLP_fact([e_s⊙W_c v_b, e_s⊙W_d q_b, e_s⊙W_c v_b⊙W_d q_b])`.
7. Masked prediction: `π_b = softmax_s(ℓ_{b,s} + log A_{b,s})` with `log 0 ≡ −1e9`.
8. `c^*_b = min_{s:A_{b,s}=1} C_{b,s}`.
9. Regret labels: `r_{b,s} = (C_{b,s} − c^*_b)/(|c^*_b|+ε)`; `y_b = softmax_s(−r_{b,s}/τ + log A_{b,s})`.
10. Loss: `L = −(1/B) Σ_b Σ_s y_{b,s} log π_{b,s}`. Prediction = argmax over `A`.

**Rules**: availability mask is used *only* in lines 7 and 9 — never fed into encoder or metadata. `ind` is for sanity checks, not training target.

---

## 2. Implementation plan (minimum viable, ~3 days for a junior)

```
code/unified_selector/
  configs/unified18.yaml         # single source of truth
  registry.py                    # 18 problems, global solver ids, masks, constraint bits, coord-dist codes, SBS/oracle metadata
  data.py                        # cached features + costs + ind + masks; balanced per-problem sampler; padding collate
  model.py                       # coord adapter + ATSP matrix adapter + shared encoder + solver embeddings + masked head + MVRP factorized add-on
  loss.py                        # regret-soft-label CE with strict masking and NaN guards
  metrics.py                     # all 观测指标.md metrics: top-1/2/3, mean cost, per-method win/lose, arm histogram, loss curve
  train.py                       # DDP/AMP loop, balanced sampler, grad accumulation, checkpointing, validation
  evaluate.py                    # loads checkpoint → per-problem tables, SBS/oracle gap, arm histograms, zero-shot summary
  sanity_checks.py               # mask / overfit-256 / cost-scale / tie / no-unavail-selection asserts
  export_fig_tables.py           # metric JSON → paper figure tables
  scripts/run_unified18.sh
  scripts/run_mvrp_zeroshot.sh
```

No plugin system. No Hydra maze. Build this first.

---

## 3. Training recipe (actual numbers)

**Sanity job first**: 256 mixed instances, dropout 0, 800 steps → must overfit to near-oracle mean cost with no unavailable selections. If it doesn't, **do not launch full training**.

**Full run** (2× RTX 3090, DDP + AMP):
- Balanced logical batch = **8 instances per active problem** (18 problems → 144; zero-shot: 14 → 112).
- Per-GPU microbatch = 1 × per problem (global microbatch 36); **grad-accum = 4**.
- Optimizer: **AdamW** (β1=0.9, β2=0.95, ε=1e-8), weight decay 1e-4.
- LR: max 3e-4, **1,500-step linear warm-up**, cosine decay to 3e-5 by step **40,000**.
- Grad clip = 1.0.
- Checkpoint + validate every **1,000** optimizer updates.
- **Early-stop / best-ckpt by macro normalized mean-cost**, NOT by cross-entropy or top-1.
- For compositional zero-shot: **do not** early-stop on the 4 held-out MVRP; select on val of the 14 active problems, evaluate held-out once.

---

## 4. First hyperparameters (just use these)

```
d_model              = 128
encoder depth        = 4 shared blocks
ATSP matrix depth    = 3 blocks
selector MLP hidden  = 256
soft-label τ         = 0.02        ← a solver 1% worse keeps mass; 5% worse is downweighted
dropout              = 0.10
problem-id dropout   = 0.25        ← anti-lookup regularizer
weight decay         = 1e-4
lr max               = 3e-4
warm-up              = 1,500
total updates        = 40,000
```

---

## 5. Three paper figures (tie to 观测指标.md)

**Fig. 1 — 18-problem mean-cost leaderboard.** Per problem: normalized mean cost of {SBS, unified selector, best ablation, VBS/oracle} + win/loss marker vs SBS. *Proves the cost objective, not accuracy.*
**Fig. 2 — Accuracy-regret diagnostic panel.** Per problem: top-1/2/3 accuracy **next to** normalized regret; training/val loss curve underneath. *Proves regret-soft CE beats CE and that top-k alone is insufficient.*
**Fig. 3 — MVRP compositional zero-shot grid.** Rows = held-out constraint combinations; cols = coord-distribution families; cells = selector/SBS ratio with inset arm-distribution. *Proves the factorized head actually generalizes rather than collapses to one arm.*

---

## 6. Claims matrix

| | **Zero-shot works: YES** | **Zero-shot works: NO** |
|---|---|---|
| **Beats SBS macro: YES** | **Strongest story.** Unified masked selector beats SBS across heterogeneous routing; factorized MVRP metadata enables compositional generalization to unseen variants. | Still publishable: unified supervised selection works on known families; MVRP compositional transfer unsolved → factorized head becomes honest negative result. |
| **Beats SBS macro: NO** | Narrow but interesting: reframe around compositional MVRP selection (not universal). | **Worst quadrant.** Fallback = benchmark/diagnostic paper titled *"When Neural Solver Selection Fails: A Controlled Study of Variable Method Pools in Routing"* — documents variable pools, regret labels, mask-leakage pitfalls, and SBS robustness. Credible ≠ glamorous. |

---

## 7. Mock ICML review (best case: 12/18 + zero-shot works)

**Summary**: Unified supervised neural selector for 18 routing problems with variable solver pools; shared instance encoders + solver embeddings + masked softmax + regret-soft labels; beats SBS on 12/18; promising MVRP compositional zero-shot.

**Strengths**: Timely & practical setting (heterogeneous pools, variable availability); sensible regret-soft-label formulation; careful handling of masks + cost-scale + mixed-problem training; zero-shot MVRP experiment is the most novel component; uses cost/regret rather than accuracy.

**Weaknesses**: Moderate algorithmic novelty (engineering combination of known components); may be exploiting problem-id / constraint metadata / solver availability rather than instance-level structure; need problem-id-only, mask-only, and per-problem NSS baselines; cost-only ignores runtime; fixed solver pool limits generality claims.

**Questions**: How much gain without problem-id? How often does the selector disagree with per-problem SBS *within a problem*? Paired-bootstrap significance? Leave-one-solver / leave-one-problem stress tests? τ sensitivity?

**Score**: **6/10**. **Confidence**: 3/5.

**Moves to accept**: decisive metadata-only baseline; per-problem NSS baseline; paired CIs; one leave-one-solver/problem stress test. If the full model clearly beats these while preserving zero-shot → **7/10**.

---

## 8. The one thing most likely to sink the paper — and the single experiment that pre-empts it

**Objection**: *"This is not learning instance-level solver selection; it is learning problem-id + availability-mask priors."*

**Pre-emptive experiment — Metadata-only baseline**: train the same architecture + same loss, but **zero out instance features `x`**, keeping problem-id, constraint bits, coord-distribution code, solver embeddings, and availability mask. Report full vs metadata-only on macro mean cost, top-1/2/3, arm distribution, per-problem win/loss.

If the full model does **not significantly beat metadata-only**, the main claim dies — better to learn this from a 2h experiment than from reviewer-2.

---

## Action checklist (ordered)
- [ ] Build `registry.py` + `data.py` + cache `labels.npz` from `raw_label.pkl`.
- [ ] Implement `model.py` following §1 equations exactly.
- [ ] Implement `loss.py` with strict mask assertions.
- [ ] Run `sanity_checks.py` (overfit-256, no-leak, cost-scale) — **gate** for the full run.
- [ ] `scripts/run_unified18.sh` with §4 hyperparameters.
- [ ] Evaluate + emit Fig. 1 / Fig. 2 tables.
- [ ] Launch `scripts/run_mvrp_zeroshot.sh` (11-train / 4-holdout) → Fig. 3.
- [ ] Metadata-only baseline (the §8 pre-emptive).
- [ ] Ablations in v2: loss study, BS-mask, problem-id dropout sweep, per-problem NSS reference.
- [ ] Paired-bootstrap CIs for mean-cost in the final table.
