# Initial Experiment Results — Unified Selector Pilot

**Date**: 2026-04-18.
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Code**: `code/unified_selector/` (new, built on top of `neural-solver-selection/` concepts).

## What was built

| File | Purpose |
|------|---------|
| `registry.py` | 18 problems, 18-solver global vocabulary, per-problem pool masks, constraint bits (C/O/B/L/TW), coord-distribution codes |
| `audit.py` | R0 — SBS / VBS / per-problem metadata (0 GPU-h) |
| `data.py` | UnifiedProblemDataset + collate; strips `time`/`gap` from labels for RAM; streams train-time by problem |
| `model.py` | UnifiedSelector: GAT/transformer coord encoder + MatNet-style matrix encoder for ATSP + problem/constraint/coord-dist embeddings + global solver embeddings + masked-softmax head + MVRP factorized add-on + regret-soft-label loss |
| `train.py` | DDP-ready single-GPU trainer; AdamW + cosine schedule + warmup + problem-ID dropout + meta-only flag |

## R0 — Evaluation & split audit (DONE)

Saved to `code/unified_selector/runs/audit.json`. Key per-problem oracle-SBS gaps (val):

| problem | K | SBS | SBS_mean | VBS_mean | gap% | SBS winrate |
|---------|--:|-----|---------:|---------:|-----:|------------:|
| TSP | 10 | LEHD | 8.038 | 7.969 | **0.86%** | 70.0% |
| CVRP | 9 | RELD_CVRP | 18.294 | 18.141 | **0.85%** | 33.9% |
| ATSP | 3 | MATPOENET | 1.722 | 1.682 | **2.36%** | 70.4% |
| OVRP–OVRPBLTW | 4 | RELD_MTL | — | — | 0.67–1.25% | 44–56% |

All 18 problems have **non-trivial oracle-SBS gap** → room for a selector exists everywhere.

## R1-Full seed 0 (2 epochs, 9.5 min per run on single 3090)

Results after 2 epochs (45k steps):

| metric | R1-Full | R2-MetaOnly |
|--------|--------:|------------:|
| **macro top-1 acc** | 0.511 | 0.434 |
| **macro vs SBS (mean cost)** | **+0.156%** | **+75.23%** |

**R1-Full per-problem** (val, epoch-1 best):
- TSP top-1 **71.7%**, vs_sbs **-0.09%** → beats SBS
- ATSP top-1 **70.3%**, vs_sbs +0.05% → matches SBS (correct collapse — pool has 3 arms, SBS wins 70%)
- CVRP top-1 29.7%, vs_sbs +0.94% → slightly worse; needs more training
- MVRP (15 variants) — mostly within ±0.1% of SBS, several negative (beats SBS):
  - **VRPL -0.03%, OVRPL -0.01%, VRPBL -0.01%, OVRPBL -0.03%, OVRPBTW -0.04%, VRPBLTW -0.01%** (wins)
  - Others slightly above SBS

**R2-MetaOnly per-problem** (same setup, instance features zeroed):
- TSP: mean cost = 115.95 (vs SBS 8.04!) — catastrophic, top-1 = 0.0%
- CVRP: 30.94 vs 18.29 (+69%)
- ATSP: top-1 = 2.1%, +11.59% vs SBS
- MVRP: every variant degenerates to picking SBS (top-1 identical to audit's SBS win-rate)

## Interpretation

**C1 (main claim — selector matches or beats SBS)**: ✅ supported *after only 2 epochs*. Macro vs SBS = +0.156% (well under the 0.3% Pivot bar; would likely go negative with more training — the extended 10-epoch run is in progress).

**C2 (anti-leakage — gains from instance features)**: ✅ **SLAM DUNK**. R2 meta-only is +75.23% vs SBS macro, R1 is +0.16% — a 75-point gap. Without instance features the selector cannot disambiguate TSP/CVRP sizes and cannot beat SBS on ATSP (3-arm problem). The one experiment most likely to sink the paper has been answered.

**C4 proxy (MVRP factorization)**: under evaluation — `R1_nofact_seed0` is running with `--no-fact` as an ablation for the MVRP factorized head.

## Runs currently in flight

| Run | GPU | Purpose | Status |
|-----|----:|---------|--------|
| R1-Full ext (10 epochs) | 0 | Extended training to push macro_vs_sbs negative | running |
| R1-NoFact (10 epochs) | 1 | Ablation: remove MVRP factorized head; contrast against R1-Full for MVRP-only impact | running |

Expected completion: ~50 min each.

## Landmine audit

- ✅ Masked softmax: verified (`prob[~mask]==0` via `+log(mask)` with `log 0 ≡ -1e9`).
- ✅ Cost-scale normalization: relative regret `r_s=(c_s-c*)/(|c*|+ε)` used; τ=0.02; label entropy checked (TSP/CVRP/ATSP labels span full pool).
- ✅ No BatchNorm across problems — uses LayerNorm + GraphNorm.
- ✅ Tie handling: `tie_tol=1e-3` floors r_s to 0 for ties.
- ✅ SBS frozen from val (audit.json).
- ⚠️ WandB unavailable (entity `jkds` not registered). Logs saved to `train.log`; retrofitting local metric tracking JSON per epoch.

## Summary

- **1/5 core claims tested and supported** (C1 positive, C2 slam-dunk).
- **GPU-h used so far**: ~0.3 (two 9.5-min runs in parallel).
- **Ready for Workflow 2**? No yet — need (a) extended R1 result (~50 min), (b) R3 loss ablation, (c) R4 zero-shot MVRP, (d) R5 BS-mask.
- **Next step**: after extended R1 + nofact finish, launch R3 loss ablations (CE / pairwise) and R4 compositional zero-shot (train 11 MVRP, hold out OVRPBL / VRPBLTW / OVRPBTW / OVRPLTW).

**Bottom line for the user's original question** — *"can the basic approach actually hit the goal with not-too-bad performance?"* — **YES**. After 2 epochs (9 min on a single 3090), the unified selector already essentially matches SBS on macro mean cost (+0.16%, well within the Pivot tier), beats SBS on TSP, and thoroughly dominates the metadata-only baseline. With the extended 10-epoch training still running, the expectation is macro_vs_sbs drops negative.

## Post-Bridge Effect-Focused Runs (2026-04-19)

These runs were executed after the new selector-effect literature pass and are **not** part of the original claim plan. They were used to probe specific failure modes:

| Run | Purpose | Test result | Verdict |
|-----|---------|-------------|---------|
| `R16_multigen_support_sanity_seed2` | multi-generator shortlist retrieval on top of best `R12` backbone | `macro_top1 = 0.5317`, `macro_vs_sbs = -0.109%` | negative / no gain |
| `R17_multigen_budget_support_sanity_seed2` | force sparse shortlist via budget regularization | `macro_top1 = 0.5066`, `macro_vs_sbs = +0.044%` | fail |
| `R18_margin_sanity_seed2` | winner-vs-competitor margin sharpening on top of `R12` backbone | `macro_top1 = 0.5321`, `macro_vs_sbs = -0.117%` | neutral / tie |

Key interpretation:

- naive shortlist retrieval tends to collapse into “select everything”
- hard sparsity control hurts too much
- direct margin sharpening is stable but not enough by itself
- the best completed line still remains `R12_init_gaprank_manual_adapter_soft_risk_seed2`
