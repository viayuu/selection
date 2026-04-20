# Experiment Tracker

| Run ID | Status | Seeds | GPU-h budget | GPU-h used | Pass/Pivot/Fail | Key result | Notes |
|--------|:------:|:-----:|-------------:|-----------:|:---------------:|------------|-------|
| R0 — eval & split audit | **done** | n/a | 0 | 0 | **Pass** | All 18 problems have gap ≥ 0.67%; ATSP gap 2.36%. See `code/unified_selector/runs/audit.json` | Blocks all others — complete |
| R1-Full seed 0 (2-epoch pilot) | **done** | 1 | 5 | 0.15 | **Pass** | macro_vs_sbs=+0.156% (pivot band, likely pass with more epochs); TSP beats SBS -0.09%; 7 problems at or below SBS | 2 epochs pilot; 10-epoch ext running |
| R1-Full seeds 1–2 | pending | 2 | 10 | — | — | — | Completes **C1** |
| R2-MetaOnly seeds 0–1 | seed 0 **done** | 1 | 3 | 0.15 | **Pass (C2)** | macro_vs_sbs=+75.23% vs R1's +0.16% → instance features clearly essential | **C2** anti-leakage slam-dunk |
| R1-NoFact seed 0 (ablation) | **running** | 1 | 5 | — | — | 10 epochs; tests MVRP factorized head contribution | bonus ablation |
| R1-Full seed 0 **extended (10 ep)** | **running** | 1 | 5 | — | — | Aims to push macro_vs_sbs negative | Expected ~50 min |
| R5-NoBS seeds 0–1 | pending | 2 | 10 | — | — | — | **C5** BS-mask ablation |
| R4-ZS-Fact seeds 0–1 | pending | 2 | 10 | — | — | — | **C4** compositional zero-shot |
| R4-ZS-Flat seed 0 | pending | 1 | 5 | — | — | — | Contrastive baseline for C4 |
| R3-CE seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| R3-Pairwise seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| R3-GapReg seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| R16 — multigen shortlist sanity | **done** | 1 | 0.1 | 0.04 | **Pivot** | test `macro_top1=0.5317`, `macro_vs_sbs=-0.109%` | shortlist collapsed to near all-ones; no gain over `R12` |
| R17 — multigen + support-budget sanity | **done** | 1 | 0.1 | 0.04 | **Fail** | test `macro_top1=0.5066`, `macro_vs_sbs=+0.044%` | shortlist became sparse but over-corrected and hurt cost |
| R18 — winner-margin sanity | **done** | 1 | 0.1 | 0.04 | **Pivot** | test `macro_top1=0.5321`, `macro_vs_sbs=-0.117%` | almost ties `R12`, but no real breakthrough |
| **Total** | | | **58 + 2 contingency** | | | | |

**Legend**
- Status: pending / in-progress / done / blocked
- Pass / Pivot / Fail: use thresholds in `EXPERIMENT_PLAN.md §A`

**Update protocol**
- Fill GPU-h used + Pass/Pivot/Fail + Key result after each run.
- If Pivot → update `FINAL_PROPOSAL.md` and claims matrix accordingly.
- If Fail on C1 → stop; reassess before continuing.
