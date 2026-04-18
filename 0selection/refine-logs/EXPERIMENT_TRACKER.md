# Experiment Tracker

| Run ID | Status | Seeds | GPU-h budget | GPU-h used | Pass/Pivot/Fail | Key result | Notes |
|--------|:------:|:-----:|-------------:|-----------:|:---------------:|------------|-------|
| R0 — eval & split audit | pending | n/a | 0 | — | — | — | Blocks all others |
| R1-Full seed 0 | pending | 1 | 5 | — | — | — | Template / stability check |
| R1-Full seeds 1–2 | pending | 2 | 10 | — | — | — | Completes **C1** |
| R2-MetaOnly seeds 0–1 | pending | 2 | 3 | — | — | — | **C2** anti-leakage |
| R5-NoBS seeds 0–1 | pending | 2 | 10 | — | — | — | **C5** BS-mask ablation |
| R4-ZS-Fact seeds 0–1 | pending | 2 | 10 | — | — | — | **C4** compositional zero-shot |
| R4-ZS-Flat seed 0 | pending | 1 | 5 | — | — | — | Contrastive baseline for C4 |
| R3-CE seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| R3-Pairwise seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| R3-GapReg seed 0 | pending | 1 | 5 | — | — | — | **C3** loss ablation |
| **Total** | | | **58 + 2 contingency** | | | | |

**Legend**
- Status: pending / in-progress / done / blocked
- Pass / Pivot / Fail: use thresholds in `EXPERIMENT_PLAN.md §A`

**Update protocol**
- Fill GPU-h used + Pass/Pivot/Fail + Key result after each run.
- If Pivot → update `FINAL_PROPOSAL.md` and claims matrix accordingly.
- If Fail on C1 → stop; reassess before continuing.
