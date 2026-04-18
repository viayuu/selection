# Review Summary — 4 Oracle-Pro Sessions

| # | Session ID | Stage | Key verdict |
|---|------------|-------|-------------|
| 1 | `selector-litreview-review-1` | Literature deep pass | Revised gap framing; G1 narrowed; G3 replaced by G3' (compositional zero-shot); added G4 (cold-start), G5 (cost-aware). Identified missing prior art: CTS (2006.00715), GINES (2302.04035), MatNet (2106.11113), MTNCO (2402.16891), PromptVRP (2405.12262). |
| 2 | `idea-creator-selector-review-1` | Idea triage | Cut cold-start + PLE-hierarchical + cost-aware (time excluded). Injected two novel angles: **MVRP solver×variant factorized scoring** and **regret-soft labels** — the single biggest novelty lever for a cost-focused metric. |
| 3 | `selector-feasibilit-review-1-2` | Feasibility | **PROCEED**. Expected outcome: 10–13/18 problems beat SBS. Landmines: mask leakage, cost-scale normalization (use relative regret), BatchNorm across mixed problems, tie handling. Three early-warning sanity checks prescribed. |
| 4 | `selector-impl-review-1` | Implementation + mock review | Full 10-line method equations; module-level code tree; concrete hyperparameters (τ=0.02, d=128, depth=4, AdamW 3e-4 cosine 40k steps, batch 144); 3 mandatory figures; claims matrix (2×2); mock ICML 6/10 best-case; **metadata-only baseline** as the single pre-emptive experiment. |
| 5 | `selector-exp-plan-review-1` | Experiment plan | Claim×Evidence×Run×Gate matrix; 58-GPU-h core; seed protocol. See `EXPERIMENT_PLAN.md`. |

## Consensus points across all reviews
- **Regret-soft labels** are the most important design decision (metric alignment).
- **Masked softmax with strict leak guards** is non-negotiable.
- **Metadata-only baseline** is the single critical defense.
- SBS baseline beats many problems already → frame "matches or beats" rather than "beats all".
- ATSP pool (3 methods) cannot carry a main claim; report separately.
- MVRP compositional zero-shot is the most novel experimental component.
