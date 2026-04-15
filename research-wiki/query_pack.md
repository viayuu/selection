# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

- **gap:G1** Benchmark/OOD gap remains unresolved, especially on `CVRPLIB`, even though the simple unified selector is already strong on standard IID `val/test`.
- **gap:G2** Current evaluation mixes truly competitive solver-selection cases with low-entropy one-solver-dominant variants, which can dilute the scientific interpretation of overall gains.
- **gap:G3** The project still lacks a principled account of which solver preferences are environment-invariant and transferable across problem families, scales, and benchmark shifts.
- **gap:G4** The current selector optimizes pooled compatibility ranking, but not explicitly residual regret against a safe anchor or robustness under environment shift.

## Failed Ideas (avoid repeating)
- **Archive: Heavy Expertized Unified Selector as Main Direction**: **Lesson:** the next paper should focus on transferability of solver preference, hard competitive subsets, and benchmark robustness — not on a larger selector trunk.

- **Archive: Generic 'One Selector for All Problems' Framing**: **Lesson:** keep the overall direction, but sharpen the next paper around environment-robust / benchmark-robust solver preference transfer instead.

## Recent Relationships (4 total)
  idea:ood_invariant_residual_regret --addresses_gap--> gap:G1
  idea:ood_invariant_residual_regret --addresses_gap--> gap:G3
  idea:competition_aware_selection --addresses_gap--> gap:G2
  idea:selective_abstaining_selector --addresses_gap--> gap:G1
