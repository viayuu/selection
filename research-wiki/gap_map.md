# Gap Map

- **gap:G1** Benchmark/OOD gap remains unresolved, especially on `CVRPLIB`, even though the simple unified selector is already strong on standard IID `val/test`.
- **gap:G2** Current evaluation mixes truly competitive solver-selection cases with low-entropy one-solver-dominant variants, which can dilute the scientific interpretation of overall gains.
- **gap:G3** The project still lacks a principled account of which solver preferences are environment-invariant and transferable across problem families, scales, and benchmark shifts.
- **gap:G4** The current selector optimizes pooled compatibility ranking, but not explicitly residual regret against a safe anchor or robustness under environment shift.
