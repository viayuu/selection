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

## Constraints carried forward

- Do NOT re-run exploratory test-set grid sweeps — they inflate Δ by ~5× compared with val-locked numbers.
- Do NOT treat val-win configurations as implying test-win — the pattern failed at R1, R2, R3 repeatedly.
- Do NOT attempt further global post-hoc correction on the R18 base — the gross decomposition proves the cancellation is structural to this class of method.
- A genuine method-improvement paper would need a new base policy (train-time collapse prevention, e.g., CQR-style distribution regularization) and is out of scope for this diagnostic paper.

## Terminal state

- Loop terminated at Round 3 with oracle-pro score 8.5/10 and verdict "Stop the loop. Write the diagnostic paper."
- Six claims in `CLAIMS_FROM_RESULTS.md` all `supported` at `high` confidence, `integrity_status = unavailable` (provisional).
- Next workflow: `/paper-plan` → `/paper-write` over the six claims + Method Description in `review-stage/AUTO_REVIEW.md`.
