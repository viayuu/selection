# CLAIMS_FROM_RESULTS — Unified Selector Diagnostic Paper

- Source: `review-stage/AUTO_REVIEW.md` fresh loop R1–R3 (terminated 2026-04-20 at 8.5/10).
- Judge: oracle-pro (GPT-5.4 Pro) via browser MCP, REVIEWER_DIFFICULTY=nightmare.
- Verdict: **Stop. Write diagnostic paper.** Six claims all `supported` at `high` confidence.
- Integrity audit: not run → claims labeled **provisional — no integrity audit**.

## Headline result table

| Method | Test top1 | Δ vs R18 | 95 % CI | Pre-reg? | Sig? |
|---|---:|---:|---|---|---|
| R18_alltail (base) | 0.5193 | — | — | — | — |
| gated blend α=0.5, δ=0.1 (representative) | 0.5231 | +0.0038 | [−0.0025, +0.0100] | NO (exploratory) | no |
| R36 gated k=32 val-lock | 0.5202 | +0.0009 | [−0.0044, +0.0065] | YES | no |
| R37 gated k=64 val-lock | 0.5200 | +0.0007 | [−0.0041, +0.0056] | YES | no |
| R37 gated k=256 val-lock | 0.5191 | −0.0002 | [−0.0061, +0.0057] | YES | no |
| R33 binary learned gate | 0.5206 | +0.0013 | [−0.0056, +0.0081] | YES | no |
| R34 prospective blind-spot | 0.5181 | −0.0012 | [−0.0047, +0.0024] | YES | no |
| R35 dual-retrieval agreement | 0.5225 | +0.0032 | [−0.0031, +0.0094] | YES | no |
| R38 train-time fusion head | 0.5222 | +0.0029 | [−0.0030, +0.0087] | YES | no |

- 10 000-sample paired stratified bootstrap, per-problem resampling.
- SBS baseline: 0.5368 → R18 is −0.0175 below SBS (unchanged through all interventions).
- Regret (mean relative SBS-oracle gap %): base 1.007 %, method 1.007–1.015 % across every config.

## Claim 1 — Performance wall and regret concentration

- **Statement**: The unified selector across 18 routing families plateaus at strict top-1 ≈ 0.52; 77 % of relative regret is concentrated in the 29.2 % of the test set where the SBS-oracle gap exceeds 1 %.
- **claim_supported**: yes
- **what_results_support**: R18_alltail test top1 = 0.5193; SBS = 0.5368 (Δ = −0.0175). Margin-bucket decomposition `margin_bucket_test/` with buckets (<0.1 %, 0.1–0.5 %, 0.5–1 %, >1 %) shows the >1 % bucket holds 29.2 % of test mass and 77 % of the cumulative relative regret.
- **what_results_dont_support**: does not establish that no architecture could cross 0.52; restricted to the R18 family and the S5 pre-training regime.
- **missing_evidence**: none needed for the diagnostic claim; a separate architecture-sweep paper could explore whether 0.52 is a fundamental ceiling.
- **suggested_claim_revision**: none.
- **next_experiments_needed**: none for this claim.
- **confidence**: high.

## Claim 2 — Support collapse of the base selector

- **Statement**: The base selector exhibits support collapse — 50 of 82 arms are zero-pick under R18 (35 are genuine collapse, 15 are essentially zero-mass arms), and 17–20 % of test instances have their oracle arm inside this collapsed set.
- **claim_supported**: yes
- **what_results_support**: `runs/R29b_zp_rescue` reports 4714 / 18 000 ≈ 20.01 % of test instances with oracle on a pick-frequency-zero arm under R18. Arm-frequency histogram gives 50 zero-pick arms, 35 of which have non-trivial oracle mass (genuine collapse) vs 15 with ≤ 3 oracle instances (benign degeneracy).
- **what_results_dont_support**: does not by itself prove the collapse is *causal* — it is a descriptive property of the learned policy.
- **missing_evidence**: none for the diagnostic framing. Causal attribution would require targeted re-training ablations (explicitly out of scope).
- **suggested_claim_revision**: none.
- **next_experiments_needed**: none.
- **confidence**: high.

## Claim 3 — Retrieval exposes and locally rescues collapse

- **Statement**: A k-NN retrieval prior over R18 pooled embeddings rescues 8–17 % of the zero-pick oracle mass (8 % for conservative gated configs, up to 17 % for pure prior); the 20 % collapse subset moves from 0 % top1 to 11.1 % under the R38 fusion head.
- **claim_supported**: yes
- **what_results_support**: R39 `gross_decomp.md` table: ZP rescue ranges 438 / 4714 = 9.3 % (val-best gated) to 797 / 4714 = 16.9 % (prior-only) with ZP harm always exactly zero (base gets 0 correct on ZP by construction). R38 fusion zp-rescue = 0.1111 on 3601 test ZP instances (flip-count definition). This is a *mechanistic* contribution — we can identify and partially repair a named failure subset.
- **what_results_dont_support**: rescue magnitude is bounded; it does not claim the full 20 % mass can be recovered.
- **missing_evidence**: none for local-rescue claim.
- **suggested_claim_revision**: none — quote the 8–17 % range with configuration-specific numbers in the paper.
- **next_experiments_needed**: none.
- **confidence**: high.

## Claim 4 — All pre-registered global interventions null

- **Statement**: Across six validation-locked / pre-registered interventions — gated blend (k∈{32, 64, 256}), R33 binary learned gate, R34 prospective blind-spot override, R35 dual-retrieval agreement, R38 train-time fusion head — none produced a statistically significant improvement in macro top-1 or macro regret under a 10 000-sample paired stratified bootstrap.
- **claim_supported**: yes
- **what_results_support**: `REVIEW_STATE.json → pre_registered_test_table` enumerates all seven pre-registered rows with Δtop1 ∈ [−0.0012, +0.0032] and every 95 % CI crossing zero. Δregret % ∈ [−0.002 %, +0.008 %], also null. wr/rw ratios 0.99–1.04, well below the 1.08 threshold set in R2's pre-registration.
- **what_results_dont_support**: does not preclude that a completely different architecture or a re-trained base selector could help.
- **missing_evidence**: none for the null claim; by design the pre-registration freezes the test protocol and prevents researcher-degrees-of-freedom.
- **suggested_claim_revision**: report as "six pre-registered interventions" in the abstract; list the seven rows in a table (R37 has two k values).
- **next_experiments_needed**: none.
- **confidence**: high.

## Claim 5 — Gross rescue / gross harm decomposition (central figure)

- **Statement**: The gross rescue/harm decomposition shows that on collapse-oracle instances retrieval provides large positive rescue with essentially zero possible harm, whereas on off-collapse instances it incurs more selection harm than rescue, canceling 80–95 % of the collapse-mass gain.
- **claim_supported**: yes
- **what_results_support**: R39 decomposition (`runs/R39_gross/gross_decomp.md`):
  - Gated α=0.5, δ=0.10: ZP rescue +682, ZP harm 0 → ΔZP = +682; non-ZP rescue 1016, non-ZP harm 1629 → Δnon-ZP = −613. Net = +69 on 18 000.
  - Prior-only α=0.5 (no gate): ZP rescue +797, ZP harm 0; non-ZP rescue 1142, non-ZP harm 1902, Δnon-ZP = −760. Net = +37.
  - Gated val-best α=0.8, δ=0.2: ZP rescue +438 vs non-ZP net −421. Net = +17.
  - Across configs, non-ZP harm / ZP rescue ratio = 0.80–0.95.
- **what_results_dont_support**: does not prove that no future method can break the cancellation — only that every global method we tried falls under the same pattern.
- **missing_evidence**: none for the diagnostic framing.
- **suggested_claim_revision**: present as the paper's central figure (scatter of harm vs rescue per config) accompanied by the four-row decomposition table.
- **next_experiments_needed**: none.
- **confidence**: high.

## Claim 6 — Regret invariance

- **Statement**: The mean relative SBS-oracle regret is unchanged across base and all interventions (1.007 % → 1.007–1.015 %), reinforcing that the small point-estimate top-1 variations are post-hoc noise rather than real improvements.
- **claim_supported**: yes
- **what_results_support**: `preregistered.json` per-config regret figures; bootstrap Δcost% CIs include zero for every intervention.
- **what_results_dont_support**: regret is cost-weighted and can mask redistribution of picks; but combined with Claim 4's top-1 nulls it is mutually reinforcing.
- **missing_evidence**: none.
- **suggested_claim_revision**: none.
- **next_experiments_needed**: none.
- **confidence**: high.

## Overall judgment

- All six claims supported; none over-reach the data.
- Paper framing: **diagnostic paper**. Title sketch — "Support Collapse in Neural Solver Selection: Retrieval Exposes but Cannot Reliably Repair."
- Abstract: verbatim oracle-pro R3 paragraph in `REVIEW_STATE.json → r3_oracle_pro_final_abstract`.
- Loop is terminated (`REVIEW_STATE.status = completed`, `last_score = 8.5`).
- Integrity audit not run; downstream workflow should either run `/experiment-audit` or keep the "provisional" label through the paper draft.
- Next workflow: `/paper-plan` → `/paper-write` over these six claims plus the Method Description in `AUTO_REVIEW.md`. Do not launch further method experiments.
