# Research Findings (Auto-Review Loop, Rounds 9-16 under nightmare difficulty)

## Headline
- [R16] paper-ready: "A unified selector matches a strong baseline across 18 routing families, and its exact-top1 errors split into two regimes: benign low-margin ambiguity and a smaller regret-dominant >1% failure subset."

## Key quantitative findings
- [R14] negative: R22 (hard-family 2× oversample from R18) failed preregistered +0.008 val top1 bar by factor 80 (achieved +0.0001).
- [R14] null: R22 ≈ R18 ≈ S5 on test (bootstrap 5000-sample 95% CIs all include zero on top1 AND cost).
- [R15] structural: bucket decomposition shows 53.7% of test is SBS≈oracle (top1 ≈ 0.81), but in non-easy 46.3% mass top1 uniformly ≈ 0.17 across all methods — the wall made visible.
- [R15] structural: in 0.1-1% buckets, fuzzy@1% stays 0.97-0.98 (low-margin ambiguity, not capacity failure).
- [R15] structural: >1% bucket (29.2% mass) has fuzzy@1% = 0.18 and contributes 76-78% of total relative regret — the paper's central cost-critical finding.
- [R15] null: R18+R21 blend helps val regret (1.0005%→0.9909%) but hurts test regret (1.0072%→1.0202%) — visual risk-coverage sign flip across all coverage levels.
- [R16] diagnostic: 17.2-20.3% of test mass has oracle on a zero-pick arm (corrected from earlier "~8%" estimate per codex R16 verification).
- [R16] diagnostic: top-2 oracle arms account for 67.4% (CVRP, lowest) to 97.6% (ATSP, highest) of oracle wins per problem — "apparent collapse" is mostly oracle skew.

## Methods that FAILED to beat S5 baseline
- Architecture (arm-attn, CoE, FiLM, depth=6, d=192).
- Loss design (focal, combined, gap_rank, winner-margin, Plackett-Luce, diversity entropy).
- Ensembles (mean_logit, mean_prob, geo_mean, plurality, Borda, calib_prob, min_cost_proxy, SWA weight-space).
- Reranker (shared features R12, raw tokens + both arms R21).
- 8-fold D4 test-time augmentation / training-time augmentation.
- Hard-family 2× oversampling (R22).
- Regression-only Huber cost-gap loss (R23, gradient collapsed).

## Contribution ceiling
- No method across 30+ variants significantly beats S5 on test.
- Codex (R16): "there is no significant method improvement over the strong baseline; the contribution is primarily negative/diagnostic."
- Scale: 6.5-7.0 respectable analysis paper (where we are at 6.8), 7.5+ clean accept (would require new signal source: solver-internal traces, runtime features, higher-precision supervision).

## [2026-04-20] Result-to-Claim Verdict (local)
- Verdict: `partial`.
- Why: the "matches baseline" part is supported only as descriptive macro-level comparability, not as superiority and not as formal equivalence; `R18 - S5` top1 is `+0.0012` with `95% CI [-0.0025, +0.0049]`, and all cost-delta CIs include `0`.
- Strongly supported: the error decomposition is real and paper-worthy. The `>1%` bucket is only `29.2%` of test mass but carries `76-78%` of total regret, with fuzzy@1% dropping to `0.18` and mean regret rising to `+2.69%`.
- Also supported: low-margin misses are mostly near-ties rather than catastrophic mistakes. In the `0.1-0.5%` and `0.5-1%` buckets, exact-top1 is only `0.17`, but fuzzy@1% stays `0.98`.
- Not supported: "beats baseline", "formal equivalence", "per-family matching", or any wording that implies the two regimes are visible in exact-top1 alone.
- Recommended paper-safe headline: a unified selector is competitive with a strong distilled baseline on 18 routing families, and its misses decompose into low-margin near-ties versus a minority `>1%` clear-gap subset that dominates regret.
- Canonical source for claim drafting: `review-stage/CLAIMS_FROM_RESULTS.md` backed by `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`.
