# Claims From Results

## Verdict

- `claim_supported`: `partial`
- `confidence`: `high`
- `next_experiments_needed`: `n/a` (review loop terminated by user)
- `evaluation_mode`: local `result-to-claim` assessment from `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`

## Primary claim

On this 18-family routing benchmark, the unified selector is competitive with a strong distilled baseline but does not significantly beat it; the strongest supported contribution is diagnostic, showing that selector misses split into a large low-margin near-tie regime and a minority `>1%`-margin regime that dominates regret.

## Supporting sub-claims

### 1. The unified selector matches the strong baseline at the macro level, but superiority is not supported.

Claim text:
The unified selector should be described as comparable to, not better than, the strong distilled baseline on the 18-family test benchmark.

Numeric evidence:

- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: `S5` macro top1 `0.5181`, `R18` macro top1 `0.5192`, `R22` macro top1 `0.5185`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: `R18 - S5` top1 delta `+0.0012`, `95% CI [-0.0025, +0.0049]`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: `R22 - R18` top1 delta `-0.0008`, `95% CI [-0.0038, +0.0022]`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: all cost-delta CIs include `0`.

Bounded-scope caveats:

- This supports descriptive comparability, not formal equivalence or non-inferiority, because no equivalence margin was preregistered.
- The summary is macro-averaged across 18 families; it does not by itself prove per-family matching.

### 2. Much of the strict top1 difficulty comes from near-tie ambiguity rather than large cost mistakes.

Claim text:
Low strict top1 is partly a label-resolution problem in near-tie regions, where the selector often misses the exact winner but still lands within a `1%`-regret neighborhood.

Numeric evidence:

- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: in the `0.1-0.5%` bucket, exact-top1 `0.17`, fuzzy@1% `0.98`, mean regret `+0.28%`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: in the `0.5-1%` bucket, exact-top1 `0.17`, fuzzy@1% `0.98`, mean regret `+0.63%`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: only `57.9%` of test instances have a uniquely-best arm at `1%` margin, even though `94.6%` do at `0.1%` margin.

Bounded-scope caveats:

- "Near-tie ambiguity" is supported more strongly than "benign": the low-margin buckets still account for about `7.8%` of total regret (`2.2% + 5.6%`), and the `<0.1%` bucket adds another `14-16%`.
- The split is in error consequence and margin structure, not in raw exact-top1 alone, which is similarly low (`0.17`) across all non-easy buckets.

### 3. The cost-critical failures are concentrated in a minority `>1%`-margin subset.

Claim text:
The paper's strongest positive finding is that most regret comes from a minority of clear-gap cases, not from the large mass of ambiguous near-ties.

Numeric evidence:

- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: the `>1%` bucket is `29.2%` of test mass.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: in that bucket, fuzzy@1% drops to `0.18` and mean regret rises to `+2.69%`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: the `>1%` bucket contributes `76-78%` of total relative regret.

Bounded-scope caveats:

- The subset is a minority, but not tiny; "minority" or "smaller subset" is safer than "rare subset."
- This statement is benchmark-conditional: the margin buckets are defined relative to the 18-solver oracle, not to global optimality.

### 4. Additional recipe complexity does not break the baseline wall, which strengthens the diagnostic interpretation.

Claim text:
Oversampling, reranking, architecture changes, loss changes, and ensembling fail to produce a reliable test improvement, suggesting the bottleneck is structural rather than a missed training trick.

Numeric evidence:

- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: `R22 - S5` top1 delta `+0.0004`, `95% CI [-0.0039, +0.0049]`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: validation regret for `R18+R21` improves `1.0005% -> 0.9909%`, but test regret worsens `1.0072% -> 1.0202%`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: SWA collapses to top1 `0.449` and `+2.55%` vs_sbs.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: architecture, loss, ensemble, and reranker variants all fail to beat `S5` on test.

Bounded-scope caveats:

- "Structural" should still be framed cautiously because the dataset is IID and relatively small (`1000` train instances per problem).
- The negative sweep rules out many obvious recipe tweaks, not every possible modeling alternative.

### 5. Apparent arm collapse is partly due to oracle concentration, but there remains a real uncovered tail.

Claim text:
The selector's concentration on a few arms is partly explained by heavily skewed oracle winners, yet a nontrivial share of test mass still places the oracle on never-picked arms.

Numeric evidence:

- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: top-2 oracle arms account for `67.4%` to `97.6%` of oracle wins per problem, mean about `90%`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: zero-pick arms with oracle wins are `33` for `S5`, `35` for `R18`, and `35` for `R22`.
- `review-stage/RESULT_TO_CLAIM_EVIDENCE_2026-04-20.md`: oracle-on-zero-pick test mass is `17.2%` for `S5`, `20.0%` for `R18`, and `20.3%` for `R22`.

Bounded-scope caveats:

- This is a diagnostic side claim, not the main paper headline.
- Because the oracle is best-of-18, the uncovered tail is relative to the candidate solver pool, not to all possible solvers.

## What the current results do not support

- Do not claim the unified selector beats or significantly improves on the strong baseline.
- Do not claim formal equivalence or non-inferiority without a stated equivalence margin.
- Do not claim family-by-family matching unless you add a per-family results table from existing evaluations.
- Do not say exact-top1 itself separates the regimes; what separates is regret and fuzzy tolerance under the same low top1.
- Do not say the low-margin regime is harmless; it is lower-regret, not zero-regret.
- Do not generalize beyond this IID benchmark or beyond the best-of-18 oracle.

## Missing evidence to disclose as limitations

- No preregistered equivalence margin for the word "matches."
- No per-family confidence intervals in the present summary.
- No OOD or distribution-shift evidence.
- Oracle is limited to the 18 available neural solvers.
- Evidence snapshot is a curated summary, not raw log files.

## Suggested paper-safe primary claim

We evaluate unified solver selection over 18 routing families and find that a single selector is competitive with a strong distilled baseline but not significantly better. The main contribution is a diagnostic decomposition of selector error: most strict top1 misses occur in low-margin near-tie regions, while a minority `>1%`-margin subset accounts for most regret.

## Suggested Section 4/5 headline statements

1. Unified selection is competitive with a strong distilled baseline, but not significantly better.
2. Strict top1 underestimates performance on near-tie instances.
3. A minority `>1%`-margin subset accounts for most selector regret.
4. Additional training and reranking machinery fail to improve test performance.
5. Oracle skew explains part, but not all, of the selector's arm concentration.
