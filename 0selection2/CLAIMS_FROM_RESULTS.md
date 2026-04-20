# Claims From Results

_Local result-to-claim judgment based on completed experiments through the strict-gate pass and the Round 5 oracle review._

## Verdict

- `claim_supported`: `yes`
- `confidence`: `medium`
- `review basis`:
  - loose multi-seed aggregate in [2026-04-19-round4-seed-aggregate.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round4-seed-aggregate.md)
  - strict multi-seed aggregate in [2026-04-19-round5-strict-gate-aggregate.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round5-strict-gate-aggregate.md)
  - best-seed full test analyses in
    - [analysis_test_fixed](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R1_seed2_serial/analysis_test_fixed/analysis_test.json)
    - [analysis_test_gated_fixed](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R1_seed2_serial/analysis_test_gated_fixed/analysis_test.json)
  - oracle reassessment in `selector-round5-review-browser`

## What The Results Support

1. A single raw unified selector across 18 routing problems is feasible, but slightly unsafe relative to SBS.
   - Evidence:
     - raw multi-seed test `macro_vs_sbs = +0.105%` mean
     - per-seed raw results: `+0.227%`, `+0.071%`, `+0.015%`

2. A conservative held-out calibrated SBS-fallback policy can reduce almost all raw-selector harm and achieve near-parity with SBS.
   - Evidence:
     - strict gated multi-seed test `macro_vs_sbs = +0.001%` mean
     - strict gated seed-bootstrap CI: `[+0.000%, +0.003%]`
     - per-seed strict gated results: `+0.001%`, `+0.003%`, `+0.000%`
     - all 3 strict-gated seeds remain at tiny positive measured degradation in unrounded arithmetic

3. Instance features matter materially; the model is not explained purely by problem-id / mask priors.
   - Evidence:
     - metadata-only baseline remained far worse than the main selector after evaluation fixes (`+7.861%` on val in Round 2)

4. The supported deployment story is **safe selective switching with SBS fallback**, not broad solver-selection improvement over SBS.
   - Evidence:
     - oracle Round 5 explicitly marked the strict gated SBS-fallback story as `READY`
     - strict gate enables only `1.0` problem per seed on average, which matches the intended conservative abstention mechanism

## What The Results Do Not Support

1. The claim that the raw unified selector beats SBS on macro cost.
2. The claim that the gated method produces a practically meaningful macro improvement over SBS.
3. The claim that switching is broadly reliable across most of the 18 problems.
4. The claim that combined loss / oversampling / longer training are beneficial directions.
5. The claim that more selector training is the most valuable next use of effort.

## Suggested Claim Revision

Use this narrowed main claim:

> A unified neural selector by itself is slightly unsafe, but a strict held-out calibrated SBS-fallback policy makes unified solver selection safe in deployment, matching SBS at near-zero macro cost while preserving a small set of auditable switching opportunities.

Avoid the following unsupported wording:

- “beats SBS across 18 problems”
- “consistently improves over SBS”
- “strong macro gains”
- “switches reliably on most problems”

## Missing Evidence

1. Final paper assembly around the conservative story:
   - strict gate protocol in the main text
   - raw selector demoted to ablation / failure mode
   - per-problem enable-or-fallback table in the main text
2. A short limitation paragraph stating:
   - only 3 seeds
   - mostly-SBS deployed behavior
   - sparse switching support

## Recommended Next Step

- `next_experiments_needed`:
  - **No new selector training.**
  - Finish the paper-facing framing around the strict gated result.
  - Keep the switch-diagnostics table and per-problem enable/fallback table as the main empirical support for deployment.

## Routing Decision

- `route`: `confirm_and_reframe`

This project should move forward as a **modest, conservative near-SBS-parity strict fallback paper**, not as a raw performance-improvement paper.
