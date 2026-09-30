# R41 Experiment Integrity Audit

Date: 2026-10-01 (Asia/Shanghai)

Auditor: fresh read-only Codex agent `01a0f345-9d96-7e91-9b4b-4d7f1346684c`, inherited model, requested reasoning ultra. `review_independence=same-family`, `acceptance_status=provisional`.

## Overall Verdict: WARN

The reviewer independently recomputed all 72,000 test selections and validated 12 solver process records, including prechecks, and 512 formal repeat costs. No blocking corruption was found. Historical source and solver-generation provenance remain incomplete; they must not be presented as fully reproduced original recipes.

| Check | Reviewer status | Evidence |
| --- | --- | --- |
| A. Ground truth provenance | WARN | `performance_targets.py` reads dataset raw costs/native ind; `original_solver_provenance.md` lists incomplete historical bindings. |
| B. Score normalization | PASS | SBS/Oracle denominators come from dataset costs, never the model's predictions. Macro percentages are averaged per problem. |
| C. Result existence/numbers | PASS | `test_predictions/`, `test_results.json`, and `test_prediction_replay.json` agree for all 72,000 predictions. |
| D. Metric execution | PASS | Saved metrics are produced by invoked classification/stability functions. Incomplete full pools are explicitly unmeasured. |
| E. Scope | PASS, qualified | Four single-seed models; only two RELD methods on 128 OVRPTW instances have formal repeats. No complete-pool winner stability claim. |
| F. Evaluation type | real_gt, qualified | Dataset-provided solver outcomes, not certified mathematical optima. Head-pair metrics are conditional and partial-pool. |

## Findings And Remediation

1. Original protocol/evaluation hashes were not overwritten. Early execution-time R41 source snapshots are unavailable. Selected-source snapshots and stage revision records now cover subsequent validation prospectively; see `SOURCE_REVISIONS.md`.
2. Saved solver results now require pinned source/weights, exact original model/decoder parameters, seed, augmentation/sample budget, independent PIDs, full sample indices, input field hashes and finite positive costs. Actual records pass; rejection tests cover invalid contracts and empty witnesses.
3. Undefined conditional pair metrics now render as `not measured`. Interpretation is conditional on measured in-pair choices; the reviewer confirmed that the no-in-pair case does not produce a representation/generalization claim.

## Claim Impact

- Four frozen 18-problem test comparisons: supported for the fixed single-seed checkpoints and dataset label convention; no reliable-advantage claim from small differences.
- RELD_MOEL/RELD_MTL repeat costs and winner-difference signs are stable: supported on the fixed 64 train + 64 val sample and pinned configuration, with historical weight/CLI limitations disclosed.
- The complete TSP/OVRPTW solver pool is stable/noisy: unsupported, not claimed. Missing methods remain NaN and complete-pool metrics remain unmeasured.
- The entire dataset has a maximum achievable Top1 inferred from repeats: unsupported, not claimed.

Raw reviews: `.aris/traces/experiment-audit/2026-10-01_r41/`. The initial request manifest is explicitly reconstructed, not a verbatim capture; later remediation requests/responses are preserved verbatim.
