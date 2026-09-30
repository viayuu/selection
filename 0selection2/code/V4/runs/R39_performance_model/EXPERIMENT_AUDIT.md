# R39 Experiment Audit

Date: 2026-09-30
Auditor: fresh read-only Codex agent, same-family/provisional
Overall reviewer verdict: **WARN** (retained, not upgraded by the executor)

## Scope and Evidence

The revised budget is one seed (2), two configurations, 18 problems and 60 epochs per run. Both completed 16,200 successful updates. Seeds 3/4 did not start. All reported comparisons are validation, not test. The excluded 20-epoch reporting-repair attempt remains archived and is disclosed in RESULTS.md.

| Check | Reviewer status | Evidence |
| --- | --- | --- |
| A. Ground truth provenance | PASS | Original train/validation labels, pools, FP64 costs, native winners and train-only fitted scales were independently checked. |
| B. Score normalization | PASS | Dataset denominators; classification and performance policies are separate. |
| C. Result existence | PASS | Two completed runs; 2,160 validation problem/epoch records and best/final/last-five summaries reproduce. |
| D. Connected evaluation and replay | WARN | Verifier safeguards were missing; the reviewer sampled checkpoint inference rather than independently repeating all full-validation forwards. |
| E. Scope | WARN | Single-seed validation conclusions only; historical absence of test access cannot be independently certified. |
| F. Evaluation type | PASS: real_gt | Dataset-provided offline solver costs and native winners. Oracle means the best available candidate, not a proven route optimum. |

## Applied Repairs

- Exact declared run inventory and complete epoch/problem coverage are required.
- Train-only scales are independently refitted from original FP64 costs; hashes, both policies and aggregates are checked.
- Result group, seed and epoch budget must match the declared run.
- best.pt and last.pt must match the selected/final record's epoch, successful updates, configuration and validation metrics; a last-to-best substitution is rejected.
- RESULTS.md identifies the archived partial attempt and keeps B's classification results diagnostic.

These changes affect verification and reporting, not the trained model, loss, data, or selected checkpoints.

## Deterministic Closeout

The complete V4 suite passed **74 tests**; see verification_unit_tests.log. Reverification passed all **2,160** validation problem/epoch records. Four selected/final checkpoints loaded strictly and reproduced the full validation logits, performance predictions and selections with **maximum output error 0**. The replay also verifies run identity and selected/final binding.

The final metadata guards were validated with tests and replay, not another semantic reviewer call. Consequently these repairs do not rewrite the original reviewer WARN or imply cross-family acceptance. Current deterministic input hashes and reviewed-input hashes are recorded separately in EXPERIMENT_AUDIT.json.

## Claim Impact

R39B did not outperform its matched R39A control on seed-2 validation; this qualified negative result is supported. No cross-seed robustness, causal mechanism or test-set improvement is claimed. Keep R34 as the established reference. No additional experiment was launched during closeout.

## Trace Limitations

Full final and remediation responses are retained under .aris/traces/experiment-audit/2026-09-30_r39/. The earlier 002-fixes request was not retained; it has not been reconstructed. The semantic review is same-family/provisional, not an independent cross-family certification.
