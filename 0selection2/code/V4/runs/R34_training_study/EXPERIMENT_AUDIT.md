# Experiment Audit

Date: 2026-09-30 (Asia/Shanghai)
Reviewer: fresh read-only Codex agent, inherited model, high reasoning.
Agent: `01a0ee70-da0a-78b3-99a4-4fd36954cf17`.
Review independence: same-family. Acceptance status: provisional.

## Final Verdict: PASS (Provisional)

No blocking or remaining P2 issue was found in the audited results or the persisted winner-frequency-policy fix.

| Check | Status | Evidence |
| --- | --- | --- |
| A. Ground truth provenance | PASS | Dataset labels; all 54 label files inspected in the preliminary review; no label-value dependency in dual-stream forward. |
| B. Metric normalization | PASS | Sample-count accuracy and dataset SBS/VBS denominators; macro averaging disclosed. |
| C. Artifact/claim correspondence | PASS | `comparison.json`, `paired_comparison.json`, saved logits and `results_summary.md` agree. |
| D. Evaluation wiring | PASS | Validation selection, frozen checkpoint, validation replay, then test are supported by code, logs and timestamps. |
| E. Scope | PASS | Claims explicitly restricted to one seed and 18 problems, not across-seed robustness. |
| F. Evaluation type | real_gt | Recorded solver-selection targets; pool VBS is not a proven routing optimum. |

## Independently Verified

- Test top1 47.3444%; improvement over R33a 1.7222 percentage points and over R32e 1.3944 points; mean cost 11.1729 versus R33a 11.1881.
- Saved logits and GT support 1996 corrected, 1686 harmed, net 310 additional correct predictions and improvement in both metrics on 16/18 problems.
- Winner-cost epoch50 was selected using validation top1 48.4667%; source/frozen checkpoint hashes agree with result records.
- Top1/top2/top3 matched exactly on all 108 model/split/problem rows. Maximum cost discrepancy was about 1.34e-6, consistent with floating-point reductions.
- Training budgets, final train/validation accuracy and macro arithmetic matched.
- Five evaluation-guard tests and the native-frequency regression passed during review.
- Historical weighted continuation now restores its frequency policy, with legacy fallback, rather than silently switching its weighting definition.

## Limits

The reviewer did not rerun checkpoint-to-logit inference, the full 25-test suite, or the training-calling policy regression. Those were executed by the main agent; the reviewer inspected the policy regression and independently compared existing resumed/uninterrupted smoke states.
No new experiments, GPU use or artifact edits were performed by the reviewer.

## Resolved Preliminary Findings

Completion/budget gates and immutable selection/hash checks were added before final testing.
All checkpoint selection trackers now update before best-checkpoint saves; same-directory history rewind is rejected.
Future native-label class frequencies are corrected, while the completed historical control retains its old frequency definition and is explicitly disclosed.
Historical frequency policy is persisted/restored. These repairs are not retroactively credited as gains of the completed control.

The independent reviewer classified all A-F checks as passing for this restricted report and approved a provisional PASS, not a multi-seed performance claim.
