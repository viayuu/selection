# R78 Round005: One Fixed A/B Probability-Average Screen

Frozen A ep13/u16146 and B ep28/u34776 with their respective original
TRAIN-derived preprocessing. Only existing complete VAL logits are read.
Masked FP32 softmax T1, fixed probability mean0.5/0.5 for every task, exact
ties by ascending global solver ID. No training, logits/weight averaging,
taskwise choice, parameter scan, GPU/model/solver/API/TEST/LIB or Git action.

One complete70110-row VAL candidate,3895/task,18 tasks, CPU chunks1024.
Two synthetic tie/mask/normalized-CE checks and necessary row/pool alignment
only; no broader audit or new VAL/model pass. B's original unrounded macro
Top1 /mean-cost /actual-regret gate is decisive. Any failed component retains
whole-version B and closes this fixed rule without rescue.

Execution: code/V4/r78_ab_equal_weight_screen.py.
Small checks: code/V4/test_r78_ab_equal_weight_screen.py.

Outputs: RESULTS.md, results.json, all18_VAL_metrics.csv,
native_method_counts.csv, decision_changes.csv,
decision_change_costs_by_category.csv, VAL_candidate_predictions.npz,
locked_candidate_masks.npz, protocol.json, progress.json and screen.log.
Pass only: ensemble_release_manifest.json pending explicit formal approval;
failure: screen_rule_manifest.json is a preserved negative screening record.

Mixture CE is descriptive after probability normalization, not evidence of
profitable routing. Real deployment uses BOTH frozen networks, roughly2x
network arithmetic. CPU cache-screen runtime is not real inference latency,
memory, or a single-network deployment claim. No hardware benchmark here.

## Completed Screen

CPU cached-screen runtime0.290172s, one candidate /70110 rows, zero optimizer
updates; this runtime excludes Python imports and is NOT deployment latency.
All three fixed gates failed; the equal-probability route is closed negative,
whole-version B is retained, and file workflow awaits locked NSS TRAIN/VAL.
No release manifest or TEST/LIB permission was issued.

| Rule | Native Top1 % | Raw mean cost | Actual regret % | CE |
|---|---:|---:|---:|---:|
| B |70.894308943|11.395558343544208|0.401326745448|0.713641610410|
| Ensemble |70.753102268|11.395610637022680|0.402752678645|0.711996995605|

2444 changed choices:751 corrected,850 harmed,843 wrong-to-wrong. All positive
and negative FP64 cost contributions are preserved per task and switch category
in the CSVs. VAL zero/winner method pairs30->33, covered native winners
2648->3268 (3.776922%->4.661247%). Lower descriptive CE does not override gates.

CHECKS.md records the two necessary checks and metadata-only startup fix.
preparation_failure.log retains the pre-VAL traceback; screen.log is the
successful sole candidate execution. Both members' checkpoint SHA and own
frozen preprocessing state/SHA are preserved in screen_rule_manifest.json.
