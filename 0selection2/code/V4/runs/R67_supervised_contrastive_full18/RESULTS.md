# R67 Results

R67 is complete: both fixed40+40epoch TRAIN pipelines, validation-only unique
checkpoint lock, ONE original18 TEST per model and ONCE CPU analysis. Official
arm B was fixed before TEST by integer validation correct count; no reselection.

| Model | Selected Stage2 Epoch | Validation Top1 % | Test Top1 % | Test Actual Regret % |
|---|---:|---:|---:|---:|
| R67A: classification pretraining | 16 | 48.705556 | 47.488889 | 0.975968 |
| R67B: contrastive pretraining, official | 36 | 49.288889 | 48.100000 | 0.955544 |
| R60B39: fixed historical reference | 39 | 49.361111 | 48.244444 | 0.940486 |

B improves matched A by0.583333 validation and0.611111 TEST percentage points,
but does not exceed the fixed R60B reference. This is limited one-seed evidence,
not a breakthrough, noise/accuracy-ceiling conclusion or70% goal completion.

Full original18/task/family/ALL metrics, best/final/last5 validation summaries,
stage1 objectives, stage2 curves and actual compute are in [REPORT.md](REPORT.md),
[comparison.csv](comparison.csv), [compute_summary.json](compute_summary.json)
and [delivery_receipt.json](delivery_receipt.json). Paired actual training:
38400updates/28.8Mselected-row presentations; TRAIN1465.96 and TEST1465.97 both
COMPLETED/0:0. Analysis ran once, return0, producing667comparison rows/331plots.

Primary Astra/max TRAIN/VAL-only review: performance4/not ready for70%, inspected
training integrity9/ready, next plan8/ready PLAN ONLY. Actual fresh nightmare
delivery review:9/ready with no actionable defects, same-family/provisional;
performance remains4/not ready. Raw responses/traces are preserved in reviews.
No corrective rerun and no new TEST feedback informed the next mechanism.

[objective_audit.json](objective_audit.json) explicitly records the original
whole-goal threshold as NOT achieved. Parent curates one completed-version
publication; [DELIVERY.md](DELIVERY.md) explains frozen launch-era documents.

Weights, complete prediction NPZ/raw costs, rawupdates, W&B and regenerable
caches remain LOCAL under user cleanup; Git alone is not all inference assets.
R58 fully deferred; eight legacy B-task feasible-domain mismatches unresolved
and disclosed. Original full18 single-solver native70% goal remains ACTIVE.
