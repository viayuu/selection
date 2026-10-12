# R68 Results

Both fixed40-epoch fine-tuning arms, validation-only checkpoint selection,
ONE original18 TEST per model and ONCE CPU analysis are complete. Official A
was fixed from integer validation correct counts before TEST; no reselection.

| Model | Selected Epoch | Validation Top1 % | Test Top1 % | Test Actual Regret % |
|---|---:|---:|---:|---:|
| R68A: ordinary AdamW, official | 28 | 48.022222 | 46.861111 | 1.015239 |
| R68B: SAM + AdamW | 35 | 47.866667 | 46.644444 | 1.042468 |
| R67B: fixed historical reference | 36 | 49.288889 | 48.100000 | 0.955544 |
| R60B39: fixed historical reference | 39 | 49.361111 | 48.244444 | 0.940486 |

SAM loses28 validation and39 TEST correct decisions against its matched
ordinary-AdamW control, and does not improve selection regret. Neither R68 arm
improves the fixed historical references. This is a negative one-seed result,
not evidence for label noise, an accuracy ceiling or70% goal completion.
No SAM coefficient scan, corrective training or TEST retry is authorized.

Both arms start from the SAME R67B terminalstage1 contrastive pretrained TRUNK
and the SAME entire fresh utility head670003, then train all selector parameters.
They are not from-scratch encoders. Final checkpoints are complete ordinary,
unperturbed AdamW selectors: one forward/query, one completed solver/query,
no projector, averaging, head transplant or task-wise checkpoint assembly.

Each arm completed9600 NEW updates/7.2Mselected rows; pair19200/14.4M.
Inheritedstage1 9600updates/7.2Mrows is counted ONCE separately. Actual TRAIN
walltime A/B was3002.024/5677.557seconds. Successful view rows14.4M/28.8M,
so rows and updates match but gradient compute does not. Real batch256/task,
three tasks/effective768; tail16/task retained. TRAIN1465.99 and TEST1465.100
both COMPLETED/0:0. CPU analysis ran once, return0,805comparison rows/330plots.

Full task/family/ALL curves, best/final/last5 validation, selected metrics and
actual compute are in [REPORT.md](REPORT.md), [comparison.csv](comparison.csv),
[compute_summary.json](compute_summary.json) and
[execution_receipt.json](execution_receipt.json). Historical references reuse
saved official reports; no reference inference or retrospective TEST occurred.

Actual Astra/max primary TRAIN/VAL-only review: performance4/notready for70%,
inspected training integrity9/ready, nextR69plan8/ready PLANONLY. No actionable
implementation defect or corrective replay. Actual FRESH nightmare delivery
review:9/ready for legacy reporting only, no actionable defects; performance
remains4/notready. Full raw responses/traces are preserved under reviews.
New R68 TEST and analysis are excluded from primary next-plan design. Delivery
readiness never establishes performance success.
[objective_audit.json](objective_audit.json) records the whole original70%
objective as NOT achieved; the goal remains ACTIVE.

R69's new full18 task-specific arm will compare against the EXACT reused R68A
shared control. The actual primary reviewer approved protocol equivalence;
there is no A retraining/recount/reselection/retest. The new B initializes from
the common PRE-fine-tuning state, never trained A28. Its best is one common-epoch
18-branch bundle; selection remains full18 validation only. Capacity and active
parameter-update counts differ. This is an adopted PLAN, not measured benefit.

Publication is one completed-version commit after review. Weights, prediction
NPZ/raw costs, rawupdates, W&B and regenerable schedules/caches remain LOCAL
under user cleanup. Git alone is not all inference assets; see [DELIVERY.md](DELIVERY.md).
R58 is fully deferred; eight legacy B-task feasible-domain mismatches remain
unresolved/disclosed, not repaired or used to veto the native70% target.
