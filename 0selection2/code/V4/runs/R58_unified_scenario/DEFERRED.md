# R58 deferred by user

On 2026-10-11 the user explicitly asked to skip R58 and continue the other
performance-improvement goal. This supersedes the earlier instruction to finish
scenario_v2 first. R58 is unfinished, not a successful or rejected model result.

Do not launch R58 task-definition work, solver adaptations, performance-table
generation, relabeling, the full baseline, or test evaluation. Existing code,
qualification/timing evidence, original inputs and historical labels are retained.
No full R58 solve or training controller was running when the work was deferred.

## Open issue retained

The existing B deployments do not all use the same backhaul feasible domain.
Affected task types are VRPB, OVRPB, VRPBL, OVRPBL, VRPBTW, OVRPBTW,
VRPBLTW and OVRPBLTW. ReLD signed-load semantics and classical class1 semantics
must not be described as a verified common task. See the R57 report and source
evidence; this discrepancy has not been repaired by the R58 preflights.

Any further experiment using the existing costs measures prediction in the
**legacy deployment scenario**, with unchanged splits and candidate pools. It
does not certify classical-backhaul feasibility or a repaired Oracle. The
discrepancy is not established as the explanation of the whole Top1 plateau.

## Continuing goal

Continue the requested performance-first auto-review loop: target full18-task
single-solver Top1 of70%, maximum100 rounds, nightmare difficulty, Astra/max
planning/review and Sol6.1/max implementation/execution. No GPU-hour cutoff;
only the designated available RTX3090 may be used. Do not change the metric,
drop difficult tasks/candidates, use held-out labels in the forward pass, or
replace single-solver selection with multiple complete solves to claim70%.
Meaningful completed experiments should be committed and pushed promptly.
