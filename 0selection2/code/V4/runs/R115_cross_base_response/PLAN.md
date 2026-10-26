# R115: finite cross-base response learning

Implements the first, CPU-only D1 step of Astra's completed
`docs/research/post_major_paradigm_results_20261026/REPORT.md`.

Use only the corrected R114 TRAIN responses: 512 bases, 2,048 endpoints,
1,536 pairs, 16 applicable problems and 24 problem/family conditions.
No new native solves, labels, validation, TEST or LIB reads.

Two folds are assigned by base, stratified approximately by true node count.
All families/arms/endpoints of one base stay together. Each fold holds out
16 bases per problem. This does not alter the frozen dataset membership.

One fixed multioutput Ridge(alpha=1, intercept), per problem/family, pools
both intervention arms. Inputs are the existing 357 input-only R92 statistics
at the original endpoint and their change. No row IDs, arm IDs, labels or
current-query costs enter the predictor. Standardization is fit-fold-only.

Predict candidate-centered cost change divided by a fixed fit-fold problem
scale: mean original cost over all candidates and unique fitting bases.
Compare with zero and fit-fold problem/family mean responses. Report all
conditions, candidate-response order signs, both arms, and unique base counts.
Do not call this a selector result or use actual original costs as a query
prediction anchor.

Predeclared continuation gate: CVRP-demand and VRPL-demand must each lower
MAE by at least 10% versus BOTH controls; OVRPL-demand and OVRPTW-tw must not
harm MAE by more than 10% versus EITHER control (OVRPTW's saved family name
is `tw_center`). No parameter search.
Failure stops this version; it does not refute all neural response learning.

Hard cap: 8 CPU-core-hours, 4 wall-hours with two CPU threads, zero GPU-hours.
Protocol/config and fold manifest are locked before fitting. Run once:

```bash
bash code/V4/run_v4_r115.sh
```

Long work runs in tmux. Deliver PDF/CSV/decision and one curated GitHub
publication after completion. Original R114 inputs and labels stay unchanged.
