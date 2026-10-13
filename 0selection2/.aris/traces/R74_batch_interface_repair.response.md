Continuation4 `1465.115` **FAILED `1:0` after 13s**. Native batch-size loading and `_reset(None)` passed; permutation creation then raised `Expected a 'cuda' device type for generator but found 'cpu'`. No retry occurred.

- One syntax invocation: exit `0`, **0.01769s**.
- Complete TSP/CVRP caches: **20,000 rows**, skipped entirely. Reused **16 finite ATSP base rows**; zero new base forwards. Cumulative: **20,016 rows / 1,729 forward batches**.
- ATSP configured **20 starts / 5-visit limit**. Native reset returned; no prefix actions or successful batch. Prefix done/finite checks were not reached.
- Cumulative Slurm charge: **1,487s**.
- Observed trainer PID `4042001`, pane `%139`, pane PID `3754598`. Step terminal; `r74_continuation4` and all prior owned sessions are absent.
- Optimizer updates: **0**. No TEST or delivery-watcher restart.

The isolated adapter omits spec resizing and calls native reset directly. Exact changes are in the [source amendment](/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition/continuation4/execution_amendment.json); failure and accounting are in the [handoff](/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition/continuation4/handoff.json). Prior evidence remains preserved. Goal70 remains active and unachieved.