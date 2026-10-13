First finite OVRP batch passed. **`1465.117` remains live** in `r74_continuation6`, trainer PID `2280579`, pane `%143`.

- Both probes: **16 rows, 50 starts**, distinct visits **6 then 11**, limit **12**, finite features, `done=false`, zero completions.
- Three complete tasks skipped: **30,000 cached rows**. Reused **16 OVRP base rows**, zero new forwards. Actual base work at first batch: **30,016 rows / 2,377 forward batches**.
- Successful generation counters: **30,032 initializations**, **1,583,566 starts**, **126,554 model calls**, **125,044 decoder calls**.
- Prior failed probe work charged separately: **32 initialization rows, 14 returned model calls, one failed call, 11 decoder computations**. These counts are source/control-flow-derived. The current replay recorded **24 model calls / 20 decoder calls**.
- Syntax: **one invocation**, exit `0`, **0.01916s**.
- Slurm at first-batch observation: **1,547 prior + 9 new = 1,556s**; the live job continues accruing time.

Optimizer updates remain **0**. No TEST or watcher restart. Records: [handoff](/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition/runtime_repair6_handoff.json). Goal70 remains active and unachieved.