The explicit CPU `randperm` fix passed. TRAIN/ATSP completed, then `1465.116` **FAILED `1:0` after 60s** in OVRP: the decoder hook called `.detach()` on a tuple. No retry; `r74_continuation5` is absent.

- First finite ATSP batch: **16 rows, 20 starts**, visits **3 then 5**, limit **5**, `done=false`, finite features, zero completions.
- Complete caches: **30,000 rows** across TSP/CVRP/ATSP. Successful counters: **2,376 batches**, **1,581,966 starts**, **126,530 model calls**, **125,024 decoder calls**.
- Reused **16** base rows; executed **10,000 new rows**, including **16 failed OVRP rows**. Actual cumulative base work: **30,016 rows / 2,377 forward batches**.
- Syntax: **one invocation**, exit `0`, **0.01776s**.
- Slurm charge: **1,487 + 60 = 1,547s**.
- Optimizer updates: **0**. No TEST or watcher restart.

All prior evidence and the completed ATSP cache are preserved. Details: [terminal handoff](/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition/continuation5/terminal_handoff.json). Goal70 remains active and unachieved.