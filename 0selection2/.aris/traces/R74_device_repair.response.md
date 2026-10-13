`1465.114` failed `1:0` after **13s**. The device adapter worked; native `ATSPEnv.load_problems` then hit a TorchRL batch-size/Composite-spec shape mismatch.

Syntax check: **1 invocation, exit 0, 0.019s**. The 20,000 complete TSP/CVRP rows remain untouched. All 16 persisted partial base rows were verified and reused, adding **zero base forwards**; cumulative actual base work remains **20,016 rows**.

Charged Slurm wall is **1,474s**. No finite ATSP batch or optimizer update occurred. No retry, TEST, or watcher refresh ran; all owned sessions are terminal.

[Full handoff, source amendment, and receipts](/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition/continuation3/handoff.json).