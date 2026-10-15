# R75 CPU/GPU Pipeline Continuation

The same seed2 run continues from logged/durable update 25380; no logged successful update was lost or replayed. The original model, optimizer, AMP retry path, RNG, data plan, batch384 and scientific protocol are unchanged.

One CPU thread prepares exactly the next real task group. Only the main thread uses pinned staging and CUDA. No additional group is prefetched into GPU memory. An isolated runtime copy removes two timing-only global CUDA barriers; the original scientific source files are unchanged.

| Measurement | RAM input before | CPU prefetch after |
|---|---:|---:|
| Mean update seconds | 0.4158 | 0.4118 |
| Update rows/s | 2770.4 | 2797.3 |
| Wall rows/s, including intervening logging/checkpoints | 2754.7 | 2715.3 |

AMP retries in the two windows: 0 before, 1 after. All reported successful updates had finite loss and gradients.

Observed update speedup: 1.010x; wall throughput ratio: 0.986x. These are balanced120-update observed windows with different batches, not a paired controlled benchmark or a guarantee of constant99-100% GPU utilization.

The first observed windows do not establish an additional material speedup: update time is almost unchanged and wall throughput did not improve. The earlier lossless RAM fix remains the main measured acceleration. Do not attribute the retry-count difference to prefetching; batches differ.

The resumed epoch is mixed pre/post-continuation; its optimization.seconds only covers the latest process segment. Use complete later epochs for epoch comparisons. The final completion metadata records both runtime continuations and partial timing scopes.

Added execution files: r75_cpu_prefetch.py, r75_pipeline_runner.py, r75_pipeline_boundary.py, run_v4_r75_pipeline.sh and r75_pipeline_report.py. The running command is `bash code/V4/run_v4_r75_pipeline.sh` in tmux `r75_train_pipeline2`; do not launch a duplicate. The original native70 goal stays paused.
