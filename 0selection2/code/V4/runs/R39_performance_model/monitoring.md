# R39 Training Checks

## 2026-09-30 19:12 Asia/Shanghai

- Source: local logs and offline W&B; no online W&B credentials are configured.
- Completed: A/seed2, 60 epochs, 16,200 successful updates. Best validation cost is epoch 53: macro Top1 0.484278, vs_SBS -1.055374%.
- Active: B/seed2, epoch 7. Performance-policy Top1 0.427278, vs_SBS -0.774377%, classification CE 1.183276, performance MSE 0.365522.
- Health: finite logged losses/gradients, no interruption; isolated AMP overflow attempts are retried with the same batch and random state.
- GPU: recent compute utilization 89-99%, reserved/physical memory about 20.2 GB in the B run, batch 640 retained for matched comparisons.
- Evidence: queue.log, gpu_telemetry.csv, winner_cost_seed2/history.json, performance_seed2/history.json. A's 1,080 saved problem/epoch prediction records match raw FP64 labels and recomputed metrics.
- Decision: CONTINUE. Complete the predeclared 60-epoch budget for all six runs; ordinary plateaus/noise do not trigger early stopping.
- Next quality check: 19:42; process telemetry remains every 30 seconds.

## Completion After the User's Single-Seed Amendment

- The user requested one fixed seed rather than seeds 2/3/4. The live B/seed2 training protocol was not changed; the remaining seeds were cancelled before starting.
- A and B each finished 60 epochs and 16,200 successful updates. Both closed their offline W&B runs before the owned queue was terminated after B's completed marker.
- Saved-prediction replay: 2,160 validation problem/epoch records passed. Four checkpoint replays passed with zero output difference and identical selections.
- Revised budget: 2/2 runs complete. B's performance policy did not beat A at best, final or last-five checkpoints.
- Decision: COMPLETE for the revised R39 budget. No test evaluation or additional training is launched. Remaining work is report/integrity review, not monitoring a detached training process.
