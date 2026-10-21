# R91 Running

Started once on2026-10-20 at14:08CST. Both first full18 cycles passed the bounded
finite-loss/gradient and actual PID/UUID checks. R89 was not interrupted.

| Arm | GPU | Session | Physical host PID | Container PID |
| --- | --- | --- | --- | --- |
| A_utility_ce | gpu03 physical3, RTX3090 | r91_A_utility_ce |1390523|2663967|
| B_utility_probabilities | gpu03 physical4, RTX3090 | r91_B_utility_probabilities |1341162|2663707|

Recent startup resource snapshot:100%/98%,18878MiB each. This is a snapshot, not
continuous utilization. Logical batch1024, microcap352; full57600 successful
updates per arm. Identical fresh seed2 parameter hash and fixed legal pools.

Progress updates automatically in `goal_progress.json` and each arm's
`status.json`; update-level metrics are in each arm's `updates.jsonl`. Launcher
stage logs are in `../R91_launch/<arm>/train.log`. The quiet CPU watcher is
`tmux r91_goal_watch`. No automatic restart or protocol change is enabled.

Each arm's own pipeline is TRAIN+uniqueVALlock+TEST, idempotent TEST, same-GPU
LIB/report, then CPU comparison after both deliveries complete. No TEST/LIB
metric selects weights. PDF reports will appear in `full_model_statistics/`
and `comparison_artifacts/` after completion.

Bounded startup evidence:
`../R91_launch/launch_startup_check.json`.
