# R80 Status

Snapshot UTC: 2026-10-15T13:13:19.721578+00:00
One-shot snapshot, not a live monitor. Refresh with --status.

| Arm | Runner Phase | Updates / Budget | Progress | ETA Hours | TEST Receipt | Reporting | Updated UTC |
|---|---|---:|---:|---:|---|---|---|
| A_ce_only | training | 9540 / 49680 | 19.20% | 3.67 | missing | not_started | 2026-10-15T13:13:14.701461+00:00 |
| B_batch512 | evaluation | 24840 / 99360 | 25.00% | 2.65 | missing | not_started | 2026-10-15T13:13:18.346649+00:00 |
| C_lr3e4 | training | 7092 / 49680 | 14.28% | 3.88 | missing | not_started | 2026-10-15T13:13:16.964359+00:00 |
| D_code_embedding | training | 9684 / 49680 | 19.49% | 2.70 | missing | not_started | 2026-10-15T13:13:16.612943+00:00 |

## Per-Arm Paths

- `A_ce_only`: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/train.log`.
- Report: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/postprocess_status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/reporting.log`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/full_model_statistics/MODEL_STATISTICS.pdf`.
- Launcher: `running`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/launch_A_ce_only.json`.
- `B_batch512`: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/B_batch512/status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/B_batch512/train.log`.
- Report: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/B_batch512/postprocess_status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/B_batch512/reporting.log`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/B_batch512/full_model_statistics/MODEL_STATISTICS.pdf`.
- Launcher: `running`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/launch_B_batch512.json`.
- `C_lr3e4`: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/C_lr3e4/status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/C_lr3e4/train.log`.
- Report: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/C_lr3e4/postprocess_status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/C_lr3e4/reporting.log`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/C_lr3e4/full_model_statistics/MODEL_STATISTICS.pdf`.
- Launcher: `running`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/launch_C_lr3e4.json`.
- `D_code_embedding`: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/D_code_embedding/status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/D_code_embedding/train.log`.
- Report: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/D_code_embedding/postprocess_status.json`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/D_code_embedding/reporting.log`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/D_code_embedding/full_model_statistics/MODEL_STATISTICS.pdf`.
- Launcher: `running`; `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/launch_D_code_embedding.json`.

W&B: https://wandb.ai/yjkds-southern-university-of-science-technology/selector
Specific run URL: runner logs / ARM/wandb/latest-run/logs/debug.log.
Offline W&B is not a training failure. This command does not sync or monitor.
