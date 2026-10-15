# R80 Progress

Stable entry: `/public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/PROGRESS.md`.

Refresh once, on demand (no persistent monitor):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easynco
python -m code.V4.r80_reporting --root code/V4/runs/R80_training_and_code_ablation --status
```

Open `STATUS.md` in this directory after refreshing. A missing status is displayed
as `not_started_or_no_status`, never inferred to be successful.

| Arm | Training Budget | Records |
|---|---:|---|
| A_ce_only | 49680 updates | `A_ce_only/status.json`, `A_ce_only/train.log` |
| B_batch512 | 99360 updates | `B_batch512/status.json`, `B_batch512/train.log` |
| C_lr3e4 | 49680 updates | `C_lr3e4/status.json`, `C_lr3e4/train.log` |
| D_code_embedding | 49680 updates | `D_code_embedding/status.json`, `D_code_embedding/train.log` |

Per-arm report progress/errors: `ARM/postprocess_status.json` and
`ARM/reporting.log`. Finished PDFs: `ARM/full_model_statistics/MODEL_STATISTICS.pdf`
and `ARM/full_model_statistics/selection_distributions.pdf`.

W&B: https://wandb.ai/yjkds-southern-university-of-science-technology/selector
Use the runner's recorded URL or `ARM/wandb/latest-run/logs/debug.log` for the
specific run. An offline W&B run does not imply training stopped.

```bash
tail -n 40 code/V4/runs/R80_training_and_code_ablation/A_ce_only/train.log
tail -n 40 code/V4/runs/R80_training_and_code_ablation/A_ce_only/reporting.log
tmux list-sessions
```

Reporting errors do not invalidate or repeat completed training. Retry only:

```bash
python -m code.V4.r80_reporting --root code/V4/runs/R80_training_and_code_ablation --arm A_ce_only
```

Full interface and metric definitions: `../../R80_README.md`.
