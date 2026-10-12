Completed one original 18-task/full-pool TEST for each frozen model. Protocol status: `complete`. Return code: **0**.

Tmux `r63_final_test` terminated. Slurm step: `1465.88`.
GPU: RTX 3090, `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`.

| Model | Macro Top1 | Top2 | Top3 | vs SBS | VBS Gap Closed |
|---|---:|---:|---:|---:|---:|
| R63A34 | 0.4734 | 0.8141 | 0.9219 | -0.980% | 27.29% |
| R63B12 | 0.4652 | 0.8061 | 0.9160 | -0.946% | 24.82% |
| R60B39 | 0.4824 | 0.8172 | 0.9251 | -1.036% | 30.94% |

Outputs: [test_results.json](/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/test_results.json), [test_protocol.json](/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/test_protocol.json), [final_test.log](/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/final_test.log).
