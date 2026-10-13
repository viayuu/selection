R71 B is **LIVE** on its fixed 40-epoch schedule. [Actual handoff](/public/home/shiys/0selection2/code/V4/runs/R71_joint_performance_mixture/handoff.json).

- **Process:** `r71_train`, pane `%96`, pane PID `1287378`; Slurm `1465.105` **RUNNING** on `gpu03`; trainer PID `1476336`.
- **GPU:** index1, UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`.
- **Actual command:** `bash code/V4/run_v4_r71.sh train`.
- **First formal update:** epoch1/global41, OVRP/CVRP/OVRPLTW, `256×3 = 768` rows; 1,536 encoder-view rows.
- **Loss:** NLL/D `1.4887570540`; mean joint NLL `10.2687127590`. These are density likelihood metrics, not winner CE.
- **Gradient/LR:** norm `0.3552532196`, clip coefficient `1`, LR `4.1666666667e-7`; **zero retries**, finite loss/gradient.
- **First-update memory:** allocated `15.404731 GiB`, reserved `19.164063 GiB`, physical used `19.487549 GiB`.
- **Checks:** one CPU invocation passed in `1.389960s`; one discarded capacity update passed in `3.752883s` (`5.965898s` total), with all **305,848,320 bytes** of bank Adam moments materialized. Capacity peak allocation `15.733320 GiB`. Formal state was recreated afterward.
- **Parameters:** B `38,231,040`; reused A `33,173,154`. Capacity and compute are not matched.

Added the six authorized isolated R71 files and run artifacts. Fixed A remains R70B23/8709 VAL correct. Historical A `9600/7.2M/3668.194599s/5 retries` and inherited stage1 once `9600/7.2M/2764.217798s` remain separate.

No TEST, freeze, analysis, Git, subordinate agents or R70 edits. Foreground sessions are settled; training remains live. Goal70 remains ACTIVE/unachieved.