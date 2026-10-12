Fresh A is running in **tmux `r64_train` / Slurm `1465.90`**, gpu03/index1, assigned UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`. B starts fresh automatically after A and replays its exact LR/horizon.

Locked **256/task, effective 768**: 40 cycles, 240 updates/epoch, tail 16/task (effective 48), warmup 720 updates. LR remains `1e-4`; maximum 9,600 updates / 7.2M presentations per arm.

| Batch/Task | PCGrad Queries/s | Physical Headroom | Result |
|---|---:|---:|---|
| 256 | 892.87 | 25.80% | Selected |
| 512 | CUDA OOM | — | Rejected |
| 384 | 899.95 | 2.94% | Rejected |

Sizing took **25.71s**; 20 successful calibration updates and 19,200 presentations were discarded. CPU checks: **4 passed in 1.026s**.

First formal update: **768 rows**, finite gradient norm **0.969037**, loss **1.951245**, **0 retries**, **0.183861s**, peak **4.401 GiB**. Checkpoint saved.

Changed `r64_experiment.py`, `test_r64.py`, `run_v4_r64.sh`, `R64_README.md`, fresh R64 artifacts, [REVIEW_STATE.json](/public/home/shiys/0selection2/review-stage/r58-to70/REVIEW_STATE.json), and [REVIEWER_MEMORY.md](/public/home/shiys/0selection2/review-stage/r58-to70/REVIEWER_MEMORY.md). Details and receipts: [handoff](/public/home/shiys/0selection2/code/V4/runs/R64_pcgrad_full18/large_batch_handoff.md).

Old archive preserved. No TEST, analysis, or Git operations.