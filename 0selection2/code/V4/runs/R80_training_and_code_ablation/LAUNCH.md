# R80 Fixed-Allocation Launch

Launcher owner: Mill. No git commit, new allocation, process replacement, solver,
embedding generation, or bulk preprocessing is part of this launcher.

| Arm | tmux | Existing Job | Node/GPU | Logical Batch | Successful Updates | TSP/CVRP Microcap |
|---|---|---:|---|---:|---:|---:|
| A_ce_only | r80_A_ce_only | 1465 | gpu03/0, RTX 3090 | 1024 | 49680 | 384 |
| B_batch512 | r80_B_batch512 | 1475 | gpu05/0, RTX 4090 | 512 | 99360 | 352 |
| C_lr3e4 | r80_C_lr3e4 | 1465 | gpu03/1, RTX 3090 | 1024 | 49680 | 384 |
| D_code_embedding | r80_D_code_embedding | 1475 | gpu05/1, RTX 4090 | 1024 | 49680 | 352 |

Full UUIDs are fixed in `code/V4/r80_launch.py` and `launch_manifest.json`.
Every process binds one UUID; a node-local GPU lock rejects duplicates. Each
overlapping srun step uses 4 CPUs and 9G RAM within the existing 10 CPU / 20G
allocation. Slurm steps inherit the allocation's GPU visibility, then
`CUDA_VISIBLE_DEVICES` selects exactly one device. No additional job is requested.

Actual resource correction: this cluster's Slurm 20 does not overlap GPU GRES.
A/B's original steps retained both logical GPU GRES despite using just UUID0.
C/D's original waiting srun clients were explicitly canceled before any runner
started. Their original controller exit records remain on disk. C/D now use
CPU-only steps (`--gres=none`, still 4 CPU / 9G) to exec into the existing,
identity-checked allocation containers `1465_gpu03` and `1475_gpu05`. Container
cgroup IDs and exact assigned GPU UUIDs are recorded; ordinary host SSH is never
used to run training. No new job/container/allocation is created. A/B are untouched.

The first C/D container preflight failed because container `/tmp` is distinct
from host `/tmp`; receipts are retained as `launch_ARM*_failed_cache_preflight.json`.
No C/D training had started. The platform's restricted `ai_docker` does not support
`cp`, so the corrected entry copies only frozen metadata and array files from
the shared R76 source into container-local `/tmp`, once per node. These copy
receipts are `launch_container_cache_gpu03.json` and `launch_container_cache_gpu05.json`.

Both nodes already have the 16G frozen R76 cache at
`/tmp/shiys_R76_R45A_single_reld_mt_cache`. The worker verifies all 54 metadata
records and array sizes without reading held-out values. No cache rebuild occurs.
MVRP/ATSP use the runner's full logical batch, not the TSP/CVRP microcaps.

## Start And Receipts

The launcher requires `launch_ready.json` to contain explicit main-thread ready
confirmation and passing runner CPU-test evidence. It refuses existing launch
receipts, tmux sessions, or training checkpoints. There is no automatic restart.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_v4_r80.sh start
```

`launch_manifest.json` records fixed protocol/resources. Each arm has
`launch_ARM_controller.json` (admin PID/srun exit) and `launch_ARM.json` (node PID,
training/test/report child PIDs, exact commands, timestamps and stage exit codes).

W&B is enabled in `selector`, entity
`yjkds-southern-university-of-science-technology`. Existing credentials, including
netrc, select online. No credential selects offline with an explicit reason.
An online initialization failure uses the runner's explicitly logged offline
fallback, without restarting training. No credential values are recorded.
Initial credential checks on both gpu03 and gpu05 found no configured credential,
including netrc. Thus the actual initial mode is offline, not online or disabled.
Project: https://wandb.ai/yjkds-southern-university-of-science-technology/selector

## Pipeline And On-Demand Progress

Each arm runs train -> locked-best TEST -> LIB + PDF report, with exit status 0
required before the next stage. Any failure remains visible in the receipts and
prevents subsequent stages; neither hyperparameters nor checkpoints are changed.
The runner's `train` command already performs locked-best TEST internally;
the subsequent `test` command verifies/reuses its saved result without re-selection
or another complete inference pass.
The reporting command is `python -m code.V4.r80_reporting --root ROOT --arm ARM
--device cuda:0` on the same assigned GPU. Reporting may be retried separately;
completed training is never repeated by the launcher.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_v4_r80.sh status
python -m code.V4.r80_reporting --root /public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation --status
tail -n 40 /public/home/shiys/0selection2/code/V4/runs/R80_training_and_code_ablation/A_ce_only/train.log
tmux attach -t r80_A_ce_only
```

Use the equivalent B/C/D arm paths and sessions for other groups. Runner progress
is `ARM/status.json`; stdout is `ARM/train.log`, `ARM/test.log`, and
`ARM/reporting.log`. PDFs are `ARM/full_model_statistics/MODEL_STATISTICS.pdf`
and `ARM/full_model_statistics/selection_distributions.pdf`.

Only one bounded real startup check is performed: A/C/D at least 18 successful
updates, B at least 36, all 18 tasks, finite losses/gradients, and verified GPU
bindings. No complete validation is required for startup handoff. No persistent
monitor, W&B sync loop, Feishu/Slack notification, or completion watcher is added.
`launch_startup_check.json` stores that handoff evidence; `launch_gpu_gpu03.json`
and `launch_gpu_gpu05.json` store the one-shot physical UUID/PID/utilization checks.
For container workers, receipts store the container PID; GPU snapshots also
record host PID namespace mappings, allowing an exact UUID/host-PID/container-PID
match. The final reporter contract/hash is in `launch_report_contract.json`;
an older reporter hash in an already running arm's launch receipt does not change
or restart that arm's frozen training implementation.

## Verified Startup Handoff

All four real training workers passed the one-shot check at
2026-10-15 12:33:10 UTC (20:33:10 Asia/Shanghai). The steps below are the saved
first complete task-cycle evidence, not an assertion that training stayed at
those steps. All18 tasks have one successful update each (B has two), every loss
component and pre-clipping gradient norm is finite, and AMP retries are zero.

| Arm | Checked Step | Loss | Grad Norm | Host Train PID | Container Train PID | GPU Utilization |
|---|---:|---:|---:|---:|---:|---:|
| A_ce_only | 18 | 0.51757495 | 1.11328159 | 3896328 | n/a | 99% |
| B_batch512 | 36 | 0.99985273 | 2.00392531 | 1637595 | n/a | 78% |
| C_lr3e4 | 18 | 0.99973294 | 2.00874860 | 983221 | 1410954 | 100% |
| D_code_embedding | 18 | 1.00280502 | 1.96762508 | 1681397 | 220045 | 96% |

GPU utilization is a single physical-GPU snapshot. UUID/PID namespaces match
the fixed arm mapping; no two arms share a physical GPU. A/B were never restarted.
C/D cache copies contain 375 necessary files / 17,026,558,189 bytes per node.
All four use eval batch 256, two torch threads, pinned two-batch prefetch, and
the original effective training batches/budgets. W&B is enabled offline because
no existing credential was found in either host or container environment.

There is no persistent monitor from this launcher. Training and the successful-
exit-gated TEST verification / LIB / PDF pipeline remain in tmux; progress is
now for the user to inspect on demand through `PROGRESS.md`, `STATUS.md`, the
per-arm `status.json`, or the commands above.
