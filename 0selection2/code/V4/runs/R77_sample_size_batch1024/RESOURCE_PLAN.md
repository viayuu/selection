# R77 / NSS Resource Handoff

Status: four independent runs are user-approved. The main coordinator reported
forwarding this approval to Aquinas and Popper for formal launch. Training and
normal runtime checks remain worker-owned; do not wait for this document or
repeat resource/cache preflight. This coordinator has not launched training.

## Approved Cards

| Run | Node / job | Exact GPU UUID | Physical index | Idle snapshot |
| --- | --- | --- | --- | --- |
| R77 small10000 | gpu05 / 1475 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | 0 | 0%, 354 MiB, no compute/graphics processes |
| R77 full70114 | gpu05 / 1475 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d | 1 | 0%, 354 MiB, no compute/graphics processes |
| NSS TSP | gpu03 / 1465 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | 3 | 0%, 422 MiB, no compute/graphics processes |
| NSS CVRP | gpu03 / 1465 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | 4 | 0%, 422 MiB, no compute/graphics processes |

GPU observations: 2026-10-15 00:02:19 +08:00 on gpu05 and 00:03:30 +08:00
on gpu03. These are approval-time snapshots, not claims of continued idleness
after workers launch. Read-only NVML queried only these UUIDs, without CUDA
contexts. Both Slurm jobs were RUNNING; each allocated 2 GPUs, 10 CPUs and
20480 MiB host memory. Detailed GRES indices matched the four UUIDs above.

## Launch Constraints

- One run per assigned card; no other cards and no killing others' processes.
- Activate easynco through /public/home/shiys/miniconda3/etc/profile.d/conda.sh
  before shell work, including worker launch shells.
- Launch only inside the existing allocation with srun --jobid=1475 or 1465,
  --overlap --exact --nodes=1 --nodelist=gpu05 or gpu03 --ntasks=1 and explicit
  CPU/thread limits. SSH is not a GPU execution path.
- Set CUDA_VISIBLE_DEVICES to the exact assigned UUID in the worker process;
  verify launch identity from the worker's own log, not from a global GPU scan.
- Keep both node-local workers' combined CPU demand within 10 CPUs and combined
  host-memory budgets within 20 GiB. Existing R77 PLAN.md specifies 4 CPUs,
  9 GiB, 2 threads and prefetch 2 per step: two steps use 8 CPUs / 18 GiB.
  This document does not override worker launch settings or certify peak memory.
- All four runs use logical batch 1024. Micro accumulation must keep exactly
  1024 actual examples per optimizer update and sample-weighted normalization.
  R77's existing large micro352 setting does not redefine the logical batch.
- No coordinator training, duplicate worker checks, cache rebuild, unrelated
  solver work, speculative R78 rules, unplanned broad retraining, or git commit.
  Future user-authorized full18-task training based on an Astra plan is allowed;
  the round001 CPU-only method is not a permanent restriction on the goal.

## Cache Evidence

Both nodes' /tmp/shiys_R76_R45A_single_reld_mt_cache copies passed at
00:05:57 (gpu05) and 00:07:11 (gpu03), 2026-10-15 +08:00. Each had 54
problem/split pairs and 321 arrays, totaling 17026116096 array bytes.
Complete metadata, rows, every array header, shape, dtype, full file size and
readable boundary bytes matched the R76 manifest; zero errors. No full payload
checksum was performed. TSP and CVRP each have 70114 TRAIN rows, 3895 VAL rows
and 3895 TEST rows. This was filesystem-only verification, not model evaluation.

## Coordination

Notification recipients: Aquinas 01a13b22-4408-77b1-8f45-8e071eef955d and
Popper 01a13b22-4462-70c3-acf0-0aa04a5afd02. Direct agent messaging is not
available in this subthread; delivery is reported by the main coordinator,
not falsely claimed as a direct message from this resource agent.

Solmax owns all execution/checks. Astra supplies the requested research/paper
recommendation. R78 round1 remains active and awaits that recommendation and
the user's execution algorithm. Research-owned GOAL.md / RESEARCH.md are not
edited here. The legacy main-thread app goal remains reported paused and its
create_goal attempt failed on the unfinished goal; it has not been resumed or
falsely completed. A separate child-thread goal was successfully created once;
the distinction is recorded in R78 STATUS.json. No further goal tool calls were
made for this handoff.

## Requested One-Shot Progress

At 2026-10-15 00:24:33 +08:00, tmux sessions R77_sample_size_batch1024
and R79_NSS_new_data_batch1024 were present with live panes. R77 A logged
1242/49680 updates and epoch001 VAL; NSS TSP logged 100/27400 updates at
batch1024/micro64. R77 B and NSS CVRP launcher logs were still retrying
"Requested nodes are busy" Slurm step creation. This is a historical
two-running/two-pending snapshot, not a fresh claim about current training.
No GPU capacity was remeasured and no worker code was reviewed or restarted.

R78 round001 now uses only local CPU saved-score fitting, 2 threads, no GPU
contexts. It does not consume or change any of the four assigned training cards.
