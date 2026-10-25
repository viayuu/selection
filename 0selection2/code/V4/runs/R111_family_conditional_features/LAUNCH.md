# R111 One-Shot Launcher

2026-10-25 03:23 UTC update: the first entry stopped before any model/training
on differing cache-creation receipts. Original `preparation/dispatch/` is
preserved verbatim. One explicit launcher-only fix ignores creation timestamp,
duration and creator-lock fields, not input identity. The active explicit entry
now uses `preparation/dispatch_recovery_01/`; source/ready archives and details
are in `preparation/recovery_01/RECOVERY.md`. No automatic retries are added.

Scope: only `r111_launch.py`, `run_v4_r111.sh`, three CPU launcher tests,
and this document. Runner/model/report and `PLAN.md` are main-owned. No Git
publication, queue kickoff, GPU probe, allocation or cache work is performed
while preparing these files.

## Fixed Contract

- One arm: `A_family_conditional`, default root
  `/public/home/shiys/0selection2/code/V4/runs/R111_family_conditional_features`.
- Wait for BOTH original R110 arms, not training markers: original
  `worker.json` must complete train/TEST/timing/LIB at rc0 and original
  `controller.json` must complete allocation_GPU_pipeline/PDF at rc0.
  Any predecessor terminal failure exits this queue; no healing/restart.
- Reuse gpu05 allocation1475, immutable container
  `76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.
  Host dispatch is `srun --jobid=1475 --overlap --exact --gres=none`
  with eight CPUs and 18G memory, never a new allocation or SSH training.
- Container exec uses the existing authorized user, original cgroup,
  and affinity `0,1,2,3,4,5,72,73`. Training UUID order is
  `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be,GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`.
- Actual original mutex paths are `r81_queue.ROOT/queue_gpu_<UUID>.json`,
  exactly matching `r81_queue.gpu_lock_path()` and original r84 controller
  line535. Existing files are opened without create/truncate/replacement.
  Main freezes their inode numbers; both locks are held through CPU report.
  Preparation-time inode observations were 563597744 and 563598119,
  respectively; these observations alone are not launch authorization.
- Train: `torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0`
  `-m code.V4.r111_experiment train`. Runner owns seed2, 57,600 updates,
  unchanged LR, local512/global1024 and microcaps TSP/CVRP256, ATSP128,
  other512. Launcher passes large cap256 and eval256, no `--resume`.
- Strict rc0 order: train, `r111_experiment finalize`,
  `r111_experiment timing`, `r111_reporting lib`, `r111_reporting report`.
  Finalize owns full selected TRAIN/VAL plus full locked same-best TEST.
  Finalize/timing/LIB see only original GPU0, use `cuda:0`; LIB batch2.
  Report is CPU-only on the original host, after the container worker exits
  through LIB at rc0. The host retains BOTH GPU locks through report rc0.
  No checkpoint or inference reselection here.
- Reuse `/tmp/shiys_R84_new90k_cache`, R84 `prepared` and NEW90k data.
  Worker only checks original container/UUIDs and existing cache metadata
  paths; it never generates, repairs, copies or hashes bulk cache arrays.
  W&B remains explicitly offline using the already-documented lookup result;
  there is no new credentials search or online availability assertion.

## Main Readiness

CPU-only inspection commands, not launch authorization:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_v4_r111.sh plan
bash /public/home/shiys/0selection2/code/V4/run_v4_r111.sh check-cli
python -B -m unittest code.V4.test_r111_launch
```

`check-cli` waits for no missing files: it fails before importing parsers if
any current file is absent. It only parses the five exact stage commands
with CUDA hidden. It prints `source_sha256` for exactly six files:
launcher, runner, model, reporter, shell wrapper and main `PLAN.md`, plus
the original `mutex_inodes`. No recursive historical source closure.

After the necessary few CPU checks and genuine Sol review, MAIN writes
`ROOT/launch_ready.json`, using the exact two maps from `check-cli`:

```json
{
  "main_thread_ready": true,
  "cpu_tests_passed": true,
  "sol_review_passed": true,
  "evidence": "Main's concrete CPU-test and Sol-review evidence references",
  "confirmed_utc": "MAIN_SUPPLIES_UTC",
  "source_sha256": {"ABSOLUTE_FILE_PATH": "EXACT_SHA256_FOR_EACH_OF_SIX_FILES"},
  "mutex_inodes": {"ABSOLUTE_ORIGINAL_MUTEX_PATH": 123}
}
```

The launcher never creates or upgrades this evidence file. A missing file
means wait; malformed, false or stale ready evidence is a terminal failure.
Changing frozen current files requires fresh explicit main evidence before
any launch, not an automatic refreeze. Tests and this guide are not runtime
freeze dependencies. No draft spec is treated as permission.

## Queue And Receipts

Only main may kick off one foreground queue in its one tmux session:

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r111.sh queue
```

The default poll is quiet every60 seconds. It waits for main readiness,
both entire original R110 pipelines and both original mutex releases.
Neither R109 nor R110 is signalled, stopped, modified or resumed.
`entry` and `worker` are internal token-gated commands, not manual bypasses.

Receipts are under `ROOT/preparation/dispatch`: `queue.json`, one exclusive
`claim.json` with main evidence/current-file freeze, `entry.json`,
`worker.json`, and per-stage logs. The worker records train/finalize/timing/LIB;
the host queue records the allocation pipeline and CPU report. They retain
state, host/PIDs, exact argv,
start/end times and exit codes. First failure stops the remaining stages.
Repeated queue invocation refuses the existing one-shot receipt even after
failure. No automatic retry, restart, resume, recovery queue or auto-heal.
Signal handling waits for its own running child before releasing locks;
as with any process-based mutex, SIGKILL or node loss is not recoverable.
