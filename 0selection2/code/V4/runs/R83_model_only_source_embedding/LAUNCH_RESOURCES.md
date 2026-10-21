# R83 Launch / Resource Handoff

Owner scope: `code/V4/r83_launch.py` and this run root's `launch_*` receipts/logs
and launch/resource notes only. Runner, reporter, shell wrapper, corpus and
embeddings belong to the other workers. No training, TEST, LIB inference or
reporting has been authorized/executed by this launch worker. No credentials
are inspected, no existing process is stopped, no allocation is requested, and
no commit is made.

## Selected Route

Use the currently fully idle RTX 3090 UUID
`GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50` (candidate `A_ce_only`; this name
identifies the old GPU slot only, **not** R83's model/loss). Existing allocation
`1465`, node `gpu03`, immutable container `1465_gpu03` /
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
The resource launcher reuses `r81_queue` constants, allocation validation, NVML
parsing, cache validation and the **same per-UUID flock files** as R81/R82.
It deliberately does not call their credential or process-environment helpers.

Initial read-only checks: both existing jobs are RUNNING, owned by `shiys(2022)`,
each with 10 CPUs, 20 GiB and two GPUs. `1475` CPUs are `0-5,72-75`; `1465`
CPUs are `12-17,64-67`. On `gpu05`, selected RTX 4090 has 2 MiB, 0% utilization
and no compute process; the other 4090 has an existing process and is untouched.
Both `gpu03` RTX 3090s also had 2 MiB and no compute process. Ordinals are
informational, UUID and hardware type are authoritative. A subsequent complete
process check found that `gpu05`'s existing T2T500 label-generation PID 570707
(container namespace PID 651548) has affinity to **all ten allocation CPUs**.
Thus its physically free 4090 is not immediately eligible under the existing
queues' disjoint-CPU safety policy. No affinity/process is changed. The two
3090 slots have four disjoint CPUs available now; use one rather than wait
indefinitely or interfere with the label workload. The fastest *immediately
eligible* route is therefore the 3090. Main can explicitly pin `B_batch512`
instead to wait for the 4090 allocation's CPUs to become eligible; no automatic
route switching occurs. The detailed dated
`launch_resources.json` is the source for the final probe/CPU/cache evidence.
Its preliminary `preferred_candidate=B_batch512` was superseded after the CPU
inventory: `launch_plan.json` is the final A-slot recommendation.
`launch_selected_resource_recheck.json` additionally proves the chosen route's
real Slurm cgroup `/slurm/uid_2022/job_1465/step_171`, the immutable Docker
cgroup, free selected UUID, CPUs `12,13,14,15` and warm cache. That read-only
probe step exited normally; the future launch gets its own step in job 1465.

No SSH training is used. The command enters the existing job with
`srun --jobid=1465 --overlap --exact --gres=none --immediate=5 -N1 -n1
--nodelist=gpu03 --cpus-per-task=4 --mem=9G bash -lc ...`, activates `easynco`,
then uses the already-authorized `sudo -n /usr/bin/ai_docker exec -u 2022`
against that exact existing container. It runs the exact activated easynco
interpreter, binds exactly one GPU UUID, verifies the Docker cgroup/container
identity and allocation owner/cpus/memory, and pins four CPUs disjoint from
existing model/GPU stages. No container/image is built or started.

## Warm Cache And Budget

Warm local cache: `/tmp/shiys_R76_R45A_single_reld_mt_cache` **inside the existing
allocation container**, validated against
`code/V4/runs/R76_R45A_single_reld_mt/cache`: 375 files, 17,026,558,189 bytes
(15.86 GiB), 54 identical completed task/split metadata files and all array
file sizes. The launcher never copies or rebuilds this cache; a missing or
invalid cache blocks launch. Heldout cost values are not read by the probe.

Use main's standard R45 bundle `ROOT/model_only_vectors.pt` and
`ROOT/model_only_vectors.lock.json`. `arm` calls the runner's CPU-only bundle
validator, checks its actual CLI, and freezes hashes of the bundle/lock,
scratch initialization/preparation and relevant source. No embedding API call
or corpus rewrite is performed here.

One fresh seed2 model: logical batch 1024, 49,680 successful updates, eval batch
256, unchanged R77/R80D engine/objective and VAL checkpoint rule. RTX 3090 large
microcap 384; RTX 4090 352 if main explicitly chooses its wait route before arming.
Microbatching preserves the logical batch and budget. No automatic OOM relaunch,
alternate seed, checkpoint resume, GPU reassignment or second model.

## Exact Operator Commands (NOT Executed)

Run on `admin01` from `/public/home/shiys/0selection2`. The first command only
creates a persistent waiting controller; without a main-ready receipt it does
not even probe an allocation. `arm` is **only for main after explicit ready**,
completed focused runner tests, final embeddings and unchanged code.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' python -B -u -m code.V4.r83_launch standby --candidate A_ce_only
```

After main's explicit readiness, with its actual evidence text substituted:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' python -B -u -m code.V4.r83_launch selftest
CUDA_VISIBLE_DEVICES='' python -B -u -m code.V4.r83_launch arm --candidate A_ce_only --runner-tests-passed --main-ready-evidence 'MAIN EXPLICIT READY: <dated message and passing focused test evidence>'
```

`arm` also creates the waiting tmux if needed; `standby` is optional. Tmux name
`r83_model_only_once`, remains on exit. An exclusive claim is irreversible:
failed launch/stage means stop with receipts, **not** retry or delete locks.
Rerunning the standby command cannot create a second live controller.
Only the readiness wait and idle-resource wait poll (60 seconds); after claim,
every stage is attempted once and a nonzero exit stops all following stages.

## Automatic Delivery

1. In the chosen owned container, `code.V4.r83_experiment train` performs the
   fixed-budget training, one immutable VAL-selected lock, and the runner's
   full TEST. Do **not** append another `test` command.
2. Only rc0 permits `r83_launch lib-worker` in the same cgroup/UUID. This calls
   the existing reporter engine's `r80_lib` with the R83 checkpoint factory,
   validates/reuses predictions and saves both TSPLIB/CVRPLIB predictions.
   It does not render PDFs in the container or change checkpoint selection.
3. Only rc0 and both same-lock saved LIB hashes permit
   `code.V4.r83_reporting --group model_only_seed2 --device cpu
   --lib-batch-size 2` **on admin01 outside containers**, with CUDA hidden.
   It uses the existing
   `/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc`. This avoids the
   known R81/R82 missing-CJK failure while reusing GPU predictions. It never
   repeats training, TEST or LIB inference. Required nonempty PDF files are
   checked before the controller declares completion.

W&B goes through the existing runner logger with `--wandb-mode auto`, project
`selector`, entity `yjkds-southern-university-of-science-technology`. Standard
logger lookup and explicit offline fallback/receipt happen only when the
authorized training runner initializes. The launcher never inspects credentials,
netrc/settings contents or another process's environment, and never logs keys.

Owned outputs: `launch_plan.json`, `launch_resources.json`,
`launch_selected_resource_recheck.json`, `launch_handoff.json`,
`launch_selftest.json`, `launch_cli_check.json`, then (only after main starts)
`launch_tmux.json`, `launch_ready.json`, `launch_status.json`, `launch_claim.json`,
`launch_controller.json`, `launch_container_entry.json`, `launch_worker.json`,
`launch_lib_status.json`, `launch_lib_complete.json`, `launch_*.lock` and
`launch_*.log`. Main-ready, claim, tmux and model outputs must remain absent
in this worker's stopped handoff.
