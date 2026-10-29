# Return18 Resource-Only Transport

Implementation is CPU-tested only. Nothing has been dispatched. Parent supplies
the real approved plan; no generated training parameters, SSH, new allocations,
R120 edits, Git operations, implicit resume, or training retries.

## Exact CLI

All paths below are absolute. Substitute the actual approved plan filename.
The wrapper activates easynco, the R120 compatibility PYTHONPATH and thread limits.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && bash /public/home/shiys/0selection2/code/V4/run_v4_return18.sh describe --plan /public/home/shiys/0selection2/code/V4/runs/Return18_post_R120/dispatch/main_approved.json
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && bash /public/home/shiys/0selection2/code/V4/run_v4_return18.sh dispatch --plan /public/home/shiys/0selection2/code/V4/runs/Return18_post_R120/dispatch/main_approved.json --wait --poll-seconds 30
```

Run the second command in a parent-owned tmux session, only when intended.
`describe` validates plan/protocol/source hashes without Slurm, GPU queries or
R120 resource access. `dispatch` is the standalone CPU controller. It freezes
the original plan bytes and refuses an existing controller receipt.

Internal resource commands are `entry --plan PATH --job ID --token TOKEN
--plan-sha256 SHA` and `worker --plan PATH --job ID --token TOKEN`. The controller
and verified entry supply these tokens, hashes and identity environment. Never
invoke a worker directly outside that transport.

## Plan Schema

Machine schema: `plan.schema.json`. Runtime validation additionally checks file
hashes, loaded transport hash, output ownership, source-pinned Python entry
points, explicit --root, CUDA/gpu-type binding and dependency order. Repository
relative paths resolve from `/public/home/shiys/0selection2`, not from plan.parent.

Required top-level keys: `schema`, `name`, `main_approved`, `approved_by`,
`approved_utc`, `protocol`, `source_sha256`, `jobs`. Unknown fields are rejected.
`protocol` is ONE existing JSON object with `{path, sha256}` under Return18 or
R121-R124. It is a shared authorization/protocol input, not an instruction to
rewrite runner-owned protocol files. `main_approved` must literally be true.

Each job has `{id, arm, slot, root, stages, depends_on?}`. Each stage has
`{id, kind, argv}`. Kinds: collect/cache/train/test/lib/report. The ordered argv
is passed unchanged with no shell. Jobs are finite, roots are distinct, and
dependencies may reference only earlier job IDs. Same-slot jobs run serially;
independent slots run concurrently. There is no requirement to fill every slot.

Slots: `gpu05_0` / `gpu05_1` are job1475 RTX4090; `gpu03_0` / `gpu03_1` are
job1465 RTX3090. Each uses a 4 CPU, 9G, --gres=none overlap CPU step, then sudo
ai_docker exec into the original container with one exact CUDA UUID. Per-node
CPU lanes are permanently pinned to disjoint slices of the allocation/container
intersection. Changed container, allocation, CPU mapping, or mutex inode fails
closed. Child stages inherit the original R81 mutex fd and environment.

R122 TRAIN-only job fragment (NOT a complete approved plan):

```json
{
  "id": "R122_model",
  "arm": "R122",
  "slot": "gpu05_0",
  "root": "code/V4/runs/R122_hard_decision_constraints",
  "stages": [{
    "id": "train",
    "kind": "train",
    "argv": ["python", "-B", "-u", "-m", "code.V4.r122_experiment", "train",
      "--root", "code/V4/runs/R122_hard_decision_constraints",
      "--arm", "model", "--device", "cuda:0", "--gpu-type", "4090"]
  }]
}
```

Parent must decide the final literal argv with the R122 owner. R122's persistent
cache-path restriction is runner-owned; transport neither relaxes it nor appends
--cache-root. An approved local-cache override, if supported by the runner, must
literally use `/tmp/shiys_R84_new90k_cache`. A TRAIN-only job does not auto TEST.

Pin at least these source_sha256 keys plus every Python runner and its required
scientific dependency files: `code/V4/return18_launch.py`,
`code/V4/run_v4_return18.sh`, `code/V4/r120_launch.py`, `code/V4/r81_queue.py`.
Keys are canonical repository-relative paths; values are lowercase SHA256.
Changes to any pinned file after approval refuse execution, including between
stages. Use the active Python executable, `python`, or `python3`; only pinned
code.V4 modules or repository code/V4 .py scripts are accepted. No python -c,
shell commands, nested transports, resume flags or environment substitution.

## R120, Cache And Waiting

No slot is released until R120 test_complete.json, reports/report_complete.json
(complete24 and lib_complete), best.pt, protocol.json, final_test_lock.json and
all four selected-best VAL/TEST/TSPLIB/CVRPLIB identities agree. Git publication
is deliberately NOT required. Parent recovery and old failed logs stay intact.

The worker locks the ORIGINAL R81 UUID mutex inode (never creates/unlinks it;
st_dev is deliberately not compared across NFS namespaces), then takes exactly
one NVML XML snapshot including process inventory. Any GPU process, memory >512
MiB or utilization >10% produces a pre-attempt waiting receipt, never a kill.

All 54 task/split caches are checked: complete metadata, expected 81k/4500/4500
rows, frozen scenario, NPY shape/dtype/header/file size and available frozen
metadata equality. A complete local cache is reused without copying. Missing
train/val uses the existing R84 input_cache. Missing TEST first uses shared R84
TEST if available, otherwise R120 `existing_test_cache/test` after its
`complete.json` proves authorized source_container, no preprocessing, zero
solver calls. Per-node copy is under flock and never overwrites existing files.
No feature computation and no binary transfer through ai_docker stdout.

Optional `cache_test_recovery: {path, sha256}` overrides the TEST source using a
pinned JSON object of this exact form (both node keys are mandatory):

```json
{"schema":"return18-frozen-test-v1","test_sources":{"gpu03":"/shared/frozen/test","gpu05":"/shared/frozen/test"}}
```

`--wait` permits repeated resource entries ONLY for a matching token/hash,
explicit waiting result, exit75, and NO attempt.json. It also waits for R120 and
the declared frozen recovery source before attempts. A stage exit75 is still a
failed attempt, never a resource retry. Unclassified Slurm/exec failures are
terminal. No arbitrary training resubmission is supported.

## Receipts And Checks

`dispatch/NAME/controller.json` is exclusive; `plan.json` preserves approved
bytes. `jobs/ID/transports/TOKEN.*` preserves every entry proof, stdout/stderr
log, result, command and actual srun exit code, including genuine waiting.
`jobs/ID/attempt.json` is exclusive and permanently consumes the job. Stage
`ID.log` and `ID.json` preserve combined stdout/stderr, literal argv, PID,
timestamps and actual child exit code. First nonzero stage stops that job.
Independent jobs may finish; failed dependencies never start. No restart based
on missing/lost/failed receipts. Worker interruption only forwards to its own
child process group and keeps the GPU mutex until that child exits.
`dispatch/train_claims/ROOT_SHA256.json` permanently claims each TRAIN output
root across different plan names too. Renaming a plan cannot restart training.
A later literal TEST-only plan can still evaluate the locked existing model.

CPU checks: 14 unittest cases passed with affinity restricted to four CPUs;
wrapper bash syntax passed. No real Slurm, GPU, container exec or cache copy was
performed by this implementation session. Validation includes one-attempt
failure preservation, actual child fd inheritance, inode-only NFS identity,
four-slot mapping, plan/source/approval tamper rejection, real R120 LIB receipt
format, TEST cache truncation, complete-local no-copy, recovery publication,
and wait-only-before-attempt dispatch. Real resource integration remains for
the parent-authorized launch.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v code.V4.test_return18_launch
```
