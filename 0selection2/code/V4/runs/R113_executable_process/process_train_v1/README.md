# R113 Full TRAIN Process Generation v1

User-authorized full generation: 18 tasks x 81,000 frozen TRAIN inputs, seed 2.
This version is independent of, and never overwrites, preparation/examples.
No native solver, training, model edits, GPU, Git, VAL, TEST or LIB access.

Host: admin01. tmux: r113_process_train_v1. easynco, two CPU threads, affinity
restricted to two allowed admin01 CPUs, nice 10, CUDA devices hidden. No gpu05
allocation, SSH, job submission or CPU-budget access occurs.

The generator and tensor schema are copied and SHA256-pinned before generation.
The exact source/compiler/reference program is unchanged from the delivered v1.
K=12, neighbors=2; 256 rows per storage shard does not change program semantics.
NumPy and CPU Torch seed 2 are fixed; target generation has no random sampling.

## Files

- run_config.json: fixed source paths, task list, row coverage, seed and CPU limits.
- source/: frozen generator and original semantic contract; not the model.
- schema.json: frozen strict tensor schema.
- sourcehash.json: immutable SHA256 for code, schema, launcher and config.
- source_manifest.json: immutable input-only TRAIN identities for all 18 tasks.
- command.log: appended actual launch command, effective environment and progress.
- launch_receipt_*.json: immutable host/PID/affinity/threads/command evidence.
- progress.json: atomic latest progress, per-task rows and committed shard count.
- shards/: exclusively published compressed TRAIN NPZ files, never overwritten.
- receipts/: immutable per-shard range/hash checkpoint, authoritative for resume.
- inflight/: temporary unpublished files; interrupted partials are preserved.
- manifest.json: building status initially; full standard shard manifest on success.
- completion_receipt.json: created only after all 1,458,000 rows are committed.
- failure_receipt_*.json: saved exception if a run fails; committed shards survive.

Additional NPZ metadata seed/source_manifest_sha256 bind the generation version;
the existing load_shard API ignores these extra metadata keys. No target tensor
or compiler input is changed. Live progress is in progress.json; manifest.json
lists the entire shard collection only at completion.

## Start Or Resume

Run on admin01. The version root was checked absent and created exclusively at
initial setup. Resume validates pinned sources/config and every committed shard
hash, skips valid ranges, and adopts an unreceipted archive only if all its arrays
equal the re-executed frozen program. It never replaces existing archive bytes.
Publication uses same-filesystem exclusive hardlinks plus atomic JSON progress.
An exclusive writer lock prevents duplicate generators.

```bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easynco
tmux new-session -d -s r113_process_train_v1 \
  'bash /public/home/shiys/0selection2/code/V4/runs/R113_executable_process/process_train_v1/launch.sh >> /public/home/shiys/0selection2/code/V4/runs/R113_executable_process/process_train_v1/command.log 2>&1'
```

Do not edit pinned code/config/schema during a run or resume. The receipt signals
data generation only, not native solver trajectories, candidate-winner quality,
training completion or scientific gains. Main owns subsequent training/staging.
