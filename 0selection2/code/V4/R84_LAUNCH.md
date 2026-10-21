# R84 resource and launcher handoff

This file describes preparation only. No R84 GPU training, cache staging, tmux
controller, or allocation has been started by this work.

## Current resources

Snapshot: `runs/R84_new90k_launch/resources.json`, 2026-10-18 01:20:52 CST.

| Slot | Existing job / node | GPU UUID | Device | CPU / memory budget per arm |
| --- | --- | --- | --- | --- |
| `3090_0` | 1465 / gpu03 | `GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50` | 3090 24 GiB | 4 CPU / 9 GiB |
| `3090_1` | 1465 / gpu03 | `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` | 3090 24 GiB | 4 CPU / 9 GiB |
| `4090_0` | 1475 / gpu05 | `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be` | 4090 24 GiB | 4 CPU / 9 GiB |
| `4090_1` | 1475 / gpu05 | `GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d` | 4090 24 GiB | 4 CPU / 9 GiB |

All four UUIDs were at 2 MiB / 0% with no compute processes. Each allocation
owns 10 CPU and 20 GiB. gpu03 CPU set is `12-17,64-67`; gpu05 is `0-5,72-75`.
Containers have about 18 GiB reclaimable file cache, but only about 57.6 / 13.3 MiB
anonymous RSS. Host-wide free RAM is not the allocation memory budget.

R83 completed 49,680 updates and its locked TEST/LIB stages; CPU PDF exited 1.
Its existing files and receipts were read only and were not repaired here.

## Main / runner contract

Runner owner: Bohr `01a14ad4-3683-75d0-8fe5-9fcff07673b5`.
Dataset owner: Locke `01a14ad2-af6b-7601-9098-534e089b113c`.
The current tool set has no inter-agent messaging API; use this shared contract
or relay the exact argv and paths through main.

Locked initial arms and suggested unique slots:

| Arm | Slot |
| --- | --- |
| `A_baseline` | `3090_0` |
| `B_attention_direct` | `4090_0` |
| `C_focal` | `3090_1` |
| `D_attention_focal` | `4090_1` |

All arms: seed 2, logical batch 1024, 57,600 successful updates, evaluation every
1,440 updates, 4,320 warmup updates. Runner owns enforcement of this training
protocol and the final R77 hand-feature versus R83 model-only feature choice.
The resource launcher does not reinterpret loss, model, batch, or validation
selection settings.

New data path supplied by main:
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.

Main must supply a completed NEW shared cache path, new preparation directory,
empty model output directory, exact GPU train/TEST/LIB commands, and CPU-only
saved-prediction reporting command. Place exact argv in a JSON spec, not a shell
string. The same new cache_source/cache_root may be used for all four arms;
staging is locked once per allocation container. Cache construction/reusing old
inputs by source_index belongs to the runner/data owner. The launcher does not
generate inputs or copy the R76 cache as if it were new data.

The known old R76 cache and subdirectories are rejected. Newly prepared files
are staged to a distinct `/tmp` path, JSON metadata hashes and every file size
are checked, and an identity stamp prevents another dataset from reusing it.
Arrays are not loaded into RAM during staging; no held-out cost inspection is
performed by the launcher.

## CLI

Every shell command must first activate easynco. The wrapper also activates it
on entry, including inside existing allocation-owned containers.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r84_launch.sh template --arm A_baseline
bash code/V4/run_v4_r84_launch.sh resources
python -B -m unittest code.V4.test_r84_launch
```

`template` prints an intentionally incomplete spec. Main replaces placeholders
and provides its final JSON file. Available argv placeholders are `{python}`,
`{data_root}`, `{prepared_root}`, `{cache_root}`, `{run_dir}`, `{device}`,
`{gpu_uuid}`, `{gpu_type}`, `{micro_cap}`, plus spec fields. `{device}` is
`cuda:0` for GPU stages and `cpu` for CPU stages. `source_files` should include
the final runner/model/objective/reporting sources main wants frozen at ready.

After main's exact CLI and completed NEW cache exist, preparation has no job
side effects:

```bash
bash code/V4/run_v4_r84_launch.sh prepare --spec /ABSOLUTE/FINAL_ARM_SPEC.json
```

Only after main explicitly approves runner CPU tests and the final plan:

```bash
bash code/V4/run_v4_r84_launch.sh start \
  --spec /ABSOLUTE/FINAL_ARM_SPEC.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_READY_EVIDENCE'
```

`start` creates `tmux r84_<run_id>` on admin01. Existing per-UUID reservation
flocks are acquired read-only; an unused slot may need a new empty mutex file.
No R80/R81/R83 receipt contents change.
CPU claims across R84 arms are serialized per node and kept disjoint. The
controller rechecks UUID idle state and the original allocation/container
identity before first training. Only existing `srun --gres=none` CPU steps are
used; actual CUDA work is inside the original allocation-owned container, with
one explicit CUDA UUID. No SSH training, new allocations, implicit resume, or
automatic failed-stage retries.

GPU stages execute in listed order. The first nonzero stage stops the pipeline.
CPU/PDF stages then run on admin01 with CUDA hidden and the installed CJK font.
Main's reporting CLI must consume saved locked predictions for those stages.

Progress and logs are separate from model output:
`runs/R84_new90k_launch/<run_id>/{plan,ready,claim,controller,worker,status}.json`
and per-stage `.log` files. No `ready.json`, `claim.json`, or R84 tmux exists at
handoff; the launcher is not armed.
