# R85 resource-only launch handoff

All four R85 training controllers started exactly once after explicit main
authorization. `runs/R85_launch/launch_startup_check.json` records the bounded
first full18-cycle verification at 2026-10-20 00:33:54 CST. Every arm has finite
loss/gradient metrics, all18 tasks covered, zero first-cycle AMP retries, and
its live process mapped to the assigned GPU UUID and allocation container.
No validation wait or persistent resource monitoring was added.

Runner owner: Einstein `01a154d3-23a9-7463-97b3-5be3b7ca0719`.
Objective owner: Peirce. Model output root:
`code/V4/runs/R85_zero_selection_research`.
The four arms are locked: `A_ovr_control`, `B_gradient_balance`,
`C_cost_profile` (conditional nonwinner KL), and `D_candidate_subset`.
Shared confirmed controls are seed 2, logical batch 1024,
57,600 successful updates, and the R84 TRAIN preparation/fresh initialization.
Fixed resource mapping: A/B use `3090_0`/`3090_1`, both with large-instance
microbatch cap 384; C/D use `4090_0`/`4090_1`. C uses cap 352; D explicitly
passes `--large-micro-cap 256` for TSP/CVRP activation headroom with its extra
subset joint branch. Logical batch 1024 and MVRP cap 1024 remain unchanged.
The per-arm protocol and parsed argv record the actual cap. No GPU memory
preflight sweep is run. A and B therefore
share hardware and microbatch execution. R84 remains a historical external
reference for the other arms; this assignment does not imply matched hardware
comparisons with every R84 arm.
Main's final `protocol_lock.json` is present at the R85 model root. EQL's
initial positive/negative gradient ratio is 100, as specified there. A locked
research protocol does not authorize training before main's separate CPU-ready.

## Live resource evidence

`runs/R85_launch/resources.json` records an actual 2026-10-19 23:40 CST check
inside the two allocation-owned containers. All four GPUs had 2 MiB allocated,
0% utilization, and no compute processes. All four R84 workers recorded rc0,
and each saved train/LIB PID was checked in the container process namespace;
none was still the original worker. This is not inferred from lock files.

| Slot | Job / node | GPU UUID | CPU / memory per arm |
| --- | --- | --- | --- |
| `3090_0` | 1465 / gpu03 | `GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50` | 4 CPUs / 9 GiB |
| `3090_1` | 1465 / gpu03 | `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` | 4 CPUs / 9 GiB |
| `4090_0` | 1475 / gpu05 | `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be` | 4 CPUs / 9 GiB |
| `4090_1` | 1475 / gpu05 | `GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d` | 4 CPUs / 9 GiB |

The two arms on each node use disjoint four-core CPU claims within its existing
10-CPU / 20-GiB allocation. gpu03 allows `12-17,64-67`; gpu05 allows
`0-5,72-75`. The launcher uses the stable per-UUID mutexes and rechecks real
processes, GPU activity, memory, allocation identity, and available CPUs when
claiming. It never stops other processes or allocates another job.

The latest check briefly observed a conda activation subprocess on gpu05; it
was not a GPU or R84 worker. A queued launch waits if any live process occupies
its proposed CPUs. There is no reason to kill such a process.

## Existing data and read-only cache

Dataset:
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`

Shared preparation:
`/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/prepared`

Shared TRAIN/VAL cache source:
`/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/input_cache`

Existing node-local cache in both allocation containers:
`/tmp/shiys_R84_new90k_cache`

The completed 23:40 resource check validated 250 TRAIN/VAL files / 19,005,947,954 bytes per node
using metadata hashes and file sizes. No arrays were opened, copied, rebuilt,
or normalized again. TEST files appended by R84 are ignored by the training
cache check. This evidence is reused without another warm-cache audit.
The launch plan derives its manifest from saved R84 records and verifies it
matches the completed snapshot. At claim time only live allocation/CPU/GPU
processes are probed. Worker entry checks the cache directories and identity
stamp but does not rescan arrays or repeat metadata hashes. The existing stamp
and cache remain unchanged. `resources` now writes `resources_live.json`,
preserving the completed `resources.json` cache evidence.

R85 must start with fresh weights. The prepared `identicalinit_seed2.pt` is a
saved random initialization, not a trained R84 checkpoint. Main's runner owns
actual initialization enforcement, optimizer/loss/budget, TRAIN-only parameter
updates, VAL selection, and the unique checkpoint lock before TEST/LIB. This
launcher does not interpret labels or select checkpoints.

## Main-thread contract and commands

All commands activate `easynco`; the shell wrapper also activates it inside
the allocation container.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r85_launch.sh resources
bash code/V4/run_v4_r85_launch.sh template --run-id MAIN_SUPPLIES_RUN_ID
bash code/V4/run_v4_r85_launch.sh specs
python -B -m unittest code.V4.test_r85_launch
```

`template` only prints an incomplete JSON spec. Main supplies a final JSON
file after the protocol and exact CLI are ready. Only the confirmed common
controls appear in the template; arm definitions belong to main. Required fixed
fields are the dataset/cache/prepared paths above, `cpu_count=4`,
`memory_gib=9`, and
`protocol.initialization="scratch"`, `protocol.training_splits=["train","val"]`.
Use a fresh model `run_dir` outside historical R84 outputs, a unique `run_id`,
one preferred slot per concurrent arm, and exact `stages[].argv` lists.

`specs` writes the four fixed-arm specs into `runs/R85_launch/specs/` and
parses all eight train/test argv against the real runner parser. It creates no
ready gate, resource claim, source freeze, tmux, or training process. The
reporter parser is now present; `specs` also parses eight actual LIB/PDF argv.
`spec_parse.json` records precisely what has passed.
Main must supply `R85_zero_selection_research/protocol_lock.json` and declare
runner/objective/model/reporting files ready before `prepare` freezes their
source hashes. Existing specs are never silently replaced.

The source freeze includes `r84_experiment.py`, `r84_reporting.py`,
`r80_experiment.py`, `r80_reporting.py`, and the R85 runner/objective/candidate/
reporter files because R85 executes their generic hooks. No historical R84
run is restarted. Objective owner reports FINAL READY / 11 CPU tests; candidate
owner reports FINAL READY / 7 CPU tests. Einstein's actual runner/reporting
integration CPU-ready and main's brief Sol review remain separate requirements.
Final PDF visual polish is not a prerequisite for launch readiness.

Stage placeholders include `{python}`, `{data_root}`, `{prepared_root}`,
`{cache_root}`, `{run_dir}`, `{device}`, `{gpu_uuid}`, `{gpu_type}`,
`{micro_cap}`, and any extra top-level spec fields. GPU `{device}` is
`cuda:0`; CPU/PDF is `cpu`. GPU stages run in order; only rc0 permits a later
stage. CPU/PDF runs on admin01 with CUDA hidden and the installed CJK font.
Main should supply its final runner/model/objective/reporting files in
`source_files` so ready freezes their hashes together with both launchers.

Preparation only, without resource reservations or training:

```bash
bash code/V4/run_v4_r85_launch.sh prepare --spec /ABSOLUTE/FINAL_SPEC.json
```

Only after main declares the final runner ready and its CPU checks passed:

```bash
bash code/V4/run_v4_r85_launch.sh start \
  --spec /ABSOLUTE/FINAL_SPEC.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_FINAL_READY'
```

`start` was executed exactly once for each arm after main's final start message.
The four `tmux r85_<run_id>` controllers run one-shot sequential GPU/CPU stages.
No implicit resume, failed-stage retry, or duplicate launch occurs.

Actual startup snapshot: A/B/C/D physical GPU utilization was 92/97/56/96%.
Physical host PIDs were 3555073/3600112/644375/645535 respectively, with the
container/NVML PID mapping recorded in the startup receipt. At the bounded
handoff the four models had 468/396/648/468 successful updates. These are
startup evidence, not validation or final model results. Progress is in each
model arm's `status.json`, `updates.jsonl`, and the corresponding launch
`train_VALlock_TEST.log`. The cached TRAIN/VAL files were reused with zero
copied bytes and no repeated warm-cache audit.

Runtime receipts/logs are isolated in
`code/V4/runs/R85_launch/<run_id>/`. Existing R84 module files and receipts are
not edited. R85 loads the stable R84 launcher into a private Python namespace,
changes only in-memory resource routing and read-only cache verification, and
uses `srun --gres=none` CPU entry into the original allocation-owned container.
There is no GPU execution through an ordinary SSH user session.
