# Isolated CVRP Acquisition

The local `CPU_READY.json` is preparation evidence only, not the main READY gate
or native qualification. No GPU has been bound or launched by this sidecar.

The frozen parent manifest is the sole authority for scope and exact rows;
the builder derives counts instead of enforcing estimated or fixed counts.
The current parent has **32 bases, 96 endpoints, 64 pairs, 10 methods, 960 label
tasks**. Original/directed/random each currently have 32 endpoints. Every
endpoint, input path/hash, pair, task ID and solver seed is retained exactly.

`scripts/` contains one corrected file, `train_100k_backends.py`. The other bridge
scripts are symlinked to the original sources. `source_bindings.json` records
both the staged path and actual resolved target/hash. In particular the symlinked
`train_100k_common.py` resolves to its real original location, preserving WORKSPACE,
SOURCE_ROOT and native assets. The existing compatibility overlay and weights
are reused and hashed. Only the CVRP host audit view is changed.

The original and corrected snapshots execute the actual native classes'
`profile()` bodies on frozen recipe metadata views without constructors or
solver calls. Both must reproduce the parent profiles exactly. Native model
initialization and profile equivalence remain mandatory runtime qualification;
these CPU snapshots do not replace it. All dependency contracts, path bindings,
inventory, cold lock, grouped lock and queue plan are newly generated.

## Commands

From `/public/home/shiys/0selection2`, CPU preparation and checks:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.r114_cvrp_recovery prepare
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.r114_cvrp_recovery verify
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v test_r114_cvrp_recovery
```

`CPU_READY.json` contains the exact dry-run argv and manifest/inventory/cold/
grouped/plan hashes. `run_native.sh` is the exact existing-runner CLI, checked
against that runner's real parser and with `bash -n`, but never executed here.

**Main only**, after the meaningful v1 terminal state: enter the assigned
container, hold the external GPU mutex, use tmux for the long task, and export
`R114_RECOVERY_HOST` (actual container hostname), `R114_RECOVERY_GPU_UUID`,
`R114_RECOVERY_CPU_IDS` (1-2 assigned IDs), and
`R114_RECOVERY_TIMEOUT_SECONDS` (unchanged owner-approved timeout). Then invoke:

```bash
bash /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/run_native.sh
```

This reuses the existing fail-fast group worker and serial runner with the new
plan root, including identical probes, cold references, replays, legal-pool
barrier and lazy-subrunner resets. Cold and grouped processes share the corrected
scripts. No automatic retry, changed budgets/seeds or candidate drops exist.
The CLI intentionally omits `--finalize`: main owns combined dataset/analysis
integration. v1 attempts/failures and its other 15 problems remain untouched.
