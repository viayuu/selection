# Ready Finite Collection CLI For Main

Full18 inputs are MATERIALIZED, rc0. Manifest:
`/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/full18_inputs_v1/manifest.json`
(`eff9cfbf922dd6ed34735ef5a26f83ea1c0fa07fc97c70d05beb985ef733d94a`).
4608 bases,23040 endpoints,113920 whole-pool task rows. No solver labels yet.
The original structural source and planned_v1 have NOT been modified.

The default `collection_v1/protocol.json` is CPU-created with4 lanes and
`cold_native`, preserving the frozen fresh-backend-per-task execution contract.
No queue, waiter or GPU process was started. Main owns all transport and releases.

## Only AFTER R120 TEST/LIB/Report And Actual GPU Release

Main supplies `R120_RELEASE_RECEIPT` and `R121_BINDING`. Their schemas are the
unchanged `SCHEMA.md` release/binding schemas. Release must name the frozen plan
SHA `06362b8fe87fe0540778b68073e8f69fb5140f9fcae8a33e73a078913ced644e`.
Binding must have actual process hostname, physical GPU UUID and exact CPU
affinity. Set `R121_GPU_UUID`, `R121_CPU_SET`, `R121_LANE` consistently; lanes0..3
retain23/22/22/22 deployment groups on their respective fixed physical binding.

One finite preflight wave per main-owned lane, at most24 actual attempts:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES="${R121_GPU_UUID:?main assigned UUID}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "${R121_CPU_SET:?main assigned CPUs}" python -B -u -m code.V4.r121_collection wave --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_v1 --phase preflight --lane "${R121_LANE:?0 through 3}" --max-jobs 24 --device cuda:0 --release "${R120_RELEASE_RECEIPT:?main release evidence}" --binding "${R121_BINDING:?physical binding JSON}" --execute
```

Only successful saved preflight qualifies a deployment for a LABEL wave:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES="${R121_GPU_UUID:?main assigned UUID}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "${R121_CPU_SET:?main assigned CPUs}" python -B -u -m code.V4.r121_collection wave --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_v1 --phase label --lane "${R121_LANE:?0 through 3}" --max-jobs 128 --device cuda:0 --release "${R120_RELEASE_RECEIPT:?main release evidence}" --binding "${R121_BINDING:?physical binding JSON}" --execute
```

Each invocation is finite. Parent's resource worker decides whether/when another
wave is launched. Completed receipts/cursors are reused; no automatic retry of
an unavailable method, no restart of a completed task, and no timer-based restart.
The controller starts an isolated child namespace per deployment, using its
correctly locked original or corrected-CVRP scripts. No SSH/allocation logic is
embedded. Each child has a finite remaining task budget and inherited affinity.

TSP8 and ATSP5 invoke real native factory adapters with singleton FP32 tensors,
original complete budgets and independent geometric/directed cost/tour audits.
They are NOT claimed runtime-qualified before these actual preflights. Historical
MatNet role draws are not replayed; the new per-base seeded native execution is
declared. Runtime failures are immutable `unavailable/<deployment>.json` with
first error, source identities and task. They block full-pool export; methods
are never removed from the legal pool or replaced by another method/default.

Cold loading may be costly; no throughput estimate is promised. Optional
`cached_native_v1` exists ONLY as a separately selected execution-layer version:
create a NEW root using `create --mode cached_native_v1`, never alter/mix the
existing cold root. It reuses one backend per deployment child, native singleton
budgets/RNG resets and R114 lazy-runner resets; it is NOT asserted cold/warm or
historical replay equivalence. Parent must explicitly choose that tradeoff.

## Saved Results And Complete Export

`preflight/`, `labels/`, `progress/`, `unavailable/`, `logs/`, `worker_receipts/`
and `waves/` retain actual state under the chosen collection root. `status` is
saved-only, not a GPU probe; it does not call a solver.

After ALL89 deployments actually complete ALL113920 label tasks:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B -m code.V4.r121_collection export --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_v1
```

This validates native certificates and complete costs without rerunning solvers,
then emits `family_labels.jsonl` and `family_labels.receipt.json` for the learner.
Until then, `NotReady` is an honest blocker, not permission for partial labels or
for excluding the13 formerly unmeasured TSP/ATSP deployments. Old90k labels and
all existing TEST/LIB/R120 artifacts remain untouched.
