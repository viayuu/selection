# Parallel Collection: Resource-Only Handoff

New files only: r121_parallel_collection.py, test_r121_parallel_collection.py,
and this note. Frozen transport, collector, structural data, native profiles,
seed/budget/batch and the full89 legal pool are unchanged. No GPU, Git, native
qualification or production was run by this implementation worker.

## Contract

The existing parent must use cached_native_v1, native singleton batch1,
113920 planned labels, 89 groups of 1280 and four lanes. Group order is read
verbatim from its frozen protocol; index modulo4 assigns the original lanes.
Exactly two DIFFERENT keys can run simultaneously inside each already assigned
4CPU/9G GPU lane. No Slurm/container allocation is created here.

Each child is a separate Python PROCESS invoking the SAME frozen
code.V4.r121_collection worker --phase label. Its finite --max-jobs is
min(remaining, --max-jobs-per-deployment), at most1280. One persistent native
worker/session per key; no threads/shared RNG, no per-label process churn.
The collector's documented RouteFinder/MoSES lazy-subrunner exception remains
unchanged. Device, physical binding, release and per-task seeds remain unchanged.
Each process retains the full four-CPU binding and single-thread BLAS/OMP;
splitting its affinity would violate the frozen binding. Inherited Return18 GPU
mutex fd is passed to both children when present.

Production requires an existing successful native preflight for every scheduled
key, matching collection/profile and exact binding. Missing qualification is
recorded as qualification_pending and is not run. Known unavailable/blocked keys
are recorded as needs_other_recovery_root, retaining their first-error reference.
They stay in group_order and the legal pool; no partial global export is made.
Parent owns the five input-placement recoveries in their separate roots.

## Required Handoff Before Launch

Parent must stop the old SERIAL wave controller for this lane from scheduling
more work and let its in-flight worker finish. This component refuses visible
same-root frozen wave/worker overlap. The frozen worker does not understand a
new sidecar mutex, so do not run a competing producer on this same root/lane.
This module never kills another collector or modifies its receipts.

The lane receipt is exclusive and permanently one-shot. New native failure or
missing/inconsistent child result returns nonzero, preserves the first error,
stops scheduling new keys and drains its other already-running child. There is
no key retry or implicit controller restart. Interrupted controllers likewise
require explicit parent handling, not deletion of their receipt.

Qualified-segment completion is separate from finite-budget completion. Small
budgets cannot claim all qualified labels complete. full89_complete is ALWAYS
false here; only the separate complete89 exporter can validate the whole pool.

## Exact CLI

Parent supplies the existing root, lane0..3 and the SAME release/binding used for
successful preflights. Commands below are NOT executed by this sidecar.

CPU-only description, no native solver execution:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.r121_parallel_collection --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 --lane 0 --concurrency 2 --max-jobs-per-deployment 1280 --describe
```

Parent-authorized production, inside its existing exact GPU/CPU lane binding:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -B -u -m code.V4.r121_parallel_collection --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 --lane "$R121_LANE" --concurrency 2 --max-jobs-per-deployment 1280 --device cuda:0 --release "$R120_RELEASE_RECEIPT" --binding "$R121_BINDING"
```

Execution is explicit when --describe is absent; there is no --execute switch.
Real production itself records wall timings. No speedup, memory-fit or89-key
qualification claim follows from CPU tests; no new microbenchmark is needed.

## Receipts And Tests

New runtime subtree under the parent:
`r121_parallel_collection_v1/laneN/controller.json` is the one bounded controller
receipt. `jobs/KEY/worker.log` and `jobs/KEY/result.json` preserve each child's
stdout/stderr and immutable frozen-worker result. Normal label/preflight/progress
formats remain wholly owned by the frozen worker; no patch or reinterpretation.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v code.V4.test_r121_parallel_collection
```

CPU mocked-Popen checks cover max2, unique keys/original lane order, exact frozen
child argv, finite remaining budgets, preserved known failures/no overwrite,
nonzero new failure/no resubmission, qualification/binding gates and no false
full89 completion. The parent owns all actual GPU execution and resource fixes.

Verification: 8 CPU tests PASS with affinity restricted to four CPUs. An actual
parent-root CPU-only lane0 describe also passed: 23 assigned keys, 21 qualified,
2 needs_other_recovery_root, solver_execution=false, full89_complete=false.
This is a read-only snapshot, not a GPU resource/throughput qualification.

SHA256 for parent pinning:

```text
r121_parallel_collection.py a9e686d13f0acb4e956178a4d21bf91c3851307a05cc1839ed411333c4bad218
test_r121_parallel_collection.py 55dbc13c85be797852fddd3d39bd270bbcab7d11f976d902f2e97246aaa62c75
```

Frozen sources rechecked unchanged:
r121_collection.py 876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674;
r121_structural_data.py 719511ae34a02b0a1db417c70761d9289cc380a6a654b473f12b69bcc32d1771.
