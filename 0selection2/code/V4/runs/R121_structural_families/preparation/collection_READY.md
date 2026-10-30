# Collector Frozen Handoff

Status: CPU full18 inputs and persistent collection protocol CREATED; no native
solver/GPU job launched by this sidecar. No collector source edits after handoff.
The full plan stays256 bases/task,4608 bases,23040 endpoints,113920 labels,89 legal
deployments. Full256 annotation budget is not approved merely by creating a plan;
main first observes finite preflight/production timings. No silent64-base shrink.

Frozen source SHA256:

- r121_collection.py:876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674
- test_r121_collection.py:ad8bc57b675ad95d6c3629ccc73d625d8a825782b3f2041b661703d858c9778e
- r121_structural_data.py:719511ae34a02b0a1db417c70761d9289cc380a6a654b473f12b69bcc32d1771
- collection_persistent_v1/protocol.json file:c0c1ad3aed88e3a03cd3dd435675d98121ee85e5a08daa8581f9a0c4f8d5975d

Minimal collector tests:6 PASS (collection_tests.log). Combined frozen structural,
collector, model and learner CPU suite:31 PASS (learner_all_tests.log). This does
not certify89 native runtimes; those require exactly one saved native preflight
per deployment. Missing runtime/source/profile/route stops that deployment at its
first fatal error with an unavailable receipt; no pool drop or generic retry.

## CPU Protocol Create

Already executed rc0. The same command is idempotent only for identical content:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_collection create --inputs /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/full18_inputs_v1/manifest.json --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 --lanes 4 --mode cached_native_v1
```

## Main-Owned JSON Contracts

Release TEMPLATE, not a sidecar-issued claim of resource release. Main fills actual
authorized full UUIDs and accessible evidence paths/digests. Commit2330198 alone
does not substitute for completed-report and actual-release evidence.

```json
{
  "owner": "main",
  "r121_plan_sha256": "06362b8fe87fe0540778b68073e8f69fb5140f9fcae8a33e73a078913ced644e",
  "TEST_complete": true,
  "LIB_complete": true,
  "report_complete": true,
  "authorized_GPUs_released": true,
  "authorized_gpu_uuids": ["GPU-FULL-AUTHORIZED-UUID"],
  "evidence": [{"path": "/ABS/ACTUAL/R120/final_receipt.json", "sha256": "ACTUAL_FILE_SHA256"}]
}
```

Binding TEMPLATE:

```json
{
  "host": "ACTUAL_WORKER_socket.gethostname()",
  "gpu_uuid": "GPU-FULL-AUTHORIZED-UUID",
  "cpu_ids": [46]
}
```

The host is the actual Python process hostname (inside the container, if used),
NOT an assumed SSH alias. CUDA_VISIBLE_DEVICES must equal the same complete UUID
string; use --device cuda:0. taskset affinity must equal the exact cpu_ids set.
Preflight and labels for a deployment must retain this SAME binding JSON identity.
These JSONs are main-owned; the sidecar did not fabricate release/binding receipts.

## Finite Main Worker Commands

Main sets R121_GPU_UUID, R121_CPU_SET, R121_BINDING, R120_RELEASE_RECEIPT,
R121_LANE0..3. Only main executes the following GPU commands after allocation.

One finite preflight wave/lane, at most24 original endpoints; four lanes cover
all89 deployments once (23/22/22/22). Saved successful preflights are not rerun:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="$R121_GPU_UUID" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "$R121_CPU_SET" python -B -u -m code.V4.r121_collection wave --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 --phase preflight --lane "$R121_LANE" --max-jobs 24 --device cuda:0 --release "$R120_RELEASE_RECEIPT" --binding "$R121_BINDING" --execute
```

First finite production observation uses the SAME command with --phase label
--max-jobs8. After actual latency/budget review, production may use --max-jobs1280
or another explicitly finite budget. No process-per-label: one child/adapter per
problem-method shard, holding the adapter through its selected task sequence.

For a generic resource worker dispatching ONE deployment, bypass the lane wrapper:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="$R121_GPU_UUID" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "$R121_CPU_SET" python -B -u -m code.V4.r121_collection worker --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 --deployment "$R121_DEPLOYMENT" --phase "$R121_PHASE" --max-jobs "$R121_FINITE_JOBS" --device cuda:0 --release "$R120_RELEASE_RECEIPT" --binding "$R121_BINDING" --result "$R121_RESULT_JSON"
```

R121_DEPLOYMENT is an exact key such as ATSP/MATNET; phases preflight/label use
budgets1/8 initially, then a reviewed finite production shard. R121_RESULT_JSON
must be a fresh collection/preparation-owned result path. Unlike wave, worker is
an execution command without an --execute switch. It still requires main's release
and exact physical binding before importing any native backend.

**Persistent exception to expose, not conceal:** RouteFinder/MoSES lazy subrunners
still reset per native R114 endpoint lifecycle. Their checkpoint loading may remain
expensive. All other adapters persist through the deployment shard; native batch1
and all budgets remain unchanged. No all89 warm-equivalence or measured speedup is
claimed. Main's real preflight/first-production receipts determine availability and
cost. See collection_PERSISTENT.md and saved-only collection_estimate.py.

Production progress:collection_persistent_v1/progress/; preflights:preflight/;
labels:labels/; first fatal:unavailable/; child stdout:logs/; wave receipts:waves/.
No partial export is permitted. The separate learner is not a launch prerequisite.
