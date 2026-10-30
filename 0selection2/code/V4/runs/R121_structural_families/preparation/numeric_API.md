# R121 Numeric Collection V3 Handoff

READY: CPU-only implementation; no native solver/GPU run or production-root
creation by this sidecar. Only the three newly owned files were added. All old
collector/model/driver/READY/isolation/input/label files remain unchanged.

## Frozen Sources And Evidence

- `r121_numeric_collection.py` SHA256: `a9c638f48cbb8164171ef95bef77ae3b159a254968eddc78a71676b7dc3df50e`.
- `test_r121_numeric_collection.py` SHA256: `856cb532303e7db0a5239aa7006d408b756678964708e248f5abaa4a9ae8a091`.
- Module version: `r121_numeric_boundary_collection_v3_20261030`.
- Independent checker version: `r121_independent_L_fp32_plus_one_ulp_v3_20261030`.
- Native `VRPBLEnv.py` SHA256: `3cacadf53e8a3235fe54e60a9609f438bb43cf61404eefab8bd9efcb9d389f28`; source AST confirms EPS=`1e-5` and depot length reset.
- Real diagnostic SHA256: `262e394393309c14b2c950df457858e1a0d98e9da6437ba9c05ad9e5069e70ee`.
- Original numeric failure SHA256: `17be5620fb2d73d9709988972e7e1e202b8a9a02c92d6eaf5faf16bd3ed3a530`.
- Actual diagnostic/native inventory profile matches `68db943b4a8bc778714b41ff676c05918592e99b938a0f1318f363600b80f2b1`; binding matches `f852b1a8d770e381b872e322804683aad5e567f96c0051bc89e4c239b963a004`.

This is a versioned independent-audit boundary recovery, NOT a solver algorithm,
seed, weight, native budget, original input, candidate-pool, or benchmark-label
change. The native collection protocol/hash remains unchanged; `numeric.json`
separately seals the new checker and source strategy.

## Exact Numeric Rule

Always run the original complete strict independent audit first. Only an exact
`ValueError('Route-length limit violation')` can enter the fallback. Clone the
frozen audit body with just that exception replaced by this conjunction:

```text
T = float32(float32(route_limit) + float32(1e-5))
U = nextafter_float32(T, +infinity)  # exactly one ULP32
original-coordinate FP64 closed/open segment distance <= float64(U)
AND independently accumulated original-environment-style FP32 distance <= T
```

FP32 edges use input coordinates cast to FP32, Euclidean norms, and sequential
FP32 additions. Closed VRPBL includes the depot-return edge; open variants omit
that charged edge, as the original independent checker does. This is an
independent CPU FP32 calculation, not a claim to have captured native CUDA state.
All later routes, signed capacity/order/coverage/TW checks and final total-cost
comparison continue through the original audit body. No cost clipping or input
repair occurs. For the real saved diagnostic:

```text
FP64 distance      = 3.0000100912493717
FP32 sequential   = 3.0000100135803223 = T
one ULP32         = 2.384185791015625e-7
FP64 upper U      = 3.0000102519989014
strict excess     = 9.124937167982157e-8
charged edges     = 23
native raw cost   = 11.819985389709473 (unchanged)
```

Every newly accepted boundary carries its raw FP64/FP32 lengths, original limit,
threshold, one-ULP upper bound, route/depot positions, charged-edge count and
checker version. The diagnostic is evidence only; it is NOT promoted into a
training label. Main's actual production worker still solves the missing task
once under the original native recipe and seed.

## Fixed Sources And Bootstrap

`create()` seals every deployment's source root BEFORE any prefix snapshot:

1. The original five keys in sealed InputV1 `recovery.json:targets` use InputV1.
2. All remaining lane1 keys use `collection_explicit_cpu_v2`.
3. Every other key uses Original `collection_persistent_v1`, including lane0's
   new Original continuation producer. Lane is original `group_order` index%4.

No min-cost, longest-prefix, or alternate-root selection exists. Mutable progress
is not pinned at create. Main must wait for each lane's source producers to reach
terminal and release their physical mutex before bootstrap. The bootstrap also
requires inherited `RETURN18_GPU_MUTEX_FD` and scans active source-root/lane/key
processes WITHOUT a controller-module whitelist, including the new Original
controller. It returns NotReady instead of stopping a live producer. Queue order
and cross-container physical exclusivity remain main/resource-worker duties.

Bootstrap takes the latest actual contiguous, independently valid prefix from
the ONE fixed root per key, including a saved valid tail just ahead of an atomic
progress update. Original qualifications and prefix records are adopted byte for
byte, never re-solved. It validates exact task/input/seed/profile/original binding,
checks source progress's last committed receipt, and preserves nested V2 source
provenance. Per-task adoption SHA/provenance lives in `adoption/labels/` and
`adoption/preflight/`; no source failure flags are copied.

Only13 TSP/ATSP CUDA-to-NumPy TypeErrors with the matching next-task identity can
cross bootstrap via the proven explicit-CPU constructor. An old numeric failure
can cross ONLY when its task/input/seed/profile/binding/failure SHA matches the
one real diagnostic proof. Unknown or hard violations stop; no generic retry.
New TSP/ATSP inputs are explicitly CPU before the unchanged CUDA backend call.
Other solvers keep original `session.solve`; only the independent checker is
temporarily wrapped inside the NEW worker process and restored in `finally`.

## API And CLI

Python signatures:

```python
create(root=ROOT, parent=PARENT, overlay=OVERLAY,
       universal_root=UNIVERSAL, diagnostic=DIAGNOSTIC)
bootstrap_lane(root, lane, release, binding)
worker(root, key, phase, max_jobs, device, release, binding)
status(root)
export(root, output=None)
load(root)  # -> recovery, unchanged_native_protocol, manifest, inventory
```

Default new root:
`/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3`.
Release/binding and per-lane physical allocation stay original. The production
commands below belong only in MAIN's resource wrapper, not sidecar GPU dispatch.

CPU create, safe while old producers are still live:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_numeric_collection create --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3
```

Bootstrap ONLY under main's original lane binding/mutex AFTER source drain:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && python -B -m code.V4.r121_numeric_collection bootstrap-lane --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3 --lane "${R121_LANE:?main lane0..3}" --release "${R121_RELEASE:?main original release}" --binding "${R121_BINDING:?main original lane binding}"
```

Finite worker, only the missing tail after adopted progress; use a UNIQUE result
path per invocation. Existing qualifications can be reused with `--phase
preflight`, but missing qualifications are NotReady and never rerun:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && python -B -u -m code.V4.r121_numeric_collection worker --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3 --deployment "${R121_DEPLOYMENT:?one original key}" --phase label --max-jobs 1280 --device cuda:0 --release "${R121_RELEASE:?main original release}" --binding "${R121_BINDING:?main original lane binding}" --result "${R121_RESULT:?unique path under numeric root}"
```

Worker results expose `numeric_version`, `numeric_sha256`; lane markers use the
same fields plus `bootstrap_sha256`. A real new failure records the full actual
traceback under the new root and stops that key. CLI exits0 success,2 unavailable,
3 NotReady,1 other failure; no automatic retry or source-root error-file writes.

Saved-only CPU status/export:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r121_numeric_collection status --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_numeric_collection export --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3
```

Export is blocked until every lane is bootstrapped and all89 successful original
qualifications plus all113920 labels are in this ONE root. It checks adoption
bytes/source chains, unchanged raw native costs/tours, and independent numeric
evidence. `family_labels.receipt.json` retains the frozen learner-compatible
complete/full18/pool/input fields and adds `independent_numeric_version`, fixed
source policy, per-source provenance counts and the full boundary-case evidence.

Main's NEW followup must explicitly pass
`--family-labels /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_numeric_boundary_v3/family_labels.jsonl`
to frozen learner preparation/protocol/train stages. The frozen learner's default
still points to V2; do NOT edit the learner or pretend the old export completed.
No changes to historical benchmark labels, official VAL/TEST, B semantics, or
R58 deferral are involved.

## Bounded CPU Verification

Eight tests PASS in2.429s, one CPU core, CUDA hidden. They cover the real saved
numeric case and exact provenance/profile-mismatch stop; both independent
numeric bounds; post-L capacity/cost/coverage rejection; all13 explicit-CPU
second calls after a simulated lazy default-device change; three fixed-source
byte-identical adoption and toy bootstrap/worker/export flow; unknown failures;
new-controller drain detection; literal CLI and old-root write protection.
The full production create was also executed with writes intercepted:89 keys,
113920 tasks, exact fixed source policy, no prefix snapshot and NO production
root/label creation. Small toy counts are unit fixtures, not production results.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 46 python -B -m unittest code.V4.test_r121_numeric_collection -v
```

No extra solver, GPU, training, old-source edits, old-root writes, READY/seal edits,
Git operation, or new label population was performed by the sidecar.
