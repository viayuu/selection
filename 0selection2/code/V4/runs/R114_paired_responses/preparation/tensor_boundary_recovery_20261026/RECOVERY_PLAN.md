# R114 CVRP Tensor Boundary Recovery

Status: source revision prepared only. Not integrated, scheduled, GPU-probed or
native-runtime-qualified. This sidecar does not change the main READY state.

## Diagnosis

The genuine first failure is the immutable cold receipt
`grouped_execution/queue_v1/attempts/runtime_group__CVRP__ICAM/attempt_0001/cold/CVRP_src53138_original__ICAM.json`.
It reports `TypeError: can't convert cuda:0 device type tensor to numpy`, not an
unsupported native candidate or native-paradigm failure. The receipt contains
no traceback and no usable native cost or route; do not infer native success.

Frozen source trace:

1. `train_100k_icam.py:28` sets the default tensor device to the solver device.
2. `r114_response_data.py:724` creates CVRP input tensors without an explicit
   device after constructing that backend. Grouped solves have the same pattern
   at `r114_group_worker.py:122`.
3. ICAM converts its selected native tour/cost to host lists, then calls
   `audit_tours("CVRP", items, tours, costs)` at `train_100k_icam.py:58`.
4. `train_100k_backends.py:83` directly calls NumPy on the original input tensor
   for the first depot conversion. The later receipt `plain()` cannot help.

This identifies the source-level boundary, not a recovered GPU traceback.
CPU tests exercise the actual frozen audit body with a CUDA-boundary double.

## Minimal Correction

The new helper `code/V4/r114_tensor_boundary_recovery.py:cpu_numpy_view` performs
`detach().cpu().numpy()` recursively while preserving shape and dtype. The
generated source patch adds only a CVRP guard and host-view conversion at the
common `audit_tours` entry. Native solver inputs, arithmetic, decoding, masks,
weights, precision, seed reset, default device and budgets remain untouched.
Route validation, tolerances and cost recomputation remain untouched.

The common audit is also imported by ICAM and the NSS CVRP adapters. The
alias-safe MoSES_RF path delegates to the original RouteFinder base `run`.
Thus the correction targets the common audit used across the full frozen
CVRP legal pool, not just ICAM. This is source coverage, not runtime proof.

## Main-Owner Integration

1. Leave running v1, all failures and all other problem groups unchanged.
   Main chooses the transition after v1 reaches a meaningful terminal state.
2. Create a new isolated, complete scripts/source freeze. Apply the generated
   patch only there and include/hash the helper. Do not place this patch folder
   on PYTHONPATH or patch the live bridge scripts. The native `SCRIPT_ROOT`,
   relative helper files and original `WORKSPACE` bindings must still resolve
   to the intended assets in the new deployment.
3. Rebuild source-dependent native profiles, inventory, cold lock, grouped
   lock and queue plan under a new execution revision. Do not copy old profile
   hashes or merely disable lock checks. The current dispatch hardcodes its
   factory entry and verifies frozen files; a standalone factory override
   cannot safely retrofit the bound v1 queue.
4. Retain exactly the parent manifest/input hashes, endpoint/task IDs, all
   10 legal CVRP candidates and solver seeds. Preserve native candidate budgets,
   weights, lazy-subrunner resets and batch size 1. No retry/seed/parameter
   search, replacement instances, candidate drops or reused failed costs.
5. Both cold and grouped processes must resolve the same revised common audit
   and helper. Run the existing exact cost/tour and intervening-instance replay
   qualification gates for every legal CVRP deployment before labeling.
6. Write only new attempts/receipts with new lock provenance. Keep v1 evidence
   intact and do not merge the error into a success. Main owns scheduling,
   complete-pool accounting, READY decisions and downstream version integration.

## CPU Verification

Run from `/public/home/shiys/0selection2`:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest -v test_r114_tensor_boundary_recovery
```

The three tests cover real CPU tensors, exact detach/CPU/NumPy order, the
actual frozen audit's failure and corrected behavior, unchanged invalid-route
rejection, unchanged non-CVRP behavior, and a fail-closed source-hash guard.
Synthetic inputs are unit-test fixtures only, never label receipts.
