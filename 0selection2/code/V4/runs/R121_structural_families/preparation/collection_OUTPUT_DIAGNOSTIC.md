# Native CUDA-To-NumPy Diagnostic, Not A Claimed Fix

The real output-serialization recovery1 FAILED on ATSP/UNICO_MatPOENet with the
same TypeError. TSP/OMNI was not reached. Its failed JSON and all earlier files
remain unchanged. The previous two CPU fake-output cases did NOT cover the real
native stack and are not runtime qualification evidence.

NEW module:code/V4/r121_native_output_diagnostic.py. No existing collector,
recovery, helper, model, profile, source failure or production label is edited.

Additive root-API revision SHA256:
be1fa2e3839c52c2504da35c0e6cdba3c6cc617051a168b5b15e99cabc49083a
One argparse-only check PASS (collection_output_diagnostic_argparse.log); it mocks
run and does not open the manifest, initialize a model, or execute a solver.

## Read-Only Source Findings, Still An Inference

- UnicoATSPBackend.run and NativeARBackend.run already turn their returned costs
  and tours into Python values before returning InferenceResult.
- Their internal audit_tours calls np.asarray(item[0], dtype=np.float64).
- UniCO __init__ sets torch.set_default_device(self.device). The frozen collector
  later constructs torch.tensor(payload['item'], dtype=float32) WITHOUT an explicit
  device. The likely issue is therefore native INPUT device under the adapter's
  changed global default, before the collector receives any output.
- This is NOT an actual-stack conclusion. Do not yet move a normalization or
  change a solver. The next main-owned diagnostic is deliberately unmodified.

## One Explicit Main-Bound Attempt

No GPU job has been launched by the sidecar. Main may run ONCE, when authorized:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="$R121_GPU_UUID" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 4,5,72,73 python -B -u -m code.V4.r121_native_output_diagnostic --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_output_diagnostic_v1 --release /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_r120_release.json --binding /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_binding_lane0.json --device cuda:0 --execute
```

Defaults select exactly the preserved failed ATSP_src10130_original__UNICO_MatPOENet
task and the SAME physical binding/seed/profile. Native backend.run is called at
most once. Original helper, input placement and native calls are unchanged. Only
the backend instance's run entry is observed, with original method restored.
Instrumentation records types/shapes/devices WITHOUT copying/converting values.

Outputs in the NEW collection_output_diagnostic_v1 tree:

The generic resource worker may pass literal --root. Its default is the same
collection_output_diagnostic_v1 tree. Output defaults to
ROOT/ATSP__UNICO_MatPOENet_attempt1.json; explicit --output must stay inside ROOT.

- ATSP__UNICO_MatPOENet_attempt1.claim.json:exclusive attempt claim; no auto retry.
- ATSP__UNICO_MatPOENet_attempt1.json:stage, native input/output device metadata,
  native_calls0/1, full exception traceback and filename/line/function frames.
- ATSP__UNICO_MatPOENet_attempt1.traceback.txt:unabridged Python traceback on failure.

A diagnostic success is NOT registered as a preflight or a production label.
No normalization/recovery, shortened solver budget or automatic second attempt
exists in this module. On failure it exits2 AFTER saving the trace. Existing claims
and outputs are never overwritten or silently rerun.

Only the instrumentation's exception-chain capture was CPU-tested. There is NO
new GPU recovery success claim. After main supplies the actual stack, use its real
failing boundary to define the next separate, explicitly versioned correction.
