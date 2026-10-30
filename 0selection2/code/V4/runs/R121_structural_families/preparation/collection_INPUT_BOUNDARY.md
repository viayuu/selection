# R121 Trace-Backed CPU Input Boundary

Ready for MAIN qualification; no GPU/solver execution by this sidecar.
The original 89 qualification attempts are now terminal: 84 successful receipts
are referenced unchanged, and exactly five actual failures are sealed as targets.
No candidate, family endpoint, budget, solver/model source, seed, original binding,
cost/winner definition, or historical label is changed.

## Frozen Identity

- New module: `/public/home/shiys/0selection2/code/V4/r121_input_boundary.py`
- Module SHA256: `0e869a9f134ba7d3eeb43ede720e7169a50b40e57a02f91ccc60a370707debbe`
- Recovery seal: `66a1106e2d4eb19d372cd3fde8bbed5f4ef2b88afe21e986817f023d18e858ec`
- New root: `/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_input_boundary_v1`
- `recovery.json` pins the new source, original source/helper hashes, actual trace,
  five immutable first-error records, per-key original bindings, and 84 successful
  qualification references. `protocol.json` is identical to the parent protocol.
- Native collection identity remains `caec0a4e5ca6d290625f4df761d86f6e99ab8ad9f8e29fa2ed81d90d47578754`.
- Original collector remains `876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674`.
- Output-recovery1 remains `af0f031cd97f938fc02ea6ad98118fefc341fa9a56f9258917fcde88db334cb3`.
- Diagnostic remains `be1fa2e3839c52c2504da35c0e6cdba3c6cc617051a168b5b15e99cabc49083a`.

## Actual Evidence And Change

The actual diagnostic JSON has SHA256
`69e859935ac576c6dfff8b402eb865f5a3acc38e5a3fad862c5dba7fe231a5bf`;
the adjacent `ATSP__UNICO_MatPOENet_attempt1.traceback.txt` has SHA256
`4a5c810ba02f6bcf170b0510b63c101d5fc7a99ca61e0f6afcd2d96342624754`.
Both remain under `collection_output_diagnostic_v1`, unchanged.

The real call reached `train_100k_atsp.py:266`, then the backend's own
`train_100k_backends.py:79 audit_tours` attempted
`np.asarray(item[0], dtype=np.float64)`. Its input was already `cuda:0`, because
native initialization set the global default device before the collector built
the input tensor. The failure precedes the collector's output-tour audit.

The new layer compiles the pinned original helper with exactly one substitution:

```python
native = torch.tensor(payload['item'], dtype=torch.float32, device='cpu')
```

Only TSP/ATSP input creation is affected. Backend `.to(self.device)`, GPU model,
native run, internal audit, reset/seed order, independent audit, output conversion,
and persistent-per-deployment session are unchanged. No global-default-device
reset or downstream output normalization is installed. Each new receipt records
the layer seal and observed actual CPU input. A new failure records a complete
traceback under the new root; no automatic retry or old-record overwrite occurs.

## Exact Binding Map

Binding files below are in the unchanged `collection_persistent_v1` root.

| Deployment | Original task | Binding file | CPU affinity | GPU |
| --- | --- | --- | --- | --- |
| ATSP/MATNET | ATSP_src10130_original__MATNET | main_binding_lane2.json | 16,17,64,65 | cb626bdb-4319-2f5e-b91f-17fcaa44a83d |
| ATSP/MATPOENET | ATSP_src10130_original__MATPOENET | main_binding_lane3.json | 0,1,2,3 | 71479e7e-f8bd-e430-66ed-f8db9f0d47be |
| ATSP/UNICO_MatPOENet | ATSP_src10130_original__UNICO_MatPOENet | main_binding_lane0.json | 4,5,72,73 | 597c682a-4b7f-7609-80a6-8ced791f5c5d |
| TSP/LEHD | TSP_src10132_original__LEHD | main_binding_lane3.json | 0,1,2,3 | 71479e7e-f8bd-e430-66ed-f8db9f0d47be |
| TSP/OMNI | TSP_src10132_original__OMNI | main_binding_lane0.json | 4,5,72,73 | 597c682a-4b7f-7609-80a6-8ced791f5c5d |

The worker rejects any different canonical physical binding before native work.
MAIN must serialize same-GPU stages under its existing mutex. This is not a
permission to execute on a different idle GPU.

## CPU Create

Already executed successfully, with solver execution false. Idempotent command:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 46 python -B -m code.V4.r121_input_boundary create \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_input_boundary_v1 \
  --parent /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1 \
  --trace /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_output_diagnostic_v1/ATSP__UNICO_MatPOENet_attempt1.json
```

Creation refuses to freeze an incomplete original qualification set or unrelated
first errors. The sealed target set is not silently extended after launch.

## MAIN Worker CLI

Literal example for UNICO, run only by MAIN in its original bound container.
For the other four stages use the exact deployment, UUID, affinity and binding
from the table, plus a distinct result basename within this NEW root.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES=GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 4,5,72,73 python -B -u -m code.V4.r121_input_boundary worker \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_input_boundary_v1 \
  --deployment ATSP/UNICO_MatPOENet --phase preflight --max-jobs 1 --device cuda:0 \
  --release /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_r120_release.json \
  --binding /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_binding_lane0.json \
  --result /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_input_boundary_v1/main_UNICO_preflight_attempt1.json
```

This attempts exactly the preserved failed original once, not a synthetic new
input and not any of the 84 already-successful originals. Only after a successful
NEW per-key preflight, use the same worker arguments with `--phase label`, an
explicit finite `--max-jobs 8` (or MAIN's approved larger budget), and a fresh
result path. The cached native session persists within that worker chunk.
`--max-jobs 1280` is the full existing endpoint shard, not a new automatic budget.
On failure preserve the new unavailable/traceback record and report its first
error; do not retry or fall back to output-recovery1.

## Complete Export

The original parent root supplies the 84 successful deployments; this new root
supplies the five repaired deployments. Full18/89 completion is mandatory:
23,040 endpoints / 113,920 labels, no partial or reduced-pool export. Old failed
parent records remain intact and are not mistaken for active overlay failures.
The existing endpoint/targets loader schema is retained, with a new export receipt
identifying both native receipt roots and the input-boundary seal.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 46 python -B -m code.V4.r121_input_boundary export \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_input_boundary_v1
```

Default output: `collection_input_boundary_v1/family_labels.jsonl` plus
`family_labels.receipt.json`. Same CLI with `status` instead of `export` reports
the combined per-deployment saved progress without solver execution.

## CPU Proof, Not GPU Recovery Evidence

Four tests PASS in 0.024 seconds; log:
`collection_INPUT_BOUNDARY_cpu_tests_20261030_v1.log`.
The test calls real `torch.set_default_device('meta')` without GPU allocation.
The implicit original constructor then produces a non-CPU tensor and the actual
native `audit_tours` source raises TypeError. The single constructor substitution
produces CPU inputs and passes that same native audit for both TSP and ATSP,
then passes the unchanged independent route/cost audit. Global default remains
`meta` throughout; it is not reset by the fix. Tests also verify literal `--root`
parsing, rejection of a different original binding before native work, and refusal
of an inferred/absent actual traceback. No model, solver, or checkpoint is loaded.

Actual recovery is PENDING the five MAIN GPU qualifications. No claim that CPU
tests alone establish GPU compatibility or completed labels is made.
