# Multidomain raw data preparation

Base commit: `04866414831d25330b0d2e575e4b63c68b4196d0` (read HEAD and packed-refs;
no Git command). Scope: CPU data preparation only; no solver, GPU, or training.

## Assumptions fixed before implementation

- Use accepted `nonrouting_v2_20261025` inputs and labels, not the older v1
  generator. Preserve source IDs, input identities, order, splits, parameters,
  seeds, and any explicit base-group fields. If no base group was supplied,
  expose the input identity as a fallback group and declare that fallback.
- Reference the frozen routing release in place. Hash its small manifests and
  split-index file; do not load/copy its instance or label data. Selector seed 2
  does not replace the nonrouting source data seed 20261025.
- Candidate columns are the lexicographically sorted, full solver IDs in the
  accepted 25-configuration inventory. L2D/DAN weights remain separate actions.
  Deferred search methods remain provenance, not invented observed columns.
- Native objectives are float64, maximized for MIS/MKP and minimized otherwise.
  Exact equality retains all ties in `all_best_mask`; missing objectives are
  NaN with `observed_mask=False`. A canonical winner is only the first best
  column in fixed solver-ID order, explicitly not a native unique winner.
  No compatibility cost, gap, regret, loss, normalization, or SBS is introduced.
- MIS/JSSP/FJSP have accepted formal 10000/1000/1000 pools with 3/6/10 columns.
  BPP has 10000 training inputs but only two all-tied candidates and no formal
  VAL/TEST. MKP (270, one candidate) and SMTWTP (375, three candidates) are
  benchmark-only. All six raw types are supported; all-six readiness stays false.
- Resumed scope: export MIS/JSSP/FJSP existing splits, including separate
  pilots. BPP/MKP/SMTWTP remain explicit pending manifest entries, to be supplied
  by the other worker. Never concatenate pilots or benchmarks into formal
  splits. Reuse completed audits and hash files; do not rerun solver audits.

## Parent-facing API

Module: `code.V4.nonrouting_data`.

`load_manifest(root)` reads the registry only. `load_nonrouting_datasets(root)`
returns `datasets[problem][split]` for ready problems and formal TRAIN/VAL/TEST
only; explicit `splits=(...)` enables separate pilot reads. Pending problems are
not silently selected. Dataset properties: `candidate_solver_ids`,
`objective_direction`, `descriptor`, `label_arrays` (read-only NumPy mmaps).

`NonRoutingDataset(root, problem, split, allow_incomplete=False)` returns
`{"inputs": ..., "labels": ..., "metadata": ...}`. `collate_nonrouting(samples)`
accepts a nonempty single-problem batch and returns the same three namespaces.
The dataset returns NumPy views; the collator returns CPU tensors for inputs
and numeric labels. IDs, source references, and certificates never enter inputs.
Pending problems raise until a subsequent manifest explicitly attaches an
export. Reading an attached incomplete pool requires `allow_incomplete=True`;
split names stay native
(`train`, `val`, `test`, `pilot_dev`, `pilot_val`, `benchmark`).

Raw input shapes (one sample; integral values, no engineered summaries):

- MIS: `num_nodes`, `edge_index[2,E]` (one native undirected edge, no reverse
  duplication). Batch masks: `node_mask`, `edge_mask`.
- JSSP/FJSP: `num_jobs`, `num_machines`, `job_ptr[J+1]`, `operation_job[O]`,
  `operation_index[O]`, `precedence_edge_index[2,P]`. JSSP additionally
  `duration[O]`, `machine[O]`; FJSP `processing_time[O,M]` and
  `compatibility[O,M]`, with incompatible times -1, never usable zero times.
  Batch masks: `job_mask`, `machine_mask`, `operation_mask`, `precedence_mask`,
  `job_ptr_mask`; compatibility remains distinct from padding validity.
- BPP: `sizes[N]`, `capacity`; batch `item_mask`.
- MKP: `profits[N]`, `resources[N,R]`, `capacities[R]`; batch `item_mask`,
  `resource_mask`. Resources transpose the native resource-by-item matrix only.
- SMTWTP: `p[J]`, `w[J]`, `d[J]`; batch `job_mask`. Preserve zero due dates.

Labels expose `native_objective[K]`, `observed_mask[K]`, `all_best_mask[K]`,
`canonical_winner_index` (-1 if no observations), and native direction. Solver
IDs, version, instance ID, identity, split, base group, and source provenance
remain metadata. Padding indices/times use -1; masks define valid entries.
Source-native scale/distribution live in `metadata['parameters']`; there is no
engineered summary input. `metadata['source_provenance']` references
`manifest['source_files']` with `[file_index, one_based_jsonl_line]`, aligned to
the candidate solver IDs. Source catalogue entries carry SHA256 and real path.

```python
from code.V4.nonrouting_data import load_nonrouting_datasets, collate_nonrouting
datasets = load_nonrouting_datasets(root)
dataset = datasets['MIS']['train']
batch = collate_nonrouting([dataset[0], dataset[-1]])
model_inputs, supervision = batch['inputs'], batch['labels']
# Pass only model_inputs to the model; metadata is not a model input.
```

Target: `/public/home/shiys/selection_data_multidomain_seed2_20261029/manifest.json`.
Creation is exclusive; the manifest is published only after the complete export.
Future new data/candidates require a new prepared version, not in-place mutation.
`make_sample(instance, labels, candidate_solver_ids, source_provenance=...)`
constructs the shared record schema. `write_prepared_split(directory, samples,
count=..., problem=..., split=..., candidate_solver_ids=...)` exports that schema
and returns an absolute-directory descriptor, reusable by a later immutable
combined manifest. Neither helper creates split memberships or runs solvers.

## Verification and build result

The single ready3 export completed in tmux `r120_nonrouting_data_ready3_v1`
with exit status 0. It is preserved as an immutable source, not rebuilt.

| Domain | TRAIN | VAL | TEST | Pilot dev/val (each) | Deployments | All labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIS | 10000 | 1000 | 1000 | 360 | 3 | 38160 |
| JSSP | 10000 | 1000 | 1000 | 80 | 6 | 72960 |
| FJSP | 10000 | 1000 | 1000 | 160 | 10 | 123200 |

Total: 37200 inputs and 234320 labels, including separate pilots; 30000/3000/
3000 formal TRAIN/VAL/TEST. Native inputs/labels occupy about 569 MiB. The
manifest records 2155 source files and keeps BPP/MKP/SMTWTP pending. Routing
is reference-only. Ready3 manifest SHA256 (unchanged after all CPU checks):
`fef91b492000966f7dbd145f6f730828c4d5875cc445777be62a9b754c9ad162`.

`python -B -m unittest code.V4.test_nonrouting_data -v`: 11 tests passed, no
skips after build. Checks include all six real types, input/label isolation,
FP64 min/max/ties/zero/missing values, identity/base-group collisions, ragged
roundtrip, missing/extra tail protection, no overwrite, mmap pickle reopening,
and all ready split tails/best masks. TRAIN index 9999 matches native source
objectives in every deployment column. Separate two-worker CPU DataLoader
spines loaded indices 0/9999 with K=3/6/10 and FP64 supervision.

## Supplement and runner handoff

The actual supplement functions are `nonrouting_supplement.iter_records(...)`,
`as_sample(record, solver_ids=None)`, and `export_prepared(...)`. There is no
`read_supplement` or `get_supplement` alias in either current data module.
`as_sample` belongs to the supplement worker, not this file. Few existing real
BPP/MKP/SMTWTP records were passed through it and our collator successfully;
this was a read-only bridge check, not new label generation or a full audit.

The join identity is `(metadata.dataset_version, problem, split, instance_id)`
plus exact `metadata.input_identity` equality. Do not substitute prepared or
supplement version for the source version: reused BPP TRAIN retains v2 IDs.
Both sample builders provide this tuple and sorted `candidate_solver_ids`.
Direction, FP64 arrays, exact best masks, and observed masks must also agree;
identity alone does not establish measured-pool readiness. Supplement integer
inputs may be int64 rather than ready3 int32; the collator returns int64 CPU
tensors for both. Source-provenance representation is source-specific: do not
interpret supplement references using the ready3 source-file index catalogue.

After supplement publication, runner-owned
`r120_experiment.merged_nonrouting_manifest(ready3_root, supplement_prepared_root)`
combines the same `nonrouting_raw_v1` descriptors. The runner writes its own
`nonrouting_manifest.json`; pass that file to `NonRoutingDataset` or
`load_nonrouting_datasets`. Descriptors use absolute directories, so no copying
or rebuilding ready3 is needed. `prepared_cpu` can leave MKP incomplete;
`prepared_with_as` is usable formally only when the supplement worker marks
all required candidate labels/splits complete. The default reader rejects
pending/incomplete pools. Certificates remain outside model inputs.

## Bounded Sol critical review

Current review outcome: no remaining concrete training-blocking or
direction/tie/mask/typed-relation defect found within this limited scope.
The earlier full-solver-ID family parser mismatch was fixed in the current
parent file (not edited here): four L2D deployments now share family `L2D`,
eight DAN deployments share `DAN`, while deployment IDs remain distinct.

Reviewed `r120_model.py` SHA256
`cb3da6862ea1d79b29837032fba4a2d7d411fe43a0110e0ef24aef82a95bd721`
and `r120_objective.py` SHA256
`a7e6084c7cfc3d0b55589eddfa1c4c344ad48bc85ed0718b50165ff9f9425883`.
Six two-record real-input CPU forwards passed; single versus padded-batch
utility differences were at most 5.97e-8. Replacing FJSP incompatible/padded
times with 123456 left utilities bit-identical. Targeted backward probes had
finite nonzero gradients through typed projection, sparse relation source,
relation values and score head for all six types. Ready3 used all K=3/6/10
columns; pending-type input spines used their accepted legacy records with
K=2/1/3, not a claim of complete supplement pools. Native objective probes
passed FP64 max-direction near ties, all-best zero objectives, missing NaN
masking and finite loss gradients. No optimizer or parameter update occurred.

Boundary: easynco has no pytest (`No module named pytest`), so the parent pytest
files were not executed as a suite here; the checks above are explicit bounded
standalone CPU probes plus the 11 data unittests. This is not a GPU/AMP/DDP,
full-scale memory/performance, full-24 readiness or experiment-result audit.
No parent-owned files, frozen supplement/dispatch source, GPU AS processes,
solver runs, or existing experiments were changed. Data worker is now idle,
available only for necessary small full24-manifest integration fixes.
