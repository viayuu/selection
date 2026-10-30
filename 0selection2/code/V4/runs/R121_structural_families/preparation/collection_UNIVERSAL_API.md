# Universal Collection V2 Interface For Resource Peer

Owner: Sol owns only `r121_universal_collection.py` and its tests/metadata.
Resource peer `01a188ed-435f-70f1-952b-842d33ccdfc4` owns parallel_v2.
No original collector, structural source, inputV1 source or live receipts are edited.

## Fixed Python API

```python
from code.V4 import r121_universal_collection as universal

result = universal.bootstrap_lane(root, lane, release, binding)
```

`root`, `release`, `binding` accept Path/string; lane is integer0..3.
Invoke once UNDER MAIN's physical GPU mutex, after that lane's old producers have
finished/failed/drained. Inherit `RETURN18_GPU_MUTEX_FD`. The function requires the
actual original host/UUID/CPU binding (`cuda:0`) and performs no solver/GPU call.
It checks local old producers read-only and refuses active assigned producers.

Return dict fields: `status` (`bootstrap_complete` or `already_bootstrapped`),
`lane`, `assigned_deployments` (list), `adopted_labels` (total),
`deployments` (dict by key with `source_root`, `adopted_labels`, `next_index`,
`preflight_sha256`), and universal/collection identity fields.

After bootstrap, ordinary `base.load(root)`, `base.task_groups(manifest)` and
`base.progress(root,key)` remain compatible with the frozen single-root schema.
Do not run an old worker on the new root. New child module is
`code.V4.r121_universal_collection`; its `worker` subcommand accepts the original
literal `--root --deployment --phase --max-jobs --device --release --binding
--result` arguments. No new native qualification runs are required: bootstrap
copies the successful qualification bytes and records source provenance.

## Root And Source Policy

New root:
`/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2`

CREATE references immutable protocols/source identities only; it never freezes
live label counts/progress. At lane bootstrap, exactly the five target keys in
inputV1's immutable recovery manifest use `collection_input_boundary_v1`; all
other84 keys use `collection_persistent_v1`. Never select a min cost, longest
prefix, or alternate root dynamically. Completed valid contiguous receipts are
adopted byte-identically, with per-task source SHA/provenance. Failure flags are
not imported. The ICAM_ATSP original prefix1 is retained, so its next task is z00.

The parent native protocol/hash, all89 deployments and113920 labels are unchanged.
All13 TSP/ATSP deployments use the same proven explicit CPU tensor constructor
on new calls, including the second call after lazy backend initialization.
Other tasks retain the exact `session.solve` path. Native/backend/model/seed/
budget/cost/winner definitions are unchanged. Adopted receipts are historical
successful solves, never claimed as fresh universal-v2 executions.

CLI forms (all support a literal `--root`):

```text
python -B -m code.V4.r121_universal_collection create --root NEW_ROOT
python -B -m code.V4.r121_universal_collection bootstrap-lane --root NEW_ROOT --lane LANE --release RELEASE --binding ORIGINAL_BINDING
python -B -u -m code.V4.r121_universal_collection worker --root NEW_ROOT --deployment KEY --phase label --max-jobs FINITE --device cuda:0 --release RELEASE --binding ORIGINAL_BINDING --result FRESH_RESULT_UNDER_NEW_ROOT
python -B -m code.V4.r121_universal_collection status --root NEW_ROOT
python -B -m code.V4.r121_universal_collection export --root NEW_ROOT
```

These are argument forms, not permission for this sidecar to launch anything.
Every executable shell stage must first source conda.sh and activate easynco.
## READY And Frozen Handoff

- Module SHA256: `92c3bc0ec8afeab1220e653106f786b746a0a1ba96fdb8a3996228f3a5e7e296`.
- Universal root seal: `24118bed2e9b4876ef504040631bbb62d31d79551aceea91e4320d3954534b11`.
- CPU CREATE succeeded; `prefixes_snapshotted=false`, no lane bootstrap/solver run
  by this sidecar. Root metadata pins only immutable source/protocol identities.
- Eight CPU tests PASS in0.156s, including all13 real tensor constructors after a
  lazy first-call default-device change, actual native audit, byte-identical
  source-prefix/qualification adoption, ICAM prefix1 continuation, original
  failure preservation, unchanged other-task session.solve and literal CLI/API.
- Final test log: `collection_UNIVERSAL_cpu_tests_final.log`, SHA256
  `fc8b420465dabf14589f85564b04690f829b0cdea36aa3ac4c6f15e310dc9903`.
- New worker patches only shared `base.solve_native`. A local compiled copy of
  the unchanged frozen worker control flow adds only `traceback.format_exc()` to
  its existing failure record, including errors raised during model loading.
  It does not patch the shared worker/session/backend or change native budgets.
- Source error flags are not copied. A preserved TSP/ATSP CUDA-to-NumPy failure
  is covered by this input correction; any unrelated original first error is
  reported explicitly, never treated as a generic retry target.
- Saved contiguous valid receipts just beyond an atomic progress update are
  validated/adopted and reported as `adopted_saved_uncommitted_tail`, never solved
  again. Holes or invalid identities fail before worker launch.
- Once bootstrap succeeds, its per-lane marker seals that actual prefix snapshot;
  subsequent calls return `already_bootstrapped`, not a new source-root selection.

CPU CREATE was executed with this exact command, without GPU visibility:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 46 python -B -m code.V4.r121_universal_collection create --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2
```

MAIN lane1 bootstrap argument form, INSIDE the resource transport's original
physical binding and inherited mutex (not a standalone permission to launch):

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && python -B -u -m code.V4.r121_universal_collection bootstrap-lane \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2 \
  --lane 1 \
  --release /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_r120_release.json \
  --binding /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_binding_lane1.json
```

Worker module: `code.V4.r121_universal_collection worker`; after bootstrap it
accepts the original frozen worker's exact flags and maintains one persistent
native session per finite problem/method shard. Any accidental preflight CLI
reuses the adopted successful qualification without a new solver invocation.
Per-task adoption JSON records are under `adoption/preflight` and `adoption/labels`.
Copied qualification/label receipt bytes are unchanged, not falsely labeled as
new universal executions. New receipts have `universal_version`,
`universal_sha256`, `universal_execution` and `native_input_placement` fields.

EXPORT only from this NEW root after all89 shards actually complete. Output is
`collection_explicit_cpu_v2/family_labels.jsonl` plus receipt. Existing endpoint,
native full-pool targets and collection hash stay compatible; additional aligned
`label_provenance` records and aggregate counts distinguish adopted original,
adopted inputV1, new explicit-CPU and new unchanged-session solves.

Learner files were NOT edited during this critical task. Their current default
export still names inputV1; when using the complete universal export supply
`--family-labels /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2/family_labels.jsonl`
until the separately coordinated learner handoff is updated. The frozen input
manifest and source-record isolation definition themselves are unchanged.

No source edits after this READY handoff without coordination with MAIN/peer.
