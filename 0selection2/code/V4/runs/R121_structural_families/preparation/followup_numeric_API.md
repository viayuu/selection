# Numeric Followup V2

Owner: only new `r121_followup_numeric.py`, its test and this API document.
No frozen source, old controller/contract, learner, finisher, model or label edit.
No production create/launch, GPU, solver, training or Git by this worker.

## Exact Adaptation

The wrapper verifies old `r121_followup.py` SHA256
`0da941d9ab693eb91fcd17be182d6ce5cf277c3129ed9061ab0fc0c106af32e2`,
compiles it in a PRIVATE namespace, and guards exactly one occurrence each of:

- `universal.json` -> `numeric.json` in the new collection's immutable inputs.
- Export module `code.V4.r121_universal_collection` -> `code.V4.r121_numeric_collection`.
- Contract schema -> `r121-principal-followup-numeric-v2`.

Only private globals/hooks change. The old file and imported canonical modules
are not patched. `ROOT=followup_principal_v2`,
`COLLECTION=collection_numeric_boundary_v3`; `LEARNER=learner_v1` is unchanged.
The resource plan is STILL `main_approved_r121_principal_v1.json`, first NN
attempt STILL `principal_v1`. Existing plan/last.pt/best.pt/training_complete
blocks create/run/protocol-freeze; no resume or restart is introduced.

Source pins retain every original hash from the old contract, including the old
observer hash, and add exactly the new numeric API, v3 collector wrapper and new
followup source. The publisher is deliberately NOT pinned: it was not in the old
source tuple and Curie owns its active path extension. Numeric API/wrapper pins:

```text
r121_numeric_collection.py
a9c638f48cbb8164171ef95bef77ae3b159a254968eddc78a71676b7dc3df50e
r121_parallel_collection_v3.py
6c8632e5fba4f45c736313773cd0540c0de9e3c75fabb7a92422d37feabd14b0
```

The new contract also pins the old contract, every old immutable anchor, the new
numeric protocol/numeric.json and main's external interruption receipt:
`followup_principal_v1/main_external_replacement_numeric_v3.json`.
Main alone confirmed/terminated PID1202384, the old childless CPU observer. This
wrapper never signals it or interrupts a NN/native worker.

## Barrier And Fixed Sources

Only numeric-root progress, all four bootstrap markers, all89 successful original
qualifications and all113920 labels release the barrier; the original23040
endpoints remain required at create and native export. No partial training.
Old Original/InputV1/V2 progress is informational only: no sum/double counting,
no old qualification substituted, and an invalid historical cursor cannot release
or block the numeric barrier.

Validate the numeric.json fixed source map against original group_order: the five
InputV1 keys first, otherwise lane1 V2, otherwise Original. No min-cost or
longest-prefix source selection. Main's four native collectors remain untouched.

The inherited pipeline is unchanged after export:
wait -> numeric export -> prepare-family -> freeze original learner protocol ->
principal train/test/lib/capture-train through pinned Return18 transport ->
saved-only CPU finish -> four-route analysis -> publisher. All family-label flags
explicitly name the NEW numeric root. Fixed14400 updates, logical1024, seed2,
micro/eval/LIB352/256/1, offline, one principal, no automatic retry/resume.

## Main-Only Ready CLI

The following commands are for MAIN; this implementation worker does NOT execute
them. Main must retain its external old-observer replacement record first.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
R=/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/followup_principal_v2
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r121_followup_numeric create --root "$R"
CUDA_VISIBLE_DEVICES='' python -B -u -m code.V4.r121_followup_numeric run --root "$R" --poll-seconds 60
```

Place the second command in main's tmux session, not in this sidecar. CPU-only
observer stages keep CUDA hidden; actual GPUs are assigned solely by the original
Return18 transport AFTER the complete full89 barrier.

Saved-only status:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r121_followup_numeric status \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/followup_principal_v2
```

## Three CPU Mocks

Tests intercept ALL production writes and children. They check private compilation,
all original plus new source pins, full-size89/113920/23040 contract counts, fixed
source policy, informational-only legacy cursors, unstarted-principal protection,
and inherited stage/CLI order with the same one principal_v1 plan. Tiny progress
fixtures are not production completion evidence.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B -m unittest code.V4.test_r121_followup_numeric -v
```
