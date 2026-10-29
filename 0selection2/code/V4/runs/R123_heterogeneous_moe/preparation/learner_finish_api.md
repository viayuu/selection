# R123 Post-Lock Finish

This is a NEW postprocessing implementation. R123 model, experiment, meta-data
and shared prediction helper remain frozen while formal training is running.
No GPU, solver, training or Git command is executed by its implementation worker.

## Fixed Meanings Before Implementation

- `--root` is the R123 bank root; the unique run is `ROOT/model`. All new
  outputs go to `ROOT/finish`, never into the frozen preparation or runner files.
- Every operational stage requires completed 7,200 updates, the unique VAL lock,
  unchanged best-checkpoint/protocol hashes, and completed selected VAL and TEST
  captures. There is no checkpoint/temperature/threshold selection here.
- TRAIN in these diagnostics means the certified 10,886 source-record-unseen
  metaTRAIN rows per task, NOT all 81,000 current TRAIN rows. Source-record
  identity does not certify exhaustive base/content novelty. Historical
  holdouts were development-exposed. The shared-unseen intersection is
  generation-cohort-conditioned, not asserted to be an IID sample of TRAIN81k.
- References use only that metaTRAIN's cached native correctness: one globally
  fixed expert (equal task weight) and a task-conditioned correctness-prior
  router (constant expert per task). Ties use the fixed bank order. The latter
  is not an instance-aware learner or an answer-dependent oracle.
- Native error is `bank cannot emit saved native winner` plus `bank can emit
  it but router misses it`. Quality regret is separately `bank action-support
  floor` plus `selected action excess over that floor`. Exact-cost ties may
  give native error and zero quality regret; these quantities are not added.
- All solver rows, including zero counts, are retained. Six zero/near-zero
  lists cover TRAIN/VAL/TEST; near-zero is disjoint from strict zero and means
  `(0, 0.1%]` selected with positive native-winner support. Missed cost is
  assigned once by the native winner, never charged again to a false-pick row.
- Fixed36 is the existing immutable R84 VAL-defined list, not reselected here.
- TRAIN capture uses cached chosen-expert scores solely for descriptive
  statistics. This cost is NOT a deployment timing. Timing and LIB call the
  locked live router without expert-score inputs; hooks require exactly one
  complete frozen-expert execution per query. Raw-input/geometry preparation,
  host transfer, warmup and all-bank cache construction are not silently free
  forward work: their inclusion/exclusion is explicit in the receipt.
- The runner's VAL/TEST NPZs lack an embedded checkpoint hash. Finish records
  this boundary: their attribution relies on the unchanged locked run, selected
  update, pinned producer code, current row/source identities and exact cached
  selected-expert scores. A later receipt is not a retroactive capture signature.
- LIB uses the original native pools/FP64 objectives and the same locked complete
  router. Existing NSS predictions are reused only after instance/pool/cost and
  checkpoint checks. Missing NSS is explicit; no NSS training or inference.

## Parent Commands (After TEST Completion Only)

All commands first source conda and activate `easynco`. Replace the device only
after the parent releases that GPU. The worker does not execute these commands.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && \
python -m code.V4.r123_finish describe --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe

source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && \
python -m code.V4.r123_finish finish --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --bank-root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --input-cache-root /tmp --device cuda:0 --predict-batch 128 --lib-batch 1
```

Restartable individual stages are `capture-train`, `lib`, `timing`, `report`.
`report` is CPU-only and requires the other stages' receipts. Its main output
is `finish/reports/R123_MODEL_REPORT.pdf`; machine-readable CSVs and a source /
input/checkpoint receipt accompany it. `describe` only describes paths/rules;
it does not open TEST results or initialize CUDA.

## Handoff Status

- New finish module is ready for the parent to execute after locked TEST.
- Eight synthetic CPU checks passed with standard-library unittest. They cover
  native ties, separate native/quality budgets, mean-ratio versus per-instance
  regret, ID joins/collisions, fixed solver-ID ties, TRAIN-only references,
  full18 equal-task aggregation, zero rows, supervision isolation, sparse row
  dispatch, no-TEST `describe`, and an explicitly marked fixture PDF.
- The one-page fixture was also rendered from the PDF, not just from the
  Matplotlib figure; CJK text is visible and the font is embedded. The installed
  older plotting stack emits Pyparsing deprecation and Poppler CFF/font-type
  warnings; no missing glyph was observed in the rendered fixture.
- No real model, LIB, natural TEST timing, GPU, training, solver or Git command
  was run by this worker. Actual reports do not exist yet; fixture artifacts
  under `learner_finish_cpu_fixture/` are explicitly NOT trained-model results.
- New finish source SHA256:
  `3f0cd5c4fa714b176d4c3e9be20e0a9c326ebe30db2f6a44e51f940d5a42d995`.
- New minimal test SHA256:
  `507ac3d651eac2249a4267ccf0b05287927645ffb143bcb1f04d0d4e876610b3`.
- Frozen sources remain unchanged, checked against the pre-freeze handoff:
  model `43a3ce251b2ff55cc8d427e40823bf69642c28e3d7a421fd22966efc70f0eecf`;
  experiment `a5e334ab047c696ac6187fee7085c35089709f9e39cc23a482b249d87844c4fb`;
  meta `86f73d45edcb5dde6655b855a9e3aa1771c686eb9d57508e18528f3a804054cb`;
  shared cache `35ac20b813e1281deac0b11429a1eca750a21256482d3e777a02022b8037b32b`.
