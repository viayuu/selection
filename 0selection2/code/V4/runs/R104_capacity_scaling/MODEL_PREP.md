# R104 Capacity Scaling: Model Preparation Only

Status: model preparation complete and frozen for handoff (2026-10-23).
This is not an experiment protocol or a launch.
Only `r104_model.py`, `test_r104_model.py`, and this note belong to this change.
R99/R101/R102 remain independent, unchanged directions. The requested width and
depth arms are available for later parallel experiments; no jobs are started.

## Decisions Before Implementation

| Item | Decision | Basis |
| --- | --- | --- |
| Parent | Full18 R88B `B_execution_anchor`, original R77 score MLP/maximin | Requested preservation; inspected `r88_model.py` and `pairwise_selector.py` |
| Original dimensions | d128, instance4/joint2, heads4, qkv32, FF512, score hidden256, method-feature hidden128 | R84 `prepared/prepared.json`, `DualStreamSelector`, `InstanceEncoder` |
| A | `A_width256`: d256, instance4/joint2, heads4, qkv64, FF1024, score hidden512, method-feature hidden256 | Width axis; retain head count and original width ratios |
| B | `B_depth8`: d128, instance8/joint4, heads4, qkv32, FF512, score hidden256, method-feature hidden128 | Depth axis; original widths unchanged |
| Fixed geometry branch | Input21 -> hidden64 -> d, original zero-initialized output projection | Inspected `GeometryResidual`; do not scale its fixed64 hidden layer |
| Capacity overrides | embedding_dim, encoder_layer_num, joint_layer_num, head_num, qkv_dim, ff_hidden_dim, head_hidden_dim, solver_feature_hidden only | Raw fresh-R84 params still describe d128/instance4; do not silently retain those dimensions |
| Fresh source | Original architecture regenerated on CPU with seed2; optionally verify/copy the in-memory R84 fresh record | No trained checkpoint or donor model |
| Added/resized parameters | Independent CPU RNG seed104002, then overwrite every same-name/same-shape parameter with fresh R84 values | Preserve common initialization without padding/cropping/tiling width-changed tensors |
| Restore API | Strict caller-supplied in-memory state restoration; no file loader | Required `make_model(params, arm, state_dict=None)` interface |
| Claim boundary | Two capacity axes against the historical original, not a parameter-matched A/B causal comparison | Requested; main owns the final protocol |

## Unchanged Semantics

Raw coordinate/node attributes, ATSP matrix processing, local geometry and
distance bias/normalization stay in the original implementations. R88B marks
the TSP execution customer in existing slot3, with its original validation and
no new geometry/cache fields. Other problems' inputs are unchanged.

Method representation remains global method ID embedding plus its fixed
34-dimensional handcrafted metadata MLP and LayerNorm. Joint layers retain
synchronous node-to-method and method-to-node updates. Final mean/max pooling,
method-memory attention, the complete score MLP, FP32 utility differences,
legal-candidate maximin, and stable global-ID ranking semantics are unchanged.

No loss implementation is added or modified. The existing `RDropObjective`
continues to own CE, comparison/focus, relative risk, and two-dropout symmetric
KL semantics. Labels/costs remain supervision, never forward features. There
is no donor/context bank, retrieved label, query winner, or cost input path.

## Deferred To Main

Budget, LR, optimizer/schedule choices, gates, validation binding/checkpoint
selection, TEST/LIB evaluation, runner/launcher, GPU allocation, and actual
parallel execution are deliberately absent. No administrative handoff issue
is inspected. No GPU/job operations, cache copying, solver/API calls, or Git
commands are part of this preparation.

## Verification

Counts below are actual sums of instantiated parameter `numel()`, including
all original registered trainables, not estimates from nominal settings.

| Setting | Original full18 R88B/R77 | A_width256 | B_depth8 |
| --- | ---: | ---: | ---: |
| Embedding width | 128 | 256 | 128 |
| Instance / synchronous joint layers | 4 / 2 | 4 / 2 | 8 / 4 |
| Heads / per-head QKV / attention projection | 4 / 32 / 128 | 4 / 64 / 256 | 4 / 32 / 128 |
| FF hidden, instance and both joint directions | 512 | 1024 | 512 |
| Original score MLP, input -> hidden -> output | 384 -> 256 -> 1 | 768 -> 512 -> 1 | 384 -> 256 -> 1 |
| Method metadata MLP | 34 -> 128 -> 128 | 34 -> 256 -> 256 | 34 -> 128 -> 128 |
| Geometry projection | 21 -> 64 -> 128 | 21 -> 64 -> 256 | 21 -> 64 -> 128 |
| Attention modules, including final method-memory decoder | 9 | 9 | 17 |
| Global method-ID embedding rows | 22 | 22 | 22 |
| Actual parameters | 1,842,953 | 7,288,969 | 3,426,057 |
| Same-name/same-shape fresh parameters reused | - | 1,417 (5 tensors) | 1,842,953 (148 tensors) |
| Independently initialized capacity parameters | - | 7,287,552 (143 resized tensors) | 1,583,104 (104 new tensors) |

A has the original tensor names but most shapes change: it is not an identical
R84 initialization. Its five compatible tensors are the two four-head distance
bias scales, the first geometry-projection weight/bias, and the scalar score
output bias. B preserves every original trainable and independently initializes
the added instance/joint layers; no layers share parameters or repeat a donor.

Read-only CPU inspection of R84's `prepared/prepared.json` confirmed its exact
dimensions above. Regenerating the original factory at seed2 matched its saved
trainable SHA256 exactly:
`dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93`.
No `.pt` file, trained checkpoint, input dataset, or cache was loaded/copied in
this preparation. The in-memory initializer additionally checks the record's
hash against this reproducible original factory, not just a `scratch` label.

## API

Exports: `ARMS`, `SPEC`, `make_model(params, arm, state_dict=None)`,
`initialize_shared(initial, geometry, arm, device='cpu')`, and
`model_dimensions(model)`. Both arms use unmodified `RoleInputSelector` modules
and forward methods, not a reduced or replacement scoring head.

```python
from code.V4.r104_model import ARMS, SPEC, initialize_shared, make_model

# initial and geometry are caller-supplied R84 records already in memory.
model, actual_params, receipt = initialize_shared(initial, geometry, ARMS[0])
restored = make_model(actual_params, ARMS[0], state_dict=model.state_dict())
```

`make_model` also accepts raw original R84 `model_params`; explicit capacity
overrides take precedence over their historical width128/instance4 settings.
Non-capacity settings are preserved, with arm/spec/role metadata added.
`model.params` contains actual dimensions; `model.capacity_report` records
incoming overrides, actual/original module dimensions and parameter counts.
`model.initialization_receipt` records hashes, copied/new/resized names, exact
shape changes, counts, and isolated seed104002. CPU construction preserves the
caller's RNG. No external state is read by the factory.

The bare factory leaves geometry moments at the original zero/one defaults;
use `initialize_shared` to install the existing TRAIN-only moments. The caller
must bind their provenance; this helper does not fit statistics. Unlike R84's
original initializer, the R104 helper intentionally permits new/resized
parameters. Do not pass R104 directly to R84's all-shapes-must-match initializer.

In-memory `state_dict` restoration is strict, including geometry/metadata
buffers. Restoration provenance is not assumed: the receipt distinguishes it
from fresh initialization and records whether trainables still match fresh.
There is no checkpoint-path loader. Only fresh states were restored in tests.

## CPU Tests

Executed in `easynco`, with GPUs hidden and bytecode writes disabled:

```bash
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r104_model -v
```

Result: **9/9 passed**, 6.166 seconds. The tests exercise actual main-arm
dimensions on tiny synthetic CPU batches, not reduced-width stand-ins.

- Exact dimensions/counts and preservation of non-capacity input parameters.
- All18 original R88B forward outputs and dropout RNG agree bit-for-bit at the
  same state; the complete score MLP, utility differences and maximin remain.
- Finite original RDropObjective loss/backward on TSP, ATSP and OVRPBLTW;
  every instance layer, both directions of every joint layer, methodID+34,
  pooling, decoder and score MLP receive finite nonzero gradients.
- Native shared/per-row candidate IDs, order equivariance, masked padding,
  compact-pool equivalence, singleton pools, invalid masks and stable ID ties.
- Labels/costs and poisoned donor/context/current-query-label fields do not
  affect predictions; no input mutation or donor/context parameter branch.
- Fresh same-name/same-shape reuse, isolated deterministic capacity weights,
  no parameter aliasing, exact geometry buffer installation and RNG isolation.
- Strict fresh-state roundtrip and rejection of incomplete/cross-arm states.
- Rejection of altered roles/heads/specs, nonfresh seed/records, a synthetic
  modified donor relabeled scratch, and invalid geometry moments.
- Factory succeeds with checkpoint loading and CUDA initialization patched to
  raise; all created parameters stay on CPU.

These checks establish local implementation behavior, not training quality,
scaling benefit, equal compute, or a capacity-versus-initialization causal
claim. No accuracy or experiment outcome is claimed. Owned work stops here.
