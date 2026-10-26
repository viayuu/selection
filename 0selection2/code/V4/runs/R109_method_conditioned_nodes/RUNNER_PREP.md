# R109 Runner Preparation

Implementation scope: `r109_experiment.py`, `test_r109_experiment.py`, and this
note only. Base commit: `89ed70dca84d97d5bade438629fcac8970e40c54`.
No historical files, launch, GPU work, or commit belongs to this worker.

## Public Interface

- Root: `code/V4/runs/R109_method_conditioned_nodes`.
- Arms: `A_tied_late`, `B_private_late`.
- One public `CAPS` configuration: `large_micro_cap=128`,
  `other_micro_cap=1024`, `eval_batch_size=128`. Main confirms or changes this
  single configuration after its kernel witness, before locking or training.
  CLI defaults, protocol validation, evaluation, and timing derive from it.
  There is no automatic OOM cap shrink.
- Canonical minimal science record: `required_lock_fields()`. Main adds
  `data_root`, `prepared_root`, and `primary_cache` to create `protocol_lock.json`.
  Main owns protocol locking, readiness, launch, LIB, reporting, and publication.
- Model interface: `models.initialize_shared(initial, geometry, arm, device)`
  returns `(model, params, receipt)`; `models.make_model(params, arm, state_dict)`
  strictly restores a checkpoint. Counts come from `models.P_COUNTS` (or its
  `PARAMETER_COUNTS` alias); metadata comes from `models.SPEC`.
  Model params bind `r109_arm` and `r109_spec`; branch seed is the model's
  `BRANCH_INIT_SEED=109002`. Verified exported counts are original 1842953,
  A 2305033, and B 10220553. Only `common_parameter_sha256` is shared across
  arms; complete hashes remain per-arm evidence with different dimensions.
- R107-compatible runner interface: `private_engine`, `evaluate`, `train`,
  `test`, `authorize_test`, `ensure_test_cache`, `load_checkpoint`, `load_model`,
  `from_checkpoint`, `selected_capture_proof`, `bind_selected_val`,
  `raw_inputs`, `cached_input_batch`, `measure_inputs`, `timing`, and `parser`.
- CLI commands: `train`, `test`, `timing`, `describe`, `status` with the same
  options as R107; fixed caps are read from `CAPS`.
- Timing metadata uses `method_conditioned_node_refinement_included=true`.
  Existing timing validation is delegated to `r109_reporting.inference_timing`
  once main provides that module, as in R107.

## Scientific Boundary

Original full18 R88B inputs, original four shared node layers, two additional
method-conditioned graph refinement layers, two original joint streams, and
original score MLP/maximin. A ties late weights across global method IDs; B has
an independent full graph-layer bank indexed by those global IDs. Both clone
original node layers 2/3 into their late prototype(s). Equal initial evaluation
function is a model contract, not equal total counts or equal complete state
hashes. Runner never constructs a second full model just to compare full hashes.
Each own graph summary occupies the first third of the original score features
and is also added to its method token before the original two joint layers.

The private R84 update engine and original stateless R77 CE/comparison/risk plus
two-forward symmetric RDrop are retained. Logical batch 1024, seed 2, budget
57600, warmup 4320, VAL every 1440, full TRAIN every 7200 plus selected capture,
AdamW 1e-4 to 1e-6, WD 1e-4, clip 1, dropout .1. Unique checkpoint: strict
minimum whole18 VAL macro_vs_sbs_pct, first exact tie. No donor, route targets,
new targets, additional input features, quotas, or post-TEST reselection.

## Decisions Declared Before Implementation

| Decision | Choice | Source |
| --- | --- | --- |
| Caps pending final kernel witness | One editable `CAPS` dict, both arms fixed at 128/1024/eval128 per main | User |
| Parameter-count source | Model exports only, no guessed counts | User |
| Complete cross-arm initialization hash | Not compared; not equal by design | User |
| Input-only timing batch | Use the locked evaluation cap, no supervision reads | Runner default |
| Capture and retry semantics | Reuse private R84 and R107 proven hooks | User |

## Verification

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r109_experiment -v`:
13/13 passed after the 128/1024/eval128 change.

Combined historical/runner regression command, with the same CPU-only environment:
`python -B -m unittest code.V4.test_r107_experiment
code.V4.test_r84_experiment code.V4.test_r109_experiment -q`:
22/22 passed. Historical R107, R84 and R76 engine sources have no Git diff.

Contracts include actual CPU initialization of both model arms, the exported
unequal parameter counts and shared original fresh tensors, a tiny mocked
checkpoint factory round trip, unchanged two-forward RDrop/loss, row-weighted
partial microbatches, one accepted optimizer step, nonfinite retry RNG replay,
complete-budget TEST authorization, preexisting full18 TEST-cache enforcement,
exact selected TRAIN/VAL capture proof, immutable archived VAL identity,
input-only cache reads, complete-model timing calls, and timing metadata with
all GPU work mocked. These are implementation contracts, not performance or
final result evidence. Main owns candidate-pool permutation and model-level
input-isolation coverage.

The only configuration edit requested after implementation was main's increase
from 64/256/eval64 to 128/1024/eval128. Logical batch remains 1024 and runtime
caps cannot shrink automatically. The final larger-cap GPU witness remains
main's responsibility. No protocol lock, main-ready receipt, training, TEST,
LIB, report, task launch, GPU work, or commit was performed by this worker.
