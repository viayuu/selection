# R105 Model-Only Handoff

Status: model and bounded CPU tests complete, 2026-10-23.
Contract: main-owned `PLAN.md`. No runner, historical source, data/cache or
checkpoint file is modified/read by this implementation. No jobs are started.

## Runner API

```python
from code.V4.r105_model import ARMS, SPEC, initialize_shared, make_model

# Both records are already in memory, from the existing R84 preparation.
model, params, receipt = initialize_shared(initial, geometry, ARMS[0])
restored = make_model(params, ARMS[0], state_dict=trained_state)
```

`ARMS = ('A_static_gate', 'B_input_gate')`, `ROLE_ARM = 'B_execution_anchor'`,
`FRESH_SEED = 2`, `EXPERT_INIT_SEED = 105002`. The tuple initializer accepts
`device='cpu'` by default. Factories preserve caller CPU RNG; no file loader or
CUDA initialization is used. `model.params` adds `expert_arm` and `expert_spec`;
all original dimensions and original return keys remain unchanged. Use this
initializer, not R84's all-parameters-must-already-match initializer.

`make_model` strictly restores all supplied parameters/buffers, including trained
routers and geometry. It does NOT require trained weights to equal fresh weights.
`model.initialization_receipt` distinguishes restoration from initialization:
`fresh_initialization=False`, `state_dict_restored=True`, actual hashes in
`trainable_sha256`/`state_sha256`, and explicit fresh-comparison booleans.
`fresh_*` hashes and transfer details describe only pre-restore construction.
The factory cannot attest where a caller's restored state was trained.

## Reporter API

```python
model.eval()  # Do not change the locked checkpoint or inference rule.
routing, outputs = model.capture_routing(input_batch)
# routing: detached FP32 tensor [B, 4, 4], on model device.
# outputs: original dict(logits, pair_margin, solver_mask, utility).
```

Axes are instance, instance-encoder layer index 0..3, expert index 0..3.
Each row is the mean of the actual softmax probabilities over valid physical
nodes in that instance. Exclude the CLS/condition/statistics prefix (3 tokens)
and padding; include a valid depot/customer exactly as the original node mask
does. No labels, winners or costs are required or read. The reporter may move
the detached result to CPU explicitly. Each layer's `ffn.last_routing` also
retains its detached `[B,4]` snapshot until the next capture.

Capture executes one ordinary forward, preserves train/eval mode and RNG
consumption, and disables its capture flags in `finally`, including on errors.
Normal forward does not install hooks, compute diagnostics, synchronize or
update the last capture. Use `capture_routing`, not persistent training hooks.
In train mode, compare outputs only with the same starting RNG because dropout
is unchanged. Capture is no-grad and not reentrant; no routing auxiliary output,
loss, extra LayerNorm, balancing term or top-k mechanism is introduced.

Main owns sampling/aggregation: after model lock, first128 VAL instances per
task, disclose coverage, group variation by exact problem/size. Entropy of this
graph-mean mixture is not mean node-level entropy and must be named accordingly.
This small input-only diagnostic cannot establish efficacy or exclude shortcuts.

## Initialization And Counts

Each of the four instance `EncoderLayer.ffn` modules contains four distinct
original `FeedForward(128,512,.1)` modules and `Linear(128,4)`. All four experts
are computed. Router logits, softmax and weighted accumulation are FP32, with
the mixture cast back to the expert-output dtype. The original surrounding
residual/normalization, joint layers, methodID+34 branch and score MLP/maximin
are not replaced or extended.

A's router consumes `zeros_like(x)` and learns a bias-softmax, NOT a fixed
uniform gate. B consumes the current post-attention representation. All router
weights and biases start at zero in both arms. Original tensors map exactly
from R84 fresh seed2; old FFNs map to expert0. Experts1..3 use isolated seed105002.
Both complete initial states/functions match, including the same dropout RNG.
The fresh trainable SHA256 is identical for both arms:
`99cac97d7618e782a3c1a2f14425c8d0efe1a214b619320645371080a72a212d`.
With the factory's default zero/one geometry buffers, both full-state hashes are
`4f85dade8ca7f7698a47c7f769bcac91c00fa04397bada83ec4c7f7b11f900ed`.
Installing caller geometry changes the full-state hash, not the trainable hash.

The exact original fresh trainable hash, verified against R84 preparation, is
`dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93`.
`initialize_shared` checks both record integrity and equality to this regenerated
factory hash; a modified donor relabeled scratch is rejected. Existing TRAIN
geometry moments are installed as buffers, never fitted here.

| Quantity | Parameters |
| --- | ---: |
| Original full18 R84/R88B | 1,842,953 |
| One FFN: 128x512 + 512 + 512x128 + 128 | 131,712 |
| Added experts: 4 layers x 3 x 131,712 | 1,580,544 |
| Routers: 4 x (128x4 + 4) | 2,064 |
| Total, each arm | 3,425,561 |
| Original same-name tensors retained | 1,316,105 |
| Original FFNs mapped to expert0 | 526,848 |
| A intentionally inactive router feature W | 2,048 |
| A learned router biases | 16 |

Receipts list inactive parameter names/counts explicitly. Equal total parameter
count is NOT equal active functional freedom; B adds 2,048 input-sensitive gate
parameters, not another expert bank. The inactive count is the intentional gate
control difference, not an audit of every input/task-specific inactive parameter.

## CPU Verification

Executed in `easynco`, with GPU visibility and bytecode writes disabled:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r105_model -v
```

Result: **14/14 passed**, 6.489 seconds. Tests use the real width128/four-layer
model and tiny in-memory input tensors, not a reduced-width substitute.

- Actual parameter arithmetic, dimensions, fixed zero routers and inactive-W disclosure.
- Exact fresh R84 reuse and expert0 mapping, isolated seed105002, distinct experts,
  no parameter aliasing, and identical complete A/B initial states.
- All18 task forwards match bit-for-bit between arms from the same dropout RNG;
  original role processing, return keys, utility differences and maximin stay intact.
- Finite backward through every expert and original stages on TSP/ATSP/OVRPBLTW;
  A router W has exactly zero gradient, B W and both biases have finite nonzero gradients.
- Nonuniform learned bias-softmax for A, state-dependent learned gates for B,
  and explicit computation/mixing of all four experts in both arms.
- Capture valid-node means, original output/RNG parity, no-grad snapshots, no
  persistent hooks, normal-forward disabled capture, and failure cleanup.
- Native candidate permutations/shared and per-row masks, candidate/node padding,
  compact-pool equivalence, role-preserving node relabeling, singletons and ID ties.
- Poisoned label/winner/cost/context fields do not affect forward/capture; no input mutation.
- Strict fresh and modified trained-state restoration, including geometry buffers,
  without false fresh provenance; missing/extra state keys and cross-arm metadata fail.
- CPU BF16 autocast with FP32 routing, finite mixture/backward, contract/geometry
  rejection, donor relabeled scratch rejection, caller RNG isolation, and successful
  construction with all file reads/checkpoint loads/CUDA initialization patched to fail.

Backward checks use an input-only scalar of model outputs, not generated solver
costs or a new training objective. No dataset scan, solver run, training job,
GPU operation, persisted data/cost generation or commit occurred. No model quality,
equal active freedom, sparse compute saving or experiment result is claimed.
