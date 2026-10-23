# R106 Model Preparation

2026-10-23. CPU implementation witnesses only. Not launched, not protocol-locked,
not an efficacy result. No ready marker or lock record was created. Work is
limited to `r106_donor.py`, `r106_model.py`, `test_r106_model.py`, and this note.
No old Rxx source, upstream native source, dataset, process, task, or protocol
was edited; no GPU, Git command, allocation, or launch was used.

## Interface

```python
from code.V4 import r106_model as models

models.ARMS == ('A_frozen', 'B_finetune')
models.SPEC
model, params, receipt = models.initialize_shared(initial, geometry, arm, device='cpu')
model = models.make_model(params, arm, state_dict=complete_state_or_none)
counts = models.parameter_counts(model)
```

`initial` is the original fresh R84 seed2 payload, not a selector checkpoint.
Its parameter digest is checked against the actual frozen R84 digest, not just
against a self-declared hash. `geometry` supplies the frozen TRAIN-only mean/std.
The receipt hashes **all initial parameters and buffers** independently of
`requires_grad`, and separately hashes original parameters, projections, and
native state. Both arms load the same three original pinned R102-B donors.

The new schema is `r106_schema_version=1`, `r106_arm`, `native_donor` with
explicit `trainable`, and `model_spec`. Frozen R102/R94 configuration fields
are rejected rather than relabeled as trainable B metadata. The selector keeps
four instance layers, two joint layers, ID34, the original score MLP/utility
and maximin, nonaffine per-node fusion norms, and physical TSP node0 role.

Each donor Parameter is tagged `_r106_native=True`; tags are reapplied after
module device conversion. Runner grouping must use `p.requires_grad` plus this
tag. A has no trainable native group. B has all native parameters trainable
immediately, with the runner applying exactly `common_lr * 0.1` at every
update-axis rate, including warmup/cosine. There is no extra unfreeze stage or
optimizer/schedule implementation here.

## Online Execution

Every public selector/donor call first copies a raw-input whitelist. Labels,
costs, winners, arbitrary targets, and unrecognized fields do not reach the
selector or donor. External `nco_*` fields and externally supplied internal
feature fields are rejected by the selector, not trusted. Raw-derived
`node_geom` is permitted; it is not a native representation cache.

All three branches execute one padded batch without an exact-N loop. There
is no ATSP exact-N fallback in the delivered implementation.

| Branch | Preserved native execution |
| --- | --- |
| TSP | ELG128 embedding, all original Q/K/V/combine/FFN/affine weights; SDPA key mask and valid-node biased-variance InstanceNorm with native eps |
| Routing | ReLD128 original depot/customer inputs and units, norm=none/norm_last; masked attention, original eval topk2 gating, SparseDispatcher, expert FFNs, and combine on valid nodes only |
| ATSP | MatNet original five bidirectional layers, zero FROM role and valid-order identity TO role, distance mixed-score weights, independent simultaneous FROM/TO updates, valid-node InstanceNorm; output FROM256 then TO256 |

All native networks and all their child modules remain eval even after
`selector.train()`. B still builds an autograd graph. No original frozen
forward, `cached_nodes`, native hidden detach, decoder, noise, or native
load-balancing objective is used. Native execution/output remains FP32 with
autocast/TF32 disabled until the normal selector fusion precision path.

Native eval did not consume CPU RNG in repeated real-weight forwards for all
three branches. The pinned definitions and adapted paths contain no active
random operation. **Six ReLD `w_noise` tensors (3,072 parameters) are naturally
unused in eval**: B sets them trainable, but they have no gradient because the
fixed no-noise graph never reads them. No fake zero-valued connection or
synthetic noise gradient is added. Nonzero final selection gradients were
witnessed in all three active native branches, including ReLD `w_gate`.
Not every native parameter is claimed to have a nonzero derivative on every
task/example; inactive branches and analytically invariant directions exist.

## Scoped R-Drop Graph

```python
from code.V4.r76_experiment import RDropObjective as OriginalRDropObjective

class R106Objective(OriginalRDropObjective):
    def forward(self, batch, consistency_weight):
        if batch is None:
            return super().forward(batch, consistency_weight)
        with self.selector.native_pair_context(models.input_only(batch)):
            return super().forward(batch, consistency_weight)
```

Use the **original R76 objective**, not the R94/R102 cached-node objective;
the latter pre-prepares private native fields. The thin wrapper above changes
no loss or reduction. Backward is on the combined returned loss, normally
after the scope exits. Pass original raw tensors to both selector forwards.
`prepare_inputs` returns a private internal batch, not a reusable public cache.

Context entry computes native features once (A no_grad, B differentiable).
It records whitelisted tensor identity/data pointer/shape/stride/dtype/device
and version, raw scalar/problem metadata, cloned mask/n witnesses, and all
model parameter/buffer identities and versions. Labels/costs are absent from
this signature. Each of at most two preparations checks the signature, grad
mode, mask/n values, and model state. Wrong batches, mutations, parameter
updates, nested scopes, and a third preparation are rejected. `finally` clears
the only model-held graph reference on both normal exit and exceptions.
Never keep the scope across an optimizer step or microbatch. Context requires
versioned raw tensors; tensors created in inference mode are intentionally
rejected there. Ordinary unscoped forward is always online, including eval.

## Actual Counts And Initial Hashes

Counts below are sums over the real constructed Parameter objects, not
declared constants used as receipts. Initialization fails if they disagree
with the expected architecture.

| Parameters | A_frozen | B_finetune |
| --- | ---: | ---: |
| Original fresh selector | 1,842,953 | 1,842,953 |
| Three bias-free projections | 98,304 | 98,304 |
| Selector trainable | 1,941,257 | 1,941,257 |
| Native total | 10,023,840 | 10,023,840 |
| Native trainable | 0 | 10,023,840 |
| Total | 11,965,097 | 11,965,097 |
| Total trainable | 1,941,257 | 11,965,097 |

Native split: ELG 1,187,712; MatNet 5,273,760; ReLD 3,562,368.

Hashes use existing `r77_data.parameter_hash`: sorted names, shape/dtype, and
contiguous CPU tensor bytes. These are tensor-state hashes, not `.pt` hashes.

| Tensor set | SHA256 |
| --- | --- |
| Full initial state, identical A/B | `7381a063ecba3c898ed2ebabcb9d06a4bf30a70db6767ecf09cf230c165b0e41` |
| Native state | `bcb04a1ce2aca695c48c608fb2d1738112e92afcdec19b3e8e212cc5be19a233` |
| Selector parameters including projections | `79a823efa328b528cad136b7b3fbc878fd34e87fac1ae108a65a7aee23befe73` |
| Original R84 parameters | `dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93` |
| Three projections | `eadae4398372b6c1c4603979dbc697d7408aa62f292e27d66a85910a405d5312` |

Read-only R84 preparation: `code/V4/runs/R84_new90k_attention/prepared/`.
`identicalinit_seed2.pt` file SHA256:
`291266bf421075ed05cd663b1c4df207063081d4562278b2d9c0068ab0b51e9b`.
`geometry_stats.json` file SHA256:
`382aba5e74ae4f49ba7dc651b66425efbaaee6bfff64095594e14bc4c388f33b`.

Both arms use original checkpoints authenticated by existing native loaders:

| Component | Original checkpoint SHA256 | Native source identity SHA256 |
| --- | --- | --- |
| ELG | `94448cce3ba413388286449886eae05f27c03f94996ff45daea37ebdce88fcdf` | `46990e346f1b1d4015f2bd0a429f38fbdeb65c7c189fa108e73446022ae29ceb` |
| MatNet | `a86d1ab1d8db62088e6a9d54118db92ebe8472c316dc046fcf755ca61280b5ca` | `ac518d326ae5c1903e414eefc1438a261c8e45eb1940f13eedfd47ae8e21ff63` |
| ReLD | `2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3` | `d3dc79e687af2aff2a1d8499f651ca6a982bda3e0c7fa56cb7328895c4d6ab0e` |

Per-source file paths and digests, checkpoint paths, native input semantics,
and unknown historical pretraining overlap are embedded in donor provenance.
The weights' identity is verified; historical pretraining data overlap is
still unknown.

## CPU Witnesses

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest code.V4.test_r106_model -v
```

Result: **10 tests passed**, 8.667 seconds in the final complete run.
Fixtures have two rows with N=5/8, non-prefix masks and 11 padded slots;
full18 raw-layout checks use the same small fixtures, not real dataset rows.

| Real native exact-N vs masked FP32 CPU | Max abs | RMS |
| --- | ---: | ---: |
| TSP | 3.84449959e-6 | 7.43146643e-7 |
| CVRP | 2.02655792e-6 | 5.05039623e-7 |
| ATSP | 4.88501787e-3 | 2.84273963e-4 |
| OVRPBLTW | 1.49011612e-6 | 4.34475993e-7 |

Reference calls use the unmodified native encoder on compact exact-N rows,
never the old frozen donor forward. Padding is excluded from attention and
normalization, and outputs are scattered to original physical slots. Compact
repacking is also checked. This is **not bitwise equivalence** or a bound for
all sizes/devices. Small MatNet graphs show FP32 reduction sensitivity;
the initial common `3e-3` absolute assertion failed at `4.885e-3`. The final
MatNet-only comparison explicitly allows `atol=6e-3, rtol=3e-4`, with RMS
below `5e-4`; ELG/ReLD use `atol=rtol=1e-5`. No hidden fallback or changed
roles/weights was used to make this pass.

| Shared vs two independent online native graphs | Loss abs | All-parameter gradient relative L2 |
| --- | ---: | ---: |
| B / TSP | 0 | 1.64261777e-7 |
| B / CVRP | 0 | 8.80780944e-8 |
| B / ATSP | 0 | 2.59856652e-5 |
| A / CVRP | 0 | 0 |

These comparisons reset the same CPU RNG and retain selector dropout=0.1.
All gradient key sets and each parameter are compared. Per-element gradient
tolerance is `atol=3e-6, rtol=3e-3`, except MatNet common pre-normalization
biases/constant mixed-score bias, where analytically zero directions can
leave FP32 cancellation dust (`atol=3e-4`). An independent global relative
L2 assertion is `<1e-3`. Native call counts are 2 unscoped versus 1 scoped.

Other witnesses: identical complete A/B state and counts; no RNG consumption
in native eval; final original loss gradients reach B's three native branches
but not A; full18 label isolation/external-cache rejection; physical TSP
anchor preservation under relabeling; distinct ATSP roles and distance
dependence; zero projections recover the original selector; context cleanup
and stale-state rejection; complete strict state-only restoration.

## Checkpoint Restore And Remaining Main Witness

`models.load_checkpoint(path, device='cpu')` accepts the familiar envelope
`{'args': {'arm': arm, 'model_params': params}, 'model': complete_state}`.
The runner must validate its own protocol/config/lock before using this
portable model loader. Restore builds native class layouts without external
weight loads, checks saved source provenance, checks exact state keys/shapes/
dtypes, then loads the entire selector/native state strictly. Tests modify a
saved native weight and prove that modified value survives restore, while
`torch.load` is allowed to read only the requested model checkpoint. Missing
native state and dtype mismatches are rejected, not filled from pretraining.
External source definitions are still required; external weight files are not.

Still requires main's actual GPU startup witness under the main-owned protocol:
online padded donor+selector forward/backward with real raw batches and size
extremes; bounded input-only cap/memory evidence; B finite native gradients and
first-update native changes versus A frozen state; disjoint safe optimizer
groups with 0.1 native LR from update one; A/B complete initial-state hash;
GPU numerical/precision behavior and scoped-vs-independent R-Drop agreement;
checkpoint/resume without external native weight I/O; and actual end-to-end
online latency. No GPU memory fit, throughput, formal successful update,
scientific result, ready status, or protocol lock is claimed here.
