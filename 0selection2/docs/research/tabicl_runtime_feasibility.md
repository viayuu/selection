# TabICL Runtime Feasibility

2026-10-22. Bounded source/API feasibility only, not literature research or model-result analysis.

## Decision

Recommend **full 81,000-row TSP TRAIN context + 32 fixed VAL inputs, without VAL labels**, as a resource-only first witness. Keep all TRAIN rows and their historical labels; no sampling, class quotas, or context reduction. Use the user-measured TRAIN-constant mask (162 remaining TSP columns). Full context is source-supported and memory-plausible, **not certified to fit CPU4 / 9 GB RAM / 24 GB GPU without a measured witness**.

Scheduling is owned by the main session: no GPU is available now; the intended opportunity is the existing `4090_1` only after R94B's entire final pipeline finishes, not merely its training loop. This turn did not inspect or change jobs, allocations, R94/R95 files, raw data, packages, or Git. Only source archives under `/tmp/tabicl-feasibility-DY3C5T` and this document were written. No weights were downloaded; no fit, prediction, or GPU operation was run.

The user's explicit permission for historical TRAIN labels in forward context supersedes AGENTS.md's input-label prohibition for this experiment only. Query labels, winners, and costs remain absolutely excluded from forward inputs. Existing split/candidate/winner conventions otherwise remain unchanged.

## Version And Isolation

| Item | Verified pin / status |
| --- | --- |
| Official upstream | [soda-inria/tabicl](https://github.com/soda-inria/tabicl) |
| Recommended Python 3.9 release | `tabicl==2.0.2`, tag `v2.0.2`, commit `ba4bdbc6bb46b9307da3dbc1685109371ec5085f` ([source archive][archive], [metadata][metadata]) |
| Current upstream snapshot checked | `c91f00df184a5097e584cb376f7e699e9f3444c4`, package version 2.2.0; requires Python >=3.10, so not a drop-in for easynco ([current metadata][current-meta]) |
| Local metadata | Python 3.9.23; Torch 2.5.1+cu124; NumPy 1.24.4; SciPy 1.10.1; scikit-learn 1.6.1; huggingface-hub 1.8.0; psutil 7.2.2; tqdm 4.70.1 |
| Missing dependency | `einops`; pin `0.8.1` (declares Python >=3.8). TabICL itself is not installed. |

**CPU import verified**, with CUDA hidden and bytecode disabled, using unmodified `v2.0.2` source plus separate [einops v0.8.1 source][einops]. `TabICLClassifier`, its signatures, and Torch 2.5.1's `sdpa_kernel` / `SDPBackend.FLASH_ATTENTION` imported successfully. Constructor creation also passed. This is import compatibility, not CUDA/kernel or checkpoint-load certification. Torch >=2.2 is declared upstream; this build satisfies it. Keep the observed sklearn 1.6.1: the source uses `sklearn.utils.validation.validate_data`, so the declared >=1.3 minimum alone is not a sufficient compatibility test.

Preferred deployment: retain an immutable copy of the commit-pinned source outside the shared environment and prepend its `src` plus a private einops source directory to the pilot process's `PYTHONPATH`. No install is needed. Alternatively, a future owner may install **only** `einops==0.8.1` into a new private `--target` directory with `--no-deps --no-compile --no-cache-dir`; never install/upgrade Torch or run a general dependency resolver in easynco. The [PyPI 2.0.2 wheel metadata][pypi] gives SHA256 `00890709ca03b128250b8dac51d1e25180254942eb0f129fe5f8f6950a67125c` if a wheel is preferred later. No install was performed here.

## Public API For The Runner

This is a future runner fragment, not an executed experiment. The checkpoint must already exist at `checkpoint_path`. `X_train` is the complete task TRAIN matrix; `X_val32` contains only 32 input rows selected by stable split order, with the same TRAIN-fitted feature mask. Retain FP32 input arrays.

```python
import torch
from tabicl import TabICLClassifier
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(4)
torch.set_num_interop_threads(1)  # Once, at process startup.
mgr = dict(offload="gpu", safety_factor=0.1, min_batch_size=1,
           use_async=False, max_pinned_memory_mb=0.0)
clf = TabICLClassifier(
    n_estimators=1, batch_size=1, norm_methods=["none"],
    model_path=checkpoint_path, allow_auto_download=False,
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    device="cuda:0", use_amp=True, use_fa3=False,
    offload_mode="gpu", random_state=2, n_jobs=None, verbose=True,
    support_many_classes=False,
    inference_config={k: dict(mgr) for k in
                      ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")},
)
clf.fit(X_train, y_train, kv_cache=False)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    probabilities = clf.predict_proba(X_val32)
```

Verified signatures: `fit(self, X, y, kv_cache=False)` and `predict_proba(self, X)`. **In 2.0.2, `kv_cache` belongs to `fit`, not the constructor**; current latest documentation describes a different API. Predicted columns follow `clf.classes_`; map them back to the frozen task candidate IDs, not assumed integer positions. Native 2-10-class operation should not require hierarchical many-class inference. [Classifier source][classifier]

Set `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` before imports; no worker pool. Leave `n_jobs=None`: 2.0.2 uses `max(host_cpu_count, n_jobs)` for positive values, so `n_jobs=4` can oversubscribe the shared CPU allowance. The current changelog records the later correction. [Thread code][threads], [changes][changes]

`use_amp=True` permits mixed-precision internals despite FP32 inputs. `use_fa3=False` avoids the optional Hopper-oriented FA3 package; Torch's built-in SDPA supplies Flash Attention. The [Torch 2.5.1 context][sdpa] disables math fallback for this call: unsupported Flash kernels must fail, not silently materialize quadratic attention. Actual 4090/cu124 execution remains untested.

## Memory Evidence And Limits

`batch_size=1` controls concurrent ensemble members, **not TRAIN row count**. COL batching splits feature/group batch axes while retaining the row sequence; ROW batching splits independent rows. ICL batching splits ensemble axes and cannot reduce the single 81K-row context. No public `max_context_rows` or ICL sequence-chunk cap was found in this pin. [Embedding][embedding], [interaction][interaction], [inference manager][manager]

The source-default structure is embedding width 128, four reserved CLS tokens, `feature_group="same"` (one circular group per input feature, **not** feature_count/3), and row representation width 512. The downloaded weights were not loaded: the owner must verify their actual `config` before applying these numbers. Bounds below are **individual tensor sizes, not end-to-end peak bounds**. The manager's merged output buffers can be FP32 even with AMP. [Structure][structure], [embedding][embedding], [manager][manager]

For one ensemble and 81,000 + 32 rows, conservative FP32 COL buffer = `(81032 * (remaining_columns + 4) * 128 * 4)` bytes:

| Input case | Columns | One COL buffer (GiB) |
| --- | ---: | ---: |
| ATSP, user-measured constant removal | 100 | 4.018 |
| TSP, first witness | 162 | 6.414 |
| CVRP | 173 | 6.839 |
| MVRP range | 174-217 | 6.878-8.539 |
| Unfiltered input statistics | 357 | 13.949 |

One full TSP TRAIN FP32 input matrix is only 50.06 MiB (357 columns: 110.31 MiB), but preprocessing/ensemble views make copies. A 512-wide FP32 row representation is another 0.155 GiB. These do not include weights, CUDA workspaces/allocator reservation, group construction, Q/K/V/FFN temporaries, Python/import overhead, or shared-container occupants. **No empirical RSS, container-memory, VRAM peak, latency, or rigorous total-memory upper bound was obtained.**

Resource controls and cautions:

- Keep one task and one estimator resident. `norm_methods=["none"]` avoids the default second power-normalized view; upstream still performs its TRAIN-fitted preprocessing/clipping. This is a frozen resource pilot configuration, not a claim about best accuracy.
- Prefer GPU-resident COL/ROW outputs for the 162-column witness. Their known tensor sizes leave plausible 24 GB headroom while avoiding a multi-GiB host copy; `safety_factor=0.1` conservatively shrinks internal COL/ROW batches. It is a heuristic, not a reservation or peak cap. The single-batch fast path has no batch-halving retry, and ICL has only one ensemble member to split. [Manager][manager]
- `offload="gpu"` is a preference, **not fail-closed**: allocation/mode selection can fall back to CPU. The runner must reject CPU fallback as a resource failure and must not assume this API enforces the 9 GB budget. Preflight GPU headroom and monitor only the new pilot's own resources; never manage sibling jobs. [Manager][manager]
- Old `auto` reads host `psutil.virtual_memory().available`, not cgroup or the practical per-slot allowance. Current source has a cgroup fix, but upgrading would also require a different Python runtime. Even cgroup-aware code would not know an informal 9 GB share inside a 20 GB container. Do not use old `auto` here. [Old memory query][manager], [current changes][changes]
- Disk mode is supported through `offload_mode="disk"`, `disk_offload_dir`, and COL/ROW/ICL manager overrides including `disk_flush_mb`, `disk_dtype`, `use_async`, and `max_pinned_memory_mb`. It is **not** a 9 GB safety guarantee: memmap page cache, CPU staging, full single-batch copies, and ordinary buffers remain. Disk shortage can fall back to CPU, including swap. No disk throughput/capacity was measured; no disk mode is proposed for the first witness. [Config keys][config], [manager][manager]
- Keep KV caching off initially. The ICL K/V cache alone, assuming 12 layers / width 512 / FP16 / 81K context / one estimator, is about 1.854 GiB (FP32: 3.708), before column caches and concatenation copies. `fit(..., kv_cache="repr")` is available later, but is not a free or measured-memory-safe streaming option. [Cache API][classifier], [cache implementation][cache]
- Flash attention avoids storing the full attention-score matrix, not its computation. A hypothetical dense 8-head 81K-square FP16 score tensor alone is 97.77 GiB. Chunking 32 query inputs outside the estimator preserves all TRAIN rows but, without caching, repeats full context processing per call. Do not promise cheap 4,500-row evaluation from the 32-row witness. [Attention][attention], [classifier][classifier]

## First Witness And Gate

1. Main stages the pinned public checkpoint privately, verifies its SHA256, and inspects `config` on CPU (`torch.load(..., map_location="cpu", weights_only=True)`). Confirm actual embedding/group/CLS/layer sizes and max_classes; recompute buffer estimates if different. Reuse only input-only statistics and the TRAIN-derived constant mask. No data scan or weight download was performed in this turn.
2. Only when the main session confirms R94B's complete pipeline has released `4090_1`, use the existing free slot with CPU4, practical RAM9 GB, and one 24 GB GPU. Do not create/resize allocations. Run the fragment once with **all 81K TSP TRAIN rows and labels**, and the first fixed 32 VAL input rows. The witness must never open VAL labels, TEST/LIB labels, or query costs. Do not compute accuracy or tune temperature/labels/context from this witness.
3. Record actual imported paths/versions, source and weight hashes, TRAIN IDs/order/count, input columns/mask hash, checkpoint config, actual class mapping, finite output shape, elapsed time, process peak RSS, whole-container memory headroom (including siblings), CUDA peak allocated/reserved, and any fallback/OOM. Use a self-only resource/time guard with margin below the practical 9 GB share and 24 GB GPU capacity; sampled guards are not hard peak guarantees. A CPU fallback, unsupported Flash kernel, or exhausted headroom is a failed resource witness, not permission for arbitrary context truncation or repeated retries.

Passing this witness establishes only this TSP/32-input configuration. For 4,500 VAL/TEST and 49/100 LIB inputs, retain the same complete TRAIN context and mapping; larger query calls need their own input-only capacity check. Fix inference/chunking rules using resource checks and, if later authorized, VAL only. Never add VAL/TEST/LIB labels to context; freeze the rule before TEST/LIB. Keep the existing frozen split and do not call old holdouts fresh blind tests. No TRAIN subset, rare-class quota, PCA, or other reduction is recommended now; if full-context execution fails, report the measured failing stage and resource requirement before changing the protocol.

## Public Weights And Exact Sources

Official weight repository [jingang/TabICL][hf] is public, `private=false`, `gated=false` in the unauthenticated metadata checked here. The model card declares BSD-3-Clause; [source LICENSE][license] is BSD-3-Clause too. Preserve notices when redistributing. No access approval/token gate was reported; successful transfer from the future execution container is still untested.

- Filename: `tabicl-classifier-v2-20260212.ckpt`; size: **110,368,038 bytes**.
- HF revision: `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
- [Revision-pinned download][weights]; set the resulting local file as `model_path`, `allow_auto_download=False` to avoid mutable default downloads/shared caches.
- SHA256: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` ([public file metadata][weight-page]).

[archive]: https://codeload.github.com/soda-inria/tabicl/tar.gz/ba4bdbc6bb46b9307da3dbc1685109371ec5085f
[metadata]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/pyproject.toml
[current-meta]: https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/pyproject.toml
[pypi]: https://pypi.org/pypi/tabicl/2.0.2/json
[einops]: https://codeload.github.com/arogozhnikov/einops/tar.gz/refs/tags/v0.8.1
[classifier]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/sklearn/classifier.py
[threads]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/sklearn/classifier.py#L681-L696
[changes]: https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/CHANGES.md
[config]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/inference_config.py
[manager]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/inference.py
[structure]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/tabicl.py
[embedding]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/embedding.py
[interaction]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/interaction.py
[cache]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/kv_cache.py
[attention]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/src/tabicl/model/attention.py
[sdpa]: https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/attention/__init__.py
[hf]: https://huggingface.co/api/models/jingang/TabICL?blobs=true
[weight-page]: https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt
[weights]: https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt
[license]: https://github.com/soda-inria/tabicl/blob/ba4bdbc6bb46b9307da3dbc1685109371ec5085f/LICENSE
