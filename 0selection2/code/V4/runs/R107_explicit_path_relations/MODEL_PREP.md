# R107 Model Contract

Scope: only `r107_model.py`, `test_r107_model.py`, and this note. No runner,
protocol, launcher, dataset, historical source, GPU job, or checkpoint writes.
Implementation assumptions are recorded here before their code is introduced.

## Fixed Contract

- `ARMS = ('A_edge_factorized', 'B_connected_paths')`.
- `make_model(params, arm, state=None)` returns a CPU selector; the older
  `state_dict=` spelling is also accepted, never simultaneously with `state`.
- `initialize_shared(initial, geometry, arm, device='cpu')` returns
  `(model, deepcopy(model.params), receipt)`.
- Export `SPEC`, `PATH_SPEC`, `ORIGINAL`, `PATH_INIT_SEED=107002`, and
  `PARAMETER_COUNTS` (arm to integer, plus `original`). Original trainables: 1,842,953.
  New trainables: 49,280. Total per arm: 1,892,233.
- Preserve all original named parameter tensors, R88B physical TSP node0 role,
  original four-layer width128 encoder, TRAIN geometry normalization, ID+34,
  two synchronous dual streams, pooling, shared readout, utility differences,
  and maximin. The runner retains the original loss and logical batch1024.
- Regenerate the exact R84 scratch seed2 trainables and validate their hash;
  copy all of them without aliasing. Initialize ONLY the new tensors under an
  isolated seed107002. Both arms start with identical complete state dictionaries.
  A caller's trained state is allowed only for explicit strict reconstruction,
  and is never described as fresh initialization.

## Assumptions And Intervention

| ID | Decision | Meaning / boundary |
| --- | --- | --- |
| A01 | Two 32d relation rounds after encoder layers 1 and 3 | Relations influence subsequent encoding, not an after-encoder adapter. |
| A02 | Four outgoing neighbors and at most sixteen connected two-edge paths per source | Current directed raw matrix for ATSP; Euclidean distances for coordinates; no dense cubic tensor. |
| A03 | Routing customers reserve one of four neighbor slots for their actual depot | Depot is raw routing slot3, never the TSP execution-role bit; depot has up to four customer neighbors. |
| A04 | Exclude a distance tie class if it overflows the neighbor boundary | This sacrifices some tied neighbors instead of introducing index-based, non-equivariant pruning; all fitting tie classes remain. No resource/feasibility filtering. |
| A05 | Paths have distinct i, j, k | Direct k-to-i closure and depot return distances remain primitive fields. A middle depot is a physical walk waypoint, not an implicit resource reset. |
| A06 | A learns phi(e,e,0,previous_edge), B learns phi(e_ij,e_jk,composition,previous_path) | A averages independently updated edge states only after its nonlinear update. No coupled path values or previous joint state enter A's phi. |
| A07 | Both use masked arithmetic-mean path-to-source injection | The original final node mean/max and candidate-conditioned readout remain unchanged. No path search, candidate path tokens, or new loss. |
| A08 | Preserve stored demand units and signs | Positive delivery, negative backhaul. Expose signed prefix extrema/range and separate positive/negative magnitudes, not classical-B phases, onboard-load feasibility, or invented capacity thresholds. |
| A09 | TW starts at time0 at the depot for the local walk, with stored service/endpoints | Explicit max(arrival, window_start), waiting, departure and signed deadline slack. Depot's stored zero TW slots are NOT a zero deadline; no feasibility filtering. Disabled TW yields zero time fields. |
| A10 | L uses stored route_limit and physical distance only; O omits final depot return | Slack is local-walk arithmetic, not proof of an extendible route. Raw L/TW values are never replaced by project defaults3.0/0.2. |
| A11 | Caller supplies frozen TRAIN geometry moments | Factory never fits moments or reads labels/files; validates twelve finite means and positive standard deviations. Dataset/split provenance still belongs to main. |

The shared node projection is bias-free Linear128to32 plus LayerNorm32
(4,160 parameters). Each round uses Linear252to64, GELU, Linear64to32,
LayerNorm32, and Linear32to128 injection (22,560 parameters). Initial content
uses the mean projected edge endpoints in A and three connected nodes in B;
later rounds use their own previous edge/path states. Thus no first-round
previous-state affine block is rendered inactive by an all-zero initial state.

The intervention matches tensor count and initialization, NOT active functional
freedom or FLOPs. A's two edge-input weight blocks are identifiable only through
their sum. Its composed-input columns (2 rounds * 64 * 32 = 4,096 weights) have
zero gradients. A evaluates the shared-arity network once per edge per round;
B once per path. A still retains all global interactions of the original trunk,
and the second round reads node states affected by the first: the additivity
claim is conditional on each round's incoming node states, not for the whole
selector function. The primitive-only differences do not identify a pure
"learnability of connections" effect separately from explicit composed fields.

## Bounded Cost

Learned content is O(batch * N * 4^2 * 32); full raw distance/bias construction
remains O(batch * N^2). The first affine layer is evaluated by separate edge,
composition and previous-state projections: never materialize a 252d feature
vector for every path. No disk feature cache is added. Relation geometry is
computed once per forward and reused across the two rounds.

Conservative, unmeasured 24GB starting recommendations: microbatch64 near N500,
microbatch16 near N1000; smaller for substantially larger LIB instances. These
are NOT measured peaks or speed guarantees. Main must retain sample-weighted
logical-batch1024 semantics and own any device capacity probe/launch.

## Evidence

Design references: `docs/research/deep_direction_review_v2_20261023/REPORT.md`
and `REPRESENTATION_AND_INPUTS.md`. Their proposal is a representation hypothesis,
not evidence of improved solver selection. No model performance claim.

CPU verification, with CUDA hidden:

```sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest code.V4.test_r107_model -v
```

All eight tests passed in 3.177 seconds. They cover all18 input-only forwards,
unchanged output/maximin algebra, during-encoder round placement, node/candidate
permutation (preserving the physical TSP role), exact boundary ties, directed
matrix differences, nonzero joint mixed difference versus exactly additive A,
current stored resource arithmetic (including negative slack and pickup before
delivery), gradients to relations/raw nodes/original trunk, padding/depot/tiny
graphs, immutable TRAIN moments, and strict restore/fresh-donor rejection.
Both arms measured 1,892,233 trainables with identical complete fresh states.
Copied original hash:
`dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93`.
No GPU launch, timing claim, dataset scan, solver, or checkpoint-file load.

Main's source closure must retain the original topology, not only R107:
`code/V4/V4Model.py`, `dual_stream.py`, `pairwise_selector.py`,
`pairwise_objective.py` (R88 input isolation), `r88_model.py`,
`local_geometry.py`, `solver_features.py`, `r77_data.py`, and
`code/unified_selector/registry.py`, plus their transitive imports. R107 imports
these through their existing package paths and replaces no global module or
historical source. Runner/loss/cache/protocol source closure remains main-owned.

## Main-thread measured execution choice

The heavier B_connected_paths arm passed actual two-dropout forward/backward on
the allocation-owned RTX3090 using the largest TRAIN rows. Final shared caps are
TSP/CVRP256, other tasks1024, evaluation128; logical batch stays1024. Peak allocated
memory was18,805/18,856MiB on TSP/CVRP and12,075/12,141MiB on ATSP/VRPBLTW;
maximum reserved memory21,618MiB on the24GB card. Both original and path gradient
groups were finite and nonzero. These figures supersede the unmeasured starting
recommendation above; they are not a throughput or model-quality result.

Receipt: `preparation/kernel_witness/B_larger256_1024/complete.json`.
The earlier128/512 witness is preserved; no formal training or optimizer update
was used to choose the caps.
