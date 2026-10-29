# Frozen Shared Prediction API: R123 / R124

Module: code.V4.r123_prediction_cache. It does NOT import the still-changing
r123_experiment or r123_model. Fixed bank order: R77B, R80A, R83.

FROZEN helper SHA256:
35ac20b813e1281deac0b11429a1eca750a21256482d3e777a02022b8037b32b

Do not edit the helper while jobs are live. All learner changes stay separate.

## Ready two-slot CLI

Parent binds the approved physical device for slot 3090_0 / 3090_1; each
process uses its visible cuda:0. Literal --root is required. Do not execute on
a device before the parent's resource release. No GPU was launched here.

3090_0:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_prediction_cache --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --split train val --shard-index 0 --shards 2 --batch-size 128 --device cuda:0
```

3090_1:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m code.V4.r123_prediction_cache --root /public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe --split train val --shard-index 1 --shards 2 --batch-size 128 --device cuda:0
```

CUDA_VISIBLE_DEVICES above illustrates parent device binding; if its slot
launcher already binds a GPU UUID, retain that binding rather than overriding
it. The remaining command/absolute root is identical on both slots.

Disjoint 9+9 partition, all18:

- shard0: TSP, ATSP, VRPB, VRPTW, OVRPB, VRPBL, VRPLTW, OVRPBTW, VRPBLTW.
- shard1: CVRP, OVRP, VRPL, OVRPTW, OVRPL, VRPBTW, OVRPBL, OVRPLTW, OVRPBLTW.

--problem TSP CVRP ... can instead specify literal problem sets. Each process
loads only ONE frozen expert at a time, handles its TRAIN+VAL problems, frees
it, then loads the next expert. FP32 eval/no_grad/no autocast/TF32 disabled.
Default batch128; --batch-size 256 is supported without changing helper source.

## Atomic Archives And Consumer API

ROOT/prediction_cache/{train,val}/{problem}/{expert_id}.npz is atomically
published after the whole problem/expert is complete. Per-archive writer
locks prevent overlapping writers. Restart validates/skips complete archives;
only an interrupted, unpublished problem/expert needs recomputation.

Each archive contains logits[N,K] FP32, row_indices[N] current split-local
int64 positions, source_indices[N] release references, source_identity[N]
qualified problem:shard_SHA256:shard_row, pool, pool_ids, metadata_json. Embedded
metadata pins model checkpoint/source-lineage/helper hashes and all identity
array digests. No costs, native winner, ind or labels are stored or forwarded
to the frozen model. TRAIN N=10886/task, VAL N=4500/task.

```python
from code.V4 import r123_prediction_cache as shared
root = '/public/home/shiys/0selection2/code/V4/runs/R123_heterogeneous_moe'
ctx = shared.context()  # create once, reuse; optional transported input-cache root
proposal = shared.read_expert(root, 'train', 'TSP', 'R77B', ctx)
# R124 reuses proposal['logits'] [N,K] and proposal['row_indices']; do NOT regenerate.
bank = shared.load_scores(root, 'train', 'TSP', ctx)
# bank['logits'] [N,3,K], bank['row_indices'] original current TRAIN81k positions.
```

R124 may read R77B as soon as that archive exists, without waiting for R80A/R83.
load_scores requires all requested experts; missing archives are not filled
with zero. Exact row ordering/pools/checkpoint hashes are validated.

--input-cache-root remaps the existing R84 cache tree; no inputs or geometry
are regenerated. --prepared-root can point to its frozen metadata directory.
--describe performs metadata/file preflight without loading models or CUDA.
TRAIN+VAL two-row CPU fixture parsing and disjoint CLI partition passed.
Fixtures under preparation/learner_prediction_smoke are marked incomplete and
cannot be used as formal TRAIN/VAL caches.

TEST is rejected before a completed R123 unique full18 VAL checkpoint lock;
use --test-lock-root RUN when the learner run is not the default location.
Current source-record certification does NOT prove exhaustive absence of base
or content copies. No new bank member, ReLD merging, candidate restoration,
cross-expert raw-logit mixing, or TEST-based expert selection is permitted.
