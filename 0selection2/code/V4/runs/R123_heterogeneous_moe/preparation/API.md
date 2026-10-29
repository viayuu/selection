# R123 handoff: frozen experts and shared metaTRAIN

## Actual readiness

- All 18 routing tasks have 10,886 source-record-certified common unseen TRAIN
  rows, totaling 195,948. These are positions in the current frozen TRAIN81k,
  not guessed global-index ranges. Original historical TRAIN membership is
  retained as explicit source references in the NPZ, qualified by the source
  catalogue and hashed generator/subset metadata.
- R77B, R80A, R80C, R80D and R83 strictly load on CPU and completed the full
  49,680-update, 18-task training budget. Their existing best.pt checkpoints
  remain unchanged. Paths, checkpoint hashes, model_params, normalization
  buffers and training provenance are in experts.json/manifest.json.
- Provisional trio: R77B, R80A, R83, selected for objective/representation
  diversity, not held-out scores. All are ScoreDifferenceSelector backbones;
  this is NOT a claim of three distinct architectures or proven complementary
  predictions. R80C/D remain alternatives. R50 is not a full18 expert.
- R94A declares the complete current TRAIN81k, including normalization
  exposure, so contributes zero unseen rows and is excluded.

## Direct integration

```python
from pathlib import Path
from code.V4.r76_cache import ScenarioCache
from code.V4.r123_meta_data import (
    PREPARATION, ExpertAdapter, load_expert, load_meta_indices, read_json,
)

manifest = read_json(PREPARATION / 'manifest.json')
pools = {p: row['candidates'] for p, row in manifest['problems'].items()}
indices = load_meta_indices('TSP')  # 10,886 local TRAIN81k positions
model, checkpoint = load_expert('R77B', device='cpu')
adapter = ExpertAdapter(model, checkpoint['args']['solver_names'], pools,
                        manifest['candidate_aliases'])

# Existing current-data cache; no routing input/label copy is needed.
cache_manifest = read_json(Path(
    '/public/home/shiys/0selection2/code/V4/runs/'
    'R84_new90k_attention/prepared/source_cache_manifest.json'))
cache = ScenarioCache(Path(cache_manifest['splits']['train']['TSP']['directory']))
try:
    batch = cache.batch(indices[:2], device='cpu')
    scores = adapter.predict('TSP', batch)  # [2, 8], exact legal pool order
finally:
    cache.close()
```

The same indices can support a future MoE gate or Top2 re-ranking metaTRAIN.
No such learner or scientific mechanism has been trained/selected here.
ExpertAdapter filters the batch through an input whitelist: labels, costs,
winners and identity metadata never reach the expert forward. Supervision
for a future meta-learner must stay separate, with its protocol fixed first.

Scores are existing classification logits: larger is better, not native
costs or calibrated probabilities. Original FP32 logits are returned as
FP64 arrays without recovering additional precision. Default normalization
is raw; center/zscore are explicit optional per-query label-free transforms,
not a choice of gating objective or calibration. Each expert retains its
own historical TRAIN-fitted geometry buffers. Candidate alignment accepts
the explicit reld_mt -> RELD_MOEL rename, but rejects changed pools and alias
collisions; it never merges deployments or restores removed methods.

## Membership and certificate boundary

For every task p, meta_train_indices.npz contains:

- p__current_train_source_index: complete current 81,000-row membership.
- p__historical_train_source_index: complete historical 70,114-row membership.
- p__meta_train_local_indices: certified intersection in current local order.
- p__meta_train_source_index: corresponding declared source references.
- p__meta_train_source_identity: problem:generator_shard_SHA256:shard_row.

source_catalogues.json and manifest.evidence_files bind generator metadata,
old subset lineage, split TRAIN index hashes, per-release input metadata,
and the historical cache snapshot. Source-record certification is positive;
strict content/base-group novelty is deliberately null, NOT certified:
there is no complete semantic base-group/content-hash inventory and the
large generator-shard byte files were not rehashed. Do not claim absence
of every possible mathematical duplicate or correlated copy. No assumption
that this remaining generation cohort represents all TRAIN81k is made.

Only TRAIN membership and small CPU interface probes were used. No TEST
scores/predictions or VAL/TEST labels were read for selection or metaTRAIN.
Existing best.pt selection historically used VAL, and historical holdouts
have participated in research; they are not fresh blind evaluation sets.

## Minimal completed verification

- Six CPU unittests passed: source collisions/relocation, lineage mismatch,
  conservative unseen intersection, exact candidate alignment, explicit
  normalization, and label isolation/frozen expert weights.
- Five strict CPU checkpoint loads; one historical TRAIN row each from
  TSP, ATSP and VRPB per expert, all finite and correctly shaped.
- Current metaTRAIN spine: R77B/R80A/R83 each consumed two actual TSP rows
  through the existing R84 TRAIN cache, returning finite [2, 8] scores.
- One TSP NPZ round-trip checked current-local -> source identity references
  and historical disjointness. All 18 certified counts agree at 10,886.

Current cache manifest SHA256:
541c46b39ae9f46b7a11c1e4d468070fccfc31c2ed8af1a8aa1d30bd828556ee

Script SHA256:
86f73d45edcb5dde6655b855a9e3aa1771c686eb9d57508e18528f3a804054cb

Indices NPZ SHA256:
de1c2eb7e2b6e925eabcb6e9769de2f742c353852cfab87d216dd0264e3157c3

Manifest SHA256:
4f5b8962d6bc6ebd46104e8b00fa7ff22805241d1db6e613f8723373986f4b24

model_sources.sha256 supplements the manifest with the full core model,
factory, solver-feature and input-adapter source paths and current hashes.
No GPU work, training, Git commands or R120 modifications were performed.
The tmux preparation session r123_cpu_meta_prepare_v1 completed with exit 0.
Existing preparation is preserved: prepare() refuses an existing manifest;
an intentional future expert selection can use a new owned subdirectory.
