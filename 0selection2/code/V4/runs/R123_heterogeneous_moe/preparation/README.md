# R123 CPU data/model preparation

Provisional trio: R77B, R80A, R83.
Source-record scope only; no mathematical-base/content-duplicate guarantee.

| Problem | Source-certified metaTRAIN | Current TRAIN | Status |
| --- | ---: | ---: | --- |
| TSP | 10886 | 81000 | source_identity_certified |
| CVRP | 10886 | 81000 | source_identity_certified |
| ATSP | 10886 | 81000 | source_identity_certified |
| OVRP | 10886 | 81000 | source_identity_certified |
| VRPB | 10886 | 81000 | source_identity_certified |
| VRPL | 10886 | 81000 | source_identity_certified |
| VRPTW | 10886 | 81000 | source_identity_certified |
| OVRPTW | 10886 | 81000 | source_identity_certified |
| OVRPB | 10886 | 81000 | source_identity_certified |
| OVRPL | 10886 | 81000 | source_identity_certified |
| VRPBL | 10886 | 81000 | source_identity_certified |
| VRPBTW | 10886 | 81000 | source_identity_certified |
| VRPLTW | 10886 | 81000 | source_identity_certified |
| OVRPBL | 10886 | 81000 | source_identity_certified |
| OVRPBTW | 10886 | 81000 | source_identity_certified |
| OVRPLTW | 10886 | 81000 | source_identity_certified |
| VRPBLTW | 10886 | 81000 | source_identity_certified |
| OVRPBLTW | 10886 | 81000 | source_identity_certified |

## Expert interfaces

| Expert | Loader | Complete full18 budget | Role |
| --- | --- | --- | --- |
| R77B | loaded_strictly_on_cpu | True | handcrafted representation; original composite objective and R-Drop |
| R80A | loaded_strictly_on_cpu | True | handcrafted representation; CE-only objective |
| R80C | loaded_strictly_on_cpu | True | handcrafted representation; composite objective; changed learning rate |
| R80D | loaded_strictly_on_cpu | True | complete source-code representation; composite objective |
| R83 | loaded_strictly_on_cpu | True | model-only source-code representation; composite objective |

## Usage

```python
from code.V4.r123_meta_data import load_meta_indices, load_expert, ExpertAdapter
indices = load_meta_indices('TSP')  # positions in current frozen TRAIN81k, not source-global IDs
model, state = load_expert('R77B', device='cpu')
# Keep labels/IDs outside ExpertAdapter.predict inputs.
```

Full paths/hashes and each checkpoint model_params are in experts.json/manifest.json.
source_catalogues.json binds generated source shards and old subset membership.
meta_train_indices.npz also stores both complete TRAIN source-reference lists and meta source identities.
Default score normalization is raw; center/zscore are optional per-query, label-free transforms.
R94A is excluded: it declares the full current TRAIN81k exposure population.
VAL/TEST are not meta training or expert-selection inputs. Old holdouts were exposed during research.
No GPU, optimizer, training, R120 modification, or Git operation occurred.
