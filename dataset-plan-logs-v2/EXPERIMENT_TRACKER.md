# Dataset and Label Tracker V2

**Status key**:

- `[ ]` pending
- `[~]` in progress
- `[x]` done

## Dataset Generation

- `[x]` `TSP` canonical source fixed as `NSS varying`
- `[x]` `CVRP` canonical source fixed as `NSS varying`
- `[ ]` extend `MVRPGenerator.py` to support varying-size + varying-geometry export
- `[ ]` extend `ATSPGenerator.py` to support multi-family generation
- `[ ]` extend `PCTSPGenerator.py` to support varying-size + varying-geometry + regime metadata
- `[ ]` export `ATSP` 10k varying dataset
- `[ ]` export `PCTSP` 10k varying dataset
- `[ ]` export all `16` CVRP-variant 10k varying datasets
- `[ ]` export manifest JSON for every new dataset
- `[ ]` export fixed-by-scale shards for every new dataset

## Solver Entry Freeze

- `[ ]` freeze `TSP` first-wave solver-entry list
- `[ ]` freeze `CVRP` first-wave solver-entry list
- `[ ]` freeze `ATSP` first-wave solver-entry list
- `[ ]` freeze `PCTSP` first-wave solver-entry list
- `[ ]` freeze `16` CVRP-variant solver-entry list
- `[ ]` record checkpoint mapping per solver entry
- `[ ]` record assigned scale shards per solver entry

## Init-Only Labeling

- `[ ]` create reusable init-only override pattern
- `[ ]` run full `TSP` label jobs
- `[ ]` run full `CVRP` label jobs
- `[ ]` run full `ATSP` label jobs
- `[ ]` run full `PCTSP` label jobs
- `[ ]` run full `16` CVRP-variant label jobs

## Merge and Packaging

- `[ ]` merge shard outputs to solver-entry-level files
- `[ ]` merge solver-entry files to final problem-level supervision tables
- `[ ]` compute feasible masks
- `[ ]` compute best-entry labels
- `[ ]` compute full rankings
- `[ ]` package final labeled datasets for supervised learning

## Final Outputs

- `[ ]` labeled `TSP` dataset ready
- `[ ]` labeled `CVRP` dataset ready
- `[ ]` labeled `ATSP` dataset ready
- `[ ]` labeled `PCTSP` dataset ready
- `[ ]` labeled `16` CVRP-variant datasets ready
- `[ ]` unified metadata schema frozen
