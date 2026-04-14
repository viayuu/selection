# Dataset and Label Tracker V3

**Status key**:

- `[ ]` pending
- `[~]` in progress
- `[x]` done

## Data Preparation

- `[x]` `TSP` source frozen as `NSS varying`
- `[x]` `CVRP` source frozen as `NSS varying`
- `[ ]` generate `ATSP` shard files at `20 / 50 / 100`
- `[ ]` write `ATSP` manifest
- `[ ]` generate all variant-`MVRP` shard files at `50 / 100`
- `[ ]` write all variant-`MVRP` manifests
- `[ ]` optionally generate `PCTSP` shard files at `20 / 100 / 500`

## Checkpoint Preparation

- `[ ]` unpack `mtpomo.zip`
- `[ ]` unpack `mvmoe.zip`
- `[ ]` freeze exact ATSP checkpoint map
- `[ ]` freeze exact MVRP checkpoint map

## Real Label Jobs

- `[ ]` launch `TSP` NSS label batch
- `[ ]` launch `CVRP` NSS label batch
- `[ ]` launch `ATSP` 9 jobs
- `[ ]` launch `MVRP` 64 jobs
- `[ ]` optionally launch `PCTSP` single-method archive

## Merge

- `[ ]` merge `TSP` solver-entry outputs
- `[ ]` merge `CVRP` solver-entry outputs
- `[ ]` merge `ATSP` solver-entry outputs
- `[ ]` merge `MVRP` solver-entry outputs
- `[ ]` compute feasible masks
- `[ ]` compute best-entry labels
- `[ ]` compute ranking labels

## Final Release

- `[ ]` labeled `TSP` dataset ready
- `[ ]` labeled `CVRP` dataset ready
- `[ ]` labeled `ATSP` dataset ready
- `[ ]` labeled `MVRP`-variant datasets ready
- `[ ]` `PCTSP` release decision recorded explicitly
