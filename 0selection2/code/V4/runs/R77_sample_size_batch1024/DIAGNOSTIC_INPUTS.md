# R77 Locked TRAIN/VAL Inputs

Canonical manifest: `selected_train_val_predictions/manifest.json` in this root.
Each arm's `manifest.json` records its immutable VAL-selected checkpoint hash,
selected successful update, completed budget49680 and preserved prediction hashes.
The CPU-only `diagnostic_watch` exports each arm after `training_complete.json`;
the root manifest becomes `complete=true` after both exports. This is not a new
TRAIN evaluation, model inference, TEST read or checkpoint choice.

## Files

For each arm `A/` or `B/` under `selected_train_val_predictions/`:

- `train.npz`: selected checkpoint on that arm's own TRAIN population.
- `val.npz`: the original full VAL at the same selected checkpoint.
- `train_metrics.json`, `val_metrics.json`: common FULL TRAIN SBS and own SBS views.
- `manifest.json`: checkpoint, original prediction paths/hashes, row counts and baselines.

The files are hard links to saved evaluations, not copied datasets or recomputed
scores. A has10000 TRAIN rows/problem; B has70114; both have3895 VAL rows/problem.
The export never opens TEST or writes Feynman's `train_val_selection_diagnosis/`.

## Array Schema

All18 tasks have keys `<problem>__<field>` in both NPZ files.

| Field | Meaning |
|---|---|
| `logits` | Raw uncalibrated scores `[rows, pool slots]` |
| `pool` | Solver names, in the original score/cost slot order |
| `pool_ids` | Global solver ID for each local pool slot |
| `costs` | Native unnormalized R76 cached solver costs `[rows, pool slots]` |
| `winner` | Original winner label in local pool slot space |
| `nodes` | Valid node counts per row, not coordinates |
| `indices` | Local evaluation row IDs `0..N-1` |
| `source_indices` | Corresponding original R76 split row IDs |

Use `pool_ids` to align solver columns, and apply the same permutation to
scores/costs/labels. Do not mistake local `winner` slots for global IDs. A's
TRAIN `source_indices` maps its sampled subset into B's full TRAIN; VAL row IDs
are unchanged for both. The pool and IDs are saved, not re-inferred from winners.

Geometry buffers are fitted independently on each arm's own TRAIN. Native
`costs` are not geometry-normalized, scaled scores or reconstructed costs.
Original input coordinates/features are still available in the shared R76 cache
referenced by the manifest; no large input export is necessary.

## Selection Boundary

Selection remains the single minimum complete VAL macro_vs_sbs_pct using the
common FULL70114 TRAIN-fitted SBS. Own-subset SBS and cold-class diagnostics are
reporting only. A resumed its own saved R77 update2484 checkpoint, not old weights
or a Git commit; there was no architecture/optimization formula change. The
diagnostic exporter changes no training formula and does not rerun TRAIN.
