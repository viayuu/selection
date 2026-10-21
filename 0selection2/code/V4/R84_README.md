# R84 runner integration

Runner owner: preparation, training, reporting, focused tests. Model, launcher,
and dataset release have separate owners. No launch is authorized by this file.

## Stable CLI contract

All commands run in `easynco`, from `/public/home/shiys/0selection2`.
The default release is
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
The default run root is `code/V4/runs/R84_new90k_attention`.

```bash
python -m code.V4.r84_prepare prepare --root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -m code.V4.r84_experiment train --root RUN --arm A_baseline --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --gpu-type 3090 --device cuda:0 --eval-batch-size 256 --wandb-mode offline
python -m code.V4.r84_experiment train --root RUN --arm B_attention_direct --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --gpu-type 4090 --device cuda:0 --eval-batch-size 256 --wandb-mode offline
python -m code.V4.r84_experiment test --root RUN --arm A_baseline --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -m code.V4.r84_reporting lib --root RUN --arm A_baseline --device cuda:0
python -m code.V4.r84_reporting report --root RUN --arm A_baseline --device cpu
python -m code.V4.r84_experiment status --root RUN --arm A_baseline
```

`train` includes unique VAL selection and the complete TEST. `test` is idempotent
for the same lock. `lib` performs same-checkpoint LIB inference and fresh
4500-row TEST predictions from the existing locked R79 NSS weights. `report`
uses saved predictions and runs on the CPU with the available CJK font.

Preparation writes `prepared.json`, `source_cache_manifest.json`,
`geometry_stats.json`, `common_full_train_sbs.json`, and
`identicalinit_seed2.pt`. TRAIN and VAL are prepared first. TEST features are
prepared only after an arm's unique VAL lock, and shared read-only thereafter.
The feature arrays use the existing R76/R75 scenario cache schema. Old cached
rows are reused by actual source ID after checking the native input values;
only newly added coordinate rows need geometric feature computation.

Each arm uses 81000 TRAIN rows per problem, seed2, natural full18 task order,
logical batch1024, 57600 successful updates, VAL every1440, warmup4320,
AdamW(1e-4, weight_decay1e-4), cosine end1e-6, and original R77/R-Drop losses.
The unique checkpoint minimizes full VAL macro_vs_sbs_pct with TRAIN SBS.
No TEST/LIB-based tuning, per-task checkpoint assembly, or training launch
belongs to the reporter.

## Model-owner interface

Expected module `code.V4.r84_attention_selector` exposes
`make_r84_model(params, state_dict=None, **optional)` for the direct-attention
arm. The runner will copy all common same-name/same-shape fresh parameters.
Please preserve original parameter names for unchanged modules and return the
existing `logits`, `pair_margin`, `solver_mask`, and optional `utility` contract.
If the final factory name differs, communicate it before launch.

The four fixed arms are `A_baseline`, `B_attention_direct`, `C_focal`, and
`D_attention_focal`. The objective owner exposes
`FocalRDropObjective(selector)`: A/B use the existing `RDropObjective`; C/D
add weight0.35 focal CE with gamma2 and no class-frequency weighting. The
objective returns the existing `(loss, PART_KEYS[:-1] tensor)` contract; total
`loss` includes the extra term. Both dropout forwards contribute. The runner
records real-row-weighted focal metrics after a successful AMP attempt only.

Coordination limitation: this worker session currently exposes no `send_input`
or agent-management tool; this shared contract is the available handoff.
