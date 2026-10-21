# R87 matched shared/private method residuals

Only the main thread/resource owner launches GPU jobs. This runner reuses the
frozen R84 engine through its existing callbacks; no R84/R85 source is edited.

## Exact launcher CLI

Run from `/public/home/shiys/0selection2` after activating `easynco`:

```bash
python -B -m code.V4.r87_experiment describe --arm A_shared_residual
python -B -u -m code.V4.r87_experiment train --root RUN --arm A_shared_residual --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --protocol-lock RUN/protocol_lock.json --gpu-type 4090 --large-micro-cap 352 --eval-batch-size 256 --device cuda:0 --wandb-mode offline
python -B -u -m code.V4.r87_experiment test --root RUN --arm A_shared_residual --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -B -u -m code.V4.r87_reporting lib --root RUN --arm A_shared_residual --device cuda:0 --lib-batch-size 2
python -B -u -m code.V4.r87_reporting report --root RUN --arm A_shared_residual --device cpu
python -B -m code.V4.r87_experiment status --root RUN --arm A_shared_residual
```

The second arm is `B_private_residual`. Default RUN:
`code/V4/runs/R87_private_method_residual`. Default PREPARED:
`code/V4/runs/R84_new90k_attention/prepared`. Default RELEASE:
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
Supply the already prepared node-local R84 cache using `--cache-root`.
No preparation, solver, relabeling, or cache rebuild command is introduced.

`train` includes complete TEST after the unique VAL lock. A later `test`
command only completes/reuses that lock; it never chooses a new checkpoint.
LIB runs the same checkpoint factory. PDF rendering reads saved predictions
on CPU and reuses the exact4500 R84 NSS reference without NSS inference.

## Model-owner and initialization contract

Model module `code.V4.r87_model` exposes:

```python
model = make_model(params, arm, state_dict=None)
```

`model.params` must contain the complete strict reconstruction parameters.
All original R77 parameter names/shapes are retained. The runner copies every
original parameter from R84's verified fresh seed2 initialization, restores its
TRAIN-only geometry buffers, and records common/new parameter names, hashes,
and counts. It never loads a trained selector. New residual input weights are
fresh; new output weights are zero as implemented by the model factory.

A uses one shared bias-free two-layer GELU residual, hidden width
`len(GLOBAL_SOLVERS)*32`. B uses hidden32 per stable global method ID. Both
read the same original `3*d` scoring features and add to the original score MLP.
The total added parameter counts must match. Full-pool maximin inference is
retained. No graph-level calibration or class-frequency quota is added.

## Protocol lock and training

Main supplies `RUN/protocol_lock.json` with `experiment: R87`, `locked: true`,
`arms: [A_shared_residual, B_private_residual]`, `seed: 2`,
`batch_size: 1024`, `max_updates: 57600`, `warmup_updates: 4320`, and
`eval_every: 1440`. Its optional numeric `peak_lr`, `end_lr`, `weight_decay`,
`dropout`, `grad_clip`, `large_micro_cap`, and `eval_batch_size` fields must
match the fixed controls below. A missing/changed lock prevents training/resume.
`describe` and `status` need no lock or training data.

Each task uses TRAIN81000 / VAL4500 / TEST4500, unchanged candidate pools,
seed2 and natural source order. Logical batch1024; 57600 successful updates;
warmup4320; evaluation1440; AdamW LR1e-4 cosine to1e-6, weight decay1e-4,
dropout.1, global grad clip1. Both arms use 4090 TSP/CVRP microcap352 and
eval cap256. The original R77 objective is unchanged: .35 CE, .35 comparison,
.02 relative risk and the original R-Drop ramp. There is no objective-only
mutable state, and R85 objective state cannot be resumed into R87.

The checkpoint strictly minimizes full18 VAL macro_vs_sbs_pct with the same
TRAIN-fitted SBS; first exact tie wins. One checkpoint/rule serves all tasks,
TEST and LIB. Historical TEST has influenced earlier research and is not blind.

## Deliverables

Each arm writes its own R87 protocol, initialization receipt, successful-update
history, full TRAIN predictions of the selected checkpoint, VAL lock, TEST and
LIB predictions. Reports include grouped model-choice/true-winner PDF bars,
NSS versus the same model on TSP TEST/CVRP TEST/TSPLIB/CVRPLIB, full18 metrics
with TRAIN SBS/Oracle, TRAIN/VAL/TEST zero-method counts, fixed R84A target
recall/rescues/false picks, and paired changes against historical R84A.

After both arms finish, `r87_reporting compare --root RUN --device cpu` also
compares B directly with matched A on their separately VAL-selected checkpoints.
It never reruns inference or chooses a checkpoint from TEST.

Focused CPU integration command:

```bash
python -B -m unittest code.V4.test_r87_experiment
```
