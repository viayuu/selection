# R85 runner and reporter integration

Preparation contract, pending the main thread's final research/analysis lock.
This worker does not launch GPU work or choose the four research arms.

## Early exact CLI for the launcher

Run in `easynco`, from `/public/home/shiys/0selection2`:

```bash
python -B -m code.V4.r85_experiment describe --arm A_ovr_control
python -B -u -m code.V4.r85_experiment train --root RUN --arm A_ovr_control --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --gpu-type 3090 --device cuda:0 --eval-batch-size 256 --wandb-mode offline
python -B -u -m code.V4.r85_experiment test --root RUN --arm A_ovr_control --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -B -u -m code.V4.r85_reporting lib --root RUN --arm A_ovr_control --device cuda:0
python -B -u -m code.V4.r85_reporting report --root RUN --arm A_ovr_control --device cpu
python -B -m code.V4.r85_experiment status --root RUN --arm A_ovr_control
```

The other prototype arm names are `B_gradient_balance`, `C_cost_profile`, and
`D_candidate_subset`. Root defaults to `runs/R85_zero_selection_research`.
Data and preparation default to the completed R84 NEW90k release and prepared
directory. The node-local cache can be supplied as `--cache-root` without any
rebuild. All arms retain the original R77 score MLP and maximin inference.

Formal `train` requires `RUN/protocol_lock.json` (or `--protocol-lock PATH`),
written by main after its research and Sol analysis. It must contain
`experiment: R85`, `locked: true`, the four `arms` in the order above,
`seed: 2`, `batch_size: 1024`, `max_updates: 57600`,
`warmup_updates: 4320`, and `eval_every: 1440`. A missing lock permits
`describe` and `status` only. The runner records the complete lock and its
digest, and refuses a changed lock on resume.

## Objective-owner API

Expected module: `code.V4.r85_objectives` (plural):
`OVRRDropObjective(selector, mode='natural' or 'eql_v2')`,
`CostProfileKLRDropObjective(selector)`, and
`CandidateSubsetRDropObjective(selector)`. Every objective returns the existing
`(loss, PART_KEYS[:-1] tensor)` from `forward(microbatch, consistency_weight)`.
There is no focal term. The following lifecycle is owned by the runner:

`begin_attempt(logical_batch, consistency_weight)` returns the full batch with
its frozen context; the runner slices this returned batch. `accumulate(rows)`
runs after every microbatch backward. `finish_attempt(success)` discards pending
state on failure or commits it and returns detached metrics on success. Before
each attempt the runner restores the same RNG, so subset draws, dropout and
frozen EQL weights replay identically. All arms expose these three methods and
`training_state_dict()` / `load_training_state_dict(state)`. Objective state is
saved atomically alongside model/optimizer/scaler/RNG and its successful-step
count must match the checkpoint. Inference reconstructs only the selector,
without objective or labels.
The candidate model owner provides `make_candidate_model(params, state_dict=None)`
in `code.V4.r85_candidate_selector`; all retained parameter names/shapes match
the fresh R84 initialization. Auxiliary subset scores are training-only.

## Fixed common controls

NEW90k: 81000 TRAIN, 4500 VAL, 4500 TEST per problem. Reuse R84's geometry
statistics, TRAIN SBS, random seed2 initialization, and existing feature cache.
Logical batch1024, 57600 successful updates, warmup4320, evaluation1440,
AdamW LR1e-4 to cosine1e-6, weight decay1e-4, clipping1. Both dropout passes
contribute. Each arm has its own R85 protocol/status/evaluation identity.
Hardware caps are prelocked: A/B on3090 use TSP/CVRP microcap384, C on4090
uses352, D on4090 uses256 for the shared auxiliary branch's activation memory.
The launcher supplies `--large-micro-cap` explicitly; these are hardware
execution limits, not VAL-selected hyperparameters. Logical batch remains1024.

Unique checkpoint: strict minimum full18 VAL macro_vs_sbs_pct, first exact tie
wins. Complete TEST follows the lock, then same-checkpoint LIB inference and
CPU PDF/CSV. Reuse R84's NSS4500 predictions only after confirming data root,
candidate pools, native labels/costs and saved prediction digest. No NSS retrain.

The optimized A/B classification loss is OVR; the original `parts.ce` remains
native maximin CE for diagnostics and is labeled accordingly. OVR, EQL weights,
C's conditional nonwinner KL and D's subset CE have separate real-row-weighted
logical-batch metrics in updates/history.

Reports include grouped selection/winner plots, full18 TRAIN/VAL/TEST metrics,
TRAIN/VAL zero-with-winner groups and missed-winner mass, false picks, and
same-split paired comparisons with historical R84A. TEST does not select rules
or checkpoints. All four arms are prototypes until main writes the lock.
