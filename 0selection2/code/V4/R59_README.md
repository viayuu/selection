# R59: Trainable routing-pretrained instance encoding

R58 is deferred by the user. Its B-constraint semantic discrepancy remains an
open issue; this experiment neither repairs nor certifies the legacy benchmark.

## Question and comparison

Does a routing-pretrained encoder, trained jointly with a full-pool selector,
generalize better than the same encoder initialized randomly?

- R59A: native ReLD-MTL encoder architecture, random initialization.
- R59B: identical architecture, existing ReLD-MTL epoch5000 encoder weights.
- Historical reference: R45A, not a newly matched training control.

Both arms train from scratch except B's explicitly transferred encoder weights.
All parameters are trainable. Shared non-native parameters have identical
initial values. This differs from R50's frozen two-method readout and the
R53--R56 local Top2 experts: every original legal solver remains selectable.

## Architecture

One shared native encoder serves CVRP and the15 MVRP tasks: six layers, width128,
eight heads, FF512. Native depot/customer inputs use coordinates, signed demand
and time-window endpoints. A local padding-aware implementation preserves the
native encoder's unpadded calculation; external solver implementations are not
modified or run as decoders.

The native node representation is fused with the existing raw/local-geometry
projection and task condition. This preserves service time, route length limit,
true size and problem constraints omitted by the native encoder interface.
TSP/ATSP retain the old InstanceEncoder path. Solver ID/34-dimensional features,
two joint layers, utility head, score differences and maximin decision remain
the R45A/R43A path.

ReLD motivates transferring knowledge acquired through routing training; it does
not establish that these weights can predict solver winners. R59 tests that
hypothesis rather than reproducing ReLD's decoder method. See the
[ReLD paper](https://arxiv.org/html/2503.00753v1).

## Fixed protocol

Only original10,000-example training sets and original validation are used, with
all18 tasks, original pools, FP64 cost targets and native winners. No expansion
inputs/costs, new labels, augmentations or solver execution. Train-only geometry
normalization is restored with checkpoints.

- Seed2, batch128, complete traversal including tails, no accumulation.
- AdamW, weight decay1e-4, gradient clipping1, existing AMP overflow retry.
- Native encoder LR3e-5; other parameters LR1e-4 in BOTH arms.
- Three-epoch warmup, fixed LR ratio through plateau scheduling.
- Existing min15/max40-epoch controller, LR patience3, stop patience8.
- Unchanged R43 comparison/risk objective and two-forward R-Drop.
- Validation selection: strict minimum macro_vs_sbs_pct.

Advance only if B improves full18 macro Top1 by at least3 percentage points over
historical R45A and2 points over A, with actual regret no worse than both. This
screen is not the70% goal. There is **no test stage in R59**, even if this screen
passes; any later final evaluation needs its own precommitted policy.

## Implementation and outputs

`r59_transfer.py` implements the encoder adapter and paired initialization.
`r59_experiment.py` reuses the established training, targets and evaluation.
`r59_analysis.py` replays complete saved validation predictions against original
FP64 costs and reports all18 tasks, task families, solver distributions, paired
corrections/harm and cost changes when both methods are wrong.

The run directory is `code/V4/runs/R59_trainable_solver_encoder/`. It retains
protocol, source provenance, focused preflight evidence, per-arm histories,
optimizer/RNG checkpoints, locked validation checkpoints and predictions,
comparison tables, curves and RESULTS.md. Commands are documented by the runner
and saved launch metadata; no historical files are overwritten.

From the workspace root:

```bash
bash code/V4/run_v4_r59.sh plan
bash code/V4/run_v4_r59.sh tests
```

After implementation reviews and an idle-device check, the designated executor
runs the full-batch preflight, then paired training in tmux:

```bash
R59_PARENT_GPU_BATON=trainable-solver-encoder bash code/V4/run_v4_r59.sh prepare
R59_PARENT_GPU_BATON=trainable-solver-encoder bash code/V4/run_v4_r59.sh train --resume
bash code/V4/run_v4_r59.sh analyze
```

`train --resume` restores the last atomically committed epoch, optimizer,
controller, counts and RNG; it does not skip samples from a partial epoch.
There is deliberately no test command.

Only designated RTX3090 UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` under
Slurm1465/gpu03 may be used. Long runs use tmux/easynco, one arm at a time,
offline W&B project `selector`, entity
`yjkds-southern-university-of-science-technology`. There is no GPU-hour cutoff.

## Interpretation limits

One training seed; historical R45A is not parameter/data-budget matched to native
pretraining. Existing solver weight provenance is recorded, but historical
pretraining examples and complete launch arguments are unavailable. Equal total
training data and certified absence of pretraining overlap cannot be claimed.
Legacy B feasible-domain mismatches remain pending. A negative result rejects
this transfer configuration, not an accuracy ceiling or the existence of signal.
