# R110 Runner Preparation

Status: both-arm describe CLI passed; all 12 CPU contract tests passed. No GPU witness,
training, TEST, timing or Git action is performed by this worker.

## Scope And Interfaces

- Exclusive write scope: `r110_experiment.py`, `r110_objective.py`,
  `test_r110_experiment.py`, `test_r110_objective.py`, and this file, under `code/V4`.
- Historical runners/objectives and `r109_model.py` are read-only. Training and
  evaluation use a separately loaded R84 module, following the proven R108 pattern.
- CLI commands: `describe`, `status`, `train`, `test`, `timing`; each requires `--arm`.
- Factory: `code.V4.r110_experiment.load_checkpoint`; no prior files, fitting,
  density objective, independent path implementation, quotas or label inputs.
- `A_tied_ce` maps to `r109_model.A_tied_late` (2,305,033 parameters).
- `B_private_ce` maps to `r109_model.B_private_late` (10,220,553 parameters).
- Factory and shared initialization delegate to R109 with the mapped model arm;
  model params retain `r109_arm` and the frozen `r109_spec` for strict restoration.
- Common scratch R84 seed2 tensors and identical graph-block prototypes do not
  imply equal full parameters/state hashes. A/B initially have the same eval
  function but are not parameter- or FLOP-matched.

## Locked Scientific Contract

Same frozen NEW90k full18 data: TRAIN 81,000 / VAL 4,500 / TEST 4,500 per problem;
original candidate exclusions and `reld_mt` naming remain unchanged. Reuse the
frozen R84 cache, TRAIN-only geometry/SBS and fresh initialization, not trained
weights. Old holdouts have participated in development, not a new blind test.

One selector forward with dropout 0.1; unweighted CE on raw utility masked to the
whole legal pool and the native TRAIN winner. Comparison, relative risk, R-Drop,
KL, priors and objective state are absent. Maximin CE during evaluation is only
an inherited diagnostic, not the raw-utility training CE. Inference is unchanged
score MLP -> utility differences -> maximin, with the inherited solver-ID tie rule.
The private engine's consistency callback is always zero; historical callbacks
and stateless objectives are not monkey-patched.

Seed2, logical batch1024, 57,600 successful updates, warmup4,320, full VAL every
1,440, full TRAIN every7,200 plus selected TRAIN if absent. AdamW peak1e-4,
cosine end1e-6, WD1e-4, clip1. Checkpoint: strict minimum full18 VAL macro_vs_sbs,
first exact tie, common full-TRAIN SBS; one fixed checkpoint/rule for full TEST
and downstream LIB. No TEST/LIB selection or per-problem model assembly.

Selected TRAIN and VAL summary bindings prove exact model-state hash, protocol
hash and successful update against immutable `best.pt`. Capture identities and
prediction archives are never rewritten. Timing covers the complete standalone
model including method-conditioned node refinement and reads only input arrays.

## Assumptions And Main Handoff

| Item | Choice | Source / Gate |
| --- | --- | --- |
| Canonical train microcaps | TSP/CVRP256; all other tasks1024 | Locked by user after main's B-arm 4090 one-pass witness PASS |
| Eval batch | 256 | Locked by user; separate from the train-mode model witness |
| Timing sample | first128 frozen TEST inputs/problem, 1 warmup + 3 measurements | R108 interface retained; eval batch256 is a separate setting |
| GPU assignment | R110 single-forward pair on 4090s, R109 composite pair on 3090s | Main owns allocation and launch; worker performs no GPU work |
| Source closure | includes `r109_model.py` and R110 objective/runner plus original runtime dependencies | Main freezes after all initialization integration; do not lock a stale hash |
| Reporting / LIB integration | Main or reporter worker ports R109 reporting later | Outside this worker's five-file scope; no final-model delivery claim |
| Model identity | Sol/max requested by user | This worker cannot independently attest the serving model identity |

Main reported B-arm one-forward/backward peaks: TSP17.33GB, CVRP19.73GB,
ATSP5.85GB, VRPBLTW5.24GB. The landed witness uses an input-only scalar, not
native-label CE or a complete AdamW update. These numbers are main-provided model
memory evidence, not independently measured by this worker or end-to-end optimizer
peak claims. Main's seven full18 model initialization/factory CPU tests are not
duplicated by this worker's runner/objective suite.

No automatic OOM cap shrink. Main owns any remaining optimizer/eval readiness gate.
Protocol lock must contain `required_lock_fields()` plus absolute `data_root`,
`prepared_root`, `primary_cache`. Restore validates the embedded protocol without
opening/refitting priors or requiring a live lock file. Main owns the formal lock,
GPU witness, tmux launch, full evaluation and PDF/CSV delivery.

## Safe CPU Entry Points

```bash
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r110_experiment describe --arm A_tied_ce
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest code.V4.test_r110_objective code.V4.test_r110_experiment
```

Both `A_tied_ce` and `B_private_ce` describe CLI calls passed, confirming actual
counts 2,305,033 / 10,220,553, mapped R109 model arms, eval256 and the R110 factory.
CPU verification: all 12 tests passed (`Ran 12 tests in 3.206s`, `OK`), with
`CUDA_VISIBLE_DEVICES=''` and a CUDA-initialization guard in every test. Coverage:
single-dropout-forward exact CE/gradients, two heterogeneous masks including
illegal NaN/Inf and singleton pools, unweighted mean, invalid labels/arms,
statelessness, unchanged historical callbacks/objectives, mapped factories,
actual A/B checkpoint round trips without prior/live-lock reads, logical1024
row weighting, selected TRAIN/VAL exact binding without archive rewrites,
preexisting full18 TEST cache authorization, and input-only standalone timing
schema. The first run found two test-fixture mistakes; both were corrected,
then the complete suite passed. No historical model source was changed.

These checks do not replace GPU memory/readiness evidence or authorize a launch.
