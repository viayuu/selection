# R108 Reporting Handoff

Report/admin implementation only. No formal readiness, spec write, source
freeze, training, GPU query, real timing/LIB/PDF or Git operation was performed
by this worker. R106/R107 sources remain read-only.

## Report Contract

- Arms: `A_prior_ce` and `B_density_ratio`; same scratch original R84/R88B,
  1,842,953 trainables, same forward/maximin and fixed TRAIN prior buffers.
- `t` is the original utility and `s=logpi+t`. A trains pure CE(s). B trains
  `mean[0.5*softplus(-t_y) + 0.5*sum_j pi_j*softplus(t_j)]`, including the
  true label in the product-noise term. Canonical normalization differs;
  no equal-gradient-scale or equal-FLOPs claim. `active_parameter_matched=False`:
  CE's common-shift gauge cancels shared last-bias data gradient; ratio anchors
  the utility origin. Same forward/parameters is not identical active gradient freedom.
- Single forward, dropout0.1; loss recipe
  `{ce_or_density_ratio:1, comparison:0, risk:0, kl:0, dropout_passes:1}`;
  RDrop weight/ramp0. TRAIN `loss` is arm-specific and TRAIN `ce` is CE(s).
  Evaluation CE uses maximin and is not silently treated as that training loss.
- Full18 NEW90k, logical1024, seed2; 57,600 updates, warmup4,320,
  VAL every1,440, TRAIN diagnostics every7,200; AdamW1e-4 to1e-6.
- Main generated `prepared/priors.json` once from all18 frozen TRAIN
  `n/winner` arrays (81,000/problem). Main-reported table identity:
  `690a8623d0f1f3ea85392aa4f5c1360ec29409b8b2e4c8c21784992bb00feb82`.
  The file SHA256 and content identity are distinct. Reporting verifies the
  file against the main lock and records source identity, TRAIN scope,
  (problem, exactN), smoothing specification and fixed-buffer semantics.
  This small learned classifier table is not a per-instance feature cache.
  No prior CLI flag, refitting, current-query labels or solver outputs.
- Private R80/R84/R85/R87 report helpers retain full18 ALL/per-problem
  Top1/cost/SBS/Oracle/ratios/actual-regret, TRAIN-fitted SBS and task-equal ALL.
  Same unique whole18 VAL checkpoint across full TRAIN/VAL/TEST/LIB/timing.
  TEST is development-exposed; no TEST/LIB reselection or per-problem assembly.
- Keep fixed36 true rescues/losses, false picks and aligned net cost/regret;
  combined choice-versus-native-winner distributions; NSS4 same-checkpoint
  Top1/cost table; saved R84A/R89A/R89B references, optional R94B.
- Timing uses `prior_lookup_and_residual_computation_included=True`, all18
  tasks, 128 frozen TEST inputs/task, three repetitions plus one warmup.
  Full GPU-resident forward includes priors/residuals, excludes I/O, H2D and
  solvers; not an end-to-end dataset latency or cached-feature proxy.

## Main Interface

The lazy factory is `code.V4.r108_experiment.load_checkpoint`; selected TRAIN
and VAL capture bindings are verified without rewriting archives. LIB defaults
to batch2. CPU report requires same-lock timing and both LIB receipts before
rendering; completion metadata is written only after all required PDFs exist.

```bash
conda activate easynco
python -B -m code.V4.r108_reporting lib --arm A_prior_ce --device cuda:0 --lib-batch-size 2
python -B -m code.V4.r108_reporting report --arm A_prior_ce --device cpu
python -B -m code.V4.r108_reporting compare --device cpu
```

These are main-owned handoff commands, not executed here. Repeat per-arm
LIB/report for B only through its assigned completed-predecessor pipeline.

## CPU Checks

```bash
conda activate easynco
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -B -m unittest code.V4.test_r108_reporting code.V4.test_r108_launch code.V4.test_r108_goal -q
```

Synthetic tests cover private-module isolation, factory/binding/count checks,
prior provenance and loss labels, all18 timing, fixed36 paired metrics,
NSS4, actual Top1 and combined two-color plots, staged failures and readiness
gates. In-memory plots only; no real report or evaluation receipt is claimed.
CJK font lookup found `NotoSansCJK-Regular.ttc` for Noto Sans CJK SC.
Final result: 45 owned CPU tests passed after the active-gradient-freedom
metadata correction; shell syntax and real runner/report stage CLI parsing
passed. Only existing Matplotlib/Pyparsing deprecation warnings appeared.
All 24 snapshotted R106/R107 source/document hashes remain unchanged.
Read-only predecessor checks returned A=true and B=false, both without errors.
Prior-builder tests are included in `test_r108_model.py`; there is no invented
mandatory standalone `test_r108_priors.py` dependency.

## Prior-Only Reference Follow-Up

`conditional_prior_reference.csv` is produced during CPU reporting from the
same locked TRAIN prior table, saved VAL/TEST `problem__nodes`, and already
loaded native winners, FP64 costs and current model predictions. It uses
`t=0`, fixed FP32 log-priors, the existing unseen-N task-prior fallback and
original-global-ID tie order. No fitting, inference, thresholds or pi tuning.
VAL/TEST each have all18 problem rows and a task-equal ALL row, with prior/model
Top1, mean_cost, actual regret and model gains (positive means better).
The reference is a TRAIN winner-frequency prior, NOT a cost-optimal SBS.
Protocol metadata and the existing diagnostics PDF show ALL gains versus
this reference; no new required PDF or launch/publication gate was added.
Follow-up verification: all 46 owned CPU tests passed, including one focused
prior-reference test and the report CSV integration assertion.
