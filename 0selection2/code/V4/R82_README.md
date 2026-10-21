# R82B: Context-Matched Relative-Regret Ranking

Only `B_context_matched_rank_seed2` is prepared. This is the requested revised
R80B proposal, not the existing R80 batch512 arm or R81 global winner-AUC arm.
R80/R81 code, runs and running jobs are unchanged. No R82A is scheduled.

## Fixed Model And Protocol

- Original R77B complete selector: handcrafted34 + trainable solver ID, original
  instance encoder, synchronous dual stream, utilities, score differences,
  maximin logits, stable global-ID tie break. No inference adjustment.
- Fresh shared seed2 initialization, not trained R77B/R81 weights.
- All 18 current tasks and unchanged legal candidate pools. No new labels,
  solver executions, embeddings, expert heads, feature queues or resampling.
- Batch1024, 49,680 successful optimizer updates; all tasks retain their
  original R77 seed2 order and full-population shuffle streams.
- AdamW LR1e-4, WD1e-4, dropout0.1, clip1. Warmup3,726 updates then cosine LR
  to1e-6. Original CE/comparison/risk/R-Drop unchanged.
- Added CMR weight: `0.10 * min(1, update / 3726)`. Original R-Drop ramp remains
  `0.35 * min(1, update / 3726)`; the two coefficients are distinct.
- Full validation every1,242 successful updates. Select one checkpoint by
  strict minimum validation `macro_vs_sbs_pct`, using common FULL TRAIN-fitted
  SBS. No early stopping or test-driven selection. Full TRAIN eval every fifth
  validation point, plus selected-best TRAIN if absent.

## Context And Supervision

`r82_context.py` fits ten quantile bins on TRAIN log(real size), merging equal
cutpoints. Equal physical sizes are never split by arbitrary row position.
Recorded `distribution` and, except TSP/ATSP, `capacity_type` complete each
problem-local context. No category is inferred from an instance index.

Raw FP64 costs give `r=(c-min(c))/min(c)`. TRAIN group means are saved for
interpretation; inside the same group, subtracting the same mean from each
instance cancels exactly. The loss uses original FP64 regret differences.

For each method, all unequal-regret unordered pairs in one context are ranked
by `a_m=u_m-logsumexp(other legal utilities)`. Pair weight is
`clip(abs(r_i-r_j)/0.01,0,1)`. Weighted losses are averaged by pair COUNT, not
sum of weights, then over nonempty contexts per method and over participating
methods. This is not the fixed eligible-method normalization used in R81.
Two dropout views receive separate CMR losses and their average.

`SupervisedCache` only attaches TRAIN group IDs and FP64 regrets; the runner
pops these before model calls. All original supervision remains loss-only.
`CachedCMRObjective` operates on the complete logical1024 utility leaves and
replays microbatches with exact saved dropout RNG. One batch produces one
optimizer update; AMP overflow retries the same batch and random state.

Main-owned support files are under `support/`. `prepared/pair_coverage.json`
records eight unchanged natural batch1024 draws per problem. The runner
requires its sufficient-coverage flag before formal training. No TRAIN-only
preparation reads heldout costs or labels. Historical R77B validation zero/
near-zero methods are frozen separately for diagnostics, never for sampling
or training weights.

## CLI

All commands run from `/public/home/shiys/0selection2` after:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
```

Preparation is idempotent after the main context helper has finished:

```bash
python -m code.V4.r82_experiment prepare
python -m code.V4.r82_experiment describe --gpu-type 4090 --eval-batch-size 128
```

The externally owned launcher binds an authorized idle GPU and runs:

```bash
python -m code.V4.r82_experiment train \
  --group B_context_matched_rank_seed2 --device cuda:0 --gpu-type 4090 \
  --cache-root /tmp/shiys_R76_R45A_single_reld_mt_cache \
  --eval-batch-size 128 --wandb-mode auto
python -m code.V4.r82_experiment test \
  --group B_context_matched_rank_seed2 --device cuda:0 --gpu-type 4090 \
  --cache-root /tmp/shiys_R76_R45A_single_reld_mt_cache \
  --eval-batch-size 128 --wandb-mode auto
python -m code.V4.r82_reporting --device cuda:0 --lib-batch-size 2
```

Use `--gpu-type 3090` when assigned that GPU. Both runner and reporter accept
`--root` for the fixed run directory. Defaults use
`code/V4/runs/R82_context_matched_ranking/`. `train` already includes locked
full TEST; the separate `test` command is an idempotent completion check.
Training is not initiated by `prepare`, `describe`, `status` or reporting.
W&B is enabled; if standard credential resolution fails, a recorded offline
fallback is used. No API credentials are written into source.

## Results And Interpretation

The run directory contains update JSONL, history/metrics/optimization CSV,
per-method/context participation counters, best/last checkpoints, native-FP64
predictions plus raw utility, one immutable VAL checkpoint lock, and TEST.
The reporter uses an explicit R82 checkpoint factory for both LIBs, never an
R81 or historical-model identity substitution.

PDFs under `B_context_matched_rank_seed2/full_model_statistics/` include:

- `MODEL_STATISTICS.pdf`: VAL/TEST ALL+18, same-model twoLIB, four-dataset NSS
  comparisons. SBS is fixed on TRAIN; Oracle is legal-pool per-instance min.
- `selection_distributions.pdf`: TSP/CVRP four panels (LIB model/winner, TEST
  model/winner), other tasks TEST model/winner, with model Top1 in selection
  panel headings and explicit native/FP64 winner convention.
- `training_and_method_diagnostics.pdf`: optimization/validation traces and
  single-method cost/Top1 references.
- `cmr_training_and_mechanism.pdf`: base+CMR losses, separate CMR ramp, active
  method/row coverage, within-context regret concordance and conditional
  winner AUC, zero/near-zero rescue/recall/precision/false-selection cost.

Machine-readable context/method diagnostics report continuous-regret
concordance separately from winner-positive/negative AUC. They apply frozen
TRAIN context bins and categories unchanged. Unseen heldout combinations
are excluded and their coverage is explicitly reported, never silently
coarsened. Score ties receive half credit; exact regret ties are excluded.
Conditional ROC uses whole score ties at FPR caps, not a fitted deployment
threshold. Historical paired reports count corrected/harmed and both-wrong
cost changes on aligned VAL/TEST; zero/near-zero membership stays the frozen
R77B VAL list.

R77B is a HISTORICAL reference. R82A has not run, so results cannot establish
a strict matched A/B causal gain. Earlier TEST outcomes have informed this
multi-round R&D; TEST is not described as untouched blind evaluation. AUC,
recall or fewer zero-selection methods alone are not acceptance criteria;
the full selector Top1, regret and cost remain decisive.

## Limited CPU Witnesses

```bash
python -m unittest code.V4.test_r82_experiment -v
```

These small synthetic fixtures exercise the real scratch -> cached update ->
VAL checkpoint lock -> TEST path and the complete PDF pipeline on CPU. They
do not read real TEST, run solvers or start CUDA. Small fixture budgets are
patched only inside temporary unittest mocks; formal CLI has no small-budget
override. Core math/gradient equivalence tests are owned separately by main.
