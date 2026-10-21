# R83: Model-Only Source Embedding

R83 is one fresh run, not an R80 checkpoint continuation or a new model/loss.
Root: `/public/home/shiys/0selection2/code/V4/runs/R83_model_only_source_embedding`.
Group: `model_only_seed2`. The runner/reporting integration does not generate a
corpus or embeddings, allocate GPUs, or launch background jobs.

## Integration Assumptions

- Main owns the freshly cleaned corpus, model-only manifest and embedding calls.
  Source must contain only solver MODEL `nn.Module` code, excluding config,
  training, solver inference and environment code. The runner checks bundle
  structure/provenance, not the source text or the upstream extraction itself.
- Main supplies `ROOT/model_only_vectors.pt` and its `.lock.json` using the
  existing production R45 Atlas tensor/provenance format (`r45-atlas-v1` or
  `r45-embeddings-v2`). All original API, corpus, cache and chunk trace is kept.
  Encoder/decision masks are true for neural-model candidates; inference/config
  masks are false, their vectors zero, and component indices all `-1` (no chunks).
  These are two active model-only neural code views, encoder and decision, with
  separately recorded source selections, not one whole-model embedding copied
  into two slots. Genuine shared components may still be deduplicated.
- Deliberate exception: `GLOBAL_SOLVERS[4] == 'GLOP'` labels are actually
  `glop_atsp_init_only` (random insertion), with no neural model. All four GLOP
  roles are false/zero with indices `-1`; no unused GLOP network, heuristic or
  inference code is encoded and no vectors are invented. Bundle
  `model_only_scope` declares `kind='model_only'`,
  `active_roles=['encoder', 'decision']` and
  `no_neural_model_solvers=['GLOP']`; additional scope details remain intact.
  Main owns the small
  backward-compatible validator/encoder changes that make the code contribution
  exactly zero for a no-role candidate. GLOP retains its original trainable ID;
  all candidates and global IDs remain unchanged. The PDF states this exception.
- R80D's unchanged `SolverCodeEncoder`, learned solver ID, dual-stream instance
  model, maximin inference and complete R77 objective are retained. No new loss,
  projection cache, model architecture, normalization fit or learned source
  vectors are introduced. Existing bundle-dependent component pooling is kept.
- The locked R77 seed2 scratch initialization, FULL TRAIN geometry/SBS, stable
  solver order and task/source order are reused, not historical trained weights.
  Data defaults to the latest full18 single-RELD release already used by R80D.
- One logical batch is 1024 real rows; budget 49680 successful updates; warmup
  3726; full VAL every 1242 updates; LR 1e-4 to 1e-6. Strict minimum full VAL
  equal-problem `macro_vs_sbs_pct` selects one checkpoint, first wins exact ties.
  TEST/LIB never select checkpoints or inference rules. Prior TEST-informed R&D
  is disclosed; this is not an untouched blind-test claim.
- W&B is required (`selector`, entity
  `yjkds-southern-university-of-science-technology`); R80's explicit offline
  fallback/receipt is reused when online initialization fails.

## Commands

Every shell first activates `easynco`. Run from
`/public/home/shiys/0selection2`. The wrapper also activates that environment.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_v4_r83.sh describe
```

The following are future operator commands, not commands executed during this
integration. For a long run, use an operator-owned tmux session and externally
assigned `CUDA_VISIBLE_DEVICES`; no queue/allocator is added here.

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r83.sh train --device cuda:0 --gpu-type 3090
bash /public/home/shiys/0selection2/code/V4/run_v4_r83.sh report --device cpu
```

`train` finishes the fixed budget, locks the unique VAL-best model, then runs
full TEST. `test` resumes/reuses only that immutable lock; `train --resume`
continues the same protocol at an R80-supported interval boundary. `report`
requires completed TEST, performs/reuses same-lock TSPLIB/CVRPLIB inference, and
generates PDFs/CSVs without training or TEST inference. Use the same LIB device
and batch size to retry unfinished LIB inference. `status` only reads receipts.

All commands accept `--root`; an omitted `--embeddings` resolves to that root's
`model_only_vectors.pt`. Training also accepts R80's `--prepared-root`,
`--data-root`, `--cache-root`, `--state-hash`, microbatch and logging options.
Reporting accepts `--nss-root`, `--device`, `--lib-batch-size` (1 or 2).

## Delivery

Outputs are confined to `ROOT/model_only_seed2/`. The checkpoint, protocol,
evaluation identities, W&B name/ID, locks and reporting factory all identify R83.
Checkpoint loading restores source/ID/geometry buffers offline from the state;
it needs neither the original embedding file nor an embedding API.

`full_model_statistics/` contains `MODEL_STATISTICS.pdf`,
`selection_distributions.pdf`, `training_and_method_diagnostics.pdf`, per-problem
figures, `metrics_by_problem.csv`, `nss_comparison.csv`,
`selection_distribution.csv`, `single_method_comparison.csv` and provenance.
TSP/CVRP distribution pages show TEST then LIB; other problems show TEST.
Each dataset has model/winner bars together on one axis with fixed candidate
order, counts/percentages and native-winner model Top1 in its title. Native/FP64
`ind` is primary; FP32 argmin is separate auxiliary information.

VAL and TEST report ALL plus all18 problems: Top1, mean cost, TRAIN-fitted SBS,
legal-pool oracle, vs_SBS, vs_Oracle and actual regret. ALL is equal-problem,
not a cross-problem raw-cost ratio; actual regret is the mean per-instance
relative regret, not a ratio of means. Saved R79 NSS comparisons cover TSP TEST,
CVRP TEST, TSPLIB and CVRPLIB with explicit missing entries; NSS is not retrained.

## CPU Verification

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest code.V4.test_r83_experiment code.V4.test_r80
bash -n /public/home/shiys/0selection2/code/V4/run_v4_r83.sh
```

Tests use small synthetic CPU fixtures and temporary outputs only. They do not
run training, real-data evaluation, embedding calls or GPU jobs. No old output
is rewritten and no commit is part of this integration.

Integration check record (2026-10-17): 11 existing R80 CPU tests passed; the 5
independent R83 CLI/status/delegation/W&B/grouped-PDF tests passed in 7.060s.
Wrapper `describe`, `report --help`, shell syntax and tracked diff whitespace
checks passed. After main's standard validator/encoder hooks landed, the 6
remaining GLOP-aware bundle/model/protocol/lock tests passed in 7.671s, using the
actual standard validator and factory, not mocks replacing those dependencies.
They verify exact ID-only GLOP output before/after offline checkpoint restore,
full scope and corpus/cache/request/chunk provenance through bundle load,
protocol JSON and checkpoint serialization, and unchanged lock protections.
Only these 6 focused CPU tests were run for the follow-up; no actual corpus,
embedding generation, training or real-data evaluation was run.

Production-bundle readiness smoke (2026-10-17): PASS, 0.507s excluding imports.
The actual locked `model_only_vectors.pt` loaded with 34 real Atlas chunks,
58,859 input tokens, maximum 4,634 tokens/chunk, zero split views and mask sums
`[21, 21, 0, 0]`. All 21 neural candidates have two active views and distinct
encoder/decision chunk-ID sequences. GLOP has no active source roles.
Using the actual locked R77 seed2 scratch initialization and geometry statistics,
CPU initialization and a temporary offline checkpoint roundtrip restored the
complete model state exactly, preserved full provenance/scope, and reproduced
all solver-encoder outputs including exact GLOP ID-only behavior. The bundle
hash remained `4a9f59442fead49c22099f8a1888aec1973b9deb7eb3deec7052e9477f38d109`.
No real instance cache, training, GPU initialization, API call or old-output edit
was involved. Shared runner/report/core files are frozen for launch; this
README does not participate in the launch source hash.
