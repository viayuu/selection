# One Publication Per Completed Return18 Version

The CPU publisher is read-only unless `--apply` is explicit. It never starts,
retries or repairs TEST, LIB, a finisher, training, source code or predictions.
R121 is unsupported until its own finish gate exists; no compatibility receipt
is invented. This implementation has not executed Git or any GPU workload.

## Commands

Run from `/public/home/shiys/0selection2`. Check the actual completed run root:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m code.V4.return18_publish --experiment R122 --root /public/home/shiys/0selection2/code/V4/runs/R122_hard_decision_constraints
```

The parent may queue this literal command only AFTER recovered TEST and locked
LIB/full18 PDF/CSV completion. This is the only Git-writing mode:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m code.V4.return18_publish --experiment R122 --root /public/home/shiys/0selection2/code/V4/runs/R122_hard_decision_constraints --apply
```

For R123 use `--experiment R123 --root .../runs/R123_heterogeneous_moe`;
for R124 use `--experiment R124 --root .../runs/R124_top2_reranking/launch_v2`.
Root means the actual version root, never its `model` or report subdirectory.

## Gates And Scope

- R122: `r122_finish.locked_inputs` and `load_libs`, then
  `model/report_complete.json` and `model/full_model_statistics/protocol.json`.
  A present recovery receipt must be complete for the same lock. Infeasible VAL
  flags, distinct hard TRAIN budgets and VAL-only thresholds remain unchanged.
- R123: `r123_finish.locked` and `require_identity`, including TRAIN/LIB/live
  timing receipts, then `finish/reports/receipt.json`. Its original VAL/TEST
  capture-attribution limitation is retained, not upgraded into a SHA signature.
- R124: `r124_finish.locked_inputs`, same-identity LIB and report receipts, and
  saved prediction/output hashes under `model/finish/reports/finish_receipt.json`.
- Explicit per-experiment code, small configuration/lock/history/report receipts,
  report PDF/CSV/JSON/Markdown, common Return18 launcher/docs/science are eligible.
  All `main_approved_EXPERIMENT_*.json` plans are included, including TEST recovery
  and finish approvals; runtime worker/token records are not included.
  R122 never stages R123/R124/R121 source or run directories. Existing unrelated
  dirty files are neither reset nor staged. No `add --all` is used.
  R124 alone includes its two shared typed-bank dependencies (`r123_meta_data.py`
  and `r123_prediction_cache.py`), not the R123 learner, model or run artifacts.
- New Return18 `R120_closeout` PDF/CSV/JSON/Markdown is included with the next
  completed version; standalone R120 `2330198` is never recommitted. The routed
  follow-up `routing_receipt.json` and its unverified actual-model identity caveat
  are published verbatim, with no provenance rewriting.
- No weights, raw inputs/labels, NPZ/NPY/PKL/JSONL, caches, wandb trees, synthetic
  previews or raw manifests are eligible. The sole manifest exception is R123's
  explicit `ROOT/preparation/manifest.json` bank metadata (currently 104342 bytes),
  not any input manifest. Each eligible file must be below 8 MiB.
  Report input hash maps are metadata; their referenced arrays are NOT staged.

## Git State

Git root is `/public/home/shiys`, branch `main`, mutex
`/public/home/shiys/.git/codex-publication.lock`, shared with R120. `--apply`
refuses any pre-existing staged changes and any unpublished local parent history.
It adds only named allowlisted paths and records the one commit before pushing.
`ROOT/github_publication.json` stays local (not staged) and is complete only after
push returns zero. A failed push retains the commit and failure receipt; a later
explicit `--apply` only retries that SAME HEAD commit. A failed/interrupted staging
or commit attempt requires manual inspection and can never auto-commit again.
An already complete publication creates no further commit or push.
