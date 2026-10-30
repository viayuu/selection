# One Publication Per Completed Return18 Version

The CPU publisher is read-only unless `--apply` is explicit. It never starts,
retries or repairs TEST, LIB, a finisher, training, source code or predictions.
R121 uses its actual principal learner delivery gate, not a compatibility receipt.
The R121 extension is CPU-fixture tested only until real final artifacts exist;
READY for the publisher does not mean R121 training/delivery is complete.
No Git or GPU workload was executed while implementing this extension.

Parent-reported existing publications are R122 `870384929`, R123 `547e531ab`,
and R124 `b590ab34e`. Leave their roots, receipts and commits unchanged; this
extension is for one eventual R121 publication, not another commit for them.

## Commands

Run from `/public/home/shiys/0selection2`. Check the actual completed run root:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m code.V4.return18_publish --experiment R121 --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1
```

The parent may queue this literal command only AFTER the original learner's
delivery, the new locked capture/saved-only finish, and four-route analysis have
completed. The final supplement receipt is
`principal/learner_finish/reports/receipt.json`; the base
`principal/delivery_manifest.json` remains required, but is no longer sufficient.
The publisher never runs a finisher or analysis itself.
This is the only Git-writing mode:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m code.V4.return18_publish --experiment R121 --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 --apply
```

For R123 use `--experiment R123 --root .../runs/R123_heterogeneous_moe`;
for R124 use `--experiment R124 --root .../runs/R124_top2_reranking/launch_v2`.
Root means the actual version root, never its `model`, `principal` or report
subdirectory. R121's arm is always `principal`, not `model`.

## Gates And Scope

- R121: `r121_experiment.locked_training`, `r80_reporting.locked_inputs` and
  saved-only `r84_reporting.saved_libs`. Require completed 14400 updates and one
  unchanged principal VAL lock, full18 VAL/TEST identities, both native LIBs with
  the same factory/rule, and the exact ten signed artifacts from the original
  `principal/delivery_manifest.json` under `principal/full_model_statistics`.
  The structural objective PDF, loss, family population, isolation/exposures and
  unmatched historical-control disclosure remain unchanged.
  If present, the locked `train_diagnostic` definition must agree in delivery and
  report protocol; fixed eligible TRAIN-subset curves are not full81000 metrics.
- The new `r121_finish` contract is relayed by main from owner Archimedes
  `01a188eb-cc88-7903-a5d6-a3a9bb90bc43`, documented in
  `preparation/learner_finish_API.md`. Use its read-only `completed_receipt`
  validator for the final supplement, including every recorded input SHA and all
  nine signed PDF/CSV/JSON outputs. Verify both linked capture receipts under
  `principal/learner_finish`, their identical model identity and artifacts, full
  TRAIN81000/task (separate from eligible rows), and 23040 family endpoints with
  zero-potential and TRAIN-only mean-potential baselines. Final reporting is
  CPU saved-only and asserts neither checkpoint reselection nor research success.
  Capture receipts may truthfully record GPU forwards; publication performs none.
- R121 additionally requires the frozen `collection_explicit_cpu_v2` single-root
  export: 113920 labels, 23040 endpoints, 89 deployments, four sealed bootstraps
  and all 89 completed progress states. Verify the export SHA, collection and
  universal seals and pinned sources. Do not reinterpret adopted label provenance
  as new solves. Publish only its small export receipt/protocol/universal metadata,
  not labels, preflight/progress/binding/worker records or the 125MB input manifest.
  No exporter, collector status scanner, feature builder or inference is invoked.
- R121 includes the explicitly named followup/finisher/tests, `return18_results.py`,
  and `return18_moe_support.py` plus its test. Followup publication contains only
  `README.md`, `contract.json`, `learner_launch.json`, and completed rc0 receipts
  `full89_export.json`, `family_cpu_features.json`, `principal_gpu_stages.json`,
  `saved_only_finish.json`, `four_route_analysis.json`. The frozen launch protocol,
  plan, source and immutable-input hashes must still match. Never require or stage
  `publish.json` or post-publication `complete.json`; both are self-recursive here.
  Live status/claims/failures/tokens/logs remain excluded.
- Require `Return18_post_R120/final_results/RESULTS.pdf`, `comparison.csv`, and
  `evidence.json`. Read-only `return18_results.collect` must reproduce the four
  completed version identities/metrics and prior R122/R123/R124 publication
  references, and match the saved evidence, CSV rows and output hashes. This reads
  existing results only; it does not render, infer, publish old versions or write.
- The completed saved-only `Return18_post_R120/moe_support` delivery is explicitly
  `RESULTS.pdf`, `support_summary.csv`, `fixed36_support.csv`, `method_support.csv`,
  `verified_bank_decomposition.csv`, `evidence.json`, plus `CHECKS.json` and
  `RESULTS.md` only if present. The actual analysis uses `evidence.json`, not an
  assumed `protocol.json`; no Markdown report is fabricated when absent. Verify
  its original R123 lock, producer SHA, offline flags and signed aggregate outputs.
  Preserve its non-deployable, answer-dependent oracle limitation; reachability
  is not model accuracy. Never stage `lp_residual_cases.json`, per-instance LP
  data, arrays, or `mplconfig`, even if referenced in evidence or below the size cap.
- R121 also includes exactly `REPORT.pdf`, `REPORT.md`, `sources.csv`,
  `ROUTING.json`, `render_notes.json`, `render_pdf.py` from
  `docs/research/frozen_selector_mixture_information_20261030`. The PDF is the
  standalone primary delivery. No recursive science-directory scan, render log,
  or `formula_assets` originals are included; identity caveats remain verbatim.
- R122: `r122_finish.locked_inputs` and `load_libs`, then
  `model/report_complete.json` and `model/full_model_statistics/protocol.json`.
  A present recovery receipt must be complete for the same lock. Infeasible VAL
  flags, distinct hard TRAIN budgets and VAL-only thresholds remain unchanged.
- R123: `r123_finish.locked` and `require_identity`, including TRAIN/LIB/live
  timing receipts, then `finish/reports/receipt.json`. Its original VAL/TEST
  capture-attribution limitation is retained, not upgraded into a SHA signature.
- R124: `r124_finish.locked_inputs`, same-identity LIB and report receipts, and
  saved prediction/output hashes under `model/finish/reports/finish_receipt.json`.
  Its seven CSVs have a fixed allowlist and separate `LATE_PUBLICATION_HASH`
  metadata; the original PDF-only receipt did not sign those CSVs.
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
  previews or raw manifests are eligible. Explicit manifest exceptions are R123's
  `ROOT/preparation/manifest.json` bank metadata (104342 bytes at implementation)
  and R121's `ROOT/principal/delivery_manifest.json` finish receipt, never an input
  manifest. Each eligible file must be below 8 MiB.
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
