# R110 Completed-Version Publication Recovery

This is an explicitly authorized, bounded CPU-only administrative recovery.
It does not train, evaluate, render reports, select a checkpoint, or update the
native research goal. Historical runtime, source, reports and metrics are frozen.

The original watcher stopped at 2026-10-25T03:22:04.961361+00:00 with
`Publication requires canonical roots`. The inherited R98 publisher compares
the supplied R110 dispatch root to its own R98 `LAUNCH` constant before touching
Git. R110's own `publication_paths` has the correct canonical checks and is
reused without patching any historical module.

The live historical gate was read once and reported that original watcher
failure as its sole error, with both arms and the comparison ready. Recovery
uses the existing saved complete pre-publication gate and rc0 comparison receipt;
it does not repeat the scientific gate or CPU/model tests. Both the original
failed receipt and the saved gate are retained.
This is a publication-only administrative exception, not scientific success.

`recover_publication.py plan` writes an exact, SHA256-bound asset manifest,
a source-closure receipt and the unmodified saved gate. No new CPU/model tests
or metrics are run, per the user's final instruction. The asset list omits weights,
NPZ/prediction arrays, raw
datasets, W&B, runtime copies, per-panel figure duplicates and all R111 assets.
The only R109 additions are the required model, its CPU tests and its frozen
model-preparation document. Existing shared dependencies must already match HEAD.

`recover_publication.py publish` requires the manifest, an empty index, canonical
`/public/home/shiys`, `main`, matching `origin/main`, and the shared
`.git/codex-publication.lock`. It stages only exact paths, creates one completed
R110 version commit, then performs one normal push and independent `ls-remote`.
No broad add, stash, reset, amend, force, checkout or automatic retry is used.

`publication_receipt.json` and the additive R110 `github_publication.json` are
written after the commit, intentionally outside that single commit so its hash
and actual remote confirmation can be recorded without an amend or second commit.
The committed manifest and plan are pre-publication evidence, not a push receipt.

Run from the project root after `conda activate easynco`, with CUDA hidden and
CPU thread limits. This recovery makes no assertion about backend model identity;
the requested executor is `gpt-6.1-sol` with `reasoning_effort=max`, and no Astra
reviewer or secondary executor is invoked by the recovery.
