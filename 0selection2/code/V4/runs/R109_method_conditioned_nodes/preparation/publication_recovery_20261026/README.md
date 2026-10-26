# R109 Publication-Only Recovery

The original watcher failed at the inherited R98 publisher's canonical-root
guard on 2026-10-26 at 03:52 Asia/Shanghai, before creating a publication receipt,
acquiring the publication lock, staging, committing, or pushing. Both complete
R109 pipelines and the saved CPU comparison had already finished successfully.

This additive recovery preserves the failed watcher and the frozen model,
runner, reporters, unique VAL-selected checkpoints, and all saved results.
It does not retrain, infer, re-TEST, remeasure timing, re-evaluate LIB, regenerate
reports, use CUDA, or touch R113/R114. The independent saved-evidence gate accepts
only the exact known administrative error; it never claims the original watcher
succeeded. Completion of publication is not achievement of the research goal.

The exact manifest contains R109 source, small metadata, CSV, and standalone PDF
reports. Existing shared dependencies must already match the parent commit.
Weights, caches, per-instance prediction/cost/label arrays, runtime packages,
credentials, logs, and other experiments are excluded. Git's existing line-ending
filters are honored without rewriting original worktree files.
Raw dispatch claim, entry, and controller receipts are withheld because command
arguments may contain allocation access tokens. `dispatch_summaries.json` retains
their original hashes and non-sensitive controller/worker stage return codes;
the original receipts remain unchanged locally.

Planning and publishing both hold `/public/home/shiys/.git/codex-publication.lock`.
Publishing additionally requires main's explicit GO bound to the final manifest
SHA256. It rechecks the empty staging index, canonical root, existing `main`,
actual origin/main, absence of a prior R109 publication commit or attempt,
source closure, saved pipeline gate, selected checkpoint bytes, and every asset.
There is one commit attempt and one ordinary push attempt, with no automatic
retry, reset, cleanup, amend, force push, or rewriting of prior failure evidence.

`publication_plan.json` is precommit intent, not proof of a successful push.
The final `publication_receipt.json` is intentionally uncommitted because it
records the newly created commit and independently checked post-push remote.
