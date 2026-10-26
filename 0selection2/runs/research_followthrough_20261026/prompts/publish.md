# Sol Only: Render And Publish One Complete Research Version

Required model: gpt-6.1-sol, reasoning_effort=max. Do not replace Astra's
scientific research with new Sol directions. Read the already saved REPORT.md,
sources.csv,directions.csv and the verified evidence pack. Verify all context
research_hashes still match. If substantive evidence contradictions remain,
block with the exact reason, not a fabricated repair or silent rewrite.

Render a self-contained, downloadable, machine-readable Chinese REPORT.pdf
from the complete report. Inspect the installed CJK font before rendering;
fix missing-glyph/overflow issues without changing scientific claims. Reuse
an existing suitable renderer or keep any new rendering code inside
monitor_root. Astra must not perform this step. No external-image-dependent
HTML as primary output; any optional HTML must be single-file/offline.
Check the PDF, tables, native winner definitions and report/version identity.

Publish ONE completed research version after PDF/CSV readiness. Hold
/public/home/shiys/.git/codex-publication.lock across verification, exact-path
staging, commit and push. Resolve Git root from actual local metadata; preserve
other writers' files and index entries. If the index has unrelated staged
changes, stop for main rather than include/reset them. No add ./-A, amend,
force push, destructive reconciliation, config/provider changes or weights,
raw arrays, secrets, caches. Curate REPORT.md, REPORT.pdf, sources.csv,
directions.csv and a small DELIVERY.md plus verified evidence copies/manifest
needed to make the completed version auditable. Source evidence stays immutable.
Confirm the actual remote received the commit and exact artifact hashes.
For REPORT.md/sources.csv/directions.csv, context.research_hashes uses SHA256
after CRLF-to-LF Git text normalization; PDF remains binary-exact. Preserve
scientific text/cells, use LF for published text, and bind the actual commit
blob hashes. Do not mistake line-ending materialization for a scientific edit.
Include the additive publication_csv_materialization.json, its authority and
the small reconcile_publication_csv.py helper with the completed version; old
strict-byte review/action receipts stay unchanged. Include this monitor's own
small Python/test/wrapper and PLAN/prompts/schema files, but no runtime action
logs, credentials, token-bearing ready files or unrelated worktree changes.
An uncertain prior commit/push requires reconciliation, not blind duplicate
publication. Do not hold this mutex while invoking a publisher that acquires
it internally.

Save research_dir/publication_receipt.json AFTER success with state=complete,
exit_status=0, actual commit, model=gpt-6.1-sol, reasoning_effort=max,
global_mutex=/public/home/shiys/.git/codex-publication.lock,
published_hashes={absolute REPORT.md/PDF/sources.csv/directions.csv paths:
their actual committed SHA256}, source_snapshot_sha256, PDF metadata and
remote confirmation. Include a write-ahead publication attempt receipt before
staging so restart cannot blindly create another commit. If unavailable or
blocked, explicitly report it and do not silently substitute a model.
