# User-Requested Progress Snapshot

This is a progress publication, not a completed R121 model release and not a
claim that the active Return18 research goal has been achieved.

- R120, R122, R123 and R124 already have completed, published locked-model
  evaluations. `results_so_far.csv` reuses their saved routing18 VAL/TEST metrics.
- R120 additionally handles six nonrouting tasks. Their any-exact-best metrics
  remain separate from native routing18 Top1; see the existing R120 closeout PDF.
- R121 remains incomplete at this snapshot. `snapshot.json` records the observed
  collection/continuation state without manufacturing missing model metrics.
  Label counts are not optimizer-update counts. Copied records are not new solves.
- R121's new collection and numerical recovery leave the historical benchmark
  labels, candidates, weights and native decoding budgets unchanged. The old
  failures remain available on the server; R58 is still deferred.
- The completed MoE support analysis is an answer-informed offline diagnosis,
  not a deployable selector or a general accuracy ceiling.
- The frozen-selector-mixture research PDF and source bibliography are included.
  They do not authorize an additional R125 experiment.

Publication includes related small source, tests, protocols, documents, PDFs and
aggregate CSVs only. Model weights, raw inputs/labels, large input/deployment manifests,
feature/prediction caches, transport tokens and runtime logs remain server-side.
Unrelated worktree changes are deliberately left untouched. Running native jobs
and the CPU continuation are not interrupted by this Git checkpoint.

`publication_manifest.json` lists the selected files and their precommit hashes;
it excludes its own hash to avoid a self-reference. Mutable runtime states are
represented only by this snapshot, not presented as completed-stage evidence.
The repository uses `core.autocrlf=input`; Git can normalize CSV CRLF to LF.
The manifest preserves server-byte SHA256 and separately records such staged
normalization. Numeric table values and the live server files are unchanged.
