# Build Note

1. Atlas endpoint/authentication and provider-isolated caches.
2. Complete-view corpus with overflow-only AST/call-aware splitting.
3. All-component bundle and trainable aggregation; legacy checkpoint compatibility.
4. Offline checks and real-source upload plan (no upload).

## Run Record

Official MongoDB documentation confirms 32,000 tokens per input and 320,000
tokens / 1,000 texts per `voyage-code-4` request. Existing Voyage caches are kept.

- Prepared real-source corpus: 22 solver IDs, 26 deployments, 104 semantic
  views, 80 unique request texts, 159923 total tokens, maximum view 29754 tokens.
  No actual view needs splitting. Historical/default provenance is retained.
- Atlas request defaults: 16 texts, 2000 RPM, 8000000 TPM, at most 320000 tokens
  per request; no upload without the explicit flag and total-token budget.
- Offline checks: 31 original R45/default/request tests and 12 new Atlas tests;
  22 R43/dual-stream/solver-feature compatibility tests. All passed.
- A temporary, simulated full pipeline assembled a locked component bundle and
  replayed a self-contained checkpoint on TSP/CVRP/ATSP/OVRPTW without network.
  Those fixtures were deleted with their temporary directory, not saved as results.
- Both launchers passed `bash -n`. The dedicated no-argument launcher printed
  the real upload plan without reading an API key or sending requests.
- The bounded same-family review found future-overflow call-identity/context
  issues. Fixed by building call context from original callable identities,
  retaining AST metadata, splitting nested statement blocks with enclosing
  class/function/control-flow context, and budgeting repeated constructor state.
  Added regression tests for original call names, conditional branches and
  class initialization. Default ready-corpus producer/consumer paths align;
  draft and legacy preparation use separate defaults. Static call resolution
  remains intentionally limited and is recorded in A-008.
- Final combined command: `PYTHONWARNINGS=ignore::DeprecationWarning python -m
  unittest code.V4.test_solver_code_atlas code.V4.test_solver_code_defaults
  code.V4.test_solver_code_requests code.V4.test_r45 code.V4.test_r43
  code.V4.test_dual_stream code.V4.test_solver_features`: 65 tests, exit 0.

## Deferred

Paid embedding generation requires the user's Atlas key and upload command.
No selector training is started.

## Blockers

None for offline implementation.
