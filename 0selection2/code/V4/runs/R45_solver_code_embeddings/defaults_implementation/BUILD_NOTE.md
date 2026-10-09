# R45 Defaults Build

| Rung | Feature | Acceptance | Status |
| --- | --- | --- | --- |
| F0 | Existing corpus/plan entry point | Offline reviewed-corpus plan | Passed: 85 chunks, 154398 tokens, ready=false |
| F1 | Source defaults plus explicit experimental corpus approval | Defaults preparation CLI and targeted unit tests | Passed: 22 solvers, 26 deployments; 17 deployments use missing-field defaults; main manifest unchanged |
| F2 | Persistent provenance and compatibility | New/existing R45 tests and unchanged source-manifest checks | Passed: 24 R45/default tests and 22 legacy regression tests |
| F3 | Prepared-corpus embedding launcher | Shell syntax, default offline plan, and combined generation/assembly opt-in check | Passed: 25 R45/default tests; ready=true plan |

## Run Record

All commands ran in `easynco`, with `HF_HUB_OFFLINE=1` for Python checks.

- `python -m code.V4.solver_code_defaults`: exit 0; 92 chunks, 155543 tokens,
  experimental `ready=true`, `historical_bindings_verified=false`.
- `python -m unittest code.V4.test_solver_code_defaults code.V4.test_r45 -v`:
  exit 0; 24 tests passed.
- `python -m unittest code.V4.test_r43 code.V4.test_dual_stream code.V4.test_solver_features -v`:
  exit 0; 22 tests passed.
- `python -m code.V4.solver_code_embeddings --stage plan --corpus code/V4/runs/R45_solver_code_embeddings/corpus_assumed/corpus.json`:
  exit 0; all 92 chunks remain uncached; 26 historical deployment warnings retained.
- Final accumulated suite:
  `python -m unittest code.V4.test_solver_code_defaults code.V4.test_r45 code.V4.test_r43 code.V4.test_dual_stream code.V4.test_solver_features -q`:
  exit 0; 46 tests passed in 20.558 seconds, including offline selector-checkpoint
  save/reload with the original vector file deleted and API credentials absent.
- `git diff --check`: exit 0.

An initial plan invocation used positional `plan` instead of `--stage plan`;
argparse rejected it before work. The corrected command passed.

No training, solver execution or API request was performed. The cache fixture in
the unit test uses a temporary directory and is never exported as real vectors.

## Assumption Review

Scoped fresh read-only review reported no material findings; this same-family
review remains provisional. It identified that assumption-specific checkpoint
replay was not yet directly tested. The existing temporary bundle test was
extended to save/reload a selector checkpoint without its external vector file
or API credentials, and the final accumulated suite passed. See
`SILENT_ASSUMPTION_SWEEP.json`. This update does not claim historical
reproduction or model benefit.

## Deferred

Historical configurations that cannot be recovered remain unknown. This change
does not establish that defaults reproduce them or improve selector performance.

## API Handoff Follow-Up

The user's follow-up permits defaults or omitted optional metadata and asks for
an embedding-ready handoff. Historical source revision/checkpoint fingerprints
are not required semantic inputs. The prepared corpus is ready without further
history completion.

`run_v4_r45_embedding.sh` explicitly selects the prepared corpus. No arguments
run the offline plan; `--stage all --allow-upload --max-tokens 200000` generates
and assembles vectors only, never training. `--stage all` was added to the
embedding CLI without changing the existing stages or upload/budget guards.

Checks in `easynco`: shell syntax exit 0; default launcher exit 0, 92 uncached
chunks / 155543 tokens / ready=true; combined R45/default suite exit 0, 25 tests
passed in 18.587 seconds. No API request or training was run.

## HTTP 429 Follow-Up

The user's first generation attempt returned HTTP 429. There were no completed
cache entries; the old first request contained 16 chunks / 22004 tokens. The
old backoff waited only 30 seconds before its final attempt. Account-specific
limits were not available from the original traceback, so no definite quota or
billing diagnosis is claimed.

The generator now defaults to at most 4 chunks/request, 3 RPM and 10000 TPM,
with token-bounded batches and a rolling-minute limiter covering retries too.
These are conservative configurable settings, not recovered account limits.
Retry-After is honored; unknown 429 responses wait a full minute. Recognized
quota/payment restrictions, impossible token batches and zero rate limits have
actionable safe errors, without exposing response bodies or credentials.
Corpus, cache identities, backend and existing vectors are unchanged.

Official reference inspected: https://docs.voyageai.com/docs/rate-limits and
https://docs.voyageai.com/docs/error-codes .

Checks in `easynco`: 6 request-specific tests passed; combined request/default/
R45 suite passed all 31 tests in 18.928 seconds after fixing the empty 401 error
response edge case. Default launcher plan still reports ready=true, 92 chunks /
155543 tokens. All HTTP responses in tests were simulated; no real API request
or training was performed.
