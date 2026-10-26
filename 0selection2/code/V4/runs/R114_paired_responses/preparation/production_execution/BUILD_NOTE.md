# Production Queue Build

| Rung | Check | Status |
| --- | --- | --- |
| F0 | queue schema CLI to artifact, no GPU code invoked | passed |
| F1 | saved exact-source queue, explicit null binding, runtime family/deployment gates | passed |
| F2 | CPU gate tests and original suite; complete 512-base conversion and independent source lock | passed |

CPU conversion ran in tmux r114_prepare_production on admin01, affinity [0,1],
numerical libraries one thread. Completed in 206.1 seconds: 512 bases, 2048
endpoints, 1536 pairs, 9600 label tasks; zero rejections or preparation failures.
No old locks or processes were modified.
Failure records belong to production_execution, construction rejections to
production_inputs/rejections.json. Neither creates replacements.

No verifiable designated-model reviewer is available; no Astra review or silent
model substitution is used. Deterministic CPU checks only.

Original 27 tests plus 12 queue tests passed together, 39 tests / exit 0.
Runtime execution tests use mocked subprocesses and receipts in deleted temporary
fixtures only. No production attempt or GPU binding was created.

The original data module and tests were not edited, preserving the old smoke
source lock. The initial unexecuted queue draft is retained as superseded after
adding immutable hardware binding; the final plan is label_queue_v2/plan.json.
All 16 protocol-family groups and all 100 deployment/bucket gates are unqualified
until real probes, not certified by these CPU tests. GPU calls remain zero.
