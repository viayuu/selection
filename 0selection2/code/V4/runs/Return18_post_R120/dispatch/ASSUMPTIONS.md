# Resource Transport Decisions

This worker owns only the three new Return18 transport files and this dispatch
directory. No GPU launch, Git operation, SSH, training defaults, or edits to
R120 and other workers are part of implementation.

| ID | Decision | Contract | Source |
| --- | --- | --- | --- |
| T01 | Literal commands | Main supplies an approved, hashed finite JSON plan and one existing hashed protocol. No generated training argv or automatic retries. | user |
| T02 | Slots | gpu03_0, gpu03_1, gpu05_0, gpu05_1 use the immutable R120 node constants and original R81 UUID mutex files. | user |
| T03 | Waiting versus attempts | Only explicit pre-attempt resource/prerequisite waiting is repeatable. Creating attempt.json permanently consumes the job; nonzero stages stop that job. | interface default |
| T04 | CPU binding | Each lane gets a deterministic disjoint four-CPU slice of the allocation/container intersection; CPU-only Slurm steps request 4 CPUs and 9G. | user |
| T05 | R120 release | Complete TEST, both saved LIB identities and lib_complete report must match the unique checkpoint lock; publication is irrelevant. | user |
| T06 | Cache | Verify all 54 train/val/test task caches. Reuse complete container-local caches. Missing TEST uses shared R84 TEST if present, otherwise R120 existing_test_cache/test only after its complete.json proves frozen container provenance; an explicit hashed test_sources override is also supported. Never preprocess or stream binary ai_docker stdout. | user |
| T07 | Command interface | Python module/script argv is literal and source-pinned; output roots stay under R121-R124 or Return18. The transport never changes loss, caps, protocol or CUDA arguments. | interface default |

Acceptance is CPU tests only. Real Slurm/container/cache behavior remains
unverified until the parent explicitly dispatches after R120 completion.
