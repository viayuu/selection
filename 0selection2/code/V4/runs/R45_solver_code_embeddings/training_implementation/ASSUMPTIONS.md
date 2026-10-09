# R45 execution decisions

| ID | Decision | Source |
|---|---|---|
| A-001 | Retain all three planned arms; C tests semantic ownership rather than additional projection capacity alone. | User's R45 plan |
| A-002 | Use seed=2 only, the exact R43A model/loss, and its validation-cost selection rule. | User's R45 plan |
| A-003 | Reuse the locked Atlas vectors and accepted source defaults; do not represent missing historical recipes as verified. | User approval and embedding provenance |
| A-004 | Use existing Slurm job 1465 on gpu03, not the GPU-less admin01 host; allocate no new hardware. | Runtime discovery |
| A-005 | Keep batch128; concurrency is across experiments, not gradient accumulation or data parallelism. | User's R45 plan |
| A-006 | Keep W&B offline because the existing environment has no configured account credentials; local curves remain authoritative. | Existing experiment convention |
| A-007 | Use disposable full-batch updates for CUDA preflight; discard these models and reinitialize for actual training. | Execution decision |
| A-008 | Reuse R43's saved R42A reference predictions for train/val, with costs/winners/pool IDs checked on loading; do not copy test predictions. This avoids redundant inference and concurrent cache writers. | Execution decision |
