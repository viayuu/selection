# Portable two-3090 queue recovery

Only the two authorized RTX3090s in allocation1465 on gpu03 are used:

- `GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50`
- `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`

The historical queue's four failures occurred before worker/CUDA initialization.
Its receipts remain under `preparation/migration_3090_20261026/` unchanged.
An actual gpu03 CPU entry established that the shared lock inode is identical,
while NFS `st_dev` is2065 on admin01 and46 on gpu03. The launcher now compares
the original inode across hosts and still compares both local device and inode
between the opened descriptor and path, with no symlinks or inode replacement.
See `RECOVERY.md` and the20 CPU regression tests. No scientific source changed.

Main approved check SHA256
`d3e840cf4945ecfcda330a46acb36c1e891a42f5776134e10c7ade0b1b529781`.
Controller tmux: `r113_3090_portable_20261026`; state: `controller.json`.

Actual training, not merely readiness, was confirmed on2026-10-26 after11:46
Asia/Shanghai: A_scratch exceeded396 successful updates and C_transition
exceeded20538. Both covered every one of the18 problems with finite loss and
gradient norms. The instantaneous authorized-GPU snapshot was79%/62%
utilization,14662/14442MiB,66/61C; these are observations, not a claim of
constant saturation. Worker receipts bind the exact UUID, allocation and
allocation-owned container.

A_scratch starts from the original fresh initialization. C_transition and then
B_snapshot resume persisted update20160 with optimizer/scaler/RNG/data-stream
state. Previously observed unsaved updates are replayed, not claimed recovered.
Their historical protocol field `gpu_type=4090` is retained for exact resume
equality; actual new worker receipts prove RTX3090. Completed process pretraining
is reused. No4090 job is restarted. R114 full-pool response acquisition waits
behind the selector jobs and is not yet a trained downstream selector.

The original plan, model, loss, seed, sampling, microbatch caps, update budget,
validation selection and locked TEST/LIB rules are unchanged. No initialized
training auto-retry is enabled. CPU-only finishers perform report/publication
after each required pipeline finishes. New attempts and original failed
attempts must remain distinguishable in the final report.

The persistent followthrough contract points here from
`runs/research_followthrough_20261026/queue_handoff.json`. It must keep checking
actual receipts/progress, finish evidence-backed version publications, then
use Sol for empirical synthesis and Astra only for paper research/next ideas.
This phase closeout is not completion of the ongoing native research goal.
