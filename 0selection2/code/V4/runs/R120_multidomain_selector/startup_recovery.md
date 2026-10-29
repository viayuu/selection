# Allocation-Network Startup Recovery

2026-10-29 20:23 CST. All 12,970 native MKP labels and all 16 worker shards
completed before the formal full-pool export and preparation succeeded.

The first model launcher reached both allocation-owned containers, but both
containers use Docker's default isolated network. Torchrun waited at static
rendezvous against the physical host address; no host listener was exposed on
29820. No model rank was spawned and no protocol, status, last/best checkpoint
or successful model update existed. This was a transport failure, not a model
result, data rejection or divergence.

Only the two confirmed R120 torchrun processes were terminated: container PIDs
2480437 (gpu05) and 3871074 (gpu03). Both launch exits (241, wrapping SIGTERM)
and original entry/submission/stage evidence are preserved in
`resources/train_startup_attempt1_bridge/`. No other experiment was stopped.

Recovery uses the original R76-style host-network DDP transport, still inside
the same authorized Slurm allocations 1475 and 1465. Each node's GPU step owns
both GPUs, eight CPUs and 18 GiB, with Slurm cgroup identity and the four exact
GPU UUIDs checked at entry. The existing frozen R84 arrays are copied once to
host-local `/tmp` because the container-local cache is a separate filesystem.
There is no preprocessing or solver rerun. The model, labels, initialization,
batch=1024, rank weights, learning rate, validation rule and update budget are
unchanged. The completed export/prepare stages are reused, not repeated.

The CPU pipeline controller is restarted after preserving the failed launch
stage; it does not repeat any successful model update, since none occurred.
