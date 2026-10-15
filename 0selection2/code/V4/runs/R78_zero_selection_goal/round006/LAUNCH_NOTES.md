# Round006 Launch Notes

The two authorized CPU checks passed (0.844 seconds). An initial attempt had
failed before either test body could run because this PyTorch stable-sort
overload requires keyword `dim=1`; the one-line compatibility fix passed the
same two tests. Both attempts remain in `cpu_checks.log`.

The first Slurm launch attempts exited before any GPU process or formal
update: this installed Slurm rejects `--gpu-bind=none` as `gpu:none`.
Removed that optional parameter only. Explicit single4090 CUDA UUID, existing
job1475, exact step, CPU4/mem9G and all scientific protocol/budget remain fixed.
The launcher hash is refreshed before training; no extra synthetic check,
resource query, GPU smoke, benchmark or optimizer update was added.
