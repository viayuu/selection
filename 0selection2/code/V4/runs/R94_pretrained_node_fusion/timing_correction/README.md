# R94 Timing Correction

This is an additive erratum for frozen R94 version `cf2f06bc4`. It does not
modify the old timing receipts/PDFs, model sources, checkpoints, selection rules
or prediction files. Both A and B are complete. Only timing is in scope.

## CPU Checks And Report

Run from `/public/home/shiys/0selection2`. Every command activates `easynco`.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s code/V4/runs/R94_pretrained_node_fusion/timing_correction -p 'test_*.py' -v
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.diagnostics.r94_online_timing_recovery cpu-smoke
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.diagnostics.r94_online_timing_recovery report
```

`cpu-smoke` audits the fixed128 inputs for all18 tasks, then loads both official
fixed checkpoints on CPU and checks four input
paths (CVRP, OVRPBLTW, TSP, ATSP) with one TEST input per path. This is functional
testing only, not a reported latency benchmark. No GPU context is initialized.

## GPU Command, Only After User Approval

Do not run this until the user approves a particular GPU and time window.
Set `R94_TIMING_GPU_UUID` in the shell to that approved single GPU UUID. A/B run
serially on that one card, not four cards. Prefer a tmux shell for execution.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="${R94_TIMING_GPU_UUID:?Set the user-approved single GPU UUID first}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python -B -u -m code.V4.diagnostics.r94_online_timing_recovery measure --arms A B --device cuda:0 --gpu-authorized
```

After the GPU process exits, run the CPU `report` command above. It fills the
new PDF/CSV from the corrected receipts; before measurement those fields say
PENDING, never zero. Existing corrected GPU receipts are not overwritten.

## Measurement Boundary

- Exactly the first128 TEST input rows per task, in original order, all18 tasks.
- Each scope gets one warmup and three synchronized wall-clock repetitions.
- Native donor: unmodified frozen FP32 encoder wrapper, no decoder; includes
  native validation/packing, excludes the FP16 feature cast.
- Selector: unmodified selector with resident FP16 features freshly generated
  by the online donor, never loaded from a donor sidecar.
- Online total: independently measured unmodified model(raw), with all nco_*
  fields explicitly excluded; includes donor, original FP16 cast and selector.
- Native wrapper/encoder call counts must match on every warmup/repetition.
  Two integer forward hooks add small included instrumentation overhead.
- TSP/ATSP are raw-only. Their donor is not applicable, with zero executed calls.
- All I/O, checkpoint loading, hashing, collation, local-geometry preparation,
  transfer and validation checks are outside timing. Geometry is prepared using
  the frozen FP64 formula on CPU because the old container /tmp caches are gone.
- Batch/128 is amortized throughput cost, not independent single-query latency.
  Independently measured total must not be replaced by a sum of scope means.

`TIMING_CORRECTION.pdf` is the primary report, with Markdown and CSV companions.
`legacy_timing_preserved.json` retains both old receipts, unedited, and hashes.
`preservation_before.json` hashes existing frozen sources/checkpoints/reports
and checks size/mtime identities of prediction arrays (without rereading their
6GB payloads). A mismatch is reported, never reverted.
