# R58 TRAIN-Only Throughput Pilot

## Scope

- Owned sources: `code/V4/r58_throughput.py`, `code/V4/test_r58_throughput.py` only.
- Original TRAIN physical inputs only. No old label/cost files or actual val/test inputs.
- Frozen scenario/environments/backends/labels and shared bridge stay unchanged.
- Only gpu03, UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, existing allocation1465.
- No generation, release, locking or training. All measurements stay in this directory.

## Decisions Before Execution

1. Use24 distinct instances per concurrency method (eight each at exact min, upper-median, max physical size). Choose indices using local random seed2 before solving; same fixed set for1/2/4 workers. Warmup reserves one additional distinct instance per scale, reused once by each worker.
2. Each fresh process owns its backend and CUDA context, thread count1 and singleton calls. All recipe settings stay inherited. Reset solver seed2 per instance, including warmups. Workers share only GPU hardware and a read-only task description; each writes its own files.
3. Warm time starts at the parent's common release marker and ends after every worker's synchronized completion marker, including output/check overhead. Cold wall includes process startup, model initialization, all warmups, full measured workload and teardown. No speedup from multiplied singleton times.
4. Independently check every returned tour against the original physical item using the new R58 scalar checker, including warmups and parent rechecks. Exact paid FP64 costs are required for equivalence. Only consecutive trailing depot0 padding is ignored, retaining the terminal depot; internal zeros remain significant. Record raw and physical tour differences.
5. Measure four sequential diffusion candidates on12 fixed TRAIN instances per method, plus one warmup per scale. Set both backend.parameters and model.args to20 denoising/100 two-opt; T2T has1 genuine guided rewrite with10 guided steps and unchanged ratio/checkpoint. Witness actual call counts. This changes algorithm budget; no quality/equivalence claim and no new recipe lock.
6. Whole-device memory is sampled only on the designated UUID at approximately0.5s; record context PID/memory and per-worker torch allocator peaks. Sampled peaks are not exact hardware maxima.
7. Maximum pilot wall budget10800s, per-phase1200s, existing-allocation startup wait600s. A failed width4 stays failed; width2 is eligible only from its own completed equivalence measurement. Never alter batch size or budgets to hide OOM.

## Applicability

Selector val/test accuracy, mean-cost rankings and training loss are not applicable: this pilot deliberately performs no selector training or validation/test evaluation. W&B records offline timing metrics only in this directory.
