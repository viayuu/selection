# R75 Runtime IO Optimization

This resumes the existing one seed2 run at successful update16860. No logged successful updates were discarded or replayed; at most one uncommitted in-flight group may have been discarded. The checkpoint, optimizer, scaler, RNG and original logs are preserved.

| Measurement | Before | After |
|---|---:|---:|
| Mean full three-task update (s) | 2.3715 | 0.4081 |
| Rows/s | 485.8 | 2822.9 |
| Mean GPU utilization, short sample (%) | 18.17 | 94.25 |
| Major faults in process probe | 2037 | 0 |
| Process input disk read in probe (bytes) | 4714536960 | 0 |

Observed update speedup: 5.81x. The120-update windows contain each of18 tasks20 times; the first6 cold restart updates are excluded. Batches differ, so this is an observed runtime comparison, not an exactly paired benchmark. GPU utilization is12 one-second samples, not sustained100% utilization.

The16GiB old process hit its memory ceiling and repeatedly faulted the22.11GB padded mmap inputs. The new19GiB process owns14.03GB losslessly packed TRAIN/VAL inputs in anonymous RAM, with small bounded pinned staging buffers. No quantization, relabeling or candidate changes are involved.

## Warm Stage Timing

| Stage | Mean Seconds |
|---|---:|
| gather_seconds | 0.009815 |
| pin_seconds | 0.001330 |
| h2d_seconds | 0.001017 |
| update_wall_seconds | 0.395663 |
| update_stream_span_seconds | 0.395631 |

CUDA stream span includes CPU launch gaps; it is not pure kernel busy time. The24 instrumented updates are official training updates, not an additional experiment.

## Execution

Current tmux: `r75_train_fast1`; monitor: `r75_monitor`; Slurm step1465.125 on the same approved GPU UUID. Source scientific files are unchanged; the IO wrapper is separately hashed in `runtime_started.json`. Four real CPU batches matched every field, dtype and padding exactly.

Epoch11 is a mixed pre/post-resume epoch. Its saved optimization.seconds reports only post-resume wall time; the saved prefix step time is in `quiesced.json`. Final compute metadata separates the last process segment from end-to-end elapsed time.

## Added Files

| File | Purpose |
|---|---|
| r75_fast_input.py | Lossless packed RAM batches and bounded pinned staging |
| r75_fast_runner.py | Resume existing run, replace input factory, profile24 real updates |
| run_v4_r75_fast.sh | Same GPU/allocator/environment;19GiB step |
| r75_checkpoint_boundary.py | Capture a logged durable state without truncation/replay |
| r75_runtime_probe.py | Read-only12-second process/IO samples and memory sizing |
| r75_runtime_report.py | Generate this report from saved observations; no forwards/replays |
