# R75 Batch Selection Only

Formal training, validation and TEST have NOT started. Await explicit user batch approval.

Model: unchanged R65B native-utility dual-stream selector; seed2. All sizing states are discarded.
Full new TRAIN has1,786,887 rows. Only deterministic real TRAIN subsets are used here.
Two maximum-size stress updates precede six representative mixed-size updates covering all18 tasks.
The minimum conservative physical-memory headroom is3GiB. Larger batch is preferred only within95% of best safe throughput.

| Per Task | Effective | Allocated GiB | Reserved GiB | Headroom GiB | Mixed Instances/s | Safe |
|---:|---:|---:|---:|---:|---:|---|
| 384 | 1152 | 19.407 | 19.709 | 3.556 | 2839.5 | True |
| 512 | 1536 | 22.774 | 23.084 | 0.181 | not measured | False |
| 448 | 1344 | 22.607 | 22.980 | 0.284 | not measured | False |
| 416 | 1248 | 20.994 | 21.449 | 1.816 | not measured | False |

Recommendation: per-task batch384 / effective batch1152.

Calibration cost: 21 successful scratch updates, 20928 native presentations, 0 AMP retries, 1 CUDA OOM attempts.
These are capacity measurements, not a loss/learning-rate/model search or performance result.
CPU input/geometry preparation is excluded from steady-state timing and reported separately in capacity_data.json.
Current-batch CPU slicing, H2D, complete updates and retry work are included in measured update time.
Historical geometry normalization is used only for sizing; formal training requires new TRAIN statistics.
Future formal training must use new sample-count-aware schedules, fresh seed2 model/optimizer/scaler and no sizing weights.
The original native70 goal remains paused. R74 was terminated at user request; disk files are preserved.
Legacy eight B-task semantics remain deferred/unfixed. No Git commit is created at this batch-selection stage.
