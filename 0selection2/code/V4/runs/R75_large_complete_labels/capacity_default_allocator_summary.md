# R75 Batch Selection Only

Formal training, validation and TEST have NOT started. Await explicit user batch approval.

Model: unchanged R65B native-utility dual-stream selector; seed2. All sizing states are discarded.
Full new TRAIN has1,786,887 rows. Only deterministic real TRAIN subsets are used here.
Two maximum-size stress updates precede six representative mixed-size updates covering all18 tasks.
The minimum conservative physical-memory headroom is3GiB. Larger batch is preferred only within95% of best safe throughput.

| Per Task | Effective | Allocated GiB | Reserved GiB | Headroom GiB | Mixed Instances/s | Safe |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 768 | 12.970 | 16.721 | 6.544 | 2746.6 | True |
| 384 | 1152 | 19.391 | 22.223 | 1.042 | not measured | False |
| 320 | 960 | 16.166 | 22.682 | 0.583 | not measured | False |
| 288 | 864 | 14.554 | 20.369 | 2.896 | not measured | False |

Recommendation: per-task batch256 / effective batch768.

Calibration cost: 11 successful scratch updates, 9120 native presentations, 0 AMP retries, 0 CUDA OOM attempts.
These are capacity measurements, not a loss/learning-rate/model search or performance result.
CPU input/geometry preparation is excluded from steady-state timing and reported separately in capacity_data.json.
Current-batch CPU slicing, H2D, complete updates and retry work are included in measured update time.
Historical geometry normalization is used only for sizing; formal training requires new TRAIN statistics.
Future formal training must use new sample-count-aware schedules, fresh seed2 model/optimizer/scaler and no sizing weights.
The original native70 goal remains paused. R74 was terminated at user request; disk files are preserved.
Legacy eight B-task semantics remain deferred/unfixed. No Git commit is created at this batch-selection stage.
