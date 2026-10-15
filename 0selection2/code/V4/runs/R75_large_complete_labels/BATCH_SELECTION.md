# R75 Batch Selection: Awaiting User Approval

## Recommendation

Use per-task **batch384**, effective **batch1152** across three distinct tasks,
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before CUDA starts.
Do not start formal training, validation or TEST until the user approves this
configuration. One seed2 full18 run is planned; no baseline retraining is started.

The selected configuration completed two maximum-size stress updates and six
real mixed-size updates covering all18 tasks. Peak allocated memory was19.407GiB,
peak reserved19.709GiB, conservative device peak20.032GiB and remaining physical
headroom3.556GiB. Mixed-size short-run throughput was2839.5 instances/s, including
CPU cached-row slicing, H2D, both forwards, backward, gradient reduction,
clipping and one official ScheduleFree step. This is not a long-run throughput
guarantee or an accuracy result.

## Measured Expanded-Segment Configurations

| Per-Task Batch | Effective Batch | Peak Allocated GiB | Peak Reserved GiB | Physical Headroom GiB | Mixed Instances/s | Decision |
|---:|---:|---:|---:|---:|---:|---|
|384|1152|19.407|19.709|3.556|2839.5|Recommended|
|416|1248|20.994|21.449|1.816|Not measured|Insufficient margin|
|448|1344|22.607|22.980|0.284|Not measured|Insufficient margin|
|512|1536|22.774|23.084|0.181|Not measured|CUDA OOM|

The512 peaks describe the partial failed attempt, not a completed512 update.
Larger unsafe candidates stop after pressure testing; no mixed throughput is
invented. The fixed safety requirement is at least3GiB of conservative device
headroom. Among safety-passing configurations, select the largest within95% of
the best measured representative throughput.

The original/default allocator measurements are retained unchanged in
`capacity_default_allocator_recommendation.json`,
`capacity_default_allocator_comparison.csv` and
`capacity_default_allocator_summary.md`. Default256 passed; default288/320/384
failed the margin requirement. Do not pool allocator regimes as if they were
one matched throughput experiment. The expanded-segment setting is a PyTorch
experimental native allocator option, not a model/loss change; its exact runtime
configuration must accompany the selected batch in the future formal run.
Official implementation reference:
https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/cuda.rst

## Scope and Cost

Data:1,786,887 complete-label TRAIN instances; original VAL/TEST unchanged and
not read in this stage. Selected fixtures use real row indices, include newly
added rows, and retain actual node counts, masks and candidate order. Maximum
input dimensions were TSP499, CVRP500, ATSP100 and MVRP101. No synthetic repeated
rows and no labels/costs entering model forward.

All sizing models/optimizers/scalers/RNG are discarded. Across both allocator
regimes:8 configurations,21 successful scratch updates,20928 native
presentations,0 AMP retries and1 CUDA OOM attempt. Combined measured GPU body
time was28.491s, excluding one-time CPU fixture/geometry preparation.
Both owned Slurm steps1465.119 and1465.120 completed0:0.
The assigned GPU returned to2MiB/0% utilization after the processes exited.

Historical geometry statistics were used ONLY for capacity. Future formal work
must fit normalization on new TRAIN, derive schedules from actual per-task
counts, initialize fresh seed2 state, and preserve all tail examples. Changing
batch/update coverage means the eventual comparison with historical R65B is not
a strictly matched data-only causal experiment. Validation-only checkpoint
selection and one final complete TEST remain required after approval.

R74 was terminated at the user's request; its disk files remain intact.
The original native70 goal is still paused. Legacy eight B-task semantics remain
deferred/unfixed. No formal TRAIN/VAL/TEST or interim Git commit was started.

## Files Added For Capacity

| File | Purpose |
|---|---|
|`code/V4/r75_data.py`|CPU TRAIN adapter and bounded geometry preparation|
|`code/V4/r75_capacity.py`|Complete-update instrumentation, peak memory and selection|
|`code/V4/run_v4_r75_capacity.sh`|Capacity-only launcher bound to the assigned UUID|
|`code/V4/R75_README.md`|Authorization boundary and reproducible commands|

Actual invocations:

```bash
bash code/V4/run_v4_r75_capacity.sh prepare
bash code/V4/run_v4_r75_capacity.sh measure
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash code/V4/run_v4_r75_capacity.sh measure --root code/V4/runs/R75_large_complete_labels/capacity_expandable_allocator
```
