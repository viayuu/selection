# R76 Locked Experiment Scope

User request: one complete fresh R45A seed2 run on the new-only, B-filtered,
single-ReLD scenario, using both RTX4090 and both RTX3090 cards. The latest
request additionally prioritizes larger batches and high GPU utilization.

No historical weights, old candidate caches, solver reruns, R58 repairs or
paused-goal continuation. ReLD identity is the fixed MOEL deployment, displayed
as reld_mt. TRAIN-only geometry statistics and TRAIN-selected SBS references.

Use four DDP ranks over the two existing nodes. The logical global batch is
9856 for every task; split it in proportion to premeasured per-device
microbatch caps, including the final tail. Large graphs accumulate bounded
microbatches with unchanged parameters and synchronize before one update.
Scale the entire microbatch loss by world_size * micro_rows / logical_rows.
Every real TRAIN row appears exactly once per epoch and no tail is dropped.
Every task has eight successful logical updates per complete epoch, avoiding
the 41/48/8 frequency imbalance that physical-capacity-only batching would
otherwise introduce. At most40 epochs therefore permit5760 logical updates.

R45A architecture, maximin scoring and original R43/R45 objective remain fixed.
AdamW1e-4/WD1e-4, dropout.1, R-Drop ramp3 epochs to.35, clip1, maximum40
epochs and original CostController. Strict lowest validation macro_vs_sbs_pct
selects the checkpoint independently of controller min_delta. Full validation
every epoch; full eval-mode TRAIN every5 epochs and at stopping.

After stopping, lock one checkpoint, prepare TEST inputs, evaluate full TEST
exactly once, and report validation and TEST together. The new scenario's
70110-row test cannot be compared as a pure model improvement to the legacy
18000-row test. No claim of a >=70% goal or accuracy ceiling is implied.

Implementation/cache execution: gpt-6.1-sol/max workers. One bounded planning
and integration review: gpt-6-astra/max, same-family/provisional. No extra
training groups or audit branches. Only one completed-version publication
would be made if publication is requested.
