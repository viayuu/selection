# R76: R45A on the Single-ReLD New Scenario

Release status (2026-10-15): completed historical baseline; the retained current
model is R77B. See the [project overview](../../README.md) and the new
[paired winner/selected gap PDF](runs/R76_R45A_single_reld_mt/missed_winner_probabilities/winner_selected_gap_report/winner_selected_gap_report.pdf).
R76 LIB values must not be substituted for the same-lock R77B LIB evaluation.

One fresh seed=2 training run. The paused historical goal and R74 are not resumed.

Data: `/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`.
All 18 tasks have 70,114 TRAIN, 3,895 validation and 3,895 TEST rows. The dataset
keeps MOEL under the name `reld_mt`; MTL is not a candidate. The historical
22-slot model ID space is retained with the MTL slot inactive. Use the scenario
manifest/reader, never sorted filenames to infer cost column order.

The model and supervised objective are R45A, not R65B/R75: handcrafted solver
features plus ID, ScoreDifferenceSelector, utility differences, maximin logits,
0.35 CE + 0.35 comparison + 0.02 scaled risk, and legal-candidate R-Drop.
Both dropout paths share the current parameters and use one optimizer update.
No solver execution, code embedding, extra seed or new labels are added.

AdamW retains LR=1e-4, WD=1e-4, dropout=0.1, clipping=1, three-epoch warmup,
and the original validation-driven LR/early-stop rules with at most 40 epochs.
Save every strict minimum validation macro_vs_sbs_pct. The SBS identity is now
selected on TRAIN and held fixed for validation/TEST; evaluation does not pick
a new method using held-out costs.

## Four-GPU Execution

The existing user allocations provide two RTX4090 cards on gpu05 (job1475)
and two RTX3090 cards on gpu03 (job1465). One model is trained by four DDP
ranks; faster cards can process larger local batches. Batch sizes are bounded
by a short actual training-path capacity measurement and fixed before formal
training. Large TSP/CVRP graphs and smaller MVRP/ATSP graphs may use different
fixed microbatch caps. The logical global batch is 9,856 for every problem;
large TSP/CVRP batches accumulate bounded microbatches while the smaller
graphs normally fit in one pass. This preserves equal task update counts
rather than implicitly favoring tasks with smaller physical batch capacity.
Tail rows are split without duplication and the full DDP objective is weighted
by actual local/global row counts. There is one optimizer update per logical
batch; accumulation does not create extra updates or independently update
the model between dropout paths.

Cross-node NCCL uses the existing `bond0` IPv4 network (12.12.12.24/26),
explicitly excluding automatically selected IPv6 link-local/overlay routes.
The initial automatic-interface attempt failed before any optimizer update;
its logs are preserved separately. No cache/probe/solver rerun was required.

These large batches are an intentional training-protocol difference from
historical R45A batch128. The original LR is not automatically scaled. Report
the chosen global batches and actual successful updates, rather than claiming
optimization-equivalent replication.

Four disjoint preprocessing workers build new, scenario-aligned caches.
Completed arrays are copied into a separate local `/tmp` cache on each node;
training uses pinned asynchronous input prefetch. Old data and caches are not
modified. W&B defaults to offline recording in project `selector`.

## Commands

```bash
bash code/V4/run_v4_r76.sh cache
bash code/V4/run_v4_r76.sh capacity
bash code/V4/run_v4_r76.sh train
bash code/V4/run_v4_r76.sh test
bash code/V4/run_v4_r76.sh analysis
```

`pipeline` executes these phases in order. Long runs use a tmux session.
`finish` uses already prepared caches and measured batch caps, then executes
training, the locked TEST and analysis without repeating preparation/probes.
The unique validation-selected checkpoint is locked before any TEST cache is
prepared or opened. TEST is evaluated once and cannot change the checkpoint,
ranking rule or task composition. Original R45A/R75 results use different
data/pools and are not matched performance controls for this scenario.

Results: `code/V4/runs/R76_R45A_single_reld_mt/`.
