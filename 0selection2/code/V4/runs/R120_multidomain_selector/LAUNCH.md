# R120: Six New Domains In One Full-Pool Selector

## What Is Being Run

One shared model covers the original 18 routing tasks plus MIS, JSSP, FJSP,
BPP, MKP and SMTWTP. Native typed graphs/items/operations/resources are inputs;
the candidate comparison is over each problem's full frozen legal pool.
Four shared 128-dimensional relation/global layers and two shared node/candidate
interaction layers are trained from seed 2. There is no query solver, trajectory
probe, route head or local MOEL/MTL replacement expert at deployment.

The scientific rationale is in
`docs/research/multidomain_selection_design_20261029/REPORT.pdf` and
`HANDOFF.pdf`. They are research artifacts, not final model results.

| Added task | Native objective | TRAIN | Validation | TEST | Candidate deployments |
| --- | --- | ---: | ---: | ---: | --- |
| MIS | Max independent-set size | 10,000 | 1,000 | 1,000 | MinDegree, DIFUSCO_ER, T2T_ER |
| JSSP | Min makespan | 10,000 | 1,000 | 1,000 | SPT, MWKR, four L2D weights |
| FJSP | Min makespan | 10,000 | 1,000 | 1,000 | SPT, MWKR, eight DAN weights |
| BPP | Min bin count | 10,000 | 1,000 | 1,000 | FFD, BFD, MBS_prime |
| MKP | Max profit | 10,000 | 1,350 | 1,350 | SurrogateGreedy, LPDualGreedy, AntSystem |
| SMTWTP | Min weighted tardiness | 10,000 | 3,750 | 3,750 | EDD, WSPT, ATC |

These are 28 selectable deployments across the added six domains, not 28
independent algorithm families. Their formal totals are 60,000/9,100/9,100;
with the unchanged routing data the totals are 1,518,000/90,100/90,100.
The declared full pool is not complete until all MKP AntSystem rows are saved
and the final immutable export succeeds.

## Live Progress

All commands start in the `easynco` environment from the repository directory:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
cat code/V4/runs/R120_multidomain_selector/pipeline_status.json
tail -n 20 code/V4/runs/R120_multidomain_selector/pipeline.log
tail -n 20 code/V4/runs/R120_multidomain_selector/train_gpu05.log
```

Annotation completed at 20:16 CST. The current controller tmux session is
`r120_full24_pipeline_host`; it replaces the zero-update container-network
startup attempt documented in `startup_recovery.md`. Do not launch a duplicate.
After training starts, the model writes `status.json`, `train_gpu05.log`,
`train_gpu03.log`,
`history.json`, validation records and W&B receipts under this run root.

## Fixed Stages

1. Finish all 16 disjoint native AntSystem worker shards. Their immutable label
   count includes 10,000 TRAIN, 1,350 validation, 1,350 TEST and 270 external
   MKP benchmark records. The public benchmark never becomes model TRAIN.
2. Export `nonrouting_supplement_20261029/prepared_with_as/manifest.json`,
   retaining the earlier incomplete `prepared_cpu` version unchanged.
3. Prepare the complete 24-task protocol with TRAIN-only references/statistics.
4. Train one distributed model on both allocated GPUs of each node. Global
   batch is 1,024; rank row weights 5:5:3:3 give 320/320/192/192 rows to
   the two RTX4090/two RTX3090. Microbatch limits MIS=96 and MKP=120 retain
   exact full-global-batch normalization.
5. Select one checkpoint by task-macro any-best validation Top1, then minimum
   TRAIN-scaled excess, with earliest-update tie breaking. Routing native Top1
   is also reported separately. Run full TEST with that lock, then LIB.
6. Produce PDF/CSV reports and one curated GitHub commit after success.

The fixed budget is at most 57,600 successful updates, AdamW at 1e-4 peak,
4,320-update warmup, cosine to 1e-6, weight decay 1e-4, dropout 0.1 and
validation every 1,440 updates. Early stopping uses six validation observations
without improvement after warmup. Smaller domains repeat more often under
uniform task cycles; this is not natural pooled-sample training.

## Resource And Failure Rules

- Existing allocations only: gpu05/job1475 (two RTX4090) and gpu03/job1465
  (two RTX3090). Annotation ran in allocation-owned containers; distributed
  model stages use GPU-owning Slurm host steps for inter-node networking.
  Cgroup/UUID evidence is recorded. Ordinary SSH is not used for training.
- Four physical GPU mutexes stay held for a stage. Annotation and model
  training do not overlap on the same card. No unrelated process is stopped.
- A nonzero stage exit is recorded and halts progression, with no automatic
  rerun, candidate removal or protocol change based on TEST.
- `PAUSE_NEW_WORK` in this run directory pauses between stages without
  interrupting an already running experiment.
- W&B is enabled with an explicit offline fallback only if its usual credential
  lookup is unavailable; the actual mode/run ID is saved when training starts.

## Outputs

`reports/RESULTS.pdf`, `reports/selection_distributions.pdf`, native per-task
CSV, separate ROUTING18/NONROUTING6/ALL24 summaries, learning curves, zero-choice
support/recall/cost statistics, and a same-population historical NSS comparison.
No completed model result exists until those stages actually finish.
