# R121 One Principal Continuation

This is a CPU waiting controller, not an extra experiment or evidence that model
training has started. The four existing native collection lanes remain untouched.

It requires the complete single-root `collection_explicit_cpu_v2` population:
18 tasks, 89 deployments, 23,040 endpoints and 113,920 labels. Old valid receipts
may be adopted byte-identically by the collector, but partial pools, copied-root
double counting and failure-based candidate deletion cannot release the learner.
`completed_labels` counts the new consumable single root. The informational
`collected_unique_labels` deduplicates deployment prefixes across the fixed old
and new roots; it does NOT release training.

The sole pipeline is complete export, CPU family features, exact learner protocol
freeze, then one existing allocation-bound RTX4090 (`gpu05_1`) for from-scratch
train, locked TEST, LIB and locked TRAIN/family capture. Saved-only reporting runs
on CPU afterwards, then one curated GitHub publication for the completed version.
No new allocation, direct-SSH training, automatic retry/resume or new model arm.

Seed2 and logical1024 are fixed:820 natural samples plus102 family pairs. The
budget remains14,400 successful updates, with full18 natural VAL every720.
TSP/CVRP micro-cap352 and eval256 reuse the existing4090 envelope, not a new
batch search. Other task micro-caps use the frozen R77 helper. All actual settings
are included in the exact protocol before the first optimizer update.

W&B is enabled in explicit offline mode because existing nodes lack credentials.
The original natural81000/4500/4500 split, labels and candidate rules are unchanged.
R58 remains deferred. TEST and LIB never select a checkpoint or inference rule.

## Main Invocation

Only after learner/finisher READY and source freeze:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r121_followup create
tmux new-session -d -s return18_r121_principal_followup 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B -u -m code.V4.r121_followup run > code/V4/runs/R121_structural_families/preparation/followup_principal_v1/controller.log 2>&1'
```

Progress: `status.json` and `controller.log` here; once dispatched, the immutable
plan is under `Return18_post_R120/dispatch/main_approved_r121_principal_v1.json`.
Stage logs and the final publication receipt identify actual completion. A waiting
JSON alone is not proof a worker is live; use the controller PID and actual native
processes when monitoring. A failed attempt is preserved for explicit diagnosis,
not retried by this controller.
