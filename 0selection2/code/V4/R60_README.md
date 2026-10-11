# R60: More Independent Comparison Data, Same Full-Pool Selector

R59 did not improve validation performance. R60 changes the amount of real,
independent supervision rather than adding another encoder or local expert.
R58 remains deferred, including its task repair and solver/cost generation.

## Experiment

- Both arms: original R45A/R43A `ScoreDifferenceSelector`, initialized from
  scratch with seed2. Original18 tasks, legal pools and full-cost labels remain.
- A: additional MOEL/MTL pair supervision by replaying old TRAIN examples.
- B: the same additional supervision from existing expanded TRAIN examples.
- Expanded rows have only two observed costs for this experiment. Their lower
  cost is **not** treated as the complete-pool winner. No missing costs are filled.
- All original candidates remain in the forward pass and final maximin decision.

Each MVRP step combines the unchanged original two-forward R-Drop objective with
`0.35 * BCE(pair_margin[MOEL, MTL], cost_MOEL < cost_MTL)` on one auxiliary batch.
FP64 costs define direction and exact ties; ties contribute no pair loss. There
is one optimizer update, with the original AMP same-batch retry behavior.

The auxiliary A/B schedule matches task, actual node counts and batch sizes.
Approximately10k auxiliary rows per task per original epoch cover B's additional
90k rows every9 epochs. Original full18 training coverage and tail batches remain.
No expanded validation/test rows, new solver runs or API calls are used.

## Training

Batch128, AdamW1e-4/WD1e-4, dropout0.1, clip1, warmup3, at least15 and at most40
epochs, original validation-driven LR and early stopping. Original-TRAIN geometry
statistics are shared. Select by strict minimum validation `macro_vs_sbs_pct`.
Evaluate full original validation each epoch and original TRAIN every5/final.
Run replay A first: its controller sets the common learning-rate trace and stop
epoch for both arms. Expanded B follows that trace through the same number of
successful updates and independently selects its validation-cost best checkpoint.
Replay rows are sampled with replacement within the corresponding actual-size
bucket. This is a matched compute comparison, not independently early-stopped runs.

The locked advancement screen requires B to beat historical R45A by3pp and replay
A by2pp, with actual regret no worse than either. It does not establish70% or test
success. R59 and R60 do not access test; held-out finalization is separate.

## Execution

Use `easynco`. The launcher binds only the designated available3090; it does not
use other GPUs or modify the user's label-generation queues. CPU caches live in
`runs/R60_expanded_partial_supervision/data_cache/`; source datasets stay read-only.

```bash
bash code/V4/run_v4_r60.sh plan
R60_PARENT_GPU_BATON=expanded-partial-supervision bash code/V4/run_v4_r60.sh prepare
tmux new-session -d -s r60_train 'env R60_PARENT_GPU_BATON=expanded-partial-supervision bash code/V4/run_v4_r60.sh train --resume'
bash code/V4/run_v4_r60.sh analyze
```

Necessary checks are limited to data identity/direction and one real update.
Do not repeat historical audits or add a preflight/review campaign. Commit code,
locked results, best weights and curves once the complete version finishes.

## Interpretation

This is expanded **partial** supervision, not a completed expanded full-pool
benchmark. Existing calibration/provenance is limited; it does not establish
complete historical deployment equivalence. Existing B feasible-domain conflicts
remain unresolved and disclosed. Full-pool validation metrics, not auxiliary
binary accuracy, determine whether this direction is worth continuing.
