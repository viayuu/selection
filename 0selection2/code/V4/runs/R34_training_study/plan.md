# R34 Training Study

Fixed architecture: R33a dual-stream, four initial encoder layers, two synchronous joint layers,
d=128, heads=4, solver metadata weight=1, seed=2, all 18 existing problems.

Common protocol corrections: disable the unverified index-derived coord_dist embedding;
use native winner labels in CE and evaluation; keep labels/data/splits unchanged.
No architecture or task-weighting changes in this study.

| Run | Objective |
| --- | --- |
| R34_original_seed2_4090 | R33a original: weighted CE 0.35, top-focused pair 0.30, sequential top-k CE 0.08, risk 0.02 |
| R34_ce_seed2_4090 | Unweighted CE 0.35 only |
| R34_winner_cost_seed2_4090 | Unweighted CE 0.35, full winner-vs-arm pair 0.10, risk 0.02; relative cost scale 0.01 |

All runs: batch=640, AdamW lr=2e-4, weight_decay=1e-4, three warmup epochs.
ReduceLROnPlateau on validation top1-safe score, patience=4, factor=0.5,
absolute threshold=0.001, minimum lr=2e-6. Maximum 60 epochs.
Early stopping requires at least 12,000 successful optimizer updates and
10 stale validation evaluations. Different stopping times and learning-rate
trajectories under this common policy will be reported, not described as identical trajectories.

Validation every epoch; full train evaluation every five epochs and at termination.
History, loss components, gradient norms, AMP skips, learning-rate curves are saved.
Checkpoint contains optimizer, scaler, scheduler, RNG, counters and selection trackers.
Primary checkpoint selected on validation only. All training launches use --skip-test.
Select the recipe on validation, then perform the final test evaluation and baseline comparison.
GPU0 RTX4090, sequential runs to avoid competing GPU caches, offline W&B.

Historical results remain unchanged. Re-evaluate the reference with the same native-winner
metric before quoting changes, because historical FP32 tie-breaking differs slightly.
This study compares complete objective recipes, not isolated loss-coefficient effects.

## Code Audit Note (Before Test)

The completed original-objective control retained historical FP32-argmin frequency
weights. Its CE targets and evaluation use native winners. The frequency definitions
differ on 77/180000 train records (74 in TSP); no labels were changed.
CE and winner-cost recipes do not use class weights and are unaffected.
The current trainer now supports native-label frequency weights for future runs;
that additional frequency correction is not an evaluated component of this control.

Checkpoint tracker ordering and same-directory rewind validation were repaired after
the original and CE processes loaded their training code. These are checkpoint/resume
metadata changes, not optimizer, loss or sampling changes. The winner-cost process
will load the corrected source. Both source versions are archived in this study.
