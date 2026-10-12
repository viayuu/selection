# R66 full18 additive correction

Adopted plan: `.aris/traces/R65_closure_R66_plan.response.md`, read in full.
Run root: `code/V4/runs/R66_additive_correction_full18/`.
Original full18, all22 solver identities/all128 legal task deployments, native
winners and original TRAIN geometry remain the active problem. The target is
70% single-completed-solver macro TEST Top1, nightmare loop100, still unachieved.
R58 is fully deferred. The eight legacy B-task feasible-domain mismatches remain
unresolved and disclosed. No new labels, repairs, scenarios, routes or solver runs.

## Fixed experiment

Both arms share frozen R65B epoch40 averaged-x `best.pt`, SHA256
`d880cb2ed4605f410742b4c37945494323213168af95d02b22794d232b9e7ad8`.
Its actual NativeUtilitySelector architecture, direct utilities and saved x
weights are retained; all base parameters are frozen and all base modules stay
in eval mode through wrapper and optimizer transitions.

One fresh NativeUtilitySelector member at seed660002 is explicitly copied and
compared before training. Its last scalar score-layer weight AND bias are zero.
The architecture is d128,4 instance layers,2 joint layers,4 heads,FFN512,
scorehidden256,dropout.1,handcrafted34+ID and original TRAIN geometry. Initial
encoder gradients are naturally zero because the final scalar layer is zero.

`A_independent_sum_seed2` trains native R-Drop on g alone.
`B_additive_correction_seed2` trains the identical objective on detached F+g.
Both evaluate and deploy exactly F+g_x, coefficient1, full legal pools, exact ties
by ascending global solver ID. There are no gates, pruning, task specialists,
coefficient/temperature tuning, risk/pair losses, PCGrad, retrieval or source inputs.

Both optimizers are the unchanged official non-wrapper R65B AdamWScheduleFree,
with ONLY member parameters: LR3e-4,internal warmup720 successful updates,
WD1e-4,betas(.9,.999),eps1e-8,r0,weight_lr_power2,inner_momentum0,foreachFalse.
Train at y, evaluate/save x; no external warmup. Two dropout forwards contribute
averaged native CE plus legal-pool symmetric FP32 KL weighted
`.35*min(successful_update/720,1)`. Both branches train. The unchanged R65 update
function averages three same-state task gradients in FP32, clips globally at1,
and updates once. Whole-triple AMP retries restore RNG and advance no cursor,
ramp or successful-update counter until success.

Real256 rows/task,three tasks/update,effective768; final16/task,effective48.
All180000 TRAIN rows are shuffled/covered each epoch. Fixed40 epochs,240
updates/epoch,9600 updates/7.2M presentations per arm. Identical task/row/dropout
plans; no dropped tails,accumulation,microbatching,early stopping or pilot.
New paired cost is19200 updates/14.4M presentations. The inherited base's earlier
9600 updates/7.2M presentations are disclosed separately.

## Cache and selection

Only logits,indices,pool names,pool IDs are extracted from R65B's existing
`train_predictions/best` and `val_predictions/best` archives. The focused CPU
check aligns every original row index and legal pool name/ID. Names of original
examples follow the existing `str(index)` dataset/label convention. Archive
labels,costs,winners,predictions and routes are never extracted as forward inputs.
The original dataset supplies supervision. Member/base raw forward inputs use an
explicit allowlist; cached logits enter only the detached additive boundary.

Every epoch evaluates complete original18000 VAL rows at FP32 batch128 including
tails, always composite F+g_x. Complete180000 TRAIN evaluation occurs every5,
final and selected checkpoint, also composite. Raw optimization loss/CE/KL carry
the objective scope so A's standalone CE cannot be mistaken for composite CE.
Existing per-task Top1/2/3,cost/regret,single-method comparisons and pick
distributions are retained in evaluation records/predictions for parent delivery.

Maximum integer VALcorrect/18000 selects each arm; exact ties retain the earliest
epoch. Each improvement saves a unique complete two-model bundle and `best.pt`,
including both F and g_x,coefficient1 and exact inference semantics. `last.pt`
retains both model states,member optimizer eval-x mode,scaler,RNG,cursors and
selection history for explicit `--resume`. Validation candidates are published
only after both arms genuinely complete; official arm uses max count,tieA.

## Execution and handoff

The single CPU check is `bash code/V4/run_v4_r66.sh cpu-check`. It includes
offset objectives,both dropout gradients,identical copied member state,zero head,
frozen base/no-label boundary,live composite deployment and official optimizer
y/x save/resume semantics. Synthetic fixtures occur only inside that check.

Formal launch uses `tmux r66_train`,existing Slurm1465,gpu03,index1,UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`,with easynco,established PYTHONPATH
bridge,threads1 and offline W&B selector/entity
`yjkds-southern-university-of-science-technology`.
The launcher queries only that UUID immediately before launch and waits when
occupied. A then B run once; persistent stdout/stderr is `pipeline.log`.
No Git,reviews,agents,GPU sweep,reference inference or TEST is part of this run.

There is no TEST entrypoint. Parent later locks both complete bundles before one
full18 TEST each. `r66_model.load_deployment_bundle` returns an eval wrapper with
caching disabled; ordinary forward computes both live F and g on current raw
input. `deployment_forward` explicitly ignores any cache. Future ordinary final
TEST wall timings must include both selectors, with no cached base TEST scores
and no separately launched latency benchmark. Two selector forwards still choose
and execute only one final solver. TRAIN/VAL cached timing is not deployment timing.

Return after the first finite formal update. `handoff.json` records exact command,
GPU,tmux/Slurm/PID,progress log,configs,CPU check,first update and changed files.
The parent takes over monitoring the live tmux A-to-B pipeline; no performance
acceptance or70% claim follows from launch alone.
