# R73 Expansion-Input Teacher Training

The adopted scientific contract is
`runs/R73_expansion_teacher_training/approved_plan.md`. This experiment trains
one warm-started shared student. It does not isolate the causal effect of input
coverage, match capacity/exposure/compute to the reused control, or promise 70%.
NEW R72 TEST and delivery outputs are excluded from design.

## Fixed teacher and student

Teacher/control A is the unchanged complete R72 offline consensus bundle:
R60B39, R67B36/global76 and R70B23/global63. All three policies run on every
expansion query, frozen FP32/eval, using their original input paths and buffers.
The existing integer doubled-midrank Borda rule selects the smallest summed
rank, with global-ID ties. Its `pred_global_id` is a hard prediction, not a
measured solver annotation or calibrated probability. There is no teacher pass
over native TRAIN, VAL or TEST, no separate member evaluation and no refresh.
Fixed A retains its saved 8923/18000 VAL count without re-evaluation.

B strictly restores the complete saved averaged-x R67B36/global76
`B_contrastive_pretrain_seed2/best.pt`, including the trained head and all
normalization buffers. The original shared NativeUtilitySelector architecture
and every trainable selector parameter are retained. There is no bank, new
head, projector or new normalization fit. The fresh optimizer is the existing
official ScheduleFree factory, not a resumed historical optimizer.

## Data and objective

`r73_data.py` opens only existing R62 input arrays: nodes, source IDs, offsets
and node/matrix values. It never opens auxiliary costs or observation masks.
Original coordinates, customer order, directed ATSP, physical roles, constraints
and full legal pools are preserved. Existing label-free geometry is calculated
for the current streamed batch using unchanged TRAIN-fitted model buffers.

The one formal teacher pass visits source IDs10000..99999 in ascending order
for all18 tasks: 1.62M rows, batch128, eighteen16-row tails, 12672 query batches,
38016 member forwards and 4.86M member-row forwards. Global hard targets and
source/task/pool metadata are stored locally. Cache completion is published
only after all rows are written. Teacher models are released before student
construction. No solver executes.

Native task/row order is the original seed2 R67 global41..80 schedule. Each
task's auxiliary order concatenates five private NumPy permutations of90000
with seed730002+problem_id and consumes400000 indices. Every expansion row
appears four times, and40000/task a fifth time.

For each task, two native dropout views give mean native CE C and legal
symmetric KL R. One equal-sized auxiliary dropout view gives hard-target CE P.
The exact objective is `C + .35*min(u/720,1)*R + P`, with FP32 loss reductions.
Targets are excluded from all model forwards. Native graphs are backpropagated
and released before the auxiliary forward. Its private seed is
`730003+18*(u-1)+problem_id`, with ordinary CPU/CUDA RNG restored afterward.

Three task gradients at unchanged parameters are averaged in FP32, clipped
once at1 and followed by one official ScheduleFree step. Config: LR3e-4,
internal warmup720, betas(.9,.999), epsilon1e-8, WD1e-4, foreachFalse, r0,
weight_lr_power2, inner_momentum0. Train at y; evaluate/save averaged x. There
is no external LR controller. A nonfinite output/loss/gradient retries the
whole triple with identical rows/RNG and reduced AMP scale, up to20 attempts.
Only successful updates advance optimizer and data cursors.

The fixed budget is40 epochs/9600 successful updates, real256 native plus256
auxiliary rows per task x3, retaining16-row task tails. This is7.2M native plus
7.2M auxiliary presentations,14.4M total and21.6M student encoder-view rows.
Only the original180000 native rows have measured full-pool winners. Teacher,
student preparation, training and evaluation expenditure are recorded
separately. Historical member training is reused; shared inherited R67/R70
stage1 is counted once, not as new R73 training.

## Selection and operation

Full original18 native VAL runs every epoch in FP32 batch128/tails104. Native
TRAIN runs every5/final/selected, remaining sparse. Maximum integer VAL
correct/18000 selects one complete shared averaged-x checkpoint, earliest tie.
B is official only if strictly above8923; otherwise unchanged A remains.
Deployment is one direct-utility selector/full legal pool/global-ID ties,
followed by one completed solver. Hard teacher targets can reinforce errors;
discarding ranking uncertainty and compressing three policies are limitations.

Run exactly one `bash code/V4/run_v4_r73.sh cpu-check`, then launch
`bash code/V4/run_v4_r73.sh train` in owned tmux `r73_train`. The launcher uses
existing Slurm1465/gpu03 and only assigned UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` (physical index1). It waits if that
device is occupied. There is no GPU pilot/capacity sweep or automatic restart.
The first handoff reports an actual formal teacher batch; this does not assert
that student training has started. The same process proceeds from the complete
cache to training without transfer of teacher optimizer state.

Protocol, source snapshots, identities, raw batch/update records, checkpoints,
last optimizer/scaler/cursors, W&B offline logs and complete native evaluations
are saved under the isolated run root. Weights, pseudo-targets, predictions,
schedules and caches remain local; Git alone is not all inference assets.
No TEST, TEST lock, delivery analysis or Git is authorized by this phase.
Parent owns later delivery, actual reviews, objective audit and publication.
Goal70 remains ACTIVE/unachieved:12600/18000 on original full18 native TEST.
R58 and eight legacy B semantic inconsistencies remain deferred/unfixed and
are not a numerical veto on that native target.
