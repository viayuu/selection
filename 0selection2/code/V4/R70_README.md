# R70: clean-plus-latent-mixup fine-tuning

Exactly ONE new TRAIN/VAL arm, `B_latent_mixup_seed2`, under
`code/V4/runs/R70_latent_mixup/`. The fixed A is the already selected complete
R69B20/global60 bank: 8604/18000 VAL correct. Its complete40E TRAIN/VAL history
and authorized TEST are reused unchanged. Reuse is not fresh replication.
R69's own official arm is unchanged. New R69 TEST, analysis, RESULTS, REPORT
and delivery metrics are excluded from this implementation and design.

## Initialization and architecture

Reuse `r69_model.make_bank`: eighteen independent complete NativeUtilitySelector
branches, one known problem ID routing each query. A and B both have33,173,154
parameters (1,842,953 per branch), all trainable. B starts from R69's ORIGINAL
PRE-fine-tuning initialization, never B20: R67B terminal stage1 averaged-x trunk
plus the entire fresh head seeded670003. No new model class or extra head.

## Objective and RNG

For each task, both ordinary clean encoder/head dropout forwards run FIRST.
Their `encode_pairs` features immediately before `score_head` retain both graphs.
One row permutation pi and scalar lambda~Uniform(0,1) are shared across views.
`Hmix = lambda*H + (1-lambda)*H[pi]` is FP32 and backpropagates through both
partners. Original legal masks and candidate solver-ID alignment are required.
The entire existing scalar head, including its dropout, scores both mixed views.

`L = 0.5*meanCEclean + 0.5*meanCEmix + .35*min(u/720,1)*KLclean`.
Mixed CE gathers the two legal FP32 log probabilities per row, weights their
losses by lambda and1-lambda, then averages over rows and two views. There is
no mixed KL, extra target, auxiliary label, layer search or coefficient scan.
Labels are excluded from encoder/head inputs and enter only the loss.

After both clean forwards, a private RNG context uses
`700002 + 18*(u-1) + problem_id`. CPU draws pi then FP32 lambda; the same seed
also controls auxiliary-head dropout on the sole visible CUDA device. Both
ordinary CPU and CUDA RNG states are restored afterward. Whole-update retries
restore the original clean RNG and reuse this successful-update seed.

## Fixed optimization and budget

Exact R67/R69 seed2 global41-80 task/row/dropout schedule,40 epochs,
240 successful logical updates/epoch,9,600 total. Real256/task times3 gives768;
six final triples/epoch have16/task, effective48. Every original TRAIN row appears
once/epoch:7.2M selected rows,14.4M encoder-view rows and14.4M additional mixed-head
rows. Additional head computation and its runtime are disclosed separately.
Each branch has1,600 active updates/400,000 selected rows.

Fresh AdamW: LR`3e-4*min(u/720,1)`, WD`1e-4`, betas`(.9,.999)`, epsilon`1e-8`,
`foreach=False`. The active disjoint task gradients are divided by3, clipped
together once at1, followed by one optimizer step. Inactive gradients stay None;
their parameters, decay, moments and counters do not advance. Any nonfinite
clean/mixed output, loss or gradient retries the WHOLE triple at unchanged rows,
weights, LR and successful clock, restoring RNG, zeroing gradients and reducing
the scaler. Limit20 attempts. There is no automatic live-job restart.

## Authorized checks and launch

ONE focused synthetic CPU invocation covers masked mixed-loss/backprop through
both partners, ordinary RNG isolation, shared draws across views, and an injected
third-task nonfinite mixed output causing a whole-update retry with identical
clean and auxiliary dropout. R69's proven bank routing/inactive handling is reused.

ONE discarded real256 TRAIN-only capacity UPDATE uses the scheduled full triple
with greatest per-task padded attention geometry, then total geometry, then
total padded nodes; schedule ties keep the earliest. The driver searches schedule
metadata only, with no GPU size sweep. All18 branches' FP32 Adam first/second
moments are explicitly MATERIALIZED, including inactive and unused paths, before
the update. The receipt reports actual complete-bank storage and peak memory.
The mutated model, optimizer, scaler, gradients and RNG are discarded/restored;
the formal bank is recreated and state-checked, with a fresh empty optimizer.
Capacity failure preserves its receipt and stops without alternate batching/GPU.

```bash
bash code/V4/run_v4_r70.sh cpu-check
tmux new-session -d -s r70_train 'bash code/V4/run_v4_r70.sh train >> code/V4/runs/R70_latent_mixup/bootstrap.log 2>&1'
```

Only existing allocation1465/gpu03/index1 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` is used. The launcher checks only that
assigned UUID immediately before capacity/formal execution and waits if occupied.
No allocation/client/device substitution. Conda easynco, existing python_compat
bridge, thread1, W&B offline selector/entity
`yjkds-southern-university-of-science-technology` are fixed.

## Evaluation, selection and handoff

Every epoch: full18 native VAL, FP32 eval, batch128 and tails. Full TRAIN every5,
final and selected-best. Save unique complete-bank checkpoints from one COMMON
epoch. Select max INTEGER VAL correct/18000, earliest exact tie. Official new B
requires strictly>8604; otherwise fixed A20. Normal deployment: one active branch,
full original legal pool, highest direct utility, global-ID ties, one completed
solver. No per-task stitching or cost-based override.

Protocol, args, source snapshot, init/schedule/control identities, parameter
counts, CPU/capacity receipts, raw component/gradient/AMP/row/memory/time updates,
W&B, histories, checkpoints and last optimizer state remain under the R70 root.
`handoff.json` is written after the first finite successful formal update and
contains tmux/pane/PIDs/Slurm step/UUID/command/actual metrics. Leave the fixed40E
run live; observation timeout never authorizes restart.

Reused A:9,600 updates/7.2M rows/3554.574466s/5 retries, historical expenditure.
Inherited stage1:9,600 updates/7.2M rows counted once separately. New B formal
runtime and the discarded capacity update are separately measured. No claim of
two new arms, fresh encoders or independent replication.

This phase performs no TEST, analysis, freeze or Git operation. After training,
parent freezes the selected complete bank and official arm by VAL, then runs ONE
new B full18 TEST regardless of improvement and reuses A TEST unchanged. Parent
alone publishes one completed-version commit. User staged cleanup/index14524,
`code/.gitignore`, R64-69/shared/data/EasyNCO/AGENTS/skills/results/checkpoints
and R69 delivery are not edited.

R58 is deferred. Eight legacy B feasible-domain mismatches remain disclosed and
unfixed; they are not a numeric70% veto. Goal70 remains ACTIVE and unachieved.
