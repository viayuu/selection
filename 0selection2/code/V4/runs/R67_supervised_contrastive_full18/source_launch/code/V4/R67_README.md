# R67: native-winner supervised contrastive pretraining

Adopted PLAN ONLY: `.aris/traces/R66_closure_R67_plan.response.md`, with the
user's locked operational clarifications encoded here and in `protocol.json`.
This is one original full18/full-legal-pool paired experiment at seed2. All
R64/R65/R66 completed sources, checkpoints and results remain immutable.

## Locked comparison

Both arms explicitly copy one fresh R64 NativeUtilitySelector initialization:
d128, four instance and two joint layers, four heads, FFN512, scalar score MLP
hidden256, dropout.1, handcrafted34+solver ID, original TRAIN-normalized geometry,
all original inputs including directed ATSP matrices. No trained state is reused.

Stage1 runs40 fixed epochs, with no validation evaluation or checkpoint selection.
Every dropout view calls `encode_pairs` once and reuses its384-dimensional pair
features for the original utility head and a legal-candidate masked mean. The
mean is L2-normalized, projected384->256->128 with ReLU, then L2-normalized again.
Projector seed670002 is initialized once and explicitly copied across arms.

- A_classification_pretrain_seed2: native CE averaged over two dropout forwards
  plus legal FP32 symmetric RDrop KL, coefficient `.35*min(stage_update/720,1)`.
- B_contrastive_pretrain_seed2: only supervised contrastive Eq.2. Both views are
  anchors; positives are every other same-native-winner view, including the paired
  view of the same input. Only the self diagonal is excluded. Denominators contain
  all nonself views from this actual task batch; there are no cross-task contrasts.
  FP32 logsumexp, mean positive log probabilities outside the log, temperature.1,
  fixed loss multiplier.1, gradients through both views, and no target in forward.

The A projector and B utility head have no objective gradient during stage1.
Their gradients remain `None`, so they receive no optimizer update or weight decay.

Stage2 transfers each terminal epoch40 averaged-x trunk. It discards both projectors
and replaces the ENTIRE scalar score MLP with one ordinary fresh initialization,
seed670003, explicitly copied across arms. No trained head or zero initialization
is used. Optimizer, scaler, stage update, warmup, RDrop and scaler-growth counters
reset. ALL selector parameters are then fine-tuned40 epochs with A's native CE/RDrop.

## Attribution and adaptations

[Khosla et al., Eq.2](https://papers.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)
defines the supervised contrastive positive averaging. The
[author implementation](https://github.com/HobbitLong/SupContrast/blob/master/losses.py)
scales by `temperature/base_temperature`, whose default base_temperature is.07.
R67's fixed multiplier.1 is a PROJECT loss-scale choice, equivalent to
base_temperature1 at temperature.1; it is not the author-default scale. The locked
numeric.1 is unchanged and is not searched. Dropout-only views and fully trainable
stage2 fine-tuning are project adaptations; the paper's frozen linear stage2 is
not reproduced. No new data, labels, augmentation, expert, source embeddings or retrieval.

## Budget and execution

Every stage/arm uses the exact official R65 ScheduleFree AdamW: LR3e-4, internal
720-successful-update warmup per stage then constant, WD1e-4, betas(.9,.999), eps1e-8,
r0, weight_lr_power2, inner_momentum0. Gradients use y; evaluation/checkpoints use x.
Stage transfer initializes the fresh optimizer from terminal x. No external warmup,
controller, earlystop, PCGrad, accumulation, microbatching or row dropping.

Real256 independent rows per task and3 distinct tasks per successful update give768
rows, with final16/task giving48. A single seed2 R64-style80-epoch task/row/dropout
plan is shared across arms: stage1 epochs1-40, stage2 global41-80. Each epoch visits
all180000 original TRAIN rows once in240 updates. Per stage:9600 updates/7.2M
presentations; per arm:19200/14.4M; pair:38400/28.8M. Contrastive task batches have512
views, with32 at the tail. Mean three same-state task gradients in FP32, clip once1,
then one update. AMP failure retries the whole identical triple with restored RNG;
no row, successful-update, warmup or RDrop counter advances on failure.

Stage2 evaluates all18000 original validation rows every epoch, FP32batch128 with
tails. Complete eval-mode TRAIN is recorded every5/final/selected stage2 epochs.
Selection maximizes INTEGER native correct count, earliest exacttie; official arm
maximizes the selected count, tieA. Saved best checkpoints contain one complete
NativeUtilitySelector in averaged-x, no projector. Inference uses one selector
forward, original full-pool utility argmax/global-ID exacttie, and one final solver.
No TEST runs here. Parent freezes one checkpoint per arm only after both complete.

## Launch and monitoring

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash code/V4/run_v4_r67.sh cpu-check
tmux new-session -d -s r67_train 'bash code/V4/run_v4_r67.sh train >> code/V4/runs/R67_supervised_contrastive_full18/pipeline.log 2>&1'
```

The launcher uses only existing Slurm1465 on gpu03/index1, UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`. It checks that UUID for ordinary idleness
immediately before launch and waits15s if occupied. It never allocates or switches
GPUs or stops another client. Threads1, established bridge PYTHONPATH, offline W&B
project selector/entity yjkds-southern-university-of-science-technology. There is no
W&B login or network preflight. The sequential pipeline is A1->A2->B1->B2 once.

The single synthetic CPU check covers scalar-reference Eq.2 masks and two-view
gradients, target isolation, one encode per view, copied fresh initialization,
unused head gradients, terminal-x transfer, whole-head reset and fresh stage2
optimizer/scaler/counters. No production data is used for that check.

Local root: `code/V4/runs/R67_supervised_contrastive_full18/`. The first finite
formal update writes `handoff.json`, real row counts, memory, command, PID/tmux/
Slurm, CPUcheck and next-monitor instruction. `pipeline.log`, `current_status.json`,
per-arm updates/history/gradient/objective/counter logs, stage transitions, runtime,
source snapshots, args/protocol and selected prediction arrays remain local.
`last.pt` checkpoints use eval-x plus optimizer eval mode and include RNG/counters;
the launcher refuses an automatic replay. No large checkpoint publication or commit.

The redundant >50MB row schedule is never written. `schedule_seed2.json` records
the config/hash. Local regeneration, after activating easynco and setting the
launcher PYTHONPATH, is:

```python
from code.unified_selector.registry import PROBLEMS
from code.V4.r64_experiment import paired_triples
schedule = paired_triples({task: 10000 for task in PROBLEMS}, epochs=80, seed=2, batch_size=256)
```

The full18 native macro TEST Top1 70% goal and nightmare loop100 remain active.
R58 is fully deferred; the eight legacy B-task semantics remain unresolved and
disclosed. Parent handles later delivery, TEST, review, plots and Git. No shared
review state or memory is modified by R67 execution.
