# R68: one AdamW/SAM fine-tuning comparison

Adopted raw plan: `.aris/traces/R67_closure_next_plan.response.md`, actual
Astra/max completed R67 TRAIN/VAL review, score8/ready PLAN ONLY. New R67 TEST
did not inform that plan. This is one optimization contrast with a shared
contrastive prior, seed2, original18 tasks/full legal pools/native winners.

Both arms explicitly copy one complete starting selector: ONLY R67B
`B_contrastive_pretrain_seed2/stage1_terminal_x.pt` terminal averaged-x trunk,
plus the ENTIRE fresh `shared_initialization.pt['stage2_head']`, seed670003.
No inherited utility head or projector enters the selector. All parameters
train. The shared inherited9600updates/7.2Mrows are disclosed once, separately
from the new paired fine-tunes; these are not fresh encoders. R64 d128,
instance4+joint2,4heads,FFN512,score256,dropout.1,handcrafted34+ID, original
TRAIN-normalized geometry, every original input and directed ATSP path remain.

`A_plain_adamw_seed2` uses plain AdamW. `B_sam_adamw_seed2` adds SAM radius.05.
Both use unchanged native two-view CE/legal FP32 symmetric RDrop with
`.35*min(u/720,1)`, fresh AdamW/scaler, LR`3e-4*min(u/720,1)`, u=1..9600,
WD1e-4,betas(.9,.999),eps1e-8,foreach=False.40fixedepochs, no controller or
early stop. R68 has ordinary weights, no ScheduleFree optimizer, mode
transition or averaging. AdamW/native CE/RDrop/shared three-task SAM are
project adaptations; no sharp-minimum or70% claim is implied.

The original R67 full80epoch schedule is regenerated with the existing helper
and its COMPLETE hash must match R67. Then only permutations/cycles/triples
`[40:80]` are used, globalepochs41-80, with original
`dropout_seed(9600+new_successful_before)`. Both arms copy the same plan. The
root `schedule_seed2.json` records full and slice identity/regeneration;
the large regenerable row cache is not saved/committed.

Real256 independent rows/task, three distinct tasks/update, effective768;
six final16/task updates/effective48. Every epoch covers180k original rows.
Each arm9600successfulupdates/7.2Mselectedrows, pair19200/14.4M. A has14.4M
successful dropout-view rows, B28.8M; examples/updates match, FLOPs do not.
Attempted view rows including retries are separately logged.

Each pass computes the unscaled FP32 arithmetic mean of all three task
gradients at one parameter state. A clips this mean once at1 then steps.
B forms one global epsilon=.05*g/max(norm(g),1e-12) BEFORE clipping; saves
exact original weights, perturbs once, restores PRE-PASS RNG, recomputes the
whole mean on identical rows/two dropout masks, COPIES originals back, clips
only the second mean once1, then takes ONE AdamW step/one weight-decay step.
Unused gradients remainNone. Exception paths restore original weights/RNG
and clear gradients. Any nonfinite legal forward/loss/gradient/reduction/norm
retries the whole identical triple after AMP backoff with no sample/LR/state
advancement;20attemptlimit. Successful-update scaler growth uses the existing
explicit counter,2000interval, reset on overflow. No first optimizer step.

Original18kVAL is evaluated every epoch FP32batch128 with104-row tails;
full180kTRAIN every5/final/selected epoch. Max INTEGER VAL correct/18000,
earliest exacttie, selects unique complete ordinary-weight checkpoint;
officialarm max/tieA. Costs/TEST never select. Full-pool highest utility and
ascending global-ID exactties remain. Deployment is ONE selectorforward and
ONE final completedsolver; training/evaluation run no solvers. Labels/costs
remain outside model forward. Per-instance prediction arrays stay local.

```bash
bash code/V4/run_v4_r68.sh cpu-check
tmux new-session -d -s r68_train 'bash code/V4/run_v4_r68.sh train >> code/V4/runs/R68_sharpness_aware_finetuning/pipeline.log 2>&1'
```

The launcher uses ONLY existing Slurm1465/gpu03/index1 UUID
GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d. Its ordinary assigned-UUID-only idle
check runs immediately before formal launch and waits30s if occupied. It
does not stop clients, query other GPUs, allocate or use SSH. easynco,
established bridge PYTHONPATH, threads1, W&B offline selector/entity are set
in the launcher. One focused synthetic CPU check validates initialization,
global perturbation/exact restoration, same masks, second-pass failure and
whole-update retry/one base step. No pilot/capacity/LR sweep is performed.

The first finite FORMAL update writes handoff.json with live handles, actual
rows/LR/gradient/memory, CPU receipt and normal180-300s monitor instruction.
Only after both actual pipelines terminal0 does parent freeze one VAL best
per arm and authorize one original18 TEST. No TEST/analysis runs here.
Parent owns later delivery/reviews/state/memory/Git; never auto-restart failed
training or replay heldout work. Source hashes/snapshot, adopted plan,
protocol/commoninit/schedule/inherited identity precede the first formalupdate.
Actual HEAD is read from filesystem at launch; actual worktree source hashes,
not an intended commit, identify the run without listing unrelated changes.

Weights, NPZ/rawcosts/rawupdates/W&B/regenerable caches remain LOCAL under user
cleanup; nothing is force-added. Git alone does not contain all inference
assets. R58 stays fully deferred. Eight legacy B-task feasible-domain
mismatches remain unresolved/disclosed and do not veto native70%. One seed
cannot support severe-overfit/noise/ceiling claims. SAM>A but<R60B would be
limited progress; the full18 single-solver70% goal remains active.
