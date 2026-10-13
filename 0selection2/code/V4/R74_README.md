# R74 bounded adaptive incomplete acquisition

## Current Status: Paused Progress Snapshot

The user paused the goal and R74 on 2026-10-13. The subsequent request to
publish current progress authorizes a Git snapshot only, not resumed execution.
See `runs/R74_adaptive_incomplete_acquisition/PAUSED.md` and `PAUSE_STATE.json`
for the last observed process state and the distinction between logged update91
and the durable update60 checkpoint. No complete policy VAL epoch, selected
checkpoint lock or new TEST result exists. Continuation8 recovery edits are
unfinished and unvalidated; do not run any launcher without explicit resumption.

## Experiment Contract

One new B versus unchanged historical R73B23/global63 A (8959/18000 VAL).
The complete R73 averaged-x selector is frozen inside B, including its head and
original normalization buffers. This is an information-acquisition package,
not a capacity/compute-matched or isolated causal test of adaptivity.

The adopted contract is `runs/R74_adaptive_incomplete_acquisition/approved_plan.md`.
Only new token/candidate adapters, two cross-attention blocks, two acquisition
gates and a zero-initialized scalar utility correction train. Full native pools
and original winner labels remain unchanged; labels never enter inference.

Locked probes: original BQ TSP greedy step; ReLD_CVRP learned starts;
MATNET ATSPPolicy with seed2 per-size shared column permutation; both original
Train_ALL ReLD policies on all15 task-specific MVRP environments. These are new
extractor recipes, not historical solver-deployment recertifications. All are
frozen FP32/eval, greedy, augmentation1. No full solver wrapper, NN initializer,
postprocessing, reward reconstruction or completed-solver cost is executed.

Each state starts at one required node then acquires two increments
`h=(n//4-1)//2`; depot visits consume selections. Every state update is guarded
before and after: no finished trajectory and at most `1+2h<=n//4` required nodes.
Every start/checkpoint stays distinct. Exact403-dimensional tokens are stored
losslessly as frozen base nodes, integer start/current identities, packed
distinct-visited identities and exact FP32 visited means/scalars. Arrays remain
CPU/local; only the current batches transfer to the assigned GPU.

The root sees only h. The second gate sees h, its first shallow adapted tokens
and first-probe identity. Hard evaluation accesses only its chosen observation
path. Training differentiates the exact three-leaf expected native CE with two
dropout views per leaf, plus weighted legal KL with detached leaf weights only.

Formal generation:180000 TRAIN+18000 VAL distinct inputs,330000+33000 probe
initializations, frozen F once/input; batch16 task/true-size/native-start groups.
This counterfactual work is distinct from selected-path deployment cost.
After probes are released, official ScheduleFree trains only new parameters at
y and evaluates/saves x:40E/global41-80,9600 updates,7.2M native presentations,
38.4M successful leaf-view rows, real256/task x3 with16-row task tails.
Full hard-policy VAL every epoch, complete native TRAIN every5/final/selected.
Integer correct count selects one common complete policy; earliest ties.
Official B requires more than8959/18000 VAL correct. No TEST is authorized here.

Commands:

```bash
conda activate easynco
bash code/V4/run_v4_r74.sh cpu-check
# Exactly one formal owned session, created by the execution handoff:
bash code/V4/run_v4_r74.sh train
```

The launcher uses only allocation1465/gpu03/assigned UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, checks that device at launch and waits
there if occupied. W&B is offline in project selector and the existing entity.
No preflight/capacity pilot, batch fallback, automatic restart or failed-run retry.

Local checkpoints, probe states, feature arrays, raw updates and W&B are required
inference/research assets; Git alone does not contain them. Deployment charges F,
all acquired prefixes/gates and one fully restarted chosen solver. This screen
executes no completed solver and measures no end-to-end latency or speedup.
Historical shared pretraining counts once, separately from new extraction and
optimization. Historical TEST exposure is disclosed and excluded from design.
R58/eight legacy B discrepancies remain deferred/unfixed. Goal70 stays PAUSED
and unachieved, requiring at least12600/18000 on the locked original native TEST
and the parent's complete objective audit/reviews/publication.
