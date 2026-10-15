# R78 Round006 Execution

Owner: Solmax, 6.1Sol max. File workflow active. CPU preparation and the two
checks passed. The coordinator subsequently authorized gpu05/job1475 and
single4090 `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be`; see
`hardware_allocation.approved.json` and R79's existing process-exit receipt.
No new resource query, allocation or R79 TEST/LIB metric inspection is needed.
No R79 TEST/LIB values are inputs to this experiment.

## Fixed Protocol

The approved mechanism is in `../review-stage/SOL_NEXT_EXPERIMENT_DECISION.md`.
Use locked B ep28/u34776, SHA256
`31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b`.
All18 tasks, seed2, logical1024, fresh AdamW, original control late LR and
exactly14904 successful updates (828/task). Natural continuous permutations;
TSP/CVRP physical cap352, otherwise1024. No new control/baseline training.

FP64 native TRAIN costs yield centered log-relative full-candidate profiles;
their mean absolute distance orders cross-instance neighborhoods. On pooled
h, use FP32 normalized cosine scores and exact >= tie denominators excluding
self. Keep the complete original two-view base objective and add
`0.10 * (profile(view1)+profile(view2))/2`. No projection or inference change.

The no-grad cache pass executes BOTH full forwards including scorehead;
replay restores per-chunk RNG and the final cache-pass RNG. Feature gradients
include all1024 anchors AND keys. Only the base micro loss has n/1024 scaling;
the cached gradient injection is already globally normalized. One AMP scale
per combined micro loss; one unscale, global clip and optimizer step.

Only two tiny CPU checks now and one real finite GPU update1 later. The real
update1 is successful budget update1, not a discarded smoke. No extra audit,
gradient certificate, coefficient scan, benchmark or GPU probe.

## Endpoint And Authorization

Lock only the unique final14904 checkpoint, SHA, model/preprocessing/native
pools/inference before any candidate VAL. Evaluate actual fitted TRAIN/VAL
once, report separately from B and whole-version B fallback. No intermediate
VAL or checkpoint choice. Native method counts and corrected/harmed/
wrong-to-wrong signed original-FP64 cost changes accompany the results.

Zero-tolerance gate: macro native Top1 strictly greater than B AND FP64 macro
raw mean cost and actual regret no greater than B. CE/coverage are descriptive.
Historical actual final14904 round003 control is a descriptive comparator,
not a fresh causal control; its unavailable TRAIN metrics are not invented.

LATEST COORDINATOR AUTHORIZATION supersedes the decision's older requirement
for a subsequent TEST approval: a gate pass and unique checkpoint lock permit
ONE complete18 TEST, with an exclusive start receipt before any TEST cache
opens. Failure retains whole B, closes this fixed trial, and opens no TEST.
No LIB authorization, TEST-based revision, rescues or additional budget.

The hard4 GPU-hour cap covers the entire allocated process, including fitting,
update1, final TRAIN/VAL and conditional TEST. An outer timeout also bounds a
stalled operation. A budget/numerical/resource stop is incomplete, not a
negative scientific conclusion, and does not authorize a restart or extension.

## Commands And Outputs

From `/public/home/shiys/0selection2`, always activate easynco first:

```bash
bash code/V4/run_v4_r78_profile.sh prepare
bash code/V4/run_v4_r78_profile.sh cpu-tests
```

After the coordinator fills an approved allocation file (not before):

```bash
bash code/V4/run_v4_r78_profile.sh start /absolute/path/to/approved_allocation.json
```

That starts tmux, then one CPU4/mem9G exact single-GPU Slurm step, no SSH launch.
The pending template has no approved UUID/job and cannot launch. No automatic
resume or second trial is implemented; partial execution requires a new owner
decision, never silent extra updates.

Current CPU outputs: `prepared.json`, `late_lr.npy`, `cpu_checks.json`,
`cpu_checks.log`, `hardware_allocation.pending.json`, `status.json`.
Future execution outputs: `train.log`, `training_started.json`,
`update1_receipt.json`, `final14904.pt`, `candidate_lock.json`,
`evaluation/{train,val}/final14904/{metrics.json,method_changes.json,native_methods.csv,*.npz}`,
`results.json`, `execution_receipt.json` (or truthful `execution_stop.json`).
A pass also produces `release_manifest.json`, `test_started.json` and the one
`evaluation/test/final14904/` readout. No TEST outputs exist at preparation.

Implementation: `code/V4/r78_profile_loss.py`, `r78_profile_experiment.py`,
`test_r78_profile.py`, `run_v4_r78_profile.sh`. R77/R79 and old rounds unmodified.
Old native/app goals remain untouched; historical research provenance remains.
