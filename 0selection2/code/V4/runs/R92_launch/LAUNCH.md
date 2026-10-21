# R92 Shared Launch Plumbing

Plumbing only. This delivery does not launch, claim GPUs, probe resources, write
specs, freeze runtime sources, or choose the scientific protocol.

## Ownership And Interfaces

- Stable generic arms: `A`, `B`, `C`, `D`. Model output: `runs/R92_broad_learning`.
- Main's chosen science: A hard-input graph hierarchy NN; B learned
  DiffPool-inspired hierarchy NN; C native-winner GPU CatBoost Multiclass;
  D full-relative-gap GPU CatBoost MultiRMSE. These are not four matched NN trials.
- Fixed routing: A -> 3090_0, B -> 4090_0, C -> 3090_1, D -> 4090_1.
- Reuse only job1465/gpu03 and job1475/gpu05's original containers and exact
  UUID mutexes. Each GPU pipeline uses an allocation-owned CPU-only srun step
  (4 CPUs, 9 GiB), then container exec. No new allocation or SSH training.
- TRAIN uses original full18 NEW90k, R84 preparation/TRAIN-only statistics and
  scratch initialization, logical batch 1024, seed2. Both existing local caches
  are reused, without copying, rebuilding, or reading/hashing array contents.
  Cache evidence is the real `runs/R85_launch/resources.json`.
- W&B stays offline. No credential lookup, login, or retry.

Interface assumptions recorded before implementation:

| ID | Interface | Decision |
| --- | --- | --- |
| I1 | Unfinished protocol | Templates retain an unresolved per-arm budget until main writes the final lock; the launcher does not write it. |
| I2 | Final protocol | Main supplies `runs/R92_broad_learning/protocol_lock.json`, `locked: true`, arms A/B/C/D, full18 scope, seed2 and logical batch1024. |
| I3 | Execution controls | Read arm budgets from the lock, never equate NN updates with trees. Latest explicit user settings supply CLI defaults352/256 for every arm (irrelevant to tree fitting), superseding176/128. Per-arm locked descriptions can override those defaults. |
| I4 | Runtime freeze | Parse final experiment/reporting CLIs first; freeze only during explicit prepare, after final runtime files exist. Main lists additional dependency files in `launch.source_files` if needed. |
| I5 | R91 report failure | Only the explicitly identified root recovery-complete receipt and its detailed unchanged-input receipt can satisfy the gate. Original failed receipts stay untouched. |
| I6 | PDF placement | train (including VAL lock + TEST), idempotent TEST, LIB run in the container; PDF runs on ADMIN01 CPU after the GPU step AND UUID lock are released. Reports must use locked predictions, not run inference. |

## Main's Final Lock

Supported budget aliases are `max_updates` / `max_successful_updates`,
`warmup_updates`, and `eval_every` / `evaluation_every_updates`.
Batch aliases include `batch_size`, `global_batch_size`, `logical_global_batch`,
and `logical_batch`; conflicting values are rejected. Scope may be declared as
`task_scope: full18` or the inherited `tasks: all18 ...` description.
Frozen data/prepared paths and rows 81000/4500/4500 must be retained.

`arm_budget(protocol, arm)` reads `arm_budgets` (or `arm_budget`) per-arm
descriptions. A/B use successful updates, warmup, and evaluation cadence. C/D
use trees per task, accepting `trees_per_task`, `iterations`, `max_trees`, or
`tree_iterations`. Integer entries also work. Tree-specific budgets never fall
back to the common NN update budget. Main's selected targets are A/B57600
updates, warm4320, VAL1440; C/D3000 trees/task on all81000 TRAIN rows, with one
GLOBAL `ntree_end` selected by18-task VAL regret, not per-task selection. The
launcher does not perform that selection or impose NN progress semantics on trees.

Micro-cap keys: `large_micro_cap` / `large_instance_micro_cap`.
Eval keys: `eval_batch_size` / `evaluation_batch`. Latest user defaults: cap352,
eval256 uniformly, including C/D's ignored NN-fit CLI arguments. No cap search
is performed; first-cycle memory verification belongs to main's later authorized run.
Per-arm settings are accepted in `launch.arms`, `arm_descriptions`, an `arms`
mapping, or top-level A/B/C/D descriptions; common settings may be top-level or
in `launch`. More-specific settings override common settings. Main chooses all
caps; no cap is inferred from model size. The runner and reporter must expose
`ARMS` in A/B/C/D order and `parser()`. The launcher never calls their
`protocol_lock()` or writes the scientific lock.

## CLI

Every command first activates `easynco`; paths below are repository-relative.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash code/V4/run_v4_r92_launch.sh template --arm A
python -B -m unittest code.V4.test_r92_launch -v
```

`template` is read-only and works before the runtime or protocol exists.
`parse` checks all 16 final stage argument lists on CPU without writing specs.
`gates` reads predecessor receipts only, never resource state.

After main finalizes the runtime/protocol and confirms focused CPU tests, the
following are explicit, separate operations (NOT performed by this delivery):

```bash
bash code/V4/run_v4_r92_launch.sh parse
bash code/V4/run_v4_r92_launch.sh specs --main-ready-evidence 'MAIN FINAL RUNTIME/PROTOCOL REVIEW' --runner-tests-passed
bash code/V4/run_v4_r92_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R92_launch/specs/A.json --main-ready-evidence 'MAIN FINAL RUNTIME/PROTOCOL REVIEW' --runner-tests-passed
bash code/V4/run_v4_r92_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R92_launch/specs/A.json --main-ready-evidence 'MAIN FINAL RUNTIME/PROTOCOL REVIEW' --runner-tests-passed
```

Repeat prepare/start explicitly per arm only when authorized. Sessions are
`r92_A`, `r92_B`, `r92_C`, `r92_D`. A source/spec change invalidates readiness.
Existing armed/claimed/failed launches are never implicitly overwritten or
retried. Resource checks occur only in the authorized claim/entry path, not at
template/spec/parse/prepare time. Internal `node-resources` and
`container-resources` commands serve that path; there is no resource-audit CLI.

## Predecessors

- A/C require R91 A_utility_ce/B_utility_probabilities train, TEST and LIB rc0,
  plus `R91_probability_link_control/report_only_recovery_complete.json` with
  `complete=true`, `state=recovered_report_only`, and its detailed report-only
  receipt. Both reports and comparison must finish rc0. Protected hashes must
  agree before/after; current checkpoint/VAL lock and original failed receipts
  must still match. No inference, retraining or checkpoint reselection is allowed.
- B/D require R89 A_adamw/B_muon_hybrid worker and controller train/LIB/PDF rc0,
  plus a complete report receipt. Training alone does not satisfy either gate.

No predecessor receipt, checkpoint, frozen helper, model or scientific module is
edited by this launcher. ai_docker's recognized stdout preamble is removed before
JSON parsing; stderr is kept separate and nonzero command status always fails.
