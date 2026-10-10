# R57: Task Contract and Selection Signal Audit

R57 does not train, relabel, or evaluate test. Numerical analysis covers all18
problems and all128 legal problem/solver combinations. It separates artifact
identity, recovered invocation settings, current-source behavior, and actual
runtime verification; none is substituted for another.

Access disclosure: numerical/runtime work uses train/val only. The independent
reviewer also read full bytes of128 historical test cost files for SHA256
identity checks, without parsing costs or evaluating test performance. This
was not metadata-only access; it is explicitly retained in the report/config.

## Run

```bash
bash code/V4/run_v4_r57.sh cpu

# Inside the existing gpu03 allocation, on the user-authorized idle3090 only:
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G \
  env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d \
  bash code/V4/run_v4_r57.sh gpu

bash code/V4/run_v4_r57.sh analysis
```

Long jobs should run under tmux. Recheck allocation/GPU availability before
reusing the example. No other GPU, source solver, data generator or experiment
queue is modified. There is no API call or additional solver-label generation.

## Evidence

Outputs are in `code/V4/runs/R57_task_signal_audit/`:

- `RESULTS.md`, `config.json`: findings, coverage boundaries, commands and hashes.
- `deployment_audit.json/csv`:128 pairs /256 train-val columns, ancestry and
  actual recovered per-split deployment descriptions. No assumed R45 corpus
  defaults are accepted as historical configuration.
- `input_inventory.csv`, `audit_indices.json`: actual raw fields/scalars and
  fixed, size-stratified16 train/16 validation indices for every problem.
- `source_findings.md`, `current_source_inventory.json`: concrete source facts
  with file/line references, and current-file hashes rather than historical ones.
- `route_checks.csv.gz`, `route_check_summary.csv`: independent checks of the
  existing300,000 training and1,920 fixed validation R54 routes; no native masks.
- `route_violation_examples.json`, `contract_scope_budget.csv`: reproducible
  examples and affected old-label decision budgets, not hypothetical new scores.
- `signal_comparison.csv`, `signal_metrics.csv`, `instance_signal.png`:
  train-only majority/cost priors, frozen R45A, and100 exact-size score permutations.
- `signal_predictions/`, `permutation_maps.npz`, `prediction_replay.json`:
  validation predictions, fixed maps and metric replay.
- `runtime/`, `runtime_summary.csv`, `runtime_coverage.csv`, `solver_audit.csv`:
  recovered-recipe preflights, independent repeats and explicit runtime blockers.
- `selector_permutation.csv`, `permutation_audit.csv`: whole-model score changes
  and ReLD pair-only winner changes; no partial-pool winner is called a full-pool winner.
- `blocked_dependencies.md`, `blocked_runtime_deployments.csv`: missing original
  invocation bindings. Current TorchRL ABI failure is retained in preflight logs.

Classical backhaul and signed-residual-load compatibility are separate checks.
Raw data do not specify `backhaul_class` or depot closing time; neither is
silently inferred. A depot deadline3 check is separately reported for closed
time-window routes. Single-task ReLD_CVRP uses learned top-k initial branches,
not the first100 customer IDs. Multi-task B environments truncate positive
starts to floor(0.8N); role-preserving permutations retain that physical subset.

ALL percentages and regret use equal problem weights. Permutation intervals
describe conditional randomization only, not confidence intervals over future
datasets or an achievable-accuracy bound. Singleton exact-size strata are
explicitly excluded in the separate eligible-only comparison. Randomized cyclic
shifts are valid but nonuniform derangements; no uniform permutation-test
p-value is claimed.
