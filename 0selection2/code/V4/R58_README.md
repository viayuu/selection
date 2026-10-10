# R58: A Unified, Versioned Solver-Selection Scenario

This experiment repairs benchmark semantics, not a selector architecture.
Original inputs/splits, labels, solver source trees, and historical results remain
unchanged. The new data live under `runs/R58_unified_scenario/scenario_v2/`.

## Legacy Scenario Boundary

Existing cost tables and R25-R57 reports describe the legacy deployment
scenario. In particular, the eight B tasks cannot be treated as a certified
common feasible domain. Their old Oracle values are not the scenario_v2 Oracle.
This does not declare every historical non-B column incorrect or explain the
entire Top1 plateau. Historical files are preserved, not retrospectively edited.

No existing cost-only column has both a complete route-feasibility witness and
a fully bound match to the new locked deployment. This implementation therefore
uses fresh solves for all128 deployment columns, rather than certifying unknown
history from current defaults. That conservative choice expands the workload
beyond repairing only the eight B tasks and requires a multi-day solve budget.

## Explicit Task Decisions

- Classical backhaul: deliveries precede pickups within each vehicle route.
  Delivery and pickup totals separately obey vehicle capacity. Pure pickup routes
  are permitted. This is not the old signed-load rule and is not called MB.
- Customer service starts must lie within their windows; waiting and service
  consume time. Raw speed is used, or an explicitly fixed speed of1 if missing.
- A missing depot deadline means infinity, not an inherited cutoff of3.
- L limits per-vehicle travel distance, excluding waiting and service.
- Open-route delimiters do not charge return distance or return time.
- All customers must be visited exactly once. Vehicles are unlimited.

The executable `task_contract.json` and independent scalar checker are the
authority. The checker imports no solver environment or native feasibility mask.

## New Deployments

All original22 identities and per-problem candidate rosters are retained for
qualification. MVRP ReLD/MTPOMO/MVMoE use a new common physical transition with
the original fixed learned weights. ReLD's old signed-load context is replaced by
remaining capacity in the current classical phase. All physical customer starts
are used, including pickup starts. These are deliberately NEW deployments, not
claims to reproduce unmodified paper performance or old scalar labels.

RouteFinder/MoSES explicitly receive classical class1, raw limits and customer
windows, and an infinite missing depot deadline. Their action masks use the same
new contract. Duplicate checkpoint parameter aliases may be restored only from
another checkpoint entry referring to the identical registered parameter.
Missing real weights are never randomly filled.

RouteFinder-family inference is FP32 with all customer starts and8 fixed
dihedral augmentations. Its deployment batch is locked to1: a train-only probe
found that even FP32 batched inference could change a physical instance's tour
and cost. See `runs/R58_unified_scenario/preflight_findings.md`. The other MVRP
deployments use fixed size-homogeneous batches of up to16, retaining the smaller tail; TSP,
CVRP and ATSP deployments use singleton inference. These batch rules are part
of the new deployment identity, not inferred historical settings.

Other wrappers use the available fixed native weights and fully documented new
decoding recipes. The GLOP identity denotes the available insertion-only
deployment, not a full trained GLOP network. All profiles are saved before any
new validation/test column is solved.

No old cost-only column is automatically certified. Every new cost must have a
saved route that passes independent feasibility and cost reconstruction.
Qualification failure stops publication; it does not silently remove a method.
A genuine unsupported deployment requires an explicitly revised eligibility
decision before the protocol is locked, never per-instance candidate removal.

## Execution

```bash
bash code/V4/run_v4_r58.sh --stage preflight
bash code/V4/run_v4_r58.sh --stage preflight --force-preflight
bash code/V4/run_v4_r58.sh --stage all
```

Use tmux for the long full run. Only the specified idle3090 UUID is used. The
script uses the existing Slurm allocation on a login host without GPU access;
override `R58_SLURM_JOB_ID` only if that allocation has expired. Existing queues
and other GPUs are untouched.

The complete fresh solve includes1,536,000 solver-instance evaluations. Current
train-only timing indicates a multi-day single-GPU workload, especially for the
TSP diffusion methods. Finish preflight and confirm this budget before starting
`--stage all`; preflight alone does not generate full labels or train a model.
Full generation handles the eight B tasks first, then the other tasks and TSP
last. It still locks every deployment before generating any validation/test
label; ordering does not change candidate pools or per-column inference plans.

The existing read-only `python_compat` overlay provides TorchRL0.6/TensorDict0.6.2
compatible with PyTorch2.5. The optional PyG accelerator is disabled when its
binary cannot load on gpu03; supported torch-sparse operators remain real.
No package is installed into the shared conda environment.

Generation is resumable by physical instance and preserves its fixed inference
batch even when resuming. Random seeds, true-size grouping, batch width, source
and checkpoint identities are saved in `deployments.lock.json`. This lock is
created before new validation/test generation. Failed instances remain in the
scenario and block release instead of being discarded.

## Full Baseline

After all54 datasets pass release validation, one R45A-handcrafted model is
trained from scratch with seed2. Original R43A comparison/R-Drop objectives,
batch128, AdamW1e-4, warmup3, and at most40epochs are retained. Selection and
early stopping use FULL18-task validation actual regret. The locked best model
is evaluated once on scenario_v2 test.

References are fitted only on the new training costs: a train-selected fixed
SBS and true-size conditional majority/mean-relative-gap priors. Old and new
scenario scores must not be presented as a matched architecture improvement.

Primary outputs: `task_contract.json`, `deployment_preflight.csv`,
`deployments.lock.json`, `scenario_v2/release.json`, saved cost/route columns,
`comparison.csv`, `learning_curves.png`, `test_predictions/`, and `RESULTS.md`.

## References

- [RouteFinder](https://arxiv.org/html/2406.15007v3): classical versus mixed backhaul.
- [Classical VRPB](https://epubs.siam.org/doi/10.1137/1.9780898718515.ch8).
- [ASlib](https://arxiv.org/abs/1506.02465): explicit algorithm-selection scenarios.
