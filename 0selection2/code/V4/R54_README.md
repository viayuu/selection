# R54: Route-supervised pair specialist

The user message starts with "r45" but specifies the R54 follow-up to R53.
This implementation uses the R54 name and preserves every historical experiment.

## Method

- Reuse the R53A four-layer, 128-dimensional instance encoder and unweighted binary head.
- Initialize from scratch with exactly the same common parameter values and sample plan as R53A.
- Add two shared-across-nodes directed successor heads, one for each ReLD method.
- Predict another customer or END; exclude self, depot-as-customer, and padding.
- Optimize binary BCE plus the average normalized successor CE. Each instance/method
  first averages over valid customers, then divides by `max(log(Ncustomers), 1)`.
- A query/key compatibility score, an input-distance MLP, and an END MLP form each
  auxiliary head. No ground-truth route, cost, winner, or successor enters forward.
- Select strictly by integrated full18 validation actual regret, with the unchanged
  frozen R45A Top2 gate. Auxiliary heads are completely skipped at inference.

The [Joshi et al. paper, section 4.1](https://arxiv.org/html/1906.01227v2)
motivates learning representations from solution connections. Its original task is
undirected TSP edge classification; these directed VRP successor/END targets and
their normalized categorical loss are project adaptations, not a reproduction.

## Route Annotations

The user explicitly authorized saving all missing training routes and a fixed
size-stratified validation subset after receiving the missing-data estimate.
The original scalar cost and winner files are never modified.

`r54_routes.py` uses the R48-pinned weights and original native bridge, preserves
argmax/no-augmentation decoding and each native environment's POMO start rules,
and retains the first minimum-cost trajectory. It saves complete best routes,
successors, start identity, native/recomputed costs, and input/label hashes.
Input fields are checked before solving, network actions are checked against native
legal-action masks, and FP64 distance sums and historical scalar costs must agree.
Any mismatch stops annotation and training instead of silently relabeling.

The original R48 64-instance training routes are reused for OVRPTW. R52 validation
trajectory records are not imported. There are 300000 training solver-instance
routes and 1920 validation routes (64 instances per task, two methods). Validation
routes are only structure diagnostics, never training targets or selection criteria.
The same 24 exact-tied training instances excluded in R53 are excluded from binary
training here as well; their route records remain stored for complete coverage.

## Commands

Run within the repository and the `easynco` environment. On the current allocation,
bind only the authorized idle 3090 UUID, not the other GPU.

```bash
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
bash code/V4/run_v4_r54.sh precheck
bash code/V4/run_v4_r54.sh routes --allow-route-generation
bash code/V4/run_v4_r54.sh train
bash code/V4/run_v4_r54.sh evaluate
```

The combined `all` stage performs route generation, training, one locked-checkpoint
test evaluation, and analysis. Long jobs are launched through the existing Slurm
allocation in tmux; no new GPU allocation, solver download, or API call is needed.
Training resumes from its own `last.pt`, without changing protocol or annotations.

## Artifacts

Results live in `runs/R54_route_supervision/`. `route_plan.json`, route receipts,
configuration, protocol, sampling hashes, model precheck, learning curves,
classification predictions, structure predictions, and CSV/Markdown comparisons
are retained. Large route archives, checkpoints, and offline W&B data remain local
under the same directory and are excluded from Git. `RESULTS.md` is generated only
after training and the locked best-checkpoint evaluation actually complete.

R53A is a historical reference, not a freshly rerun strict ablation. The same
controller rules may yield different actual LR trajectories and stopping epochs.
Unrecoverable historical solver provenance remains explicitly documented.
