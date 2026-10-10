# R58: Unified scenario_v2

**Status: not complete. No new model result is claimed.**

- Qualified deployment preflights: 128/128.
- Full independently checked scenario published: False.
- Full baseline training completed: False.
- Locked checkpoint test completed: False.

## Locked semantic decisions
- Classical backhaul, separate delivery/pickup capacity, delivery-before-pickup within each route.
- Pure pickup routes are allowed. No implicit finite depot deadline.
- L limits travel distance, not waiting/service. Open routes exclude depot return distance/time.
- Fixed original weights with explicitly NEW adapted deployments, not historical reproduction.
- Every saved route must pass a solver-independent simulation before a cost can be published.
- All old instances, labels, results and solver source trees remain unchanged.

## Completion gates
All candidate preflights must pass before deployment lock; all54 original train/val/test datasets
must have complete feasible cost vectors before training. Missing or invalid rows stop publication;
they do not remove difficult instances or change per-instance candidate eligibility.

See task_contract.json, deployment_preflight.csv, preflight/*.json, and pipeline_state.json.
The provisional timing sum includes only probed deployments and is not a total-budget guarantee.
