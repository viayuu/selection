# R58 Implementation Validation

Date:2026-10-11. These checks validate implementation, not selector performance.
No real test instances or test costs were used by these checks.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -m unittest code.V4.test_r58 code.V4.test_r43 code.V4.test_r45 -q
bash -n code/V4/run_v4_r58.sh
python -m compileall -q code/V4/r58_scenario.py code/V4/r58_environments.py \
  code/V4/r58_backends.py code/V4/r58_labels.py code/V4/r58_data.py \
  code/V4/r58_experiment.py code/V4/r58_analysis.py code/V4/r58_pipeline.py
```

The53-test combined suite passed. Syntax checks passed. Existing R45 AST
fixtures emit a non-fatal invalid-escape DeprecationWarning; no unrelated
historical code was modified to suppress it.

## Covered Boundaries

- Classical backhaul precedence and independent delivery/pickup capacities.
- Pickup-only routes, open return billing, distance limits and explicit depot time.
- Customer service-start windows, exact visits and directed ATSP cost.
- No label input to physical solver state; forbidden actions fail closed.
- Nonfinite physical inputs and incorrectly scaled capacity tolerances rejected.
- New cost columns require matching deployment/input provenance.
- Synthetic54-dataset publication regenerates winners/ranks and refuses a missing row.
- Existing original labels remain unchanged in the publication fixture.
- Interrupted best/last saves, early-stop resume and test-checkpoint selection binding.
- Registry order, per-deployment inference batch rules and retained tail batches.
- Failed forced preflight cannot fall back to an older passing qualification.

Two read-only reviewer passes identified release, checkpoint-resume and stale
qualification edge cases. Those cases were fixed and covered by tests; the
focused final gate review found no remaining blocking finding.

The real train-only candidate preflight is a separate stage. Its current status
is recorded in`progress.json`, `deployment_preflight.csv`, `preflight/*.json`
and`pipeline_state.json`. Full fresh cost generation, a complete scenario
release, baseline training and locked test evaluation are not test-suite outputs
and must not be claimed from this document.

## Real Training Input Feasibility

All80000 unchanged training instances across the eight B tasks were checked
without running a solver or reading a cost label. Every customer can be served
on its own route under the new capacity, time and distance rules. Together with
unlimited vehicles and permitted pickup-only routes, this supplies a feasible
construction for each input; it does not certify any model-generated route.
Counts and exact rule scope are recorded in`single_customer_feasibility.json`.

The17 CPU-only budget estimator/controller tests also passed. They cover current/stale
qualification, warm versus cold timing, true-size weighting and honest incomplete
budget reporting, including rejection before locking or generation. A real
CPU-only`python -m code.V4.r58_pipeline --stage labels` also exited3 on the
181.637GPUh projection; no deployment lock or label column was created. These
checks do not establish new solver or selector performance.
