# R58 Preflight Findings

These are deployment diagnostics, not selector results. No full scenario label
release, model training, or test evaluation has completed at this stage.

## Batch Sensitivity

The initial FP32 RouteFinder probe on VRPTW used seed2 and the same100-customer
training instance9825. Singleton cost was33.24198739559521; in a16-instance batch,
its cost was34.801507205081215. The selected tours differed. Both costs were
recomputed by the independent scenario_v2 checker from their saved feasible
routes. This observation does not identify the internal cause or imply a label
error; it demonstrates sensitivity to the inference batch.

The batch indices were9825,9804,9805,9806,9807,9808,9809,9810,9811,9812,9813,
9814,9815,9816,9817,9818. The measured batch took0.4055775245651603seconds.
The deployment used ALL N starts,8 augmentations, multistart greedy and FP32
with matmul precision highest. Source fingerprints for that initial probe:

- `r58_scenario.py`: `aeec3cc5228b2d590e0ae370f1f60936326ca9a859bf6220eaf3e51bb0455bf3`
- `r58_environments.py`: `ada9bd601f6e423bfccb2c6695e913ccb7fa09435b1fadf38f405ef8a7fb5a8c`
- `r58_backends.py`: `14eb9468cea874736331fd1181378f5f2aad14a9a5cbfd34cc389bbace1438b9`
- `r58_labels.py`: `3428239313e467585f3ddedbdad162c9c27723b02b714a8a8f735252d8f9f28d`

Decision made before any scenario_v2 validation/test generation: RouteFinder,
MoSES_RF and MoSES_CaDA use singleton inference. All candidates must be checked
again after the final source freeze. This is a NEW deployment protocol, not a
claim to reproduce historical scalar costs.

## Release Boundaries

- Classical B, separate capacity counters, pickup-only routes permitted.
- No implicit finite depot deadline; explicit depot departure times are kept.
- Shared weights are frozen; no solver training or selector continuation.
- Missing or infeasible route rows block publication and training.
- Existing labels, solver source trees and other experiment queues are unchanged.
- Full relabeling is a multi-day workload; it is not included in the completed
  small-sample preflight or synthetic regression tests.
