# R121 VRPBL/reld_mt Route-Limit Failure: Read-Only Diagnosis

Scope: source localization only, 2026-10-30. No solver/GPU execution, real-tour
replay, label recomputation, source patch, retry, or active pin modification.
Main owns recovery of the actual failed tour and the numerical recomputation.

## Incident And Conclusion

- Main confirmed the 07:07 v2 lane0 bootstrap fatal originates from
  `collection_persistent_v1/unavailable/VRPBL__reld_mt.json`, after902 successful
  production labels; the903rd endpoint is `VRPBL_src98457_original`, NOT a CF/z
  endpoint. These are main-provided observations, not a second receipt audit.
- The exact exception `Route-length limit violation` is raised by
  `r114_response_data.check_route`, called after `backend.run` returns. The
  actual function is `check_route`, not a separate `audit_solution` function.
- Source review finds matching closed-route geometry, one depot return per
  route, and per-route length resets. The supplied `route_limit` is forwarded;
  no missing-L-field, open/closed-route swap, total-vs-per-route limit comparison,
  or accumulation across the902 preceding instances was identified here.
- A concrete audit mismatch exists: bridge audit uses FP64 and `+1e-4`; R114
  independent audit uses FP64 and `+1e-5`. ReLD's actual L mask uses FP32 and
  `+1e-5`. A near-boundary numerical/tolerance disagreement is therefore a
  supported hypothesis, NOT yet the measured explanation for this tour.
- No actual overrun magnitude or native mask witness is available to this
  sidecar. Substantive infeasibility, numerical boundary, or an unforeseen
  runtime/adapter mismatch cannot be conclusively classified from source alone.

## Actual Call And Input Chain

1. [r121_collection.py](/public/home/shiys/0selection2/code/V4/r121_collection.py:156)
   validates the pinned input. For VRPBL it calls `session.solve(task_id)`;
   universal explicit-CPU TSP/ATSP input construction does not change this branch.
2. [r114_group_worker.py](/public/home/shiys/0selection2/code/V4/r114_group_worker.py:114)
   reads the endpoint item unchanged for VRPBL, resets the task seed, calls
   `backend.run([native])`, then calls `data.check_route` at line133. The
   successful receipt is constructed AFTER that audit; a failure here need not
   retain the just-returned tour/cost in a successful per-task record.
3. [r114_response_data.py](/public/home/shiys/0selection2/code/V4/r114_response_data.py:96)
   maps current `reld_mt` to native `RELD_MOEL`, not a free choice of ReLD
   checkpoint or an alias for a per-instance minimum.
4. [train_100k_backends.py](/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py:245)
   calls `collate_problem_batch`, runs the original no-augmentation environment,
   obtains the winning POMO tour and native cost, and calls `audit_tours`
   BEFORE returning `InferenceResult` at line251.
5. [reld_bridge_common.py](/public/home/shiys/easynco_v3_bridge/scripts/reld_bridge_common.py:312)
   takes `float(sample['route_limit'])`, then returns the VRPBL tuple
   `(depot_xy, node_xy, node_demand, torch.tensor(route_limit, dtype=float32))`.
   `normalize_tensor_like` at line289 casts coordinates/demands to FP32 but
   does not geometrically rescale them. `_run_no_aug_batch` at line444 moves
   this tuple to the bound device, creates a NEW environment on every call,
   and executes `load_problems(..., problems=batch)` followed by `reset()`.
6. [VRPBLEnv.py](/public/home/shiys/reld-nco-main/Multi-Task/envs/VRPBLEnv.py:112)
   unpacks the supplied fourth field, reshapes it to `[batch,1]`, and stores
   `self.route_limit`. The random-generator default is NOT substituted when
   `problems` is supplied. The environment class is selected by
   [utils.py](/public/home/shiys/reld-nco-main/Multi-Task/utils.py:106).

For this frozen protocol, `route_limit == 3.0` is also explicitly checked by
[R114 input validation](/public/home/shiys/0selection2/code/V4/r114_response_data.py:217),
including a singleton-route feasibility certificate. This is a fact about the
current input contract, not a universal VRP/ReLD distribution assumption.

## Geometry, Reset, Cost And Precision

| Layer | Length semantics | Precision and acceptance |
| --- | --- | --- |
| R114 independent `check_route` | Split at every depot0; reset `distance=0` per nonempty segment; sum depot-to-customers and final customer-to-depot once | Original input coordinates promoted to FP64; reject `distance > R + 1e-5` |
| Native bridge `audit_tours` | Same depot boundaries and closed edge sequence for VRPBL | FP64 edge calculation; reject `distance > R + 1e-4` |
| ReLD `VRPBLEnv` L mask | Within-route prefix plus current-to-next plus next-to-depot; reset prefix on depot | FP32 coordinates, state and arithmetic; reject `candidate_length > R_tensor + 1e-5` |

R114 details are at
[check_route](/public/home/shiys/0selection2/code/V4/r114_response_data.py:365):
missing leading/trailing depot is supplied, consecutive empty depot segments are
ignored, `distance` resets at line390, return edge is added at line406, L check
is line409. `fields` at line181 promotes original coordinates to FP64. B signed
load accounting is separate and unchanged; this failure is specifically the L
check, not a new interpretation of B.

Native bridge details are at
[audit_tours](/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py:99):
the closed-route edge array includes the return edge and is summed per segment;
the looser L check is line116. Neither audit limits the SUM across all routes to3.

Native state details are at
[VRPBLEnv.step](/public/home/shiys/reld-nco-main/Multi-Task/envs/VRPBLEnv.py:230):
each selected edge updates `self.length`, then depot arrival resets it to0.
The preceding lookahead mask at line274 includes the necessary return-to-depot
edge, so reset does not itself imply that return distance was omitted.
The epsilon is defined at line263. Native reward at line306 sums the cyclic
selected sequence, including interspersed depot returns, across ALL routes.
The reset constraint-state length and total native objective are different
quantities; comparing only the total objective with3 would be a false diagnosis.

## What Follows Before Seeing The Real Tour

Conditional on the pinned runtime using the reviewed chain and the same returned
item/tour, the error location implies bridge `audit_tours` already passed. Thus
the violating segment's original-coordinate FP64 overrun should satisfy:

```text
delta64 = independently_recomputed_closed_segment_length - item.route_limit
1e-5 < delta64 <= 1e-4     # up to ordinary FP64 summation-order rounding
```

This is a source-derived bound, not a measured interval for the missing tour.
It makes an order-one length overrun inconsistent with the ordinary call chain,
but DOES NOT show that every overrun below1e-4 is harmless FP32 rounding.
ReLD does not intentionally have a1e-4 L-mask tolerance: its actual mask uses
1e-5. Its FP32 threshold is itself rounded, and prefix additions/norms can differ
from the independent FP64 calculation. That difference must be measured.

## Small Evidence Package For Main, No Duplicate Replay Here

For the one actual failed returned tour, retain its input/task/seed/profile/source
and physical-binding identity, native cost, backend audit and unmodified raw tour
BEFORE the independent check raises. Main's sidecar recomputation should record:

1. Every depot-delimited nonempty route, its customer count and original FP64
   closed length; identify the first violating route and maximum `delta64`.
2. The same edge sequence with input coordinates first rounded to FP32 but
   distances accumulated in FP64, separating input casting from FP32 arithmetic.
3. Sequential FP32 prefix lengths and return-lookahead values, the FP32
   `R_tensor + 1e-5` threshold, and the residual at the violating route's final
   customer/return. After depot arrival `self.length` has reset, so that value
   alone cannot diagnose the preceding route. CPU FP32 emulation is a diagnostic,
   not proof of bitwise identity with the real CUDA mask/norm implementation.
4. Check original/cast FP64 lengths against both1e-5 and1e-4, and compare the
   sum of closed segment lengths with the reported native total separately.

A measured native-FP32-accepted margin that explains the FP64 crossing supports
a numerical boundary diagnosis. A materially incompatible margin, differing
input/limit, or a selected node violating the ACTUAL native mask requires a
versioned adapter/runtime investigation instead. Neither branch authorizes
changing frozen tolerance, repairing the route, trying another seed, or counting
an unverified label as complete.

Keep the902 valid prefix and original failure bytes. Full89 completeness remains
blocked for this deployment; do not drop the candidate or train on a partial
pool. Other authorized GPU tasks need not be interrupted. No historical labels
have been shown wrong by this source-only check; R58 remains deferred and no
full-benchmark relabeling is proposed.

## Read-Only Source Snapshot

These are current local source hashes, not a new experiment/protocol lock or a
claim to have revalidated every remote deployment pin:

| File | SHA256 |
| --- | --- |
| `r114_response_data.py` | `d56214f7bd26367eab1db41c4053afd2f9a59bd4030ce7b5ad8f895a789b1345` |
| `r114_group_worker.py` | `a64a4a13f50f8b6584af03984f7619f325fdab4a7da36d1db229949971369276` |
| bridge `train_100k_backends.py` | `46a1dd32b169366673218a6315aee1f44d256393b7bacb54968ec0122500d117` |
| bridge `reld_bridge_common.py` | `93ae0a5c765e32314b8d868c722b3a0d76353b2298d713c5a7db497a4c3b84a2` |
| native `Multi-Task/envs/VRPBLEnv.py` | `3cacadf53e8a3235fe54e60a9609f438bb43cf61404eefab8bd9efcb9d389f28` |

Only this new diagnosis document was written. Frozen collector, learner, model,
driver, READY, isolation seal, input/label files and existing failure records were
not edited. No GPU inspection/execution, solver restart, Git operation or new
experiment was performed.
