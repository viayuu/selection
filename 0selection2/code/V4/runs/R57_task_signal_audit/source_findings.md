# R57 Independent Source Findings

Read-only current-source review, alongside train/val artifact checks. Current
source behavior is not automatically a certified historical invocation. No
historical defaults, solver/core edits, or checkpoint substitutions were used.

## Backhaul Contract

- `/public/home/shiys/easynco_v3_bridge/scripts/prepare_paper_source_data.py:70`
  splits signed demands into positive linehaul/backhaul fields. At101 the
  returned batch has no `backhaul_class`; capacity is1, depot TW is `[0,inf]`.
- `/public/home/shiys/methods论文_mineru_output/source_code/routefinder/routefinder/envs/mtvrp/env.py:213`
  defaults to class1. At309-351 it bounds delivery/pickup separately and forbids
  delivery after pickup within a route. The corresponding MoSES defaults/mask
  are `moses_vrp/envs/mtvrp/env.py:235` and392.
- `/public/home/shiys/reld-nco-main/Multi-Task/envs/VRPBEnv.py:221`
  subtracts signed demand from residual load. At247-270 it resets to1 at depot
  while deliveries remain (otherwise0) and bounds signed residual load. There
  is no delivery-after-pickup prohibition. `VRPBLTWEnv.py:271` is analogous.
- `/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py:614`
  installs a later RF mask/start-selector replacement. This later expansion
  adapter is NOT the original paper-source label bridge and is not substituted.

The independent saved-route check quantifies this discrepancy. It does not
pretend raw signed-demand fields alone unambiguously define the intended class.
Both ReLD methods share the signed-load contract; the discrepancy alone cannot
explain their within-pair prediction difficulty.

## Runtime Roles And Decoding

- `/public/home/shiys/reld-nco-main/Multi-Task/envs/VRPBEnv.py:130`
  truncates B starts to `min(floor(0.8N), configured)` positive-demand indices.
  When positive-customer count exceeds that number, indexing can change the
  physical start subset. All eight B environments follow this pattern.
- `/public/home/shiys/reld-nco-main/CVRP/CVRPModel.py:89`
  chooses learned top-k first customers, then argmax; this is NOT a prefix of IDs.
- `/public/home/shiys/methods论文_mineru_output/source_code/bq-nco/model/model.py:46`
  uses endpoint slots; `learning/cvrp/decoding.py:237` reformats the next subproblem.
  BQ/LEHD TSP candidate source has physical endpoint/index roles, but the
  historical NSS invocation is not bound; no historical effect size is claimed.
- `/public/home/shiys/EasyNCO/neural_solvers/methods/lehd/lehd_decoder.py:67`
  reads selected endpoints. CVRP depot0 is a real input role, unlike a generic
  positional token. The available implementation is not proof of the NSS recipe.
- `/public/home/shiys/EasyNCO/neural_solvers/methods/matnet/policy.py:172`
  assigns random column slots; greedy decoding does not remove initialization RNG.
  MATPOENET uses NN reordering/position encoding in `envs/ATSPEnv.py:285`.
- `/public/home/shiys/EasyNCO/eval.py:89` does not forward the top-level decoder
  setting into the MTPOMO/MVMOE module. Current `phases/rl/ar_reinforce.py:256`
  defaults to sampling, ultimately `neural_solvers/utils/post_search.py:46`.
  The historical job/source binding remains missing; neither greedy nor sampling
  is asserted as the recovered historical answer.

## Other Constraint And Input Boundaries

- ReLD closed TW checks return to depot3 in
  `/public/home/shiys/reld-nco-main/Multi-Task/envs/VRPBLTWEnv.py:311`.
  `OVRPTWEnv.py:269` explicitly disables that depot cutoff for open routes.
  Thus `[0,3]` versus `[0,inf]` alone is not an OVRPTW feasibility contradiction.
- Open scoring excludes every edge entering depot, including route separators:
  RF `routefinder/envs/mtvrp/env.py:373`, ReLD `OVRPTWEnv.py:309`.
- `code/unified_selector/data.py:138` marks depot and passes raw customer fields,
  but does not mark native endpoint slots, runtime seed, depot closing time or
  every method's physical start subset. Capacity omission is harmless only while
  the checked input convention remains the fixed1; do not assume this outside it.
- The loader's index-derived coordinate-distribution field is explicitly ignored
  by the fixed R45A (`args.json:347`, `ignore_coord_dist=true`). It is not used to
  form R57 strata or train priors.
- `solver_source_manifest.json` and the R45 completion evidence distinguish
  exact cost ancestry from invocation binding. In particular OMNI TSP train is
  EasyNCO-derived while validation is NSS-derived; this is a gap, not proof the
  deployed algorithms were different.

## Current Runtime Blockers

The existing easynco TorchRL native extension fails with an undefined ATen
`clamp` symbol before GLOP/MATNET/MATPOENET/ICAM inference. RF additionally needs
unrecovered runtime/import bindings (`rl4co` is absent in this environment).
NSS original generation recipes and effective MTPOMO/MVMOE decoder/environment
bindings are missing. Full history is not certified for any pair merely because
its source or a local weight exists. See per-pair ledger and actual logs.
