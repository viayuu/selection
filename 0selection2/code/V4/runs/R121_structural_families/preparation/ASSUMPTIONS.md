# CPU-Only D1 Structural Family Preparation

This is execution preparation by Sol, not a new scientific result. Astra owns
the scientific refinement. R120 keeps all four GPUs through TEST/LIB/report;
no solver, model inference, GPU probe, waiter, or Git command is run here.
Only the two new R121 Python files and this preparation directory are owned.

## Decisions Before Implementation

| ID | Decision |
| --- | --- |
| S01 | Use the frozen new90k release:18 tasks,81000/4500/4500,89 legal task/candidate deployments. Keep global IDs, B filtering, single reld_mt alias and R58 deferral. Never read individual old labels for base selection or copy them to new endpoints. |
| S02 | TRAIN bases only, input-stratified selection. Default prospective budget256 bases/task is a versioned collection budget, not an executed experiment or a claim that this is sufficient for learning. All endpoints of a source base AND exact shared base geometry/matrix stay in one internal TRAIN-only family partition. Natural VAL/TEST membership is unchanged. |
| S03 | One original endpoint plus a2x2 factorial of TWO substantial task-specific axes, five endpoints/base. Original endpoints also require fresh full-pool labels under the new collection version. No simple demand micro-noise, rare-winner-only sampling, family endpoint as independent base, or Ridge-to-MLP substitution. |
| S04 | TSP axes: cluster separation/connectivity and anisotropy. ATSP axes: inter-block connectivity and competing directed cycles, followed by directed shortest-path closure. They change costs/geometry, not just relabel nodes or add a potential that cancels on tours. |
| S05 | Routing axis1 is cluster separation. Axis2 is TW compatibility for TW tasks, signed-role spatial segregation for B tasks without TW, fixed-budget radial exposure for L tasks without B/TW, and whole-population demand/spatial coupling for CVRP/OVRP. Keep per-node B signs fixed; no classical backhaul repair or new precedence. |
| S06 | Reuse native generation entry points from generate_train_100k.build_shard/load_generators for a per-base proposal/latent, and R114 plain/load_pool/read_mapping/validation/adapters. Endpoints preserve native fields, node count, capacity1, route_limit3 and service durations. Routing geometry is constructed within original singleton reachability; new TW endpoints have explicit singleton certificates under unchanged depot deadline3. |
| S07 | Full candidate cost vectors, masks, native FP64-promoted costs, all-best ties and stable-global-ID winner remain separate from model inputs. Log-centered advantages require complete positive costs; zero/missing/invalid costs are not epsilon-filled. Family responses derive from full endpoint vectors and are TRAIN supervision only. |
| S08 | Use R114 production deployment metadata for non-CVRP and the corrected CVRP cold lock for CVRP. Copy budgets/checkpoint SHA/profile identities, not current defaults. This CPU pass checks metadata and file presence, NOT all weight bytes or runtime. No deployment is called currently runtime-qualified. |
| S09 | R11416-task coverage does not include TSP/ATSP responses. Their13 deployment recipes exist, but new runtime/profile/route checks remain required. Historical MATNET role draws are not restored; the new version explicitly uses per-base seeded native roles, NOT historical scalar replay or fixed identity roles. |
| S10 | Future worker is explicitly gated by a main-authored R120 TEST/LIB/report resource-release receipt, a new-protocol acknowledgement and a chosen device. Fresh backend per task, native singleton execution, recorded common initial seed per base/method. No semantic/pathwise RNG alignment claim. No automatic retries or live-chain edits. |
| S11 | Structural input coverage is not proved candidate-advantage coverage. Until new labels exist, every candidate's observed-win/advantage-region coverage is UNKNOWN, never fabricated. Keep all legal candidates regardless of historical win frequency. |

## Fixed Native Interfaces Located

- `r114_response_data.load_pool/read_mapping/plain/validate_input/check_route`
  are reusable. Its `verify_manifest/dispatch` are16-task intervention-specific
  and cannot validate the new TSP/ATSP/factorial manifests unchanged.
- Native generator: `easynco_v3_bridge/scripts/generate_train_100k.py`:
  `load_generators`, `build_shard`; TSP/CVRP use NSS generators, ATSP uses the
  original CPU matrix generator, MVRP uses the native diverse generator.
- Future factory: `train_100k_sampling:make_confirmed_backend`, with
  `r114_group_worker.NativeSession` initialization/profile binding. Input and
  tour audits need the new TSP/ATSP branch; the routing branch stays R114-native.
- Old R114 corrected complete evidence:512 bases/2048 endpoints/9600 labels,
 16 tasks. R115's24-condition Ridge result failed its gate. These are provenance,
  not new D1 results, and are not recomputed in this subtask.

The family definitions are explicit engineering proposals for Astra/main to
refine. Input differences and feasibility alone do not establish useful winner
changes or a scientifically sufficient population.
