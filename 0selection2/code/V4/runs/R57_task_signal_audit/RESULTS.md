# R57: Task Contracts and Instance-Level Selection Signal

## Scope

No selector training, test evaluation, new instances, labels or checkpoint changes. Seed=2.
All18 tasks / 22 solver names / 128 legal task-solver pairs / 256 train-val cost columns audited.
Numerical results use original FP64 costs and native winner; ALL is an equal-weight problem macro.
All 256 checked train/val cost columns match their saved result files exactly; native winner is a minimum-cost method for every checked instance. These checks establish alignment, not common task semantics.
Only the previously idle RTX3090 was used. Existing unrelated work was not stopped.

Access disclosure: numerical analysis and runtime probes use train/val only. The independent reviewer additionally read full bytes of 128 historical test cost files for SHA256 identity checks; no test cost values were parsed and no test performance was evaluated. This was file-content access, not metadata-only.

## Main Findings

### 1. A concrete backhaul task-contract discrepancy, not a cost-replay failure

Independently checked 300,000 saved training routes and 1,920 fixed validation diagnostic routes. Under classical B (deliveries before pickups, separate per-route delivery/pickup capacities), 159,993/160,000 training B routes and 1,024/1,024 validation B routes fail.
All routes pass the independently simulated native signed-residual-load rule. No customer-TW or route-length violations were found; maximum training cost error is 7.65812726e-06 (original replay tolerance: atol=5e-5, rtol=2e-6).
The existing ReLD B environments allow interleaved deliveries/pickups and permit pickup demand to restore residual load. RF/MoSES default to classical backhaul class1; the original adapter does not override it. These are different feasible-route contracts, not alternative names for the same capacity test.
Raw data record signed demand but no backhaul_class. Therefore first decide and version the intended task contract; do not silently pick one, overwrite costs, or declare every historical label corrupt.
The eight affected problem types account for 53.57% of R45A old-validation regret. Cross-current-contract selected/winner comparisons account for 6.05%. These are scope budgets, NOT recoverable improvement estimates. Full historical RF/MoSES deployments remain unclosed.
**This does not explain the MOEL/MTL binary plateau by itself: both ReLD methods use the same signed-load contract.**
See route_check_summary.csv, route_checks.csv.gz, route_violation_examples.json and contract_scope_budget.csv.

### 2. The baseline does use instance information beyond problem/size priors

| Strategy | Val Top1 (%) | Actual regret (%) | Mean raw cost | Net cost vs R45A |
|---|---:|---:|---:|---:|
| R45A | 49.1167 | 0.926048 | 11.16851657 | +0.00000000 |
| conditional_majority | 42.4667 | 1.804249 | 11.33505458 | +0.16653801 |
| conditional_mean_gap | 41.2056 | 1.660354 | 11.31366971 | +0.14515315 |
| permuted_R45A | 39.7110 | 2.148471 | 11.38035353 | +0.21183696 |

Priors are fit only on train: exact actual size, any varying scalar global constraints, and fixed 20-observation problem-global smoothing. In these actual datasets capacity=1 and route_limit=3 are constant; no coordinate distribution was guessed from indices. Majority native winner and minimum mean relative gap are different objectives and are reported separately.
100 fixed-seed score-row derangements are within problem x exact size. Singleton rows remain unchanged in full-scope metrics and are excluded in the separately reported permutable_only comparison. The sampler uses randomized cyclic shifts, yielding valid but not uniformly sampled derangements; no uniform-permutation-test p-value is claimed. Coverage: TSP 88.9%, CVRP 89.3%, other tasks100%; equal-task macro98.7889%. Reported permutation quantiles describe this conditional randomization, not a population confidence interval or accuracy ceiling.
Per-problem results are below. Variation across problems matters: a common ALL plateau is not proof of a common information-free representation or one universal network bug.

| Problem | R45A Top1 | Majority Top1 | Gap-prior Top1 | Permuted Top1 | R45A regret | Permuted regret |
|---|---:|---:|---:|---:|---:|---:|
| TSP | 38.50% | 29.80% | 14.90% | 28.18% | 0.5616% | 1.2609% |
| CVRP | 47.90% | 27.60% | 21.90% | 27.98% | 0.6764% | 1.8848% |
| ATSP | 78.50% | 77.70% | 70.50% | 77.22% | 0.3133% | 0.3725% |
| OVRP | 38.00% | 37.40% | 37.30% | 34.96% | 0.6627% | 0.7444% |
| VRPB | 55.40% | 49.80% | 50.40% | 44.93% | 0.6953% | 1.1781% |
| VRPL | 33.50% | 29.50% | 29.40% | 28.59% | 0.4511% | 0.5840% |
| VRPTW | 47.00% | 24.50% | 27.20% | 24.04% | 1.3735% | 5.5892% |
| OVRPTW | 51.00% | 47.80% | 47.20% | 47.66% | 0.8586% | 1.0436% |
| OVRPB | 58.30% | 54.10% | 54.50% | 49.80% | 0.9067% | 1.3249% |
| OVRPL | 40.20% | 38.40% | 36.40% | 35.15% | 0.7206% | 0.8266% |
| VRPBL | 52.00% | 52.00% | 52.80% | 44.64% | 0.8154% | 1.2600% |
| VRPBTW | 44.40% | 35.70% | 37.20% | 30.05% | 1.6355% | 6.0052% |
| VRPLTW | 45.70% | 30.20% | 30.40% | 25.28% | 1.2364% | 5.6693% |
| OVRPBL | 58.80% | 54.70% | 55.50% | 49.69% | 0.9032% | 1.3674% |
| OVRPBTW | 50.00% | 44.50% | 45.70% | 44.81% | 1.1592% | 1.4279% |
| OVRPLTW | 50.80% | 47.90% | 49.10% | 46.06% | 0.8850% | 1.0740% |
| VRPBLTW | 44.00% | 34.80% | 34.70% | 29.96% | 1.6550% | 5.6412% |
| OVRPBLTW | 50.10% | 48.00% | 46.60% | 45.80% | 1.1594% | 1.4184% |

### 3. Runtime verification is explicitly partial

Original-input runtime completed for 31/128 legal pairs. The full matrix remains covered by static input/deployment/label inspection. No row is called fully historically certified.
EasyNCO GLOP/MATNET/MATPOENET/ICAM preflights fail in the existing environment at TorchRL native-library ABI (undefined ATen clamp symbol). We did not install another dependency set and present it as the historical environment.
NSS import recipes, MTPOMO/MVMOE effective decoder, and RF/MoSES runtime/import bindings are still unclosed. ICAM_ATSP/UniCO have current bridge-derived settings but no historical resolved snapshot; no default substitution was performed.
Original runs are independent processes. Permutations synchronize node attributes and keep the depot fixed. For B ReLD, preserve_roles keeps the physical truncated positive-demand start set, while reassign_roles may change it. For single-task ReLD_CVRP starts are learned top-k, NOT first100 IDs; its start-set field is left unverified.
All 992 original solver-instance costs reproduce the historical scalar costs within atol=5e-5, rtol=2e-6. This does not certify unrecorded historical source/dependency revisions.
No full-pool winner change is inferred from partial candidates. ReLD pair changes are explicitly labeled pair-only.
See runtime_coverage.csv, runtime_summary.csv, selector_permutation.csv, permutation_audit.csv and runtime/* logs.

| Probe | Solver-instance rows | Cost changed beyond tolerance | Pair comparisons | Pair winner flips (strict / numeric) |
|---|---:|---:|---:|---:|
| original_1 | 992 | 0 | 480 | 0 / 0 |
| preserve_roles | 992 | 1 | 480 | 0 / 0 |
| reassign_roles | 992 | 16 | 480 | 2 / 2 |

Frozen R45A: 1152 synchronized node-permutation comparisons; 0 greedy choices change; maximum absolute logit difference 1.04904175e-05. ATSP is measured rather than assumed invariant. Runtime affected historical-regret budgets are in runtime_affected_budget.csv, limited to actually inspected validation rows; not extrapolated.

### 4. A measured execution-role mismatch, with limited observed scope

Unlike a pure renumbering, reassign_roles can change the physical subset of positive-demand POMO starts when the B environment truncates that set. The selector has no corresponding marker. The following pair winners actually flip beyond numerical tolerance; these are not hypothetical examples.

| Problem / split / index | Original MOEL / MTL cost | Reassigned MOEL / MTL cost |
|---|---:|---:|
| OVRPB / train / 278 | 5.4303050 / 5.4693179 | 5.4303050 / 5.3209229 |
| OVRPBL / train / 667 | 5.0614266 / 5.0323811 | 5.0614266 / 5.0775280 |

Both observed flips are training instances; no ReLD pair winner flips in the sampled validation instances. Preserving the physical start set gives zero pair flips. These small probes establish that execution roles can matter, not that this effect explains the overall plateau. Original-input repeats remain stable. Costs from different start subsets are not substituted into the original labels, and no full-pool winner is claimed.

## Next Decision

First settle the backhaul class/capacity contract and bind all deployment protocols, retaining old data as a separate scenario. Only then decide affected columns needing re-evaluation. Do not begin another attention-model experiment on the claim that all current costs are already semantically comparable.
The prior/permutation evidence supports genuine instance-dependent baseline signal. It does not prove unused signal is easy to learn, labels are random, or 47% is an upper bound. Missing historical TSP role/recipe information remains a concrete audit gap, not a demonstrated cause of its errors.

## Literature Boundary

[NSS Appendix A.9/Table8](https://arxiv.org/html/2410.09693v2#A9) reports greedy TSP classification36%/ranking35%, CVRP61%/62%; its candidate pool is different. This contextualizes Top1, not a target or ceiling for this project. Multi-solver Top-k/rejection/Top-p changes runtime budget and is not a single-method fix.
[ASlib](https://arxiv.org/abs/1506.02465) motivates explicit algorithm-selection scenarios and execution records. [Alissa et al.](https://link.springer.com/article/10.1007/s10732-022-09505-4) and [Smith-Miles and Bowly](https://research.monash.edu/en/publications/generating-new-test-instances-by-evolving-in-instance-space/) motivate examining instance/algorithm structure and coverage; neither establishes a property of these local labels.

## Reproduction

Commands and environment: config.json and run_v4_r57.sh. Source evidence: source_findings.md. Prediction replay: prediction_replay.json. Focused tests: test_r57.py. Large existing datasets, weights and route caches remain local; hashes/paths are recorded, not duplicated.
