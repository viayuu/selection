# R52: validation error sources and multi-start winner mechanism

## Main findings

No selector was trained. No test data or test predictions were read. Fixed seed2.
R45A validation-best epoch38 is the unchanged baseline: Top1 49.1167%, actual regret 0.926048%.

- Head-to-head ranking errors with the native winner at predicted rank2 account for 65.50% of errors and 67.51% of regret. Winners outside predicted Top3 account for only 12.95% of regret. Full-pool candidate identification remains relevant but is not the largest aggregate loss source.
- Across all problems, MOEL/MTL reversals account for 4207 errors and 51.55% of regret. OVRPTW alone explains 4.52%. The pair is important, but OVRPTW alone is not representative of its full cost budget.
- The highest-contribution specific problem/pair is OVRPBLTW / MOEL vs MTL: 419 errors, 5.90% of total regret. Both methods were replayed on all1000 existing validation instances, not selected errors.
- At most two winning starts: 375/1000 instances; these account for 18.89% of the selected problem's MOEL/MTL error regret. At least one strict reversal in20 shared80%-start subsets: 272/1000, accounting for 13.09% of that pair's error regret. Their union accounts for 18.89%; the complementary stable, more-than-two-supported group still accounts for 81.11%. Thus these particular extreme-start/sensitivity mechanisms are not the dominant measured cost explanation.

## 1. Full18-task error budget

Ranks and correctness use native ind. Costs are independently reread from raw_label.pkl in FP64 and match saved validation predictions exactly. Scores use greedy argmax, ties by global solver ID. Every problem contributes1/18, every instance within it1/N_problem. Percentages below are regret shares, not shares of summed raw route cost.

| Error category | Errors | Error share % | Macro regret pp | Regret share % |
|---|---|---|---|---|
| winner_rank2 | 5999 | 65.4984 | 0.6251 | 67.5057 |
| winner_rank3 | 1857 | 20.2751 | 0.1810 | 19.5418 |
| winner_outside_top3 | 1303 | 14.2264 | 0.1199 | 12.9526 |

### Largest problem-specific unordered pairs

Opposite directions of the same pair are combined here; `error_budget_by_problem_pair.csv` retains the full problem x selected x native-winner directions. A row counts only actual selector errors, not every binary contest between those methods.

| Problem | Solver A | Solver B | Errors | Regret share % |
|---|---|---|---|---|
| OVRPBLTW | RELD_MOEL | RELD_MTL | 419 | 5.9019 |
| OVRPBTW | RELD_MOEL | RELD_MTL | 413 | 5.8283 |
| OVRPB | RELD_MOEL | RELD_MTL | 385 | 5.0389 |
| OVRPBL | RELD_MOEL | RELD_MTL | 367 | 4.9456 |
| OVRPLTW | RELD_MOEL | RELD_MTL | 419 | 4.6486 |
| VRPBTW | RELD_MOEL | RELD_MTL | 309 | 4.5802 |
| OVRPTW | RELD_MOEL | RELD_MTL | 423 | 4.5154 |
| VRPBL | RELD_MOEL | RELD_MTL | 394 | 4.2562 |
| VRPBLTW | RELD_MOEL | RELD_MTL | 292 | 4.2224 |
| VRPB | RELD_MOEL | RELD_MTL | 363 | 3.4313 |

### Diagnostic correction upper bounds (NOT deployable results)

If an oracle corrected just the existing errors in the largest problem-specific pairs to native winner, leaving all other decisions unchanged:

| Pairs | Corrected | Diagnostic Top1 % | Diagnostic regret % | Recovered loss % |
|---|---|---|---|---|
| 1 | 419 | 51.4444 | 0.8714 | 5.9019 |
| 3 | 1217 | 55.8778 | 0.7708 | 16.7691 |
| 5 | 2003 | 60.2444 | 0.6819 | 26.3633 |
| 10 | 3784 | 70.1389 | 0.4874 | 47.3688 |

Correcting every MOEL/MTL reversal across all tasks would add 23.3722pp Top1: diagnostic Top1 72.4889% and regret 0.448633%. This uses true winners offline, not an achieved model score or an accuracy ceiling. Native-winner-vs-FP64-min checks are in `baseline_by_problem.csv`; no winner relabeling was performed.

## 2. Full-trajectory audit on OVRPBLTW

Selection is ranking-first, not automatically OVRPTW. The highest-ranked pair itself has the original no-augmentation export and recoverable bridge recipe, so no lower-ranked fallback or substituted default was needed. Sampling is size-only, seed2, covers all1000 original validation instances with50..100 customers. Eight size-stratified instances were checked before the full replay.

The original bridge supplies FP32 depot/coordinates/signed demand/route limit/service time/customer time windows, normalized capacity1, no augmentation, one argmax construction, native environment dispatch. Both R48-pinned epoch5000 Train_ALL weights and strict architecture are reused; SHA256 bindings are in `audit_plan.json`. Historical complete CLI and source/environment revision are not archived: matching exports, explicit bridge parameters and successful present replay do not manufacture missing historical metadata.

Important native detail: declared pomo_size=N is capped by OVRPBLTWEnv to floor(0.8*N), using positive-demand customer IDs in original order. This audit preserves ALL native enabled starts (40..80), not all customers including invalid backhaul starts. Both methods' actual start IDs match exactly. No start or trajectory is chosen using final cost before solving. Open-route cost excludes every return-to-depot segment, exactly as the original environment.

Every complete trajectory cost is saved in `start_costs.npz` with original instance index, customer count, start customer ID, valid-start mask and historical pair costs. Padding is NaN and excluded from all reductions. Native reward computation is FP32; saving those values as FP64 does not recover precision. Historical label costs and all diagnostic arithmetic are FP64.

- Historical scalar reproduction: all2000 full-run costs pass atol5e-05, rtol2e-06; maximum absolute difference, including precheck, 9.53675293403e-07. Exact pair-winner disagreements: 0. Historical ties: 0; replay ties: 0.
- Actual additional solving: 2016 solver-instance runs including8-instance precheck for each method, full replay 16.461s, total replay stage 19.718s, peak allocated GPU 78.6MiB on only NVIDIA GeForce RTX 3090 (GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d). This is not zero-solve analysis.
- Number of starts beating the opponent's best (strict pair wins only): min/q25/median/q75/max = [1.0, 2.0, 4.0, 11.0, 79.0]. Support fractions = [0.0125, 0.0273972602739726, 0.075, 0.1947463768115942, 1.0].
- Min-winner opposite to median winner: 27.20%; opposite to q10 winner: 22.20%; opposite to q25 winner: 25.10%. Exact ties are separate, not counted as opposition.
- Offline subset sensitivity: same retained starting-customer IDs for both methods, ceil(0.8*actual_start_count),20 subsets per instance, seed2 with per-index streams. 27.20% of strict-win instances have at least one strict reversal; mean reversal frequency 4.28%. 0 instances have a subset tie. This changes only the reduction over previously computed full costs, NOT POMO execution size, decoding, or20 new solver runs.

### Association with actual selector loss

Groups below overlap; do not add their percentages. `Target-pair loss %` is the share of419 actual MOEL/MTL selector-error losses in this problem. `Problem loss %` uses all seven-method-pool selector decisions. `ALL loss %` is the contribution to the original18-task macro loss. Neither support counts nor subset repetitions are independent data samples. The final four support-fraction bins are disjoint descriptive summaries, not additional prespecified hypothesis tests.

| Group | Instances | Pool errors | Pair errors | Target-pair loss % | Problem loss % | ALL loss % |
|---|---|---|---|---|---|---|
| all | 1000 | 499 | 419 | 100.0000 | 100.0000 | 6.9555 |
| strict_pair_winner | 1000 | 499 | 419 | 100.0000 | 100.0000 | 6.9555 |
| exactly_one_supporting_start | 231 | 125 | 102 | 9.8093 | 11.3393 | 0.7887 |
| at_most_two_supporting_starts | 375 | 201 | 164 | 18.8950 | 21.4146 | 1.4895 |
| more_than_two_supporting_starts | 625 | 298 | 255 | 81.1050 | 78.5854 | 5.4660 |
| subset_sensitive | 272 | 149 | 123 | 13.0943 | 14.7294 | 1.0245 |
| sparse_or_sensitive | 380 | 201 | 164 | 18.8950 | 21.4146 | 1.4895 |
| stable_more_than_two | 620 | 298 | 255 | 81.1050 | 78.5854 | 5.4660 |
| median_opposes_min | 272 | 154 | 125 | 20.6846 | 22.5232 | 1.5666 |
| q10_opposes_min | 222 | 137 | 113 | 15.2880 | 16.9701 | 1.1804 |
| q25_opposes_min | 251 | 141 | 111 | 16.8164 | 19.4978 | 1.3562 |
| support_fraction_le5pct | 416 | 225 | 180 | 23.0089 | 26.0844 | 1.8143 |
| support_fraction_5to10pct | 168 | 89 | 74 | 14.3309 | 16.1726 | 1.1249 |
| support_fraction_10to25pct | 209 | 99 | 84 | 25.8556 | 25.1827 | 1.7516 |
| support_fraction_gt25pct | 207 | 86 | 81 | 36.8046 | 32.5603 | 2.2647 |

## Interpretation and next direction

The error budget supports focusing on competitive head-to-head decisions across MVRP, particularly MOEL/MTL, rather than another architecture search limited to OVRPTW. It does not support attributing the platform mainly to winners absent from Top3.

The replay supplies direct evidence for the distinction between average trajectory quality and best-of-many quality: the measured median/q10 disagreements above cannot be inferred from mean prefix features. However, this distinction does NOT explain most of the measured selection loss. The stable, more-than-two-supported share is 81.11%; more than10% of starts beat the opponent's best on instances contributing 62.66% of pair-error regret. More than two does not automatically mean broad support, so the full counts/fractions are reported rather than equated with an easy decision.

The first priority supported here is transferable instance-method representation/supervision for the high-loss MVRP pairs, including their stable supported wins. Do not automatically escalate to a larger trajectory predictor on the claim that isolated exceptional starts cause most remaining cost. Per-start modeling may still be tested as a different information source, but it would need its own controlled evidence; neither another pooled prefix vector nor longer probing is justified by this audit alone. The two original questions are answered: the dominant pair is real at the full-system level, whereas this selected pair's loss is not primarily located in the prespecified rare/sensitive groups.

No label error, randomness ceiling, or global predictability limit is established. A winner change after removing starts is not a corrupted label. The original complete-start minimum remains the target. This is one problem, two methods,1000 instances, not a claim about every solver's mechanism. No training, relabeling, seed search, or test-based selection was done.

Paper context: [Leader Reward for POMO-Based Neural Combinatorial Optimization, section4.1](https://arxiv.org/html/2405.13947v1) distinguishes average generated quality from the best generated solution. R52 measures this distinction in existing ReLD outputs; it does not implement that paper's training method or assume its findings prove selector difficulty.

## Reproduction and files

Run from `/public/home/shiys/0selection2` in easynco:

```bash
bash code/V4/run_v4_r52.sh budget
R52_GPU_UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d bash code/V4/run_v4_r52.sh replay
bash code/V4/run_v4_r52.sh analysis
bash code/V4/run_v4_r52.sh tests
```

Replay must run on the GPU node inside the existing allocation; the recorded srun/tmux command is in `execution.json`. Source scripts are `r52_error_budget.py`, `r52_start_audit.py`, `r52_analysis.py`, `test_r52.py`, `run_v4_r52.sh`. Neither old solver code nor historical experiment outputs were edited.

Required deliverables: `error_budget_by_problem_pair.csv`, `topk_error_decomposition.csv`, `start_costs.npz`, `winner_mechanism.csv`, this report. Supplemental pair rankings, correction bounds, per-instance errors, historical replay CSV, frozen checkpoint/recipe hashes, start subset masks/winners and mechanism summaries allow independent recalculation without loading a selector or rerunning solvers.
