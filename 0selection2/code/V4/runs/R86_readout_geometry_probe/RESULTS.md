# R86 frozen readout geometry diagnostic

R84A update37440; seed2; 64 TRAIN + 64 VAL per task; CPU FP32, 2 threads.

| Split | Rule | Top1 % | Actual regret % | mean_cost | Target recall % | Corrected / harmed |
|---|---|---:|---:|---:|---:|---:|
| TRAIN | original_maximin | 70.8333 | 0.366116 | 11.537210737 | 0.0000 | 0 / 0 |
| TRAIN | angle_only_counterfactual | 71.0069 | 0.365688 | 11.537138810 | 0.0000 | 8 / 6 |
| VAL | original_maximin | 71.2674 | 0.415504 | 11.364720761 | 0.0000 | 0 / 0 |
| VAL | angle_only_counterfactual | 71.3542 | 0.402227 | 11.363103823 | 0.0000 | 6 / 5 |

## Interpretation guards

- Frozen R84A shared final Linear: there are no method-specific classifier weight norms. Only hidden-feature radius and angle to the same w are measured.
- CPU FP32, 64 TRAIN + 64 VAL samples/task (or the recorded input/timing-only budget reduction), seed2. Not a formal AMP evaluation, fitted model, threshold search, TEST or LIB result.
- Maximin utilities permit common shifts. Angle-only argmax is a fixed-parameter counterfactual, not a calibrated/Bayes-equivalent inference rule.
- A favorable change is only priority evidence for a new end-to-end normalized-head experiment, not causal proof or an accepted inference change. A negative result does not rule out normalized architectures.
- No native winners in a sampled method group means undefined recall, never zero. All18 macros equally weight tasks; mixed raw mean_cost is descriptive only.

The exact symmetric pair decomposition is radial = ||w||(r_y-r_s)(cos_y+cos_s)/2 and directional = ||w||(cos_y-cos_s)(r_y+r_s)/2. Their sum equals u_y-u_s. This is descriptive within the frozen representation, not causal attribution.

All group/rank feature statistics and undefined-recall groups are in method_geometry.csv. Sample IDs, recorded input strata, TRAIN-only boundaries, timing-only budget decision and pair summaries are in sample_manifest.json.
