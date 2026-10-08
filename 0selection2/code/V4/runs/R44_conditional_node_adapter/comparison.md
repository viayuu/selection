# R44: Constraint/size-conditioned node residual

## Experiment

- Baseline: R43A epoch 35, SHA256 `1b99faac4d31097003403c39cdf4b7d7cb0960a66419101ae3c5b267f9058810`.
- One insertion: after the solver-independent InstanceEncoder and before the first node/solver joint block.
- C0: continuation only. C1: shared bottleneck. C2: existing 16 semantic fields plus train-normalized log(real nodes).
- 34 solver features, local geometry, maximin scores, native labels, FP64 cost targets, comparison loss and R-Drop are unchanged.
- All original parameters trainable; fresh AdamW; original/new peak LR 1e-5/1e-4; 5% warmup then cosine to 10%.
- Each arm: 14,220 successful updates, 10 complete data epochs, batch128, full tails retained.
- Borrowed from PEPNet: two-layer 2*sigmoid multiplicative gate. GELU, metadata-only gate, zero-initialized bottleneck residual are project adaptations, not full PEPNet.
- The 3090 step0 differs from historical 4090 predictions on four near-boundary instances; data/state are identical and every R44 arm shares the same local step0. See baseline_replay.json.

## Validation Summary

| Run | Point | Update | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0_seed2 | step0 | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C0_seed2 | best | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C0_seed2 | final | 14220 | 0.4877 | 0.8228 | 0.9284 | 11.169594 | -1.062553% | 0.933672% |
| C1_seed2 | step0 | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C1_seed2 | best | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C1_seed2 | final | 14220 | 0.4878 | 0.8228 | 0.9284 | 11.169593 | -1.062487% | 0.933876% |
| C2_seed2 | step0 | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C2_seed2 | best | 0 | 0.4883 | 0.8224 | 0.9278 | 11.169102 | -1.066516% | 0.928151% |
| C2_seed2 | final | 14220 | 0.4881 | 0.8228 | 0.9284 | 11.169555 | -1.062822% | 0.933257% |

Last five nonzero common validation points:

| Run | Top1 | CE | mean_cost | vs_SBS | actual regret |
|---|---:|---:|---:|---:|---:|
| C0_seed2 | 0.4880 | 1.089047 | 11.169617 | -1.063311% | 0.932704% |
| C1_seed2 | 0.4880 | 1.089044 | 11.169544 | -1.063762% | 0.932412% |
| C2_seed2 | 0.4880 | 1.089047 | 11.169558 | -1.063665% | 0.932376% |

## Stage-A Decision

Passed: **False**. Locked screen threshold: 0.001 percentage points (operational, not statistical significance).
- C2 vs C0: best cost improvement +0.000000 pp; last-five mean improvement +0.000355 pp. Positive means C2 is better.
- C2 vs C1: best cost improvement +0.000000 pp; last-five mean improvement -0.000097 pp. Positive means C2 is better.

## Attribution

| Seed | Point | Control | Corrected | Harmed | Both wrong, cost down | Both wrong, cost up | Net mean cost change | Net actual regret change |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | best | C0 | 0 | 0 | 0 | 0 | +0.00000000 | +0.000000 pp |
| 2 | best | C1 | 0 | 0 | 0 | 0 | +0.00000000 | +0.000000 pp |
| 2 | final | C0 | 9 | 3 | 0 | 2 | -0.00003930 | -0.000415 pp |
| 2 | final | C1 | 10 | 6 | 0 | 3 | -0.00003827 | -0.000619 pp |

Negative cost/regret changes favor C2. Native-winner corrections alone do not explain all cost changes.

## Per-Problem Best Validation

| Run | Problem | Top1 | Top2 | Top3 | mean_cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0_seed2 | ALL | 0.4883 | 0.8224 | 0.9278 | 11.169102 | 11.341290 (-1.06652%) | 11.058917 (+0.89685%) | 0.92815% |
| C0_seed2 | TSP | 0.3740 | 0.6100 | 0.7590 | 7.917031 | 7.957803 (-0.51235%) | 7.869479 (+0.60426%) | 0.60056% |
| C0_seed2 | CVRP | 0.4830 | 0.6950 | 0.8310 | 18.202360 | 18.294333 (-0.50274%) | 18.109962 (+0.51021%) | 0.66496% |
| C0_seed2 | ATSP | 0.7850 | 0.9680 | 0.9940 | 1.634786 | 1.673524 (-2.31475%) | 1.629583 (+0.31927%) | 0.31546% |
| C0_seed2 | OVRP | 0.3640 | 0.6490 | 0.8420 | 7.453180 | 7.461660 (-0.11365%) | 7.403457 (+0.67162%) | 0.68300% |
| C0_seed2 | VRPB | 0.5510 | 0.9180 | 0.9950 | 9.335617 | 9.340269 (-0.04980%) | 9.268251 (+0.72685%) | 0.73414% |
| C0_seed2 | VRPL | 0.3180 | 0.5870 | 0.7320 | 12.401934 | 12.407213 (-0.04255%) | 12.344029 (+0.46909%) | 0.47007% |
| C0_seed2 | VRPTW | 0.4620 | 0.7600 | 0.9320 | 17.889557 | 18.657081 (-4.11385%) | 17.660252 (+1.29843%) | 1.38849% |
| C0_seed2 | OVRPTW | 0.5210 | 0.9330 | 1.0000 | 10.038997 | 10.048435 (-0.09393%) | 9.958517 (+0.80815%) | 0.82821% |
| C0_seed2 | OVRPB | 0.5770 | 0.9680 | 0.9860 | 6.318393 | 6.321981 (-0.05675%) | 6.260687 (+0.92172%) | 0.93115% |
| C0_seed2 | OVRPL | 0.4060 | 0.6890 | 0.8610 | 7.430228 | 7.437562 (-0.09860%) | 7.379566 (+0.68653%) | 0.69743% |
| C0_seed2 | VRPBL | 0.5420 | 0.9160 | 0.9880 | 9.355611 | 9.359268 (-0.03907%) | 9.284586 (+0.76498%) | 0.77223% |
| C0_seed2 | VRPBTW | 0.4300 | 0.8060 | 0.9270 | 19.037385 | 19.669740 (-3.21486%) | 18.744386 (+1.56313%) | 1.60070% |
| C0_seed2 | VRPLTW | 0.4560 | 0.8050 | 0.9370 | 17.622218 | 18.509149 (-4.79185%) | 17.419424 (+1.16418%) | 1.24248% |
| C0_seed2 | OVRPBL | 0.5840 | 0.9550 | 0.9830 | 6.302610 | 6.310260 (-0.12122%) | 6.245926 (+0.90754%) | 0.92488% |
| C0_seed2 | OVRPBTW | 0.4970 | 0.9130 | 0.9970 | 10.497205 | 10.506777 (-0.09110%) | 10.379805 (+1.13104%) | 1.15703% |
| C0_seed2 | OVRPLTW | 0.5090 | 0.9270 | 0.9990 | 10.123086 | 10.127205 (-0.04068%) | 10.039835 (+0.82920%) | 0.84620% |
| C0_seed2 | VRPBLTW | 0.4220 | 0.7850 | 0.9390 | 18.952084 | 19.519778 (-2.90830%) | 18.651532 (+1.61141%) | 1.69289% |
| C0_seed2 | OVRPBLTW | 0.5080 | 0.9200 | 0.9990 | 10.531556 | 10.541174 (-0.09124%) | 10.411234 (+1.15569%) | 1.15684% |
| C1_seed2 | ALL | 0.4883 | 0.8224 | 0.9278 | 11.169102 | 11.341290 (-1.06652%) | 11.058917 (+0.89685%) | 0.92815% |
| C1_seed2 | TSP | 0.3740 | 0.6100 | 0.7590 | 7.917031 | 7.957803 (-0.51235%) | 7.869479 (+0.60426%) | 0.60056% |
| C1_seed2 | CVRP | 0.4830 | 0.6950 | 0.8310 | 18.202360 | 18.294333 (-0.50274%) | 18.109962 (+0.51021%) | 0.66496% |
| C1_seed2 | ATSP | 0.7850 | 0.9680 | 0.9940 | 1.634786 | 1.673524 (-2.31475%) | 1.629583 (+0.31927%) | 0.31546% |
| C1_seed2 | OVRP | 0.3640 | 0.6490 | 0.8420 | 7.453180 | 7.461660 (-0.11365%) | 7.403457 (+0.67162%) | 0.68300% |
| C1_seed2 | VRPB | 0.5510 | 0.9180 | 0.9950 | 9.335617 | 9.340269 (-0.04980%) | 9.268251 (+0.72685%) | 0.73414% |
| C1_seed2 | VRPL | 0.3180 | 0.5870 | 0.7320 | 12.401934 | 12.407213 (-0.04255%) | 12.344029 (+0.46909%) | 0.47007% |
| C1_seed2 | VRPTW | 0.4620 | 0.7600 | 0.9320 | 17.889557 | 18.657081 (-4.11385%) | 17.660252 (+1.29843%) | 1.38849% |
| C1_seed2 | OVRPTW | 0.5210 | 0.9330 | 1.0000 | 10.038997 | 10.048435 (-0.09393%) | 9.958517 (+0.80815%) | 0.82821% |
| C1_seed2 | OVRPB | 0.5770 | 0.9680 | 0.9860 | 6.318393 | 6.321981 (-0.05675%) | 6.260687 (+0.92172%) | 0.93115% |
| C1_seed2 | OVRPL | 0.4060 | 0.6890 | 0.8610 | 7.430228 | 7.437562 (-0.09860%) | 7.379566 (+0.68653%) | 0.69743% |
| C1_seed2 | VRPBL | 0.5420 | 0.9160 | 0.9880 | 9.355611 | 9.359268 (-0.03907%) | 9.284586 (+0.76498%) | 0.77223% |
| C1_seed2 | VRPBTW | 0.4300 | 0.8060 | 0.9270 | 19.037385 | 19.669740 (-3.21486%) | 18.744386 (+1.56313%) | 1.60070% |
| C1_seed2 | VRPLTW | 0.4560 | 0.8050 | 0.9370 | 17.622218 | 18.509149 (-4.79185%) | 17.419424 (+1.16418%) | 1.24248% |
| C1_seed2 | OVRPBL | 0.5840 | 0.9550 | 0.9830 | 6.302610 | 6.310260 (-0.12122%) | 6.245926 (+0.90754%) | 0.92488% |
| C1_seed2 | OVRPBTW | 0.4970 | 0.9130 | 0.9970 | 10.497205 | 10.506777 (-0.09110%) | 10.379805 (+1.13104%) | 1.15703% |
| C1_seed2 | OVRPLTW | 0.5090 | 0.9270 | 0.9990 | 10.123086 | 10.127205 (-0.04068%) | 10.039835 (+0.82920%) | 0.84620% |
| C1_seed2 | VRPBLTW | 0.4220 | 0.7850 | 0.9390 | 18.952084 | 19.519778 (-2.90830%) | 18.651532 (+1.61141%) | 1.69289% |
| C1_seed2 | OVRPBLTW | 0.5080 | 0.9200 | 0.9990 | 10.531556 | 10.541174 (-0.09124%) | 10.411234 (+1.15569%) | 1.15684% |
| C2_seed2 | ALL | 0.4883 | 0.8224 | 0.9278 | 11.169102 | 11.341290 (-1.06652%) | 11.058917 (+0.89685%) | 0.92815% |
| C2_seed2 | TSP | 0.3740 | 0.6100 | 0.7590 | 7.917031 | 7.957803 (-0.51235%) | 7.869479 (+0.60426%) | 0.60056% |
| C2_seed2 | CVRP | 0.4830 | 0.6950 | 0.8310 | 18.202360 | 18.294333 (-0.50274%) | 18.109962 (+0.51021%) | 0.66496% |
| C2_seed2 | ATSP | 0.7850 | 0.9680 | 0.9940 | 1.634786 | 1.673524 (-2.31475%) | 1.629583 (+0.31927%) | 0.31546% |
| C2_seed2 | OVRP | 0.3640 | 0.6490 | 0.8420 | 7.453180 | 7.461660 (-0.11365%) | 7.403457 (+0.67162%) | 0.68300% |
| C2_seed2 | VRPB | 0.5510 | 0.9180 | 0.9950 | 9.335617 | 9.340269 (-0.04980%) | 9.268251 (+0.72685%) | 0.73414% |
| C2_seed2 | VRPL | 0.3180 | 0.5870 | 0.7320 | 12.401934 | 12.407213 (-0.04255%) | 12.344029 (+0.46909%) | 0.47007% |
| C2_seed2 | VRPTW | 0.4620 | 0.7600 | 0.9320 | 17.889557 | 18.657081 (-4.11385%) | 17.660252 (+1.29843%) | 1.38849% |
| C2_seed2 | OVRPTW | 0.5210 | 0.9330 | 1.0000 | 10.038997 | 10.048435 (-0.09393%) | 9.958517 (+0.80815%) | 0.82821% |
| C2_seed2 | OVRPB | 0.5770 | 0.9680 | 0.9860 | 6.318393 | 6.321981 (-0.05675%) | 6.260687 (+0.92172%) | 0.93115% |
| C2_seed2 | OVRPL | 0.4060 | 0.6890 | 0.8610 | 7.430228 | 7.437562 (-0.09860%) | 7.379566 (+0.68653%) | 0.69743% |
| C2_seed2 | VRPBL | 0.5420 | 0.9160 | 0.9880 | 9.355611 | 9.359268 (-0.03907%) | 9.284586 (+0.76498%) | 0.77223% |
| C2_seed2 | VRPBTW | 0.4300 | 0.8060 | 0.9270 | 19.037385 | 19.669740 (-3.21486%) | 18.744386 (+1.56313%) | 1.60070% |
| C2_seed2 | VRPLTW | 0.4560 | 0.8050 | 0.9370 | 17.622218 | 18.509149 (-4.79185%) | 17.419424 (+1.16418%) | 1.24248% |
| C2_seed2 | OVRPBL | 0.5840 | 0.9550 | 0.9830 | 6.302610 | 6.310260 (-0.12122%) | 6.245926 (+0.90754%) | 0.92488% |
| C2_seed2 | OVRPBTW | 0.4970 | 0.9130 | 0.9970 | 10.497205 | 10.506777 (-0.09110%) | 10.379805 (+1.13104%) | 1.15703% |
| C2_seed2 | OVRPLTW | 0.5090 | 0.9270 | 0.9990 | 10.123086 | 10.127205 (-0.04068%) | 10.039835 (+0.82920%) | 0.84620% |
| C2_seed2 | VRPBLTW | 0.4220 | 0.7850 | 0.9390 | 18.952084 | 19.519778 (-2.90830%) | 18.651532 (+1.61141%) | 1.69289% |
| C2_seed2 | OVRPBLTW | 0.5080 | 0.9200 | 0.9990 | 10.531556 | 10.541174 (-0.09124%) | 10.411234 (+1.15569%) | 1.15684% |

## Gate and Scope

Gate statistics are in gate_summary.csv and each history.json. Same semantic descriptor and node count imply the same gate; this is expected. Train-only normalization/bins and exact fields are in manifest.json.
Instance uncertainty was not estimated; seed variation is continuation randomness from one fixed R43A, not independent from-scratch training.
Percentages are arithmetic macro means over problems; ALL mean_cost ratios are not used to replace those percentages. Size-group references are recomputed within each fixed group, identically for all arms.
No claim that multi-domain negative transfer was proven. A negative result constrains this insertion, gate, recipe and budget, not every conditional architecture.

## Conclusion

This conditional residual did not pass the prespecified comparison against both continued training and parameter-matched shared capacity. Retain R43A; do not keep C2 as an established improvement.
Per protocol, no test data, additional seeds, gate/rank/LR search were run. All results above are validation results.

