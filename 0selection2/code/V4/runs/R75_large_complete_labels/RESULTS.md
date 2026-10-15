# R75 Single-Group Full18 Results

Seed2; selected epoch 40, frozen solely by full18 native validation count 8345/18000. Checkpoint: `/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/single_seed2/best.pt`; declared SHA256: `31e1ef7696c0876ba9dfcf557ffd592e919634081289b764a6eeae30d9504cd6`. Checkpoint bytes were not opened by analysis.

One completed locked TEST is reported; analysis does not run inference or select again. Historical comparisons are descriptive, not matched data-only causal evidence. Legacy eight B-task semantics remain deferred/unfixed. The source TSP/CVRP snapshot misses mainly large-size tails. The prior goal remains paused; no 70% threshold claim or goal resumption is made. R74 remains terminated.

## Selected TRAIN / VAL / TEST

| Model | Split | Epoch | Native correct/total | Native Top1 % | Task macro Top1 % | Top2 % | Top3 % | CE | Mean cost | vs SBS % | vs oracle % | Actual regret % |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R75 | train | 40 | 915469/1.78689e+06 | 51.232619 | 51.181345 | 84.352514 | 94.417445 | 1.025647 | 11.325301 | -0.244859 | 0.865755 | 0.879254 |
| R75 | val | 40 | 8345/18000 | 46.361111 | 46.361111 | 77.305556 | 89.305556 | 1.249806 | 11.340649 | -0.160888 | 1.852900 | 1.727817 |
| R75 | test | 40 | 8076/18000 | 44.866667 | 44.866667 | 76.438889 | 89.005556 | 1.257597 | 11.328395 | -0.177214 | 1.816882 | 1.700811 |
| R65B | train | 40 | 88888/180000 | 49.382222 | 49.382222 | 82.696667 | 93.186667 | 1.066195 | 11.173045 | -1.094776 | N/A | 0.926387 |
| R65B | val | 40 | 8770/18000 | 48.722222 | 48.722222 | 82.150000 | 92.483333 | 1.090114 | 11.169640 | -1.056037 | N/A | 0.945528 |
| R65B | test | 40 | 8575/18000 | 47.638889 | 47.638889 | 81.700000 | 92.394444 | 1.097159 | 11.171845 | -0.999267 | N/A | 0.978280 |
| R45A | val | 38 | N/A | N/A | 49.116667 | 82.444444 | 92.761111 | 1.089768 | 11.168517 | -1.070041 | N/A | 0.926048 |
| R45A | test | 38 | N/A | N/A | 47.911111 | 81.744444 | 92.433333 | 1.101119 | 11.168366 | -1.032003 | N/A | 0.943221 |
| R60B | train | 39 | N/A | N/A | 50.770556 | 83.176111 | 93.585556 | 1.041664 | 11.167328 | -1.143916 | N/A | 0.872533 |
| R60B | val | 39 | N/A | N/A | 49.361111 | 82.238889 | 92.722222 | 1.092423 | 11.168642 | -1.073544 | N/A | 0.922789 |
| R60B | test | 39 | N/A | N/A | 48.244444 | 81.722222 | 92.511111 | 1.102036 | 11.168404 | -1.035671 | N/A | 0.940486 |
| R73B | train | 23 | 93988/180000 | 52.215556 | 52.215556 | 83.996111 | 94.175556 | 1.120606 | 11.162747 | -1.173014 | N/A | 0.840356 |
| R73B | val | 23 | 8959/18000 | 49.772222 | 49.772222 | 82.450000 | 92.555556 | 1.252106 | 11.167655 | -1.080398 | N/A | 0.917354 |
| R73B | test | 23 | 8641/18000 | 48.005556 | 48.005556 | 81.505556 | 92.338889 | 1.277240 | 11.169605 | -1.028878 | N/A | 0.951462 |

Native Top1 uses integer correct/total; task macros weight each of18 tasks equally. These differ for unequal TRAIN sizes. MVRP is a15-task macro. Accuracies in CSV are fractions; regret/vs-SBS/vs-oracle retain saved percent units. Mixed-task mean cost has heterogeneous units.

## Full18 And Families

| Scope | Kind | TRAIN Top1 % | VAL Top1 % | TEST native correct/total | TEST Top1 % | TEST Top2 % | TEST Top3 % | TEST CE | TEST mean cost | TEST vs SBS % | TEST vs oracle % | TEST actual regret % |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | aggregate | 51.181345 | 46.361111 | 8076/18000 | 44.866667 | 76.438889 | 89.005556 | 1.257597 | 11.328395 | -0.177214 | 1.816882 | 1.700811 |
| TSP | family | 43.691982 | 28.000000 | 284/1000 | 28.400000 | 50.500000 | 66.600000 | 1.958565 | 8.059188 | -0.293016 | 0.807261 | 0.841655 |
| CVRP | family | 50.169221 | 49.400000 | 455/1000 | 45.500000 | 68.000000 | 83.100000 | 1.384011 | 18.261515 | -0.386298 | 0.566741 | 0.733949 |
| ATSP | family | 74.891000 | 76.400000 | 753/1000 | 75.300000 | 96.000000 | 99.300000 | 0.624693 | 1.634959 | -2.079482 | 0.640176 | 0.635481 |
| MVRP | family | 50.167467 | 45.380000 | 6584/15000 | 43.893333 | 77.426667 | 90.206667 | 1.244632 | 11.730363 | -0.028737 | 2.045980 | 1.893568 |
| TSP | task | 43.691982 | 28.000000 | 284/1000 | 28.400000 | 50.500000 | 66.600000 | 1.958565 | 8.059188 | -0.293016 | 0.807261 | 0.841655 |
| CVRP | task | 50.169221 | 49.400000 | 455/1000 | 45.500000 | 68.000000 | 83.100000 | 1.384011 | 18.261515 | -0.386298 | 0.566741 | 0.733949 |
| ATSP | task | 74.891000 | 76.400000 | 753/1000 | 75.300000 | 96.000000 | 99.300000 | 0.624693 | 1.634959 | -2.079482 | 0.640176 | 0.635481 |
| OVRP | task | 40.220000 | 38.100000 | 370/1000 | 37.000000 | 65.200000 | 83.600000 | 1.447651 | 7.461113 | -0.112413 | 0.707556 | 0.709484 |
| VRPB | task | 56.131000 | 56.400000 | 530/1000 | 53.000000 | 91.900000 | 99.000000 | 0.882642 | 9.283667 | -0.122613 | 0.714713 | 0.717897 |
| VRPL | task | 34.578000 | 35.700000 | 322/1000 | 32.200000 | 54.700000 | 72.700000 | 1.631947 | 12.351644 | -0.073214 | 0.460360 | 0.463132 |
| VRPTW | task | 49.686000 | 27.100000 | 242/1000 | 24.200000 | 47.900000 | 72.800000 | 2.116666 | 18.562536 | 0.688125 | 6.534557 | 5.774159 |
| OVRPTW | task | 52.971000 | 52.600000 | 490/1000 | 49.000000 | 90.900000 | 99.900000 | 0.911957 | 10.115921 | -0.072884 | 0.820372 | 0.819605 |
| OVRPB | task | 58.768000 | 61.000000 | 602/1000 | 60.200000 | 96.400000 | 99.000000 | 0.789047 | 6.287772 | -0.135636 | 0.880602 | 0.891897 |
| OVRPL | task | 40.632000 | 41.100000 | 388/1000 | 38.800000 | 68.600000 | 86.200000 | 1.380580 | 7.443100 | -0.141542 | 0.668483 | 0.671670 |
| VRPBL | task | 55.716000 | 54.200000 | 560/1000 | 56.000000 | 93.700000 | 99.300000 | 0.855468 | 9.243876 | -0.057340 | 0.725522 | 0.726905 |
| VRPBTW | task | 49.685000 | 34.500000 | 357/1000 | 35.700000 | 65.900000 | 83.200000 | 1.613990 | 19.519678 | -0.033958 | 4.436199 | 3.879407 |
| VRPLTW | task | 49.663000 | 26.900000 | 261/1000 | 26.100000 | 49.300000 | 74.100000 | 2.032670 | 18.431820 | 0.026906 | 6.097889 | 5.650287 |
| OVRPBL | task | 58.808000 | 59.800000 | 586/1000 | 58.600000 | 97.100000 | 99.600000 | 0.758489 | 6.297780 | -0.143409 | 0.897269 | 0.924865 |
| OVRPBTW | task | 51.560000 | 51.500000 | 516/1000 | 51.600000 | 90.100000 | 99.900000 | 0.885557 | 10.598592 | -0.097973 | 1.149071 | 1.167293 |
| OVRPLTW | task | 52.897000 | 53.900000 | 509/1000 | 50.900000 | 92.100000 | 100.000000 | 0.863134 | 10.095840 | -0.076827 | 0.784872 | 0.783087 |
| VRPBLTW | task | 49.542000 | 35.400000 | 349/1000 | 34.900000 | 65.200000 | 83.900000 | 1.628074 | 19.678417 | 0.007424 | 4.736198 | 4.120383 |
| OVRPBLTW | task | 51.655000 | 52.500000 | 502/1000 | 50.200000 | 92.400000 | 99.900000 | 0.871607 | 10.583696 | -0.085694 | 1.076035 | 1.103446 |

## Best / Final / Last Five

| Summary | Split | Actual epochs | Native Top1 % | Macro Top1 % | Actual regret % |
|---|---|---|---:|---:|---:|
| best | train | 40 | 51.232619 | 51.181345 | 0.879254 |
| best | val | 40 | 46.361111 | 46.361111 | 1.727817 |
| final | train | 40 | 51.232619 | 51.181345 | 0.879254 |
| final | val | 40 | 46.361111 | 46.361111 | 1.727817 |
| last5_mean | val | 36,37,38,39,40 | 46.281111 | 46.281111 | 1.725468 |
| last5_population_sd | val | 36,37,38,39,40 | 0.063983 | 0.063983 | 0.001278 |
| last5_min | val | 36,37,38,39,40 | 46.211111 | 46.211111 | 1.724014 |
| last5_max | val | 36,37,38,39,40 | 46.361111 | 46.361111 | 1.727817 |

Last-five statistics describe temporal variation, not independent-seed uncertainty. Sparse TRAIN points are not interpolated; a last-five TRAIN summary requires five actual measurements.

## Recorded Compute

```json
{
  "amp_retries": 27,
  "complete": true,
  "completed_utc": "2026-10-14T08:31:48.013261+00:00",
  "discarded_uncommitted_inflight_groups_upper_bound": 2,
  "epochs": 40,
  "expected_presentations": 71475480,
  "logged_successful_updates_replayed": 0,
  "partial_epoch_timings": [
    {
      "epoch_index": 10,
      "next_step": 1200,
      "prior_saved_step_seconds": 2806.340373510495
    },
    {
      "epoch_index": 16,
      "next_step": 324,
      "prior_saved_step_seconds": 134.66459162160754
    }
  ],
  "presentations": 71475480,
  "runtime_continuations": 2,
  "selected_epoch": 40,
  "selected_val_correct": 8345,
  "successful_updates": 62640,
  "total_elapsed_scope": "includes original input preparation and both runtime optimization pauses",
  "total_formal_pipeline_elapsed_seconds": 59754.377463,
  "training_wall_seconds": 15704.904015968554,
  "training_wall_seconds_scope": "last process segment after pipeline continuation only"
}
```

Training compute above is the saved completion summary, not a new benchmark. Saved evaluation seconds and optimizer metrics are exported separately. No solver executions, reference reruns, model/data/cache loads or prediction recounts were performed.

Instance correction/harm is unavailable from scalar metrics and is not inferred. Saved per-method comparisons, best-method metrics and arm distributions are retained in method_metrics.csv/task_diagnostics.csv. Missing historical fields stay N/A.

Sources, byte hashes, pointers and CPU analysis timings: analysis_receipt.json. All18 task and ALL/TSP/CVRP/ATSP/MVRP rows: comparison.csv. All saved epoch curves: learning_curves.csv/optimization_curves.csv. Figures: learning_curves.png and per_task_family_curves.png.

Availability: No last-five TRAIN statistics: fewer than five actual epoch measurements.
