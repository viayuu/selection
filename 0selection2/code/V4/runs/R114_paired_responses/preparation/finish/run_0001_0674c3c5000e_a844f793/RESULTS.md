# R114 Paired Response Pilot Closeout

Closeout: **partial_negative_closeout**; scheduler=blocked; labeling_complete=False.

Actual verified candidate labels in assembly: **8640/9600**; full-pool winner rows: **1952/2048**; full-pool pairs: **1472/1536**.

A partial/negative closeout is NOT labeling finished. Missing costs/routes mask full-pool winners; raw stopped-problem attempts stay on the server. No solver retries, repairs or new analysis runs.

| problem | endpoints | actual_verified_labels | full_pool_winners | planned_pairs | full_pool_pairs | stopped |
| --- | --- | --- | --- | --- | --- | --- |
| CVRP | 96 | 0 | 0 | 64 | 0 | True |
| OVRP | 96 | 576 | 96 | 64 | 64 | False |
| OVRPB | 96 | 288 | 96 | 64 | 64 | False |
| OVRPBL | 96 | 288 | 96 | 64 | 64 | False |
| OVRPBLTW | 160 | 480 | 160 | 128 | 128 | False |
| OVRPBTW | 160 | 480 | 160 | 128 | 128 | False |
| OVRPL | 96 | 576 | 96 | 64 | 64 | False |
| OVRPLTW | 160 | 960 | 160 | 128 | 128 | False |
| OVRPTW | 160 | 960 | 160 | 128 | 128 | False |
| VRPB | 96 | 288 | 96 | 64 | 64 | False |
| VRPBL | 96 | 288 | 96 | 64 | 64 | False |
| VRPBLTW | 160 | 480 | 160 | 128 | 128 | False |
| VRPBTW | 160 | 480 | 160 | 128 | 128 | False |
| VRPL | 96 | 576 | 96 | 64 | 64 | False |
| VRPLTW | 160 | 960 | 160 | 128 | 128 | False |
| VRPTW | 160 | 960 | 160 | 128 | 128 | False |

## Stopped Problems

```json
{
  "CVRP": [
    {
      "job_id": "runtime_group__CVRP__ICAM",
      "evidence": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/attempts/runtime_group__CVRP__ICAM/attempt_0001/worker_result.json",
      "detail": {
        "error": "Cold reference failed for CVRP_src53138_original__ICAM, exit=1",
        "error_type": "ValueError",
        "stage": "cold_reference",
        "task_id": "CVRP_src53138_original__ICAM"
      }
    },
    {
      "job_id": "runtime_group__CVRP__ICAM",
      "evidence": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/attempts/runtime_group__CVRP__ICAM/attempt_0001/problem_stop.json",
      "detail": {
        "error": "Cold reference failed for CVRP_src53138_original__ICAM, exit=1",
        "error_type": "ValueError",
        "stage": "cold_reference",
        "task_id": "CVRP_src53138_original__ICAM"
      }
    },
    {
      "job_id": "runtime_group__CVRP__ICAM",
      "evidence": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/attempts/runtime_group__CVRP__ICAM/attempt_0001/queue_problem_stop.json",
      "detail": "Grouped worker exit=1; stop CVRP"
    },
    {
      "job_id": "runtime_group__CVRP__ICAM",
      "evidence": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/attempts/runtime_group__CVRP__ICAM/attempt_0001/job_result.json",
      "detail": "Grouped worker exit=1; stop CVRP"
    }
  ]
}
```

## Directed Versus Random

Same problem/family/TRAIN-base full-pool intersection only; missing is not a zero response.

| problem | family | planned_bases | matched_full_pool_bases | mean_rms_delta_difference_pp | winner_flip_difference_pp | mean_amplitude_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| CVRP | demand | 32 | 0 |  |  |  |
| OVRP | demand | 32 | 32 | 0.5410261073883018 | 6.25 | 1.0900606158747264 |
| OVRPB | demand | 32 | 32 | 3.1037166176777458 | 0.0 | 1.1486358181169032 |
| OVRPBL | demand | 32 | 32 | 2.7273763138413973 | 6.25 | 1.1031597915705595 |
| OVRPBLTW | demand | 32 | 32 | 0.06580499434678859 | 15.625 | 1.1123996508453549 |
| OVRPBLTW | tw_center | 32 | 32 | -2.0979686776887374 | 18.75 | 1.1156815576615446 |
| OVRPBTW | demand | 32 | 32 | 0.17131069969598028 | 6.25 | 1.1620638490172865 |
| OVRPBTW | tw_center | 32 | 32 | 0.8005920492685471 | 6.25 | 1.1632728069524294 |
| OVRPL | demand | 32 | 32 | 0.19794330194686197 | -12.5 | 1.1399969957939304 |
| OVRPLTW | demand | 32 | 32 | -0.3792318020178457 | 0.0 | 1.1029720986192348 |
| OVRPLTW | tw_center | 32 | 32 | -0.44802529300895794 | 6.25 | 1.1076551732626458 |
| OVRPTW | demand | 32 | 32 | 0.41831983321330046 | 3.125 | 1.0951343669955076 |
| OVRPTW | tw_center | 32 | 32 | 1.2203092568926615 | -15.625 | 1.1206753020020754 |
| VRPB | demand | 32 | 32 | -0.03449899071749547 | -3.125 | 1.154693056252341 |
| VRPBL | demand | 32 | 32 | 2.2170797889703087 | 9.375 | 1.1440791848907272 |
| VRPBLTW | demand | 32 | 32 | 1.1050004172794632 | 6.25 | 1.1106714449421664 |
| VRPBLTW | tw_center | 32 | 32 | 1.8923676661969826 | -9.375 | 1.1039248398601917 |
| VRPBTW | demand | 32 | 32 | -0.7324443971088324 | 3.125 | 1.1157415379615208 |
| VRPBTW | tw_center | 32 | 32 | 0.1531308607224512 | 6.25 | 1.1338333521127315 |
| VRPL | demand | 32 | 32 | 1.363307824485928 | 15.625 | 1.1153821849354506 |
| VRPLTW | demand | 32 | 32 | -0.20045834249182226 | 3.125 | 1.1045446782036061 |
| VRPLTW | tw_center | 32 | 32 | 1.322405696026315 | -3.125 | 1.0782664817269265 |
| VRPTW | demand | 32 | 32 | 0.019492337774837803 | 3.125 | 1.0893116381693881 |
| VRPTW | tw_center | 32 | 32 | 0.021976469513795636 | 0.0 | 1.0970148012074326 |

## Provenance And Scope

Data version: r114_train_paired_v1_seed2; fixed TRAIN bases: 512.

Manifest canonical SHA256: 0fc7ad4797bf4e4953c1c1f47ba6c75d7cc2c3be5598df7857966d792b676277.

Full server manifest file SHA256: d414f0c4cb4bf3d03049e95b632faf21ee70a1acbb926ea20a8d722f6f57ef68.

Deployed lock SHA256: f935df8d7e94738f1251cf4b970194c3c3f9ac8640489fdc22ae0547f97c110a; selection SHA256: 2bf36d4c53a17b5684afc014e0b1e44244759a3495e66a71da8f66693d6d79b6.

Native float(cost) is promoted to FP64; exact ties use stable global ID. Both endpoints were required to be fresh under the same locked native protocol. This is a 16-applicable-problem/512-base pilot design, not an all18 model evaluation; TSP/ATSP interventions are not applicable. No training, accuracy, TEST/LIB or generalization claim. All counts, failure reasons and matched comparisons above are self-contained; no external images are required.
