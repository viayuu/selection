# R114 Combined Saved-Label Closeout

**complete_labels**; labeling_complete=True. Actual verified costs: 9600/9600; full-pool winners: 2048/2048; full-pool pairs: 1536/1536.

Only CVRP endpoint/pair rows were replaced by the separately locked correction (10 methods, 32 bases, 96 endpoints, 64 pairs, 960 tasks). The original other 15 problems and every metadata identity remain unchanged. Original ICAM/CVRP failed at a CUDA-to-NumPy host-audit boundary, not solver capability. Original failed attempts remain untouched.

Both schedulers are terminal; both actual assemblies and saved analyses were verified. Corrected execution has actual RTX3090 worker/physical-UUID/binding evidence. Missing costs stay null with false masks; a partial-negative closeout is NOT labeling finished.

| problem | endpoints | planned_labels | actual_verified_labels | full_pool_winners | planned_pairs | full_pool_pairs |
| --- | --- | --- | --- | --- | --- | --- |
| CVRP | 96 | 960 | 960 | 96 | 64 | 64 |
| OVRP | 96 | 576 | 576 | 96 | 64 | 64 |
| OVRPB | 96 | 288 | 288 | 96 | 64 | 64 |
| OVRPBL | 96 | 288 | 288 | 96 | 64 | 64 |
| OVRPBLTW | 160 | 480 | 480 | 160 | 128 | 128 |
| OVRPBTW | 160 | 480 | 480 | 160 | 128 | 128 |
| OVRPL | 96 | 576 | 576 | 96 | 64 | 64 |
| OVRPLTW | 160 | 960 | 960 | 160 | 128 | 128 |
| OVRPTW | 160 | 960 | 960 | 160 | 128 | 128 |
| VRPB | 96 | 288 | 288 | 96 | 64 | 64 |
| VRPBL | 96 | 288 | 288 | 96 | 64 | 64 |
| VRPBLTW | 160 | 480 | 480 | 160 | 128 | 128 |
| VRPBTW | 160 | 480 | 480 | 160 | 128 | 128 |
| VRPL | 96 | 576 | 576 | 96 | 64 | 64 |
| VRPLTW | 160 | 960 | 960 | 160 | 128 | 128 |
| VRPTW | 160 | 960 | 960 | 160 | 128 | 128 |

## Matched Directed Minus Random

| problem | family | planned_bases | matched_full_pool_bases | mean_rms_delta_difference_pp | winner_flip_difference_pp | mean_amplitude_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| CVRP | demand | 32.0 | 32.0 | 0.4240282674882587 | 9.375 | 1.1169506648195466 |
| OVRP | demand | 32.0 | 32.0 | 0.5410261073883018 | 6.25 | 1.0900606158747264 |
| OVRPB | demand | 32.0 | 32.0 | 3.1037166176777458 | 0.0 | 1.1486358181169032 |
| OVRPBL | demand | 32.0 | 32.0 | 2.7273763138413973 | 6.25 | 1.1031597915705595 |
| OVRPBLTW | demand | 32.0 | 32.0 | 0.06580499434678859 | 15.625 | 1.1123996508453549 |
| OVRPBLTW | tw_center | 32.0 | 32.0 | -2.0979686776887374 | 18.75 | 1.1156815576615446 |
| OVRPBTW | demand | 32.0 | 32.0 | 0.17131069969598028 | 6.25 | 1.1620638490172865 |
| OVRPBTW | tw_center | 32.0 | 32.0 | 0.8005920492685471 | 6.25 | 1.1632728069524294 |
| OVRPL | demand | 32.0 | 32.0 | 0.19794330194686197 | -12.5 | 1.1399969957939304 |
| OVRPLTW | demand | 32.0 | 32.0 | -0.3792318020178457 | 0.0 | 1.1029720986192348 |
| OVRPLTW | tw_center | 32.0 | 32.0 | -0.44802529300895794 | 6.25 | 1.1076551732626458 |
| OVRPTW | demand | 32.0 | 32.0 | 0.41831983321330046 | 3.125 | 1.0951343669955076 |
| OVRPTW | tw_center | 32.0 | 32.0 | 1.2203092568926615 | -15.625 | 1.1206753020020754 |
| VRPB | demand | 32.0 | 32.0 | -0.03449899071749547 | -3.125 | 1.154693056252341 |
| VRPBL | demand | 32.0 | 32.0 | 2.2170797889703087 | 9.375 | 1.1440791848907272 |
| VRPBLTW | demand | 32.0 | 32.0 | 1.1050004172794632 | 6.25 | 1.1106714449421664 |
| VRPBLTW | tw_center | 32.0 | 32.0 | 1.8923676661969826 | -9.375 | 1.1039248398601917 |
| VRPBTW | demand | 32.0 | 32.0 | -0.7324443971088324 | 3.125 | 1.1157415379615208 |
| VRPBTW | tw_center | 32.0 | 32.0 | 0.1531308607224512 | 6.25 | 1.1338333521127315 |
| VRPL | demand | 32.0 | 32.0 | 1.363307824485928 | 15.625 | 1.1153821849354506 |
| VRPLTW | demand | 32.0 | 32.0 | -0.20045834249182226 | 3.125 | 1.1045446782036061 |
| VRPLTW | tw_center | 32.0 | 32.0 | 1.322405696026315 | -3.125 | 1.0782664817269265 |
| VRPTW | demand | 32.0 | 32.0 | 0.019492337774837803 | 3.125 | 1.0893116381693881 |
| VRPTW | tw_center | 32.0 | 32.0 | 0.021976469513795636 | 0.0 | 1.0970148012074326 |

## Provenance And Scope

```json
{
  "parent": {
    "plan_sha256": "b314d2d643bbac142b6890ae6aa3b8792a7a6469be1c217af63bb7c219120ea2",
    "manifest_sha256": "0fc7ad4797bf4e4953c1c1f47ba6c75d7cc2c3be5598df7857966d792b676277",
    "lock_sha256": "f935df8d7e94738f1251cf4b970194c3c3f9ac8640489fdc22ae0547f97c110a",
    "full_manifest_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/production_inputs/manifest.json",
      "bytes": 9533385,
      "sha256": "d414f0c4cb4bf3d03049e95b632faf21ee70a1acbb926ea20a8d722f6f57ef68"
    },
    "labels_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/scheduler_runs/run_0001/labels.json",
      "bytes": 11340088,
      "sha256": "4edf80aa9cdec2dd3335e7264fa1cf6d779a6c542f95cd98b8f529dad692e06f"
    },
    "progress_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/scheduler_progress.json",
      "bytes": 1690,
      "sha256": "0674c3c5000e619fb5d19f975c7a4df3a563c0463c04fb16f536dd176448ccfb"
    }
  },
  "corrected_CVRP": {
    "plan_sha256": "0235752d2ad5a829de261fe55a49f310973cf02997bb2c5bdec8315b91c3eeae",
    "manifest_sha256": "62c3f4c9410cd91c72ff76a4c5be009e69d57e2c8119818a03b714796cfd56d3",
    "lock_sha256": "164ebc8f6cae456891895875c3aacab4960c3f3abd36d594e7bf88a147137593",
    "full_manifest_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/manifest.json",
      "bytes": 828589,
      "sha256": "59eae384a6cb1f78f019cce85284a9934e24a433c94615fc1c518c144eb04aa5"
    },
    "labels_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/scheduler_runs/run_0001/labels.json",
      "bytes": 1346377,
      "sha256": "7d21d97c96134b01fe9edc4cb2da69e3ebc0512ed1e4d35a7af2d5d167146a6d"
    },
    "progress_file": {
      "path": "/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/scheduler_progress.json",
      "bytes": 1747,
      "sha256": "8c84ddf218d3171d40d059cfa423c93986cd0c5264198f80fe0f6e13fece93d0"
    }
  }
}
```

Merged `lock_sha256=null`; `execution_lock` is mapped per problem. The full parent manifest file SHA256 and both source manifests/labels/progress hashes are recorded above. Server-only merged labels are excluded from publication. Baseline reports remain at /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/finish/run_0001_0674c3c5000e_a844f793.

Native FP64/stable-global-ID winners only. Same problem/family/TRAIN-base full-pool intersection only; missing response is not zero. Related endpoints are not independent samples. No trained function learner, TSP/ATSP response, full18 model, accuracy, TEST/LIB or generalization claim. This finisher executes no solver/model/GPU code. RESULTS.pdf is the primary delivery.
