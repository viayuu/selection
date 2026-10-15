# R79 Original NSS: New Data

Both task-specific checkpoints were selected using VALID only.
Each new TEST and original LIB was evaluated once with its corresponding frozen checkpoint.

| Task/split | n | Top1 | Top2 | Top3 | Mean cost | TRAIN-fixed SBS | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP/test | 3895 | 0.4395 | 0.6737 | 0.8290 | 7.634391 | 7.648986 | 7.602818 |
| TSP/TSPLIB | 49 | 0.2041 | 0.6122 | 0.7347 | 8.217907 | 8.240435 | 8.142167 |
| CVRP/test | 3895 | 0.4621 | 0.6508 | 0.7669 | 17.895818 | 18.013673 | 17.810107 |
| CVRP/CVRPLIB | 100 | 0.7000 | 0.8100 | 0.9000 | 67.101112 | 67.245364 | 66.945093 |

FP64 raw costs are retained in reports; original FP32 RankingLoss and FP32 winner metrics are unchanged.
New-data gap is retained-pool Oracle regret, not an optimal-tour gap. Solver time labels are placeholders.
Post-hoc split SBS is descriptive only; deployment SBS comes from TRAIN.
