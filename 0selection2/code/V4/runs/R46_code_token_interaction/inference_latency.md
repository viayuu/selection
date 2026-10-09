# Inference latency on the authorized RTX3090

FP32 eval; same128 validation instances; median of20 CUDA-event measurements after3 warmups.
Forward only; no data preparation, geometry caching or I/O. Peak allocation includes the resident input batches.

| Model | Problem | Batch | Max nodes | Median ms | p90 ms | Peak GiB |
|---|---|---:|---:|---:|---:|---:|
| R45A | TSP | 128 | 497 | 54.369 | 55.026 | 1.750 |
| R45A | CVRP | 128 | 500 | 54.919 | 55.228 | 1.765 |
| R45A | ATSP | 128 | 29 | 6.080 | 6.113 | 0.150 |
| R45A | OVRPTW | 128 | 57 | 6.252 | 6.274 | 0.173 |
| R45B | TSP | 128 | 497 | 54.831 | 55.393 | 1.749 |
| R45B | CVRP | 128 | 500 | 55.401 | 55.802 | 1.765 |
| R45B | ATSP | 128 | 29 | 7.195 | 7.213 | 0.151 |
| R45B | OVRPTW | 128 | 57 | 7.335 | 7.361 | 0.175 |
| R46A | TSP | 128 | 497 | 55.487 | 55.888 | 1.749 |
| R46A | CVRP | 128 | 500 | 57.385 | 58.102 | 1.765 |
| R46A | ATSP | 128 | 29 | 6.429 | 6.463 | 0.156 |
| R46A | OVRPTW | 128 | 57 | 6.786 | 6.812 | 0.175 |
