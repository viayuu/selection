# Method By Dataset

- Source: `method_by_dataset.csv`
- Meaning: for each dataset, list every method's mean cost and top1/top2/top3 counts.

## ATSP

### ATSPtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATPOENET | 1.715922 | 1 | 701 | 295 | 4 | 0.7010 | 0.2950 | 0.0040 | 1 |
| GLOP | 1.920294 | 2 | 24 | 575 | 401 | 0.0240 | 0.5750 | 0.4010 | 3 |
| MATNET | 2.313879 | 3 | 275 | 130 | 595 | 0.2750 | 0.1300 | 0.5950 | 2 |

### ATSPtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATPOENET | 1.726253 | 1 | 7005 | 2972 | 23 | 0.7005 | 0.2972 | 0.0023 | 1 |
| GLOP | 1.931549 | 2 | 169 | 5760 | 4071 | 0.0169 | 0.5760 | 0.4071 | 3 |
| MATNET | 2.310640 | 3 | 2826 | 1268 | 5906 | 0.2826 | 0.1268 | 0.5906 | 2 |

### ATSPval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATPOENET | 1.722144 | 1 | 704 | 293 | 3 | 0.7040 | 0.2930 | 0.0030 | 1 |
| GLOP | 1.921757 | 2 | 21 | 584 | 395 | 0.0210 | 0.5840 | 0.3950 | 3 |
| MATNET | 2.323148 | 3 | 275 | 123 | 602 | 0.2750 | 0.1230 | 0.6020 | 2 |

## CVRP

### CVRPLIB

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICAM | 67.678618 | 1 | 75 | 18 | 7 | 0.7500 | 0.1800 | 0.0700 | 1 |
| OMNI | 69.520695 | 2 | 12 | 30 | 38 | 0.1200 | 0.3000 | 0.3800 | 2 |
| INVIT | 73.765692 | 3 | 0 | 0 | 11 | 0.0000 | 0.0000 | 0.1100 | 5 |
| LEHD | 75.928254 | 4 | 9 | 36 | 21 | 0.0900 | 0.3600 | 0.2100 | 3 |
| DACT | 81.096688 | 5 | 0 | 1 | 3 | 0.0000 | 0.0100 | 0.0300 | 6 |
| GLOP | 85.326888 | 6 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7 |
| UDC | 93.577304 | 7 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| ELG | 147.921702 | 8 | 4 | 15 | 20 | 0.0400 | 0.1500 | 0.2000 | 4 |

### CVRPtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICAM | 18.547143 | 1 | 291 | 429 | 232 | 0.2910 | 0.4290 | 0.2320 | 2 |
| LEHD | 18.565324 | 2 | 408 | 242 | 268 | 0.4080 | 0.2420 | 0.2680 | 1 |
| OMNI | 18.630295 | 3 | 279 | 273 | 346 | 0.2790 | 0.2730 | 0.3460 | 3 |
| INVIT | 20.820360 | 4 | 0 | 9 | 29 | 0.0000 | 0.0090 | 0.0290 | 5 |
| GLOP | 22.071736 | 5 | 0 | 1 | 11 | 0.0000 | 0.0010 | 0.0110 | 6 |
| DACT | 22.137536 | 6 | 0 | 0 | 5 | 0.0000 | 0.0000 | 0.0050 | 7 |
| UDC | 31.329928 | 7 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| ELG | 57.489077 | 8 | 22 | 46 | 109 | 0.0220 | 0.0460 | 0.1090 | 4 |

### CVRPtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 18.278036 | 1 | 4278 | 2395 | 2348 | 0.4278 | 0.2395 | 0.2348 | 1 |
| ICAM | 18.288666 | 2 | 2727 | 4231 | 2563 | 0.2727 | 0.4231 | 0.2563 | 3 |
| OMNI | 18.349252 | 3 | 2761 | 2782 | 3398 | 0.2761 | 0.2782 | 0.3398 | 2 |
| INVIT | 20.558923 | 4 | 12 | 52 | 271 | 0.0012 | 0.0052 | 0.0271 | 5 |
| GLOP | 21.705130 | 5 | 6 | 21 | 101 | 0.0006 | 0.0021 | 0.0101 | 6 |
| DACT | 21.864802 | 6 | 5 | 9 | 32 | 0.0005 | 0.0009 | 0.0032 | 7 |
| UDC | 30.907904 | 7 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| ELG | 55.658332 | 8 | 211 | 510 | 1287 | 0.0211 | 0.0510 | 0.1287 | 4 |

### CVRPval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 18.491459 | 1 | 428 | 253 | 226 | 0.4280 | 0.2530 | 0.2260 | 1 |
| ICAM | 18.529983 | 2 | 245 | 422 | 283 | 0.2450 | 0.4220 | 0.2830 | 3 |
| OMNI | 18.545915 | 3 | 300 | 266 | 327 | 0.3000 | 0.2660 | 0.3270 | 2 |
| INVIT | 20.787117 | 4 | 3 | 8 | 31 | 0.0030 | 0.0080 | 0.0310 | 5 |
| GLOP | 21.895809 | 5 | 0 | 5 | 10 | 0.0000 | 0.0050 | 0.0100 | 6 |
| DACT | 22.056863 | 6 | 0 | 2 | 3 | 0.0000 | 0.0020 | 0.0030 | 7 |
| UDC | 30.937143 | 7 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| ELG | 58.600401 | 8 | 24 | 44 | 120 | 0.0240 | 0.0440 | 0.1200 | 4 |

## OVRP

### OVRPtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.430148 | 1 | 683 | 317 | 0 | 0.6830 | 0.3170 | 0.0000 | 1 |
| MVMOE | 9.148324 | 2 | 317 | 683 | 0 | 0.3170 | 0.6830 | 0.0000 | 2 |

### OVRPtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.362867 | 1 | 6922 | 3078 | 0 | 0.6922 | 0.3078 | 0.0000 | 1 |
| MVMOE | 9.185701 | 2 | 3078 | 6922 | 0 | 0.3078 | 0.6922 | 0.0000 | 2 |

### OVRPval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.304029 | 1 | 684 | 316 | 0 | 0.6840 | 0.3160 | 0.0000 | 1 |
| MVMOE | 9.099308 | 2 | 316 | 684 | 0 | 0.3160 | 0.6840 | 0.0000 | 2 |

## OVRPB

### OVRPBtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.306141 | 1 | 581 | 419 | 0 | 0.5810 | 0.4190 | 0.0000 | 1 |
| MVMOE | 8.141364 | 2 | 419 | 581 | 0 | 0.4190 | 0.5810 | 0.0000 | 2 |

### OVRPBtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.321386 | 1 | 5729 | 4271 | 0 | 0.5729 | 0.4271 | 0.0000 | 1 |
| MVMOE | 8.260250 | 2 | 4271 | 5729 | 0 | 0.4271 | 0.5729 | 0.0000 | 2 |

### OVRPBval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.325026 | 1 | 584 | 416 | 0 | 0.5840 | 0.4160 | 0.0000 | 1 |
| MVMOE | 8.215832 | 2 | 416 | 584 | 0 | 0.4160 | 0.5840 | 0.0000 | 2 |

## OVRPBL

### OVRPBLtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.218438 | 1 | 563 | 437 | 0 | 0.5630 | 0.4370 | 0.0000 | 1 |
| MVMOE | 7.867767 | 2 | 437 | 563 | 0 | 0.4370 | 0.5630 | 0.0000 | 2 |

### OVRPBLtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.210580 | 1 | 5691 | 4309 | 0 | 0.5691 | 0.4309 | 0.0000 | 1 |
| MVMOE | 7.883809 | 2 | 4309 | 5691 | 0 | 0.4309 | 0.5691 | 0.0000 | 2 |

### OVRPBLval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 7.199121 | 1 | 577 | 423 | 0 | 0.5770 | 0.4230 | 0.0000 | 1 |
| MVMOE | 7.848755 | 2 | 423 | 577 | 0 | 0.4230 | 0.5770 | 0.0000 | 2 |

## OVRPBLTW

### OVRPBLTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 12.040483 | 1 | 872 | 128 | 0 | 0.8720 | 0.1280 | 0.0000 | 1 |
| MTPOMO | 15.283536 | 2 | 128 | 872 | 0 | 0.1280 | 0.8720 | 0.0000 | 2 |

### OVRPBLTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.890103 | 1 | 8777 | 1223 | 0 | 0.8777 | 0.1223 | 0.0000 | 1 |
| MTPOMO | 15.166317 | 2 | 1223 | 8777 | 0 | 0.1223 | 0.8777 | 0.0000 | 2 |

### OVRPBLTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.795510 | 1 | 882 | 118 | 0 | 0.8820 | 0.1180 | 0.0000 | 1 |
| MTPOMO | 15.131536 | 2 | 118 | 882 | 0 | 0.1180 | 0.8820 | 0.0000 | 2 |

## OVRPBTW

### OVRPBTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.858678 | 1 | 889 | 111 | 0 | 0.8890 | 0.1110 | 0.0000 | 1 |
| MTPOMO | 15.253699 | 2 | 111 | 889 | 0 | 0.1110 | 0.8890 | 0.0000 | 2 |

### OVRPBTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.851035 | 1 | 8856 | 1144 | 0 | 0.8856 | 0.1144 | 0.0000 | 1 |
| MTPOMO | 15.208726 | 2 | 1144 | 8856 | 0 | 0.1144 | 0.8856 | 0.0000 | 2 |

### OVRPBTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.804059 | 1 | 883 | 117 | 0 | 0.8830 | 0.1170 | 0.0000 | 1 |
| MTPOMO | 15.155493 | 2 | 117 | 883 | 0 | 0.1170 | 0.8830 | 0.0000 | 2 |

## OVRPL

### OVRPLtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.299554 | 1 | 681 | 319 | 0 | 0.6810 | 0.3190 | 0.0000 | 1 |
| MVMOE | 8.902560 | 2 | 319 | 681 | 0 | 0.3190 | 0.6810 | 0.0000 | 2 |

### OVRPLtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.300787 | 1 | 6976 | 3024 | 0 | 0.6976 | 0.3024 | 0.0000 | 1 |
| MVMOE | 9.016318 | 2 | 3024 | 6976 | 0 | 0.3024 | 0.6976 | 0.0000 | 2 |

### OVRPLval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 8.226538 | 1 | 671 | 329 | 0 | 0.6710 | 0.3290 | 0.0000 | 1 |
| MVMOE | 9.045072 | 2 | 329 | 671 | 0 | 0.3290 | 0.6710 | 0.0000 | 2 |

## OVRPLTW

### OVRPLTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.312225 | 1 | 888 | 112 | 0 | 0.8880 | 0.1120 | 0.0000 | 1 |
| MTPOMO | 14.455833 | 2 | 112 | 888 | 0 | 0.1120 | 0.8880 | 0.0000 | 2 |

### OVRPLTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.375585 | 1 | 8740 | 1260 | 0 | 0.8740 | 0.1260 | 0.0000 | 1 |
| MTPOMO | 14.509354 | 2 | 1260 | 8740 | 0 | 0.1260 | 0.8740 | 0.0000 | 2 |

### OVRPLTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.359611 | 1 | 872 | 128 | 0 | 0.8720 | 0.1280 | 0.0000 | 1 |
| MTPOMO | 14.467725 | 2 | 128 | 872 | 0 | 0.1280 | 0.8720 | 0.0000 | 2 |

## OVRPTW

### OVRPTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.403935 | 1 | 877 | 123 | 0 | 0.8770 | 0.1230 | 0.0000 | 1 |
| MTPOMO | 14.507176 | 2 | 123 | 877 | 0 | 0.1230 | 0.8770 | 0.0000 | 2 |

### OVRPTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.301199 | 1 | 8797 | 1203 | 0 | 0.8797 | 0.1203 | 0.0000 | 1 |
| MTPOMO | 14.459164 | 2 | 1203 | 8797 | 0 | 0.1203 | 0.8797 | 0.0000 | 2 |

### OVRPTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 11.380933 | 1 | 869 | 131 | 0 | 0.8690 | 0.1310 | 0.0000 | 1 |
| MTPOMO | 14.350275 | 2 | 131 | 869 | 0 | 0.1310 | 0.8690 | 0.0000 | 2 |

## TSP

### TSPLIB

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 8.258462 | 1 | 29 | 17 | 1 | 0.5918 | 0.3469 | 0.0204 | 1 |
| ELG | 8.512219 | 2 | 5 | 13 | 19 | 0.1020 | 0.2653 | 0.3878 | 3 |
| OMNI | 8.578038 | 3 | 3 | 8 | 20 | 0.0612 | 0.1633 | 0.4082 | 4 |
| INVIT | 8.706001 | 4 | 1 | 9 | 1 | 0.0204 | 0.1837 | 0.0204 | 5 |
| GLOP | 8.933969 | 5 | 1 | 0 | 4 | 0.0204 | 0.0000 | 0.0816 | 6 |
| DIFUSCO | 9.111516 | 6 | 10 | 2 | 4 | 0.2041 | 0.0408 | 0.0816 | 2 |
| T2T | 9.452321 | 7 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7 |
| DACT | 10.069326 | 8 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| UDC | 13.192162 | 9 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 9 |
| LIH | 113.687182 | 10 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 10 |

### TSPtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 8.157863 | 1 | 699 | 118 | 75 | 0.6990 | 0.1180 | 0.0750 | 1 |
| OMNI | 8.318756 | 2 | 199 | 444 | 254 | 0.1990 | 0.4440 | 0.2540 | 2 |
| ELG | 8.461461 | 3 | 29 | 225 | 420 | 0.0290 | 0.2250 | 0.4200 | 4 |
| GLOP | 8.687625 | 4 | 13 | 61 | 112 | 0.0130 | 0.0610 | 0.1120 | 5 |
| INVIT | 8.704566 | 5 | 7 | 105 | 93 | 0.0070 | 0.1050 | 0.0930 | 6 |
| DIFUSCO | 9.235499 | 6 | 53 | 47 | 40 | 0.0530 | 0.0470 | 0.0400 | 3 |
| T2T | 9.465946 | 7 | 0 | 0 | 6 | 0.0000 | 0.0000 | 0.0060 | 7 |
| DACT | 10.128731 | 8 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| UDC | 13.084382 | 9 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 9 |
| LIH | 116.108027 | 10 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 10 |

### TSPtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 8.127289 | 1 | 6947 | 1197 | 758 | 0.6947 | 0.1197 | 0.0758 | 1 |
| OMNI | 8.278670 | 2 | 1827 | 4659 | 2369 | 0.1827 | 0.4659 | 0.2369 | 2 |
| ELG | 8.428071 | 3 | 422 | 2093 | 4073 | 0.0422 | 0.2093 | 0.4073 | 4 |
| GLOP | 8.636439 | 4 | 140 | 519 | 1232 | 0.0140 | 0.0519 | 0.1232 | 6 |
| INVIT | 8.642378 | 5 | 166 | 1040 | 1207 | 0.0166 | 0.1040 | 0.1207 | 5 |
| DIFUSCO | 9.198087 | 6 | 497 | 487 | 340 | 0.0497 | 0.0487 | 0.0340 | 3 |
| T2T | 9.446305 | 7 | 1 | 5 | 18 | 0.0001 | 0.0005 | 0.0018 | 7 |
| DACT | 10.045222 | 8 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0003 | 8 |
| UDC | 13.003409 | 9 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 9 |
| LIH | 115.854567 | 10 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 10 |

### TSPval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEHD | 8.038186 | 1 | 684 | 120 | 77 | 0.6840 | 0.1200 | 0.0770 | 1 |
| OMNI | 8.182173 | 2 | 178 | 464 | 231 | 0.1780 | 0.4640 | 0.2310 | 2 |
| ELG | 8.308660 | 3 | 54 | 203 | 428 | 0.0540 | 0.2030 | 0.4280 | 3 |
| GLOP | 8.537977 | 4 | 13 | 62 | 110 | 0.0130 | 0.0620 | 0.1100 | 6 |
| INVIT | 8.558233 | 5 | 18 | 99 | 113 | 0.0180 | 0.0990 | 0.1130 | 5 |
| DIFUSCO | 9.098994 | 6 | 53 | 51 | 38 | 0.0530 | 0.0510 | 0.0380 | 4 |
| T2T | 9.349034 | 7 | 0 | 1 | 3 | 0.0000 | 0.0010 | 0.0030 | 7 |
| DACT | 9.923410 | 8 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8 |
| UDC | 12.925378 | 9 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 9 |
| LIH | 115.952039 | 10 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 10 |

## VRPB

### VRPBtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.701428 | 1 | 407 | 593 | 0 | 0.4070 | 0.5930 | 0.0000 | 2 |
| MVMOE | 11.584618 | 2 | 593 | 407 | 0 | 0.5930 | 0.4070 | 0.0000 | 1 |

### VRPBtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.715950 | 1 | 4020 | 5980 | 0 | 0.4020 | 0.5980 | 0.0000 | 2 |
| MVMOE | 11.603011 | 2 | 5980 | 4020 | 0 | 0.5980 | 0.4020 | 0.0000 | 1 |

### VRPBval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.822326 | 1 | 392 | 608 | 0 | 0.3920 | 0.6080 | 0.0000 | 2 |
| MVMOE | 11.803180 | 2 | 608 | 392 | 0 | 0.6080 | 0.3920 | 0.0000 | 1 |

## VRPBL

### VRPBLtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.599956 | 1 | 411 | 589 | 0 | 0.4110 | 0.5890 | 0.0000 | 2 |
| MVMOE | 11.212746 | 2 | 589 | 411 | 0 | 0.5890 | 0.4110 | 0.0000 | 1 |

### VRPBLtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.572554 | 1 | 4022 | 5978 | 0 | 0.4022 | 0.5978 | 0.0000 | 2 |
| MVMOE | 11.134370 | 2 | 5978 | 4022 | 0 | 0.5978 | 0.4022 | 0.0000 | 1 |

### VRPBLval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 10.705363 | 1 | 413 | 587 | 0 | 0.4130 | 0.5870 | 0.0000 | 2 |
| MVMOE | 11.321271 | 2 | 587 | 413 | 0 | 0.5870 | 0.4130 | 0.0000 | 1 |

## VRPBLTW

### VRPBLTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.506060 | 1 | 860 | 140 | 0 | 0.8600 | 0.1400 | 0.0000 | 1 |
| MTPOMO | 24.793583 | 2 | 140 | 860 | 0 | 0.1400 | 0.8600 | 0.0000 | 2 |

### VRPBLTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.572153 | 1 | 8590 | 1410 | 0 | 0.8590 | 0.1410 | 0.0000 | 1 |
| MTPOMO | 24.899868 | 2 | 1410 | 8590 | 0 | 0.1410 | 0.8590 | 0.0000 | 2 |

### VRPBLTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.470786 | 1 | 850 | 150 | 0 | 0.8500 | 0.1500 | 0.0000 | 1 |
| MTPOMO | 24.715246 | 2 | 150 | 850 | 0 | 0.1500 | 0.8500 | 0.0000 | 2 |

## VRPBTW

### VRPBTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.310201 | 1 | 864 | 136 | 0 | 0.8640 | 0.1360 | 0.0000 | 1 |
| MTPOMO | 24.685686 | 2 | 136 | 864 | 0 | 0.1360 | 0.8640 | 0.0000 | 2 |

### VRPBTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.489391 | 1 | 8670 | 1330 | 0 | 0.8670 | 0.1330 | 0.0000 | 1 |
| MTPOMO | 24.878005 | 2 | 1330 | 8670 | 0 | 0.1330 | 0.8670 | 0.0000 | 2 |

### VRPBTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 21.619582 | 1 | 863 | 137 | 0 | 0.8630 | 0.1370 | 0.0000 | 1 |
| MTPOMO | 24.888895 | 2 | 137 | 863 | 0 | 0.1370 | 0.8630 | 0.0000 | 2 |

## VRPL

### VRPLtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 13.343418 | 1 | 667 | 333 | 0 | 0.6670 | 0.3330 | 0.0000 | 1 |
| MVMOE | 14.163921 | 2 | 333 | 667 | 0 | 0.3330 | 0.6670 | 0.0000 | 2 |

### VRPLtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 13.385177 | 1 | 6546 | 3454 | 0 | 0.6546 | 0.3454 | 0.0000 | 1 |
| MVMOE | 14.169946 | 2 | 3454 | 6546 | 0 | 0.3454 | 0.6546 | 0.0000 | 2 |

### VRPLval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MTPOMO | 13.454017 | 1 | 655 | 345 | 0 | 0.6550 | 0.3450 | 0.0000 | 1 |
| MVMOE | 14.360135 | 2 | 345 | 655 | 0 | 0.3450 | 0.6550 | 0.0000 | 2 |

## VRPLTW

### VRPLTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.012325 | 1 | 865 | 135 | 0 | 0.8650 | 0.1350 | 0.0000 | 1 |
| MTPOMO | 23.163571 | 2 | 135 | 865 | 0 | 0.1350 | 0.8650 | 0.0000 | 2 |

### VRPLTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.451790 | 1 | 8597 | 1403 | 0 | 0.8597 | 0.1403 | 0.0000 | 1 |
| MTPOMO | 23.493168 | 2 | 1403 | 8597 | 0 | 0.1403 | 0.8597 | 0.0000 | 2 |

### VRPLTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.274024 | 1 | 856 | 144 | 0 | 0.8560 | 0.1440 | 0.0000 | 1 |
| MTPOMO | 23.279750 | 2 | 144 | 856 | 0 | 0.1440 | 0.8560 | 0.0000 | 2 |

## VRPTW

### VRPTWtest

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.243225 | 1 | 853 | 147 | 0 | 0.8530 | 0.1470 | 0.0000 | 1 |
| MTPOMO | 23.240192 | 2 | 147 | 853 | 0 | 0.1470 | 0.8530 | 0.0000 | 2 |

### VRPTWtrain

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.504243 | 1 | 8576 | 1424 | 0 | 0.8576 | 0.1424 | 0.0000 | 1 |
| MTPOMO | 23.544962 | 2 | 1424 | 8576 | 0 | 0.1424 | 0.8576 | 0.0000 | 2 |

### VRPTWval

| method | mean_cost | mean_cost_rank | top1_count | top2_count | top3_count | top1_ratio | top2_ratio | top3_ratio | top1_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVMOE | 20.522358 | 1 | 869 | 131 | 0 | 0.8690 | 0.1310 | 0.0000 | 1 |
| MTPOMO | 23.643270 | 2 | 131 | 869 | 0 | 0.1310 | 0.8690 | 0.0000 | 2 |
