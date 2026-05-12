# Method Dominance Statistics

## ATSP

- Datasets: `ATSPtest, ATSPtrain, ATSPval`
- Oracle top1 mean cost: `1.633721`
- Mean-cost dataset wins: `{"UNICO_MatPOENet": 3}`
- Top1 dataset wins: `{"UNICO_MatPOENet": 3}`
- Same-method top1+mean wins: `{"UNICO_MatPOENet": 3}`

### ATSPtest

- Instances: `1000`; methods: `5`; tie instances: `17`
- Oracle top1 mean cost: `1.624559`
- Mean-cost leader: `UNICO_MatPOENet` (`1.669680`)
- Top1 leader: `UNICO_MatPOENet` (`458`)
- Same method wins both: `UNICO_MatPOENet`

| method          | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| --------------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| UNICO_MatPOENet |              1.624559 |  1.669680 |         1 |  458 |  269 |  219 |         1 |
| ICAM_ATSP       |              1.624559 |  1.705685 |         2 |  301 |  266 |  320 |         2 |
| MATPOENET       |              1.624559 |  1.715922 |         3 |   88 |  360 |  324 |         4 |
| GLOP            |              1.624559 |  1.920294 |         4 |    5 |   19 |   93 |         5 |
| MATNET          |              1.624559 |  2.313879 |         5 |  148 |   86 |   44 |         3 |

### ATSPtrain

- Instances: `10000`; methods: `5`; tie instances: `121`
- Oracle top1 mean cost: `1.635050`
- Mean-cost leader: `UNICO_MatPOENet` (`1.679478`)
- Top1 leader: `UNICO_MatPOENet` (`4694`)
- Same method wins both: `UNICO_MatPOENet`

| method          | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| --------------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| UNICO_MatPOENet |              1.635050 |  1.679478 |         1 | 4694 | 2522 | 2288 |         1 |
| ICAM_ATSP       |              1.635050 |  1.716322 |         2 | 3059 | 2667 | 3160 |         2 |
| MATPOENET       |              1.635050 |  1.726253 |         3 |  819 | 3650 | 3122 |         4 |
| GLOP            |              1.635050 |  1.931549 |         4 |   21 |  137 |  994 |         5 |
| MATNET          |              1.635050 |  2.310640 |         5 | 1407 | 1024 |  436 |         3 |

### ATSPval

- Instances: `1000`; methods: `5`; tie instances: `10`
- Oracle top1 mean cost: `1.629583`
- Mean-cost leader: `UNICO_MatPOENet` (`1.673524`)
- Top1 leader: `UNICO_MatPOENet` (`480`)
- Same method wins both: `UNICO_MatPOENet`

| method          | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| --------------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| UNICO_MatPOENet |              1.629583 |  1.673524 |         1 |  480 |  256 |  211 |         1 |
| ICAM_ATSP       |              1.629583 |  1.714536 |         2 |  302 |  258 |  325 |         2 |
| MATPOENET       |              1.629583 |  1.722144 |         3 |   76 |  376 |  321 |         4 |
| GLOP            |              1.629583 |  1.921757 |         4 |    3 |   15 |  101 |         5 |
| MATNET          |              1.629583 |  2.323148 |         5 |  139 |   95 |   42 |         3 |

## CVRP

- Datasets: `CVRPLIB, CVRPtest, CVRPtrain, CVRPval`
- Oracle top1 mean cost: `18.336564`
- Mean-cost dataset wins: `{"RELD_CVRP": 4}`
- Top1 dataset wins: `{"OMNI": 2, "RELD_CVRP": 2}`
- Same-method top1+mean wins: `{"RELD_CVRP": 2}`

### CVRPLIB

- Instances: `100`; methods: `10`; tie instances: `0`
- Oracle top1 mean cost: `66.945093`
- Mean-cost leader: `RELD_CVRP` (`67.245364`)
- Top1 leader: `RELD_CVRP` (`45`)
- Same method wins both: `RELD_CVRP`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_CVRP   |             66.945093 | 67.245364 |         1 |   45 |   39 |   11 |         1 |
| ICAM        |             66.945093 | 67.678618 |         2 |   38 |   32 |    6 |         2 |
| ELG         |             66.945093 | 68.792637 |         3 |    3 |    5 |   26 |         4 |
| OMNI        |             66.945093 | 69.053286 |         4 |    5 |    4 |   23 |         3 |
| RouteFinder |             66.945093 | 70.263184 |         5 |    1 |    2 |    4 |         8 |
| MoSES_RF    |             66.945093 | 71.061856 |         6 |    0 |    4 |    4 |        10 |
| MoSES_CaDA  |             66.945093 | 71.245238 |         7 |    2 |    1 |   10 |         6 |
| BQ          |             66.945093 | 72.449576 |         8 |    3 |    7 |    8 |         5 |
| MVMOE       |             66.945093 | 76.174769 |         9 |    1 |    0 |    2 |         9 |
| LEHD        |             66.945093 | 77.294544 |        10 |    2 |    6 |    6 |         7 |

### CVRPtest

- Instances: `1000`; methods: `10`; tie instances: `0`
- Oracle top1 mean cost: `18.158602`
- Mean-cost leader: `RELD_CVRP` (`18.332332`)
- Top1 leader: `RELD_CVRP` (`254`)
- Same method wins both: `RELD_CVRP`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_CVRP   |             18.158602 | 18.332332 |         1 |  254 |  293 |  188 |         1 |
| OMNI        |             18.158602 | 18.502841 |         2 |  244 |  158 |  107 |         2 |
| ICAM        |             18.158602 | 18.547143 |         3 |  113 |  139 |  191 |         5 |
| BQ          |             18.158602 | 18.687272 |         4 |  122 |  115 |  136 |         4 |
| LEHD        |             18.158602 | 18.723970 |         5 |  147 |  123 |  117 |         3 |
| ELG         |             18.158602 | 18.729156 |         6 |   15 |   36 |   92 |         9 |
| RouteFinder |             18.158602 | 18.971997 |         7 |   14 |   27 |   59 |        10 |
| MoSES_CaDA  |             18.158602 | 19.372128 |         8 |   26 |   51 |   47 |         7 |
| MVMOE       |             18.158602 | 19.726057 |         9 |   26 |   27 |   26 |         8 |
| MoSES_RF    |             18.158602 | 20.844264 |        10 |   39 |   31 |   37 |         6 |

### CVRPtrain

- Instances: `10000`; methods: `10`; tie instances: `2`
- Oracle top1 mean cost: `17.890935`
- Mean-cost leader: `RELD_CVRP` (`18.070454`)
- Top1 leader: `OMNI` (`2533`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_CVRP   |             17.890935 | 18.070454 |         1 | 2234 | 2984 | 2120 |         2 |
| OMNI        |             17.890935 | 18.219834 |         2 | 2533 | 1447 | 1159 |         1 |
| ICAM        |             17.890935 | 18.288666 |         3 | 1160 | 1310 | 1781 |         5 |
| BQ          |             17.890935 | 18.384555 |         4 | 1297 | 1242 | 1262 |         4 |
| LEHD        |             17.890935 | 18.447959 |         5 | 1608 | 1250 | 1050 |         3 |
| ELG         |             17.890935 | 18.448131 |         6 |  187 |  372 |  792 |         8 |
| RouteFinder |             17.890935 | 18.708003 |         7 |  168 |  362 |  608 |         9 |
| MoSES_CaDA  |             17.890935 | 19.110649 |         8 |  314 |  418 |  526 |         7 |
| MVMOE       |             17.890935 | 19.454079 |         9 |  150 |  240 |  322 |        10 |
| MoSES_RF    |             17.890935 | 20.469599 |        10 |  349 |  375 |  380 |         6 |

### CVRPval

- Instances: `1000`; methods: `10`; tie instances: `0`
- Oracle top1 mean cost: `18.109962`
- Mean-cost leader: `RELD_CVRP` (`18.294333`)
- Top1 leader: `OMNI` (`281`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_CVRP   |             18.109962 | 18.294333 |         1 |  219 |  280 |  225 |         2 |
| OMNI        |             18.109962 | 18.416695 |         2 |  281 |  142 |  103 |         1 |
| ICAM        |             18.109962 | 18.529983 |         3 |  102 |  134 |  172 |         5 |
| BQ          |             18.109962 | 18.622535 |         4 |  124 |  126 |  112 |         4 |
| LEHD        |             18.109962 | 18.655634 |         5 |  150 |  146 |  104 |         3 |
| ELG         |             18.109962 | 18.664323 |         6 |   19 |   46 |   86 |         9 |
| RouteFinder |             18.109962 | 18.917150 |         7 |   22 |   24 |   59 |         8 |
| MoSES_CaDA  |             18.109962 | 19.329173 |         8 |   25 |   53 |   52 |         7 |
| MVMOE       |             18.109962 | 19.695440 |         9 |   18 |   21 |   42 |        10 |
| MoSES_RF    |             18.109962 | 20.609603 |        10 |   40 |   28 |   45 |         6 |

## OVRP

- Datasets: `OVRPtest, OVRPtrain, OVRPval`
- Oracle top1 mean cost: `7.401085`
- Mean-cost dataset wins: `{"MoSES_RF": 3}`
- Top1 dataset wins: `{"MoSES_CaDA": 2, "MoSES_RF": 1}`
- Same-method top1+mean wins: `{"MoSES_RF": 1}`

### OVRPtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `7.408692`
- Mean-cost leader: `MoSES_RF` (`7.469509`)
- Top1 leader: `MoSES_CaDA` (`319`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.408692 |  7.469509 |         1 |  317 |  295 |  211 |         2 |
| MoSES_CaDA  |              7.408692 |  7.478137 |         2 |  319 |  277 |  213 |         1 |
| RouteFinder |              7.408692 |  7.491797 |         3 |  194 |  248 |  316 |         3 |
| RELD_MTL    |              7.408692 |  7.609439 |         4 |   83 |   72 |  121 |         4 |
| RELD_MOEL   |              7.408692 |  7.645092 |         5 |   69 |   88 |   99 |         5 |
| MTPOMO      |              7.408692 |  8.430148 |         6 |   14 |   12 |   22 |         6 |
| MVMOE       |              7.408692 |  9.148324 |         7 |    4 |    8 |   18 |         7 |

### OVRPtrain

- Instances: `10000`; methods: `7`; tie instances: `24`
- Oracle top1 mean cost: `7.400087`
- Mean-cost leader: `MoSES_RF` (`7.459079`)
- Top1 leader: `MoSES_RF` (`3240`)
- Same method wins both: `MoSES_RF`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.400087 |  7.459079 |         1 | 3240 | 2951 | 2191 |         1 |
| MoSES_CaDA  |              7.400087 |  7.466570 |         2 | 3211 | 2693 | 2159 |         2 |
| RouteFinder |              7.400087 |  7.477161 |         3 | 2139 | 2518 | 3037 |         3 |
| RELD_MTL    |              7.400087 |  7.591940 |         4 |  745 |  904 | 1234 |         4 |
| RELD_MOEL   |              7.400087 |  7.637597 |         5 |  537 |  710 |  970 |         5 |
| MTPOMO      |              7.400087 |  8.362867 |         6 |   83 |  152 |  257 |         6 |
| MVMOE       |              7.400087 |  9.185701 |         7 |   45 |   72 |  152 |         7 |

### OVRPval

- Instances: `1000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `7.403457`
- Mean-cost leader: `MoSES_RF` (`7.461660`)
- Top1 leader: `MoSES_CaDA` (`326`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.403457 |  7.461660 |         1 |  302 |  293 |  240 |         2 |
| MoSES_CaDA  |              7.403457 |  7.466386 |         2 |  326 |  267 |  235 |         1 |
| RouteFinder |              7.403457 |  7.478650 |         3 |  213 |  266 |  288 |         3 |
| RELD_MTL    |              7.403457 |  7.591531 |         4 |   85 |   90 |  109 |         4 |
| RELD_MOEL   |              7.403457 |  7.649086 |         5 |   61 |   66 |   91 |         5 |
| MTPOMO      |              7.403457 |  8.304029 |         6 |    8 |   10 |   24 |         6 |
| MVMOE       |              7.403457 |  9.099308 |         7 |    5 |    8 |   13 |         7 |

## OVRPB

- Datasets: `OVRPBtest, OVRPBtrain, OVRPBval`
- Oracle top1 mean cost: `6.255231`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPBtest

- Instances: `1000`; methods: `7`; tie instances: `5`
- Oracle top1 mean cost: `6.232885`
- Mean-cost leader: `RELD_MTL` (`6.296312`)
- Top1 leader: `RELD_MTL` (`560`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.232885 |  6.296312 |         1 |  560 |  367 |   60 |         1 |
| RELD_MOEL   |              6.232885 |  6.338776 |         2 |  404 |  509 |   70 |         2 |
| MTPOMO      |              6.232885 |  7.306141 |         3 |   20 |   60 |  381 |         3 |
| MoSES_CaDA  |              6.232885 |  7.438149 |         4 |    0 |    3 |   80 |         7 |
| MoSES_RF    |              6.232885 |  7.442370 |         5 |    1 |    0 |   55 |         5 |
| RouteFinder |              6.232885 |  7.455197 |         6 |    1 |    4 |   44 |         6 |
| MVMOE       |              6.232885 |  8.141364 |         7 |   14 |   57 |  310 |         4 |

### OVRPBtrain

- Instances: `10000`; methods: `7`; tie instances: `38`
- Oracle top1 mean cost: `6.256920`
- Mean-cost leader: `RELD_MTL` (`6.321635`)
- Top1 leader: `RELD_MTL` (`5385`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.256920 |  6.321635 |         1 | 5385 | 3881 |  609 |         1 |
| RELD_MOEL   |              6.256920 |  6.364881 |         2 | 4252 | 4674 |  847 |         2 |
| MTPOMO      |              6.256920 |  7.321386 |         3 |  164 |  660 | 3656 |         4 |
| MoSES_CaDA  |              6.256920 |  7.464470 |         4 |   12 |   67 |  752 |         5 |
| MoSES_RF    |              6.256920 |  7.470292 |         5 |    7 |   46 |  590 |         7 |
| RouteFinder |              6.256920 |  7.482950 |         6 |   11 |   29 |  472 |         6 |
| MVMOE       |              6.256920 |  8.260250 |         7 |  169 |  643 | 3074 |         3 |

### OVRPBval

- Instances: `1000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `6.260687`
- Mean-cost leader: `RELD_MTL` (`6.321981`)
- Top1 leader: `RELD_MTL` (`545`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.260687 |  6.321981 |         1 |  545 |  387 |   59 |         1 |
| RELD_MOEL   |              6.260687 |  6.367115 |         2 |  423 |  466 |   92 |         2 |
| MTPOMO      |              6.260687 |  7.325026 |         3 |   14 |   70 |  361 |         4 |
| MoSES_RF    |              6.260687 |  7.467467 |         4 |    2 |    8 |   77 |         5 |
| MoSES_CaDA  |              6.260687 |  7.471561 |         5 |    1 |    5 |   78 |         6 |
| RouteFinder |              6.260687 |  7.482710 |         6 |    0 |    3 |   41 |         7 |
| MVMOE       |              6.260687 |  8.215832 |         7 |   15 |   61 |  292 |         3 |

## OVRPBL

- Datasets: `OVRPBLtest, OVRPBLtrain, OVRPBLval`
- Oracle top1 mean cost: `6.242798`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPBLtest

- Instances: `1000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `6.241775`
- Mean-cost leader: `RELD_MTL` (`6.306825`)
- Top1 leader: `RELD_MTL` (`544`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.241775 |  6.306825 |         1 |  544 |  381 |   64 |         1 |
| RELD_MOEL   |              6.241775 |  6.344342 |         2 |  427 |  469 |   89 |         2 |
| MTPOMO      |              6.241775 |  7.218438 |         3 |    9 |   74 |  365 |         4 |
| MoSES_CaDA  |              6.241775 |  7.445964 |         4 |    0 |    5 |   67 |         7 |
| MoSES_RF    |              6.241775 |  7.446542 |         5 |    2 |    6 |   60 |         5 |
| RouteFinder |              6.241775 |  7.463310 |         6 |    2 |    3 |   43 |         6 |
| MVMOE       |              6.241775 |  7.867767 |         7 |   16 |   62 |  312 |         3 |

### OVRPBLtrain

- Instances: `10000`; methods: `7`; tie instances: `30`
- Oracle top1 mean cost: `6.242588`
- Mean-cost leader: `RELD_MTL` (`6.307886`)
- Top1 leader: `RELD_MTL` (`5415`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.242588 |  6.307886 |         1 | 5415 | 3821 |  662 |         1 |
| RELD_MOEL   |              6.242588 |  6.349055 |         2 | 4179 | 4739 |  842 |         2 |
| MTPOMO      |              6.242588 |  7.210580 |         3 |  205 |  636 | 3632 |         3 |
| MoSES_CaDA  |              6.242588 |  7.446621 |         4 |    9 |   85 |  727 |         6 |
| MoSES_RF    |              6.242588 |  7.449945 |         5 |   12 |   56 |  636 |         5 |
| RouteFinder |              6.242588 |  7.463744 |         6 |    9 |   22 |  467 |         7 |
| MVMOE       |              6.242588 |  7.883809 |         7 |  171 |  641 | 3034 |         4 |

### OVRPBLval

- Instances: `1000`; methods: `7`; tie instances: `4`
- Oracle top1 mean cost: `6.245926`
- Mean-cost leader: `RELD_MTL` (`6.310260`)
- Top1 leader: `RELD_MTL` (`555`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              6.245926 |  6.310260 |         1 |  555 |  366 |   65 |         1 |
| RELD_MOEL   |              6.245926 |  6.355338 |         2 |  400 |  494 |   86 |         2 |
| MTPOMO      |              6.245926 |  7.199121 |         3 |   24 |   67 |  368 |         3 |
| MoSES_CaDA  |              6.245926 |  7.473451 |         4 |    2 |    7 |   54 |         5 |
| MoSES_RF    |              6.245926 |  7.477525 |         5 |    0 |    2 |   60 |         6 |
| RouteFinder |              6.245926 |  7.494897 |         6 |    0 |    4 |   53 |         7 |
| MVMOE       |              6.245926 |  7.848755 |         7 |   19 |   60 |  314 |         4 |

## OVRPBLTW

- Datasets: `OVRPBLTWtest, OVRPBLTWtrain, OVRPBLTWval`
- Oracle top1 mean cost: `10.424513`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPBLTWtest

- Instances: `1000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `10.471024`
- Mean-cost leader: `RELD_MTL` (`10.592773`)
- Top1 leader: `RELD_MTL` (`478`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.471024 | 10.592773 |         1 |  478 |  421 |  101 |         1 |
| RELD_MOEL   |             10.471024 | 10.627977 |         2 |  447 |  420 |  129 |         2 |
| MVMOE       |             10.471024 | 12.040483 |         3 |   75 |  154 |  517 |         3 |
| RouteFinder |             10.471024 | 12.676970 |         4 |    0 |    4 |  162 |         4 |
| MoSES_RF    |             10.471024 | 13.219413 |         5 |    0 |    0 |   34 |         5 |
| MoSES_CaDA  |             10.471024 | 14.423974 |         6 |    0 |    1 |   55 |         6 |
| MTPOMO      |             10.471024 | 15.283536 |         7 |    0 |    0 |    2 |         7 |

### OVRPBLTWtrain

- Instances: `10000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `10.421190`
- Mean-cost leader: `RELD_MTL` (`10.547633`)
- Top1 leader: `RELD_MTL` (`4848`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.421190 | 10.547633 |         1 | 4848 | 3994 | 1142 |         1 |
| RELD_MOEL   |             10.421190 | 10.580266 |         2 | 4301 | 4460 | 1215 |         2 |
| MVMOE       |             10.421190 | 11.890103 |         3 |  838 | 1492 | 5324 |         3 |
| RouteFinder |             10.421190 | 12.664670 |         4 |    6 |   33 | 1520 |         4 |
| MoSES_RF    |             10.421190 | 13.194019 |         5 |    3 |   10 |  300 |         6 |
| MoSES_CaDA  |             10.421190 | 14.429474 |         6 |    4 |   11 |  472 |         5 |
| MTPOMO      |             10.421190 | 15.166317 |         7 |    0 |    0 |   27 |         7 |

### OVRPBLTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `10.411234`
- Mean-cost leader: `RELD_MTL` (`10.541174`)
- Top1 leader: `RELD_MTL` (`471`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.411234 | 10.541174 |         1 |  471 |  421 |  106 |         1 |
| RELD_MOEL   |             10.411234 | 10.573320 |         2 |  449 |  414 |  132 |         2 |
| MVMOE       |             10.411234 | 11.795510 |         3 |   79 |  158 |  527 |         3 |
| RouteFinder |             10.411234 | 12.636378 |         4 |    0 |    4 |  147 |         5 |
| MoSES_RF    |             10.411234 | 13.137875 |         5 |    1 |    2 |   33 |         4 |
| MoSES_CaDA  |             10.411234 | 14.326487 |         6 |    0 |    1 |   51 |         6 |
| MTPOMO      |             10.411234 | 15.131536 |         7 |    0 |    0 |    4 |         7 |

## OVRPBTW

- Datasets: `OVRPBTWtest, OVRPBTWtrain, OVRPBTWval`
- Oracle top1 mean cost: `10.455047`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPBTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `10.478190`
- Mean-cost leader: `RELD_MTL` (`10.608986`)
- Top1 leader: `RELD_MTL` (`460`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.478190 | 10.608986 |         1 |  460 |  425 |  115 |         1 |
| RELD_MOEL   |             10.478190 | 10.644272 |         2 |  444 |  412 |  143 |         2 |
| MVMOE       |             10.478190 | 11.858678 |         3 |   95 |  159 |  526 |         3 |
| RouteFinder |             10.478190 | 12.740773 |         4 |    1 |    2 |  148 |         4 |
| MoSES_RF    |             10.478190 | 13.316464 |         5 |    0 |    1 |   32 |         5 |
| MoSES_CaDA  |             10.478190 | 14.581801 |         6 |    0 |    1 |   32 |         6 |
| MTPOMO      |             10.478190 | 15.253699 |         7 |    0 |    0 |    4 |         7 |

### OVRPBTWtrain

- Instances: `10000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `10.460256`
- Mean-cost leader: `RELD_MTL` (`10.589365`)
- Top1 leader: `RELD_MTL` (`4736`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.460256 | 10.589365 |         1 | 4736 | 4052 | 1195 |         1 |
| RELD_MOEL   |             10.460256 | 10.617405 |         2 | 4361 | 4331 | 1278 |         2 |
| MVMOE       |             10.460256 | 11.851035 |         3 |  882 | 1561 | 5333 |         3 |
| RouteFinder |             10.460256 | 12.722504 |         4 |   13 |   34 | 1488 |         4 |
| MoSES_RF    |             10.460256 | 13.291783 |         5 |    5 |   10 |  320 |         5 |
| MoSES_CaDA  |             10.460256 | 14.567701 |         6 |    3 |   11 |  362 |         6 |
| MTPOMO      |             10.460256 | 15.208726 |         7 |    0 |    1 |   24 |         7 |

### OVRPBTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `10.379805`
- Mean-cost leader: `RELD_MTL` (`10.506777`)
- Top1 leader: `RELD_MTL` (`461`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.379805 | 10.506777 |         1 |  461 |  407 |  130 |         1 |
| RELD_MOEL   |             10.379805 | 10.536261 |         2 |  452 |  423 |  123 |         2 |
| MVMOE       |             10.379805 | 11.804059 |         3 |   86 |  161 |  514 |         3 |
| RouteFinder |             10.379805 | 12.625727 |         4 |    1 |    7 |  159 |         4 |
| MoSES_RF    |             10.379805 | 13.194635 |         5 |    0 |    1 |   32 |         5 |
| MoSES_CaDA  |             10.379805 | 14.416342 |         6 |    0 |    1 |   38 |         6 |
| MTPOMO      |             10.379805 | 15.155493 |         7 |    0 |    0 |    4 |         7 |

## OVRPL

- Datasets: `OVRPLtest, OVRPLtrain, OVRPLval`
- Oracle top1 mean cost: `7.393033`
- Mean-cost dataset wins: `{"MoSES_RF": 3}`
- Top1 dataset wins: `{"MoSES_CaDA": 2, "MoSES_RF": 1}`
- Same-method top1+mean wins: `{"MoSES_RF": 1}`

### OVRPLtest

- Instances: `1000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `7.393675`
- Mean-cost leader: `MoSES_RF` (`7.453650`)
- Top1 leader: `MoSES_CaDA` (`331`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.393675 |  7.453650 |         1 |  307 |  317 |  222 |         2 |
| MoSES_CaDA  |              7.393675 |  7.460694 |         2 |  331 |  268 |  224 |         1 |
| RouteFinder |              7.393675 |  7.472738 |         3 |  220 |  241 |  309 |         3 |
| RELD_MTL    |              7.393675 |  7.592861 |         4 |   81 |   81 |  115 |         4 |
| RELD_MOEL   |              7.393675 |  7.634092 |         5 |   51 |   70 |   94 |         5 |
| MTPOMO      |              7.393675 |  8.299554 |         6 |    4 |   16 |   25 |         7 |
| MVMOE       |              7.393675 |  8.902560 |         7 |    6 |    7 |   11 |         6 |

### OVRPLtrain

- Instances: `10000`; methods: `7`; tie instances: `22`
- Oracle top1 mean cost: `7.394315`
- Mean-cost leader: `MoSES_RF` (`7.452648`)
- Top1 leader: `MoSES_CaDA` (`3316`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.394315 |  7.452648 |         1 | 3229 | 3000 | 2212 |         2 |
| MoSES_CaDA  |              7.394315 |  7.458763 |         2 | 3316 | 2764 | 2031 |         1 |
| RouteFinder |              7.394315 |  7.474169 |         3 | 2056 | 2391 | 3054 |         3 |
| RELD_MTL    |              7.394315 |  7.588017 |         4 |  742 |  892 | 1292 |         4 |
| RELD_MOEL   |              7.394315 |  7.630860 |         5 |  505 |  745 | 1019 |         5 |
| MTPOMO      |              7.394315 |  8.300787 |         6 |   96 |  125 |  274 |         6 |
| MVMOE       |              7.394315 |  9.016318 |         7 |   56 |   83 |  118 |         7 |

### OVRPLval

- Instances: `1000`; methods: `7`; tie instances: `4`
- Oracle top1 mean cost: `7.379566`
- Mean-cost leader: `MoSES_RF` (`7.437562`)
- Top1 leader: `MoSES_RF` (`334`)
- Same method wins both: `MoSES_RF`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |              7.379566 |  7.437562 |         1 |  334 |  304 |  219 |         1 |
| MoSES_CaDA  |              7.379566 |  7.445771 |         2 |  333 |  258 |  202 |         2 |
| RouteFinder |              7.379566 |  7.460539 |         3 |  194 |  261 |  325 |         3 |
| RELD_MTL    |              7.379566 |  7.583246 |         4 |   68 |   91 |  117 |         4 |
| RELD_MOEL   |              7.379566 |  7.619351 |         5 |   56 |   68 |  105 |         5 |
| MTPOMO      |              7.379566 |  8.226538 |         6 |    8 |   13 |   21 |         6 |
| MVMOE       |              7.379566 |  9.045072 |         7 |    7 |    5 |   11 |         7 |

## OVRPLTW

- Datasets: `OVRPLTWtest, OVRPLTWtrain, OVRPLTWval`
- Oracle top1 mean cost: `10.025259`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPLTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `10.017218`
- Mean-cost leader: `RELD_MTL` (`10.103602`)
- Top1 leader: `RELD_MTL` (`492`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.017218 | 10.103602 |         1 |  492 |  407 |  100 |         1 |
| RELD_MOEL   |             10.017218 | 10.127295 |         2 |  429 |  451 |  117 |         2 |
| MVMOE       |             10.017218 | 11.312225 |         3 |   78 |  132 |  519 |         3 |
| RouteFinder |             10.017218 | 11.851977 |         4 |    0 |    2 |   97 |         5 |
| MoSES_RF    |             10.017218 | 12.051635 |         5 |    0 |    1 |   56 |         6 |
| MoSES_CaDA  |             10.017218 | 13.011646 |         6 |    1 |    7 |  111 |         4 |
| MTPOMO      |             10.017218 | 14.455833 |         7 |    0 |    0 |    0 |         7 |

### OVRPLTWtrain

- Instances: `10000`; methods: `7`; tie instances: `6`
- Oracle top1 mean cost: `10.024605`
- Mean-cost leader: `RELD_MTL` (`10.113588`)
- Top1 leader: `RELD_MTL` (`4764`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.024605 | 10.113588 |         1 | 4764 | 4141 | 1085 |         1 |
| RELD_MOEL   |             10.024605 | 10.138449 |         2 | 4458 | 4315 | 1188 |         2 |
| MVMOE       |             10.024605 | 11.375585 |         3 |  766 | 1440 | 5055 |         3 |
| RouteFinder |             10.024605 | 11.886656 |         4 |    0 |   31 |  970 |         5 |
| MoSES_RF    |             10.024605 | 12.093293 |         5 |    0 |   15 |  516 |         6 |
| MoSES_CaDA  |             10.024605 | 13.085449 |         6 |   12 |   58 | 1185 |         4 |
| MTPOMO      |             10.024605 | 14.509354 |         7 |    0 |    0 |    1 |         7 |

### OVRPLTWval

- Instances: `1000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `10.039835`
- Mean-cost leader: `RELD_MTL` (`10.127205`)
- Top1 leader: `RELD_MTL` (`494`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.039835 | 10.127205 |         1 |  494 |  399 |  107 |         1 |
| RELD_MOEL   |             10.039835 | 10.154016 |         2 |  433 |  442 |  120 |         2 |
| MVMOE       |             10.039835 | 11.359611 |         3 |   72 |  149 |  505 |         3 |
| RouteFinder |             10.039835 | 11.888721 |         4 |    0 |    1 |   93 |         5 |
| MoSES_RF    |             10.039835 | 12.077141 |         5 |    0 |    3 |   53 |         6 |
| MoSES_CaDA  |             10.039835 | 13.032188 |         6 |    1 |    6 |  121 |         4 |
| MTPOMO      |             10.039835 | 14.467725 |         7 |    0 |    0 |    1 |         7 |

## OVRPTW

- Datasets: `OVRPTWtest, OVRPTWtrain, OVRPTWval`
- Oracle top1 mean cost: `10.011256`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### OVRPTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `10.033608`
- Mean-cost leader: `RELD_MTL` (`10.123299`)
- Top1 leader: `RELD_MTL` (`479`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.033608 | 10.123299 |         1 |  479 |  412 |  109 |         1 |
| RELD_MOEL   |             10.033608 | 10.143173 |         2 |  429 |  437 |  134 |         2 |
| MVMOE       |             10.033608 | 11.403935 |         3 |   91 |  141 |  500 |         3 |
| RouteFinder |             10.033608 | 11.925309 |         4 |    1 |    1 |   92 |         4 |
| MoSES_RF    |             10.033608 | 12.126630 |         5 |    0 |    2 |   69 |         5 |
| MoSES_CaDA  |             10.033608 | 13.235798 |         6 |    0 |    7 |   96 |         6 |
| MTPOMO      |             10.033608 | 14.507176 |         7 |    0 |    0 |    0 |         7 |

### OVRPTWtrain

- Instances: `10000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `10.014294`
- Mean-cost leader: `RELD_MTL` (`10.103682`)
- Top1 leader: `RELD_MTL` (`4691`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             10.014294 | 10.103682 |         1 | 4691 | 4241 | 1064 |         1 |
| RELD_MOEL   |             10.014294 | 10.120903 |         2 | 4550 | 4260 | 1169 |         2 |
| MVMOE       |             10.014294 | 11.301199 |         3 |  750 | 1424 | 5255 |         3 |
| RouteFinder |             10.014294 | 11.901452 |         4 |    3 |   12 | 1018 |         5 |
| MoSES_RF    |             10.014294 | 12.109828 |         5 |    2 |   19 |  612 |         6 |
| MoSES_CaDA  |             10.014294 | 13.206879 |         6 |    4 |   43 |  879 |         4 |
| MTPOMO      |             10.014294 | 14.459164 |         7 |    0 |    1 |    3 |         7 |

### OVRPTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `9.958517`
- Mean-cost leader: `RELD_MTL` (`10.048435`)
- Top1 leader: `RELD_MTL` (`476`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.958517 | 10.048435 |         1 |  476 |  422 |  102 |         1 |
| RELD_MOEL   |              9.958517 | 10.066680 |         2 |  457 |  427 |  113 |         2 |
| MVMOE       |              9.958517 | 11.380933 |         3 |   66 |  141 |  512 |         3 |
| RouteFinder |              9.958517 | 11.827152 |         4 |    0 |    1 |  115 |         5 |
| MoSES_RF    |              9.958517 | 12.039227 |         5 |    0 |    1 |   62 |         6 |
| MoSES_CaDA  |              9.958517 | 13.133154 |         6 |    1 |    8 |   96 |         4 |
| MTPOMO      |              9.958517 | 14.350275 |         7 |    0 |    0 |    0 |         7 |

## TSP

- Datasets: `TSPLIB, TSPtest, TSPtrain, TSPval`
- Oracle top1 mean cost: `7.955244`
- Mean-cost dataset wins: `{"DIFUSCO500": 2, "T2T500": 2}`
- Top1 dataset wins: `{"BQ": 4}`
- Same-method top1+mean wins: `{}`

### TSPLIB

- Instances: `49`; methods: `8`; tie instances: `5`
- Oracle top1 mean cost: `8.142167`
- Mean-cost leader: `T2T500` (`8.233445`)
- Top1 leader: `BQ` (`22`)
- Same method wins both: `none`

| method     | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ---------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| T2T500     |              8.142167 |  8.233445 |         1 |    5 |    4 |   11 |         3 |
| T2T        |              8.142167 |  8.240435 |         2 |    5 |    6 |    3 |         4 |
| DIFUSCO500 |              8.142167 |  8.247271 |         3 |    3 |    8 |    7 |         6 |
| LEHD       |              8.142167 |  8.259872 |         4 |    4 |   17 |    3 |         5 |
| DIFUSCO    |              8.142167 |  8.264511 |         5 |    3 |    5 |    7 |         7 |
| BQ         |              8.142167 |  8.273128 |         6 |   22 |    5 |    6 |         1 |
| ELG        |              8.142167 |  8.384216 |         7 |    7 |    4 |   11 |         2 |
| OMNI       |              8.142167 |  8.578038 |         8 |    0 |    0 |    1 |         8 |

### TSPtest

- Instances: `1000`; methods: `8`; tie instances: `38`
- Oracle top1 mean cost: `7.994650`
- Mean-cost leader: `DIFUSCO500` (`8.082872`)
- Top1 leader: `BQ` (`280`)
- Same method wins both: `none`

| method     | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ---------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| DIFUSCO500 |              7.994650 |  8.082872 |         1 |  132 |  155 |  199 |         4 |
| T2T500     |              7.994650 |  8.086042 |         2 |  156 |  144 |  173 |         3 |
| T2T        |              7.994650 |  8.094400 |         3 |  114 |  152 |  186 |         5 |
| BQ         |              7.994650 |  8.115787 |         4 |  280 |  198 |   96 |         1 |
| DIFUSCO    |              7.994650 |  8.126897 |         5 |   52 |   83 |  122 |         7 |
| LEHD       |              7.994650 |  8.157513 |         6 |  185 |  180 |  111 |         2 |
| ELG        |              7.994650 |  8.215440 |         7 |   78 |   74 |   84 |         6 |
| OMNI       |              7.994650 |  8.318756 |         8 |    3 |   14 |   29 |         8 |

### TSPtrain

- Instances: `10000`; methods: `8`; tie instances: `417`
- Oracle top1 mean cost: `7.958964`
- Mean-cost leader: `DIFUSCO500` (`8.048286`)
- Top1 leader: `BQ` (`3077`)
- Same method wins both: `none`

| method     | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ---------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| DIFUSCO500 |              7.958964 |  8.048286 |         1 | 1283 | 1645 | 1887 |         4 |
| T2T500     |              7.958964 |  8.049768 |         2 | 1637 | 1587 | 1633 |         2 |
| T2T        |              7.958964 |  8.059866 |         3 | 1090 | 1363 | 1787 |         5 |
| BQ         |              7.958964 |  8.083008 |         4 | 3077 | 1738 | 1021 |         1 |
| DIFUSCO    |              7.958964 |  8.087364 |         5 |  521 |  908 | 1405 |         7 |
| LEHD       |              7.958964 |  8.128164 |         6 | 1623 | 1854 | 1165 |         3 |
| ELG        |              7.958964 |  8.184486 |         7 |  724 |  763 |  838 |         6 |
| OMNI       |              7.958964 |  8.278670 |         8 |   45 |  142 |  264 |         8 |

### TSPval

- Instances: `1000`; methods: `8`; tie instances: `45`
- Oracle top1 mean cost: `7.869479`
- Mean-cost leader: `T2T500` (`7.957803`)
- Top1 leader: `BQ` (`297`)
- Same method wins both: `none`

| method     | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ---------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| T2T500     |              7.869479 |  7.957803 |         1 |  167 |  147 |  143 |         3 |
| DIFUSCO500 |              7.869479 |  7.958014 |         2 |  116 |  157 |  188 |         4 |
| T2T        |              7.869479 |  7.964552 |         3 |  109 |  149 |  188 |         5 |
| DIFUSCO    |              7.869479 |  7.993877 |         4 |   48 |   84 |  131 |         7 |
| BQ         |              7.869479 |  7.995270 |         5 |  297 |  172 |   95 |         1 |
| LEHD       |              7.869479 |  8.036497 |         6 |  172 |  162 |  107 |         2 |
| ELG        |              7.869479 |  8.089358 |         7 |   66 |   89 |   86 |         6 |
| OMNI       |              7.869479 |  8.111762 |         8 |   25 |   40 |   62 |         8 |

## VRPB

- Datasets: `VRPBtest, VRPBtrain, VRPBval`
- Oracle top1 mean cost: `9.241954`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### VRPBtest

- Instances: `1000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `9.217786`
- Mean-cost leader: `RELD_MTL` (`9.295063`)
- Top1 leader: `RELD_MTL` (`483`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.217786 |  9.295063 |         1 |  483 |  394 |  110 |         1 |
| RELD_MOEL   |              9.217786 |  9.330275 |         2 |  435 |  405 |  142 |         2 |
| MTPOMO      |              9.217786 | 10.701428 |         3 |   22 |   50 |  275 |         4 |
| MoSES_RF    |              9.217786 | 11.289215 |         4 |    1 |    1 |   48 |         5 |
| MoSES_CaDA  |              9.217786 | 11.295189 |         5 |    0 |    2 |   41 |         6 |
| RouteFinder |              9.217786 | 11.314893 |         6 |    0 |    0 |   21 |         7 |
| MVMOE       |              9.217786 | 11.584618 |         7 |   59 |  148 |  363 |         3 |

### VRPBtrain

- Instances: `10000`; methods: `7`; tie instances: `13`
- Oracle top1 mean cost: `9.241741`
- Mean-cost leader: `RELD_MTL` (`9.316559`)
- Top1 leader: `RELD_MTL` (`5131`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.241741 |  9.316559 |         1 | 5131 | 3713 | 1055 |         1 |
| RELD_MOEL   |              9.241741 |  9.367410 |         2 | 4100 | 4250 | 1441 |         2 |
| MTPOMO      |              9.241741 | 10.715950 |         3 |  187 |  572 | 2657 |         4 |
| MoSES_RF    |              9.241741 | 11.325045 |         4 |    5 |   11 |  393 |         5 |
| MoSES_CaDA  |              9.241741 | 11.325868 |         5 |    5 |   26 |  442 |         6 |
| RouteFinder |              9.241741 | 11.344009 |         6 |    1 |    6 |  240 |         7 |
| MVMOE       |              9.241741 | 11.603011 |         7 |  571 | 1422 | 3772 |         3 |

### VRPBval

- Instances: `1000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `9.268251`
- Mean-cost leader: `RELD_MTL` (`9.340269`)
- Top1 leader: `RELD_MTL` (`504`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.268251 |  9.340269 |         1 |  504 |  390 |  103 |         1 |
| RELD_MOEL   |              9.268251 |  9.396019 |         2 |  413 |  423 |  148 |         2 |
| MTPOMO      |              9.268251 | 10.822326 |         3 |   17 |   55 |  262 |         4 |
| MoSES_RF    |              9.268251 | 11.362704 |         4 |    0 |    0 |   34 |         5 |
| MoSES_CaDA  |              9.268251 | 11.368544 |         5 |    0 |    0 |   41 |         6 |
| RouteFinder |              9.268251 | 11.382166 |         6 |    0 |    2 |   32 |         7 |
| MVMOE       |              9.268251 | 11.803180 |         7 |   66 |  130 |  380 |         3 |

## VRPBL

- Datasets: `VRPBLtest, VRPBLtrain, VRPBLval`
- Oracle top1 mean cost: `9.241595`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### VRPBLtest

- Instances: `1000`; methods: `7`; tie instances: `2`
- Oracle top1 mean cost: `9.177293`
- Mean-cost leader: `RELD_MTL` (`9.249180`)
- Top1 leader: `RELD_MTL` (`537`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.177293 |  9.249180 |         1 |  537 |  356 |   99 |         1 |
| RELD_MOEL   |              9.177293 |  9.309500 |         2 |  402 |  438 |  135 |         2 |
| MTPOMO      |              9.177293 | 10.599956 |         3 |   13 |   63 |  266 |         4 |
| MoSES_RF    |              9.177293 | 11.212035 |         4 |    0 |    2 |   44 |         5 |
| MoSES_CaDA  |              9.177293 | 11.212075 |         5 |    0 |    1 |   51 |         6 |
| MVMOE       |              9.177293 | 11.212746 |         6 |   48 |  140 |  375 |         3 |
| RouteFinder |              9.177293 | 11.230750 |         7 |    0 |    0 |   30 |         7 |

### VRPBLtrain

- Instances: `10000`; methods: `7`; tie instances: `14`
- Oracle top1 mean cost: `9.243726`
- Mean-cost leader: `RELD_MTL` (`9.320910`)
- Top1 leader: `RELD_MTL` (`5010`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.243726 |  9.320910 |         1 | 5010 | 3841 | 1021 |         1 |
| RELD_MOEL   |              9.243726 |  9.375451 |         2 | 4116 | 4140 | 1489 |         2 |
| MTPOMO      |              9.243726 | 10.572554 |         3 |  206 |  553 | 2708 |         4 |
| MVMOE       |              9.243726 | 11.134370 |         4 |  657 | 1415 | 3649 |         3 |
| MoSES_CaDA  |              9.243726 | 11.295607 |         5 |    3 |   25 |  430 |         6 |
| MoSES_RF    |              9.243726 | 11.298111 |         6 |    5 |   15 |  431 |         5 |
| RouteFinder |              9.243726 | 11.315840 |         7 |    3 |   11 |  272 |         7 |

### VRPBLval

- Instances: `1000`; methods: `7`; tie instances: `2`
- Oracle top1 mean cost: `9.284586`
- Mean-cost leader: `RELD_MTL` (`9.359268`)
- Top1 leader: `RELD_MTL` (`528`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |              9.284586 |  9.359268 |         1 |  528 |  355 |  106 |         1 |
| RELD_MOEL   |              9.284586 |  9.424117 |         2 |  386 |  436 |  162 |         2 |
| MTPOMO      |              9.284586 | 10.705363 |         3 |   21 |   59 |  256 |         4 |
| MVMOE       |              9.284586 | 11.321271 |         4 |   65 |  145 |  354 |         3 |
| MoSES_CaDA  |              9.284586 | 11.342973 |         5 |    0 |    2 |   48 |         5 |
| MoSES_RF    |              9.284586 | 11.346653 |         6 |    0 |    2 |   44 |         6 |
| RouteFinder |              9.284586 | 11.366481 |         7 |    0 |    1 |   30 |         7 |

## VRPBLTW

- Datasets: `VRPBLTWtest, VRPBLTWtrain, VRPBLTWval`
- Oracle top1 mean cost: `18.751829`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### VRPBLTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `18.788554`
- Mean-cost leader: `RELD_MTL` (`19.676957`)
- Top1 leader: `RELD_MTL` (`352`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.788554 | 19.676957 |         1 |  352 |  289 |  174 |         1 |
| RELD_MOEL   |             18.788554 | 19.799754 |         2 |  295 |  308 |  180 |         2 |
| RouteFinder |             18.788554 | 20.622837 |         3 |  133 |   88 |  111 |         3 |
| MoSES_RF    |             18.788554 | 20.814425 |         4 |   80 |  110 |   90 |         5 |
| MVMOE       |             18.788554 | 21.506060 |         5 |   92 |  143 |  324 |         4 |
| MoSES_CaDA  |             18.788554 | 22.476457 |         6 |   47 |   62 |  117 |         6 |
| MTPOMO      |             18.788554 | 24.793583 |         7 |    1 |    0 |    4 |         7 |

### VRPBLTWtrain

- Instances: `10000`; methods: `7`; tie instances: `2`
- Oracle top1 mean cost: `18.758186`
- Mean-cost leader: `RELD_MTL` (`19.666086`)
- Top1 leader: `RELD_MTL` (`3574`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.758186 | 19.666086 |         1 | 3574 | 2988 | 1661 |         1 |
| RELD_MOEL   |             18.758186 | 19.779709 |         2 | 3020 | 3084 | 1737 |         2 |
| RouteFinder |             18.758186 | 20.592864 |         3 | 1403 |  798 | 1106 |         3 |
| MoSES_RF    |             18.758186 | 20.873503 |         4 |  625 | 1194 |  925 |         5 |
| MVMOE       |             18.758186 | 21.572153 |         5 |  902 | 1413 | 3398 |         4 |
| MoSES_CaDA  |             18.758186 | 22.575468 |         6 |  473 |  516 | 1126 |         6 |
| MTPOMO      |             18.758186 | 24.899868 |         7 |    3 |    7 |   47 |         7 |

### VRPBLTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `18.651532`
- Mean-cost leader: `RELD_MTL` (`19.519778`)
- Top1 leader: `RELD_MTL` (`347`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.651532 | 19.519778 |         1 |  347 |  314 |  161 |         1 |
| RELD_MOEL   |             18.651532 | 19.667521 |         2 |  315 |  302 |  181 |         2 |
| RouteFinder |             18.651532 | 20.494722 |         3 |  136 |   70 |  114 |         3 |
| MoSES_RF    |             18.651532 | 20.820598 |         4 |   54 |  112 |   92 |         6 |
| MVMOE       |             18.651532 | 21.470786 |         5 |   92 |  141 |  336 |         4 |
| MoSES_CaDA  |             18.651532 | 22.464162 |         6 |   56 |   60 |  110 |         5 |
| MTPOMO      |             18.651532 | 24.715246 |         7 |    0 |    1 |    6 |         7 |

## VRPBTW

- Datasets: `VRPBTWtest, VRPBTWtrain, VRPBTWval`
- Oracle top1 mean cost: `18.757172`
- Mean-cost dataset wins: `{"RELD_MTL": 3}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 3}`

### VRPBTWtest

- Instances: `1000`; methods: `7`; tie instances: `1`
- Oracle top1 mean cost: `18.690529`
- Mean-cost leader: `RELD_MTL` (`19.526309`)
- Top1 leader: `RELD_MTL` (`357`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.690529 | 19.526309 |         1 |  357 |  279 |  191 |         1 |
| RELD_MOEL   |             18.690529 | 19.630883 |         2 |  297 |  338 |  165 |         2 |
| MoSES_RF    |             18.690529 | 20.557189 |         3 |   97 |  120 |   93 |         5 |
| RouteFinder |             18.690529 | 20.595841 |         4 |  106 |   89 |   99 |         3 |
| MVMOE       |             18.690529 | 21.310201 |         5 |   99 |  138 |  328 |         4 |
| MoSES_CaDA  |             18.690529 | 22.589613 |         6 |   44 |   36 |  111 |         6 |
| MTPOMO      |             18.690529 | 24.685686 |         7 |    0 |    0 |   13 |         7 |

### VRPBTWtrain

- Instances: `10000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `18.765115`
- Mean-cost leader: `RELD_MTL` (`19.692419`)
- Top1 leader: `RELD_MTL` (`3625`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.765115 | 19.692419 |         1 | 3625 | 2957 | 1699 |         1 |
| RELD_MOEL   |             18.765115 | 19.809165 |         2 | 3030 | 3181 | 1691 |         2 |
| MoSES_RF    |             18.765115 | 20.674577 |         3 |  835 | 1158 | 1032 |         5 |
| RouteFinder |             18.765115 | 20.688680 |         4 | 1193 |  842 |  989 |         3 |
| MVMOE       |             18.765115 | 21.489391 |         5 |  896 | 1391 | 3447 |         4 |
| MoSES_CaDA  |             18.765115 | 22.769545 |         6 |  420 |  470 | 1105 |         6 |
| MTPOMO      |             18.765115 | 24.878005 |         7 |    1 |    1 |   37 |         7 |

### VRPBTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `18.744386`
- Mean-cost leader: `RELD_MTL` (`19.669740`)
- Top1 leader: `RELD_MTL` (`372`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             18.744386 | 19.669740 |         1 |  372 |  297 |  143 |         1 |
| RELD_MOEL   |             18.744386 | 19.775302 |         2 |  301 |  324 |  152 |         2 |
| MoSES_RF    |             18.744386 | 20.627514 |         3 |   80 |  101 |  130 |         4 |
| RouteFinder |             18.744386 | 20.629229 |         4 |  123 |   89 |  110 |         3 |
| MVMOE       |             18.744386 | 21.619582 |         5 |   77 |  129 |  345 |         5 |
| MoSES_CaDA  |             18.744386 | 22.648913 |         6 |   47 |   59 |  115 |         6 |
| MTPOMO      |             18.744386 | 24.888895 |         7 |    0 |    1 |    5 |         7 |

## VRPL

- Datasets: `VRPLtest, VRPLtrain, VRPLval`
- Oracle top1 mean cost: `12.331137`
- Mean-cost dataset wins: `{"MoSES_CaDA": 2, "MoSES_RF": 1}`
- Top1 dataset wins: `{"MoSES_CaDA": 1, "MoSES_RF": 2}`
- Same-method top1+mean wins: `{"MoSES_CaDA": 1, "MoSES_RF": 1}`

### VRPLtest

- Instances: `1000`; methods: `7`; tie instances: `5`
- Oracle top1 mean cost: `12.295043`
- Mean-cost leader: `MoSES_CaDA` (`12.360694`)
- Top1 leader: `MoSES_RF` (`256`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_CaDA  |             12.295043 | 12.360694 |         1 |  252 |  270 |  194 |         2 |
| MoSES_RF    |             12.295043 | 12.361744 |         2 |  256 |  259 |  219 |         1 |
| RouteFinder |             12.295043 | 12.384464 |         3 |  159 |  179 |  265 |         3 |
| RELD_MTL    |             12.295043 | 12.456806 |         4 |  138 |  121 |  134 |         4 |
| RELD_MOEL   |             12.295043 | 12.489746 |         5 |  136 |   93 |  110 |         5 |
| MTPOMO      |             12.295043 | 13.343418 |         6 |   33 |   48 |   49 |         6 |
| MVMOE       |             12.295043 | 14.163921 |         7 |   26 |   30 |   29 |         7 |

### VRPLtrain

- Instances: `10000`; methods: `7`; tie instances: `30`
- Oracle top1 mean cost: `12.333457`
- Mean-cost leader: `MoSES_RF` (`12.395734`)
- Top1 leader: `MoSES_RF` (`2747`)
- Same method wins both: `MoSES_RF`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |             12.333457 | 12.395734 |         1 | 2747 | 2558 | 1996 |         1 |
| MoSES_CaDA  |             12.333457 | 12.397482 |         2 | 2658 | 2502 | 2159 |         2 |
| RouteFinder |             12.333457 | 12.419502 |         3 | 1465 | 1911 | 2531 |         3 |
| RELD_MTL    |             12.333457 | 12.487517 |         4 | 1459 | 1234 | 1315 |         4 |
| RELD_MOEL   |             12.333457 | 12.530213 |         5 | 1118 | 1085 | 1072 |         5 |
| MTPOMO      |             12.333457 | 13.385177 |         6 |  309 |  401 |  555 |         6 |
| MVMOE       |             12.333457 | 14.169946 |         7 |  244 |  309 |  372 |         7 |

### VRPLval

- Instances: `1000`; methods: `7`; tie instances: `4`
- Oracle top1 mean cost: `12.344029`
- Mean-cost leader: `MoSES_CaDA` (`12.407213`)
- Top1 leader: `MoSES_CaDA` (`283`)
- Same method wins both: `MoSES_CaDA`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_CaDA  |             12.344029 | 12.407213 |         1 |  283 |  268 |  185 |         1 |
| MoSES_RF    |             12.344029 | 12.408451 |         2 |  276 |  253 |  202 |         2 |
| RouteFinder |             12.344029 | 12.430872 |         3 |  169 |  192 |  254 |         3 |
| RELD_MTL    |             12.344029 | 12.508747 |         4 |  117 |  113 |  140 |         4 |
| RELD_MOEL   |             12.344029 | 12.546506 |         5 |  105 |   98 |  131 |         5 |
| MTPOMO      |             12.344029 | 13.454017 |         6 |   28 |   44 |   60 |         6 |
| MVMOE       |             12.344029 | 14.360135 |         7 |   22 |   32 |   28 |         7 |

## VRPLTW

- Datasets: `VRPLTWtest, VRPLTWtrain, VRPLTWval`
- Oracle top1 mean cost: `17.534410`
- Mean-cost dataset wins: `{"RELD_MTL": 1, "RouteFinder": 2}`
- Top1 dataset wins: `{"RELD_MTL": 3}`
- Same-method top1+mean wins: `{"RELD_MTL": 1}`

### VRPLTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `17.372466`
- Mean-cost leader: `RELD_MTL` (`18.426862`)
- Top1 leader: `RELD_MTL` (`271`)
- Same method wins both: `RELD_MTL`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RELD_MTL    |             17.372466 | 18.426862 |         1 |  271 |  245 |  195 |         1 |
| RELD_MOEL   |             17.372466 | 18.495057 |         2 |  219 |  275 |  203 |         2 |
| RouteFinder |             17.372466 | 18.518156 |         3 |  155 |  128 |   99 |         4 |
| MoSES_RF    |             17.372466 | 18.565982 |         4 |  123 |  169 |  113 |         5 |
| MoSES_CaDA  |             17.372466 | 19.857557 |         5 |  158 |   91 |  134 |         3 |
| MVMOE       |             17.372466 | 20.012325 |         6 |   74 |   92 |  255 |         6 |
| MTPOMO      |             17.372466 | 23.163571 |         7 |    0 |    0 |    1 |         7 |

### VRPLTWtrain

- Instances: `10000`; methods: `7`; tie instances: `5`
- Oracle top1 mean cost: `17.562103`
- Mean-cost leader: `RouteFinder` (`18.671991`)
- Top1 leader: `RELD_MTL` (`2582`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RouteFinder |             17.562103 | 18.671991 |         1 | 1504 | 1424 | 1209 |         4 |
| MoSES_RF    |             17.562103 | 18.734001 |         2 | 1333 | 1591 | 1173 |         5 |
| RELD_MTL    |             17.562103 | 18.743242 |         3 | 2582 | 2475 | 1896 |         1 |
| RELD_MOEL   |             17.562103 | 18.818920 |         4 | 2279 | 2501 | 1836 |         2 |
| MoSES_CaDA  |             17.562103 | 20.081265 |         5 | 1609 |  939 | 1428 |         3 |
| MVMOE       |             17.562103 | 20.451790 |         6 |  692 | 1069 | 2455 |         6 |
| MTPOMO      |             17.562103 | 23.493168 |         7 |    1 |    1 |    3 |         7 |

### VRPLTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `17.419424`
- Mean-cost leader: `RouteFinder` (`18.509149`)
- Top1 leader: `RELD_MTL` (`257`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| RouteFinder |             17.419424 | 18.509149 |         1 |  149 |  154 |  131 |         4 |
| RELD_MTL    |             17.419424 | 18.576974 |         2 |  257 |  254 |  177 |         1 |
| MoSES_RF    |             17.419424 | 18.612038 |         3 |  118 |  170 |  110 |         5 |
| RELD_MOEL   |             17.419424 | 18.652349 |         4 |  240 |  227 |  191 |         2 |
| MoSES_CaDA  |             17.419424 | 19.832355 |         5 |  173 |   91 |  145 |         3 |
| MVMOE       |             17.419424 | 20.274024 |         6 |   63 |  104 |  246 |         6 |
| MTPOMO      |             17.419424 | 23.279750 |         7 |    0 |    0 |    0 |         7 |

## VRPTW

- Datasets: `VRPTWtest, VRPTWtrain, VRPTWval`
- Oracle top1 mean cost: `17.573785`
- Mean-cost dataset wins: `{"MoSES_RF": 3}`
- Top1 dataset wins: `{"RELD_MOEL": 1, "RELD_MTL": 2}`
- Same-method top1+mean wins: `{}`

### VRPTWtest

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `17.423958`
- Mean-cost leader: `MoSES_RF` (`18.435676`)
- Top1 leader: `RELD_MOEL` (`241`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |             17.423958 | 18.435676 |         1 |  188 |  141 |  104 |         3 |
| RELD_MTL    |             17.423958 | 18.574120 |         2 |  238 |  267 |  190 |         2 |
| RELD_MOEL   |             17.423958 | 18.627283 |         3 |  241 |  252 |  182 |         1 |
| RouteFinder |             17.423958 | 18.632179 |         4 |  113 |  147 |  129 |         5 |
| MoSES_CaDA  |             17.423958 | 19.920975 |         5 |  154 |   93 |  148 |         4 |
| MVMOE       |             17.423958 | 20.243225 |         6 |   66 |  100 |  247 |         6 |
| MTPOMO      |             17.423958 | 23.240192 |         7 |    0 |    0 |    0 |         7 |

### VRPTWtrain

- Instances: `10000`; methods: `7`; tie instances: `3`
- Oracle top1 mean cost: `17.580121`
- Mean-cost leader: `MoSES_RF` (`18.594872`)
- Top1 leader: `RELD_MTL` (`2549`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |             17.580121 | 18.594872 |         1 | 1681 | 1618 | 1130 |         3 |
| RouteFinder |             17.580121 | 18.775406 |         2 | 1325 | 1477 | 1130 |         5 |
| RELD_MTL    |             17.580121 | 18.781431 |         3 | 2549 | 2495 | 1897 |         1 |
| RELD_MOEL   |             17.580121 | 18.852350 |         4 | 2244 | 2509 | 1960 |         2 |
| MoSES_CaDA  |             17.580121 | 20.139286 |         5 | 1572 |  887 | 1462 |         4 |
| MVMOE       |             17.580121 | 20.504243 |         6 |  629 | 1014 | 2418 |         6 |
| MTPOMO      |             17.580121 | 23.544962 |         7 |    0 |    0 |    3 |         7 |

### VRPTWval

- Instances: `1000`; methods: `7`; tie instances: `0`
- Oracle top1 mean cost: `17.660252`
- Mean-cost leader: `MoSES_RF` (`18.657081`)
- Top1 leader: `RELD_MTL` (`236`)
- Same method wins both: `none`

| method      | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
| ----------- | --------------------: | --------: | --------: | ---: | ---: | ---: | --------: |
| MoSES_RF    |             17.660252 | 18.657081 |         1 |  169 |  153 |  118 |         3 |
| RouteFinder |             17.660252 | 18.844298 |         2 |  150 |  150 |   82 |         4 |
| RELD_MTL    |             17.660252 | 18.850502 |         3 |  236 |  266 |  195 |         1 |
| RELD_MOEL   |             17.660252 | 18.895425 |         4 |  231 |  239 |  205 |         2 |
| MoSES_CaDA  |             17.660252 | 20.205531 |         5 |  143 |   90 |  162 |         5 |
| MVMOE       |             17.660252 | 20.522358 |         6 |   71 |  102 |  238 |         6 |
| MTPOMO      |             17.660252 | 23.643270 |         7 |    0 |    0 |    0 |         7 |