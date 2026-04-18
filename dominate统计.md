# Method Dominance Statistics

- Data root: `/public/home/zhoucl/shiys/0selection/data`
- Datasets analyzed: `56`

## Rule

- `top1/top2/top3` are exact ranks within each instance, sorted by lower cost first.
- Ties are broken deterministically by method name for ranking counts.
- `tie_instances` reports how many instances had at least one equal-cost tie across methods.
- `dominant_method_same_top1_and_mean` means the same method wins both `top1_count` and `mean_cost` on that dataset.
- Report display name `RELD` is kept only for `CVRP`, where it corresponds to `RELD_CVRP`.
- For `MVRP`, `RELD_MTL` and `RELD_MOEL` are reported as two separate methods in this file.
- `oracle_top1_mean_cost` means the mean cost if each instance could always choose the best displayed method.

## ATSP

- Datasets: `ATSPtest, ATSPtrain, ATSPval`
- Oracle top1 mean cost: `1.685046`
- Mean-cost dataset wins: `{'MATPOENET': 3}`
- Top1 dataset wins: `{'MATPOENET': 3}`
- Same-method top1+mean wins: `{'MATPOENET': 3}`

### ATSPtest

- Instances: `1000`; methods: `3`; tie instances: `0`
- Oracle top1 mean cost: `1.675761`
- Mean-cost leader: `MATPOENET` (`1.715922`)
- Top1 leader: `MATPOENET` (`701`)
- Same method wins both: `MATPOENET`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| MATPOENET | 1.675761 | 1.715922 | 1 | 701 | 295 | 4 | 1 |
| GLOP | 1.675761 | 1.920294 | 2 | 24 | 575 | 401 | 3 |
| MATNET | 1.675761 | 2.313879 | 3 | 275 | 130 | 595 | 2 |

### ATSPtrain

- Instances: `10000`; methods: `3`; tie instances: `1`
- Oracle top1 mean cost: `1.686235`
- Mean-cost leader: `MATPOENET` (`1.726253`)
- Top1 leader: `MATPOENET` (`7005`)
- Same method wins both: `MATPOENET`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| MATPOENET | 1.686235 | 1.726253 | 1 | 7005 | 2972 | 23 | 1 |
| GLOP | 1.686235 | 1.931549 | 2 | 169 | 5760 | 4071 | 3 |
| MATNET | 1.686235 | 2.310640 | 3 | 2826 | 1268 | 5906 | 2 |

### ATSPval

- Instances: `1000`; methods: `3`; tie instances: `1`
- Oracle top1 mean cost: `1.682441`
- Mean-cost leader: `MATPOENET` (`1.722144`)
- Top1 leader: `MATPOENET` (`704`)
- Same method wins both: `MATPOENET`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| MATPOENET | 1.682441 | 1.722144 | 1 | 704 | 293 | 3 | 1 |
| GLOP | 1.682441 | 1.921757 | 2 | 21 | 584 | 395 | 3 |
| MATNET | 1.682441 | 2.323148 | 3 | 275 | 123 | 602 | 2 |

## CVRP

- Datasets: `CVRPLIB, CVRPtest, CVRPtrain, CVRPval`
- Oracle top1 mean cost: `18.366862`
- Mean-cost dataset wins: `{'RELD': 4}`
- Top1 dataset wins: `{'RELD': 4}`
- Same-method top1+mean wins: `{'RELD': 4}`

### CVRPLIB

- Instances: `100`; methods: `9`; tie instances: `0`
- Oracle top1 mean cost: `66.949631`
- Mean-cost leader: `RELD` (`67.245364`)
- Top1 leader: `RELD` (`50`)
- Same method wins both: `RELD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD | 66.949631 | 67.245364 | 1 | 50 | 34 | 16 | 1 |
| ICAM | 66.949631 | 67.678618 | 2 | 37 | 47 | 11 | 2 |
| OMNI | 66.949631 | 69.520695 | 3 | 4 | 11 | 37 | 4 |
| INVIT | 66.949631 | 73.765692 | 4 | 0 | 0 | 2 | 5 |
| LEHD | 66.949631 | 75.928254 | 5 | 9 | 8 | 33 | 3 |
| DACT | 66.949631 | 81.096688 | 6 | 0 | 0 | 1 | 6 |
| GLOP | 66.949631 | 85.326888 | 7 | 0 | 0 | 0 | 7 |
| UDC | 66.949631 | 93.577304 | 8 | 0 | 0 | 0 | 8 |
| ELG | 66.949631 | 161.885690 | 9 | 0 | 0 | 0 | 9 |

### CVRPtest

- Instances: `1000`; methods: `9`; tie instances: `0`
- Oracle top1 mean cost: `18.192689`
- Mean-cost leader: `RELD` (`18.332332`)
- Top1 leader: `RELD` (`413`)
- Same method wins both: `RELD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD | 18.192689 | 18.332332 | 1 | 413 | 335 | 184 | 1 |
| ICAM | 18.192689 | 18.547143 | 2 | 154 | 243 | 373 | 4 |
| LEHD | 18.192689 | 18.565324 | 3 | 261 | 217 | 197 | 2 |
| OMNI | 18.192689 | 18.630295 | 4 | 171 | 202 | 226 | 3 |
| INVIT | 18.192689 | 20.820360 | 5 | 0 | 2 | 10 | 6 |
| GLOP | 18.192689 | 22.071736 | 6 | 0 | 0 | 1 | 7 |
| DACT | 18.192689 | 22.137536 | 7 | 0 | 0 | 0 | 8 |
| UDC | 18.192689 | 31.329928 | 8 | 0 | 0 | 0 | 9 |
| ELG | 18.192689 | 61.452071 | 9 | 1 | 1 | 9 | 5 |

### CVRPtrain

- Instances: `10000`; methods: `9`; tie instances: `0`
- Oracle top1 mean cost: `17.921057`
- Mean-cost leader: `RELD` (`18.070454`)
- Top1 leader: `RELD` (`3741`)
- Same method wins both: `RELD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD | 17.921057 | 18.070454 | 1 | 3741 | 3574 | 2012 | 1 |
| LEHD | 17.921057 | 18.278036 | 2 | 2936 | 2108 | 1911 | 2 |
| ICAM | 17.921057 | 18.288666 | 3 | 1490 | 2343 | 3603 | 4 |
| OMNI | 17.921057 | 18.349252 | 4 | 1821 | 1926 | 2291 | 3 |
| INVIT | 17.921057 | 20.558923 | 5 | 0 | 16 | 62 | 8 |
| GLOP | 17.921057 | 21.705130 | 6 | 1 | 7 | 20 | 7 |
| DACT | 17.921057 | 21.864802 | 7 | 2 | 4 | 8 | 6 |
| UDC | 17.921057 | 30.907904 | 8 | 0 | 0 | 0 | 9 |
| ELG | 17.921057 | 59.714816 | 9 | 9 | 22 | 93 | 5 |

### CVRPval

- Instances: `1000`; methods: `9`; tie instances: `0`
- Oracle top1 mean cost: `18.140807`
- Mean-cost leader: `RELD` (`18.294333`)
- Top1 leader: `RELD` (`339`)
- Same method wins both: `RELD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD | 18.140807 | 18.294333 | 1 | 339 | 398 | 199 | 1 |
| LEHD | 18.140807 | 18.491459 | 2 | 301 | 204 | 199 | 2 |
| ICAM | 18.140807 | 18.529983 | 3 | 143 | 213 | 361 | 4 |
| OMNI | 18.140807 | 18.545915 | 4 | 217 | 176 | 219 | 3 |
| INVIT | 18.140807 | 20.787117 | 5 | 0 | 4 | 7 | 5 |
| GLOP | 18.140807 | 21.895809 | 6 | 0 | 0 | 6 | 6 |
| DACT | 18.140807 | 22.056863 | 7 | 0 | 0 | 2 | 7 |
| UDC | 18.140807 | 30.937143 | 8 | 0 | 0 | 0 | 8 |
| ELG | 18.140807 | 62.822259 | 9 | 0 | 5 | 7 | 9 |

## OVRP

- Datasets: `OVRPtest, OVRPtrain, OVRPval`
- Oracle top1 mean cost: `7.531299`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.538907`
- Mean-cost leader: `RELD_MTL` (`7.609439`)
- Top1 leader: `RELD_MTL` (`481`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.538907 | 7.609439 | 1 | 481 | 347 | 128 | 1 |
| RELD_MOEL | 7.538907 | 7.645092 | 2 | 382 | 391 | 185 | 2 |
| MTPOMO | 7.538907 | 8.430148 | 3 | 90 | 182 | 455 | 3 |
| MVMOE | 7.538907 | 9.148324 | 4 | 47 | 80 | 232 | 4 |

### OVRPtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.530298`
- Mean-cost leader: `RELD_MTL` (`7.591940`)
- Top1 leader: `RELD_MTL` (`5078`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.530298 | 7.591940 | 1 | 5078 | 3196 | 1406 | 1 |
| RELD_MOEL | 7.530298 | 7.637597 | 2 | 3638 | 4090 | 1738 | 2 |
| MTPOMO | 7.530298 | 8.362867 | 3 | 904 | 1805 | 4615 | 3 |
| MVMOE | 7.530298 | 9.185701 | 4 | 380 | 909 | 2241 | 4 |

### OVRPval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.533698`
- Mean-cost leader: `RELD_MTL` (`7.591531`)
- Top1 leader: `RELD_MTL` (`508`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.533698 | 7.591531 | 1 | 508 | 316 | 156 | 1 |
| RELD_MOEL | 7.533698 | 7.649086 | 2 | 366 | 413 | 173 | 2 |
| MTPOMO | 7.533698 | 8.304029 | 3 | 91 | 184 | 440 | 3 |
| MVMOE | 7.533698 | 9.099308 | 4 | 35 | 87 | 231 | 4 |

## OVRPB

- Datasets: `OVRPBtest, OVRPBtrain, OVRPBval`
- Oracle top1 mean cost: `6.255882`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPBtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.233093`
- Mean-cost leader: `RELD_MTL` (`6.296312`)
- Top1 leader: `RELD_MTL` (`561`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.233093 | 6.296312 | 1 | 561 | 368 | 61 | 1 |
| RELD_MOEL | 6.233093 | 6.338776 | 2 | 405 | 513 | 73 | 2 |
| MTPOMO | 6.233093 | 7.306141 | 3 | 20 | 62 | 508 | 3 |
| MVMOE | 6.233093 | 8.141364 | 4 | 14 | 57 | 358 | 4 |

### OVRPBtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.257640`
- Mean-cost leader: `RELD_MTL` (`6.321635`)
- Top1 leader: `RELD_MTL` (`5397`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.257640 | 6.321635 | 1 | 5397 | 3913 | 613 | 1 |
| RELD_MOEL | 6.257640 | 6.364881 | 2 | 4263 | 4759 | 867 | 2 |
| MTPOMO | 6.257640 | 7.321386 | 3 | 171 | 677 | 4984 | 3 |
| MVMOE | 6.257640 | 8.260250 | 4 | 169 | 651 | 3536 | 4 |

### OVRPBval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.261093`
- Mean-cost leader: `RELD_MTL` (`6.321981`)
- Top1 leader: `RELD_MTL` (`548`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.261093 | 6.321981 | 1 | 548 | 387 | 59 | 1 |
| RELD_MOEL | 6.261093 | 6.367115 | 2 | 423 | 478 | 95 | 2 |
| MTPOMO | 6.261093 | 7.325026 | 3 | 14 | 73 | 500 | 4 |
| MVMOE | 6.261093 | 8.215832 | 4 | 15 | 62 | 346 | 3 |

## OVRPBL

- Datasets: `OVRPBLtest, OVRPBLtrain, OVRPBLval`
- Oracle top1 mean cost: `6.243311`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPBLtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.242088`
- Mean-cost leader: `RELD_MTL` (`6.306825`)
- Top1 leader: `RELD_MTL` (`546`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.242088 | 6.306825 | 1 | 546 | 382 | 66 | 1 |
| RELD_MOEL | 6.242088 | 6.344342 | 2 | 428 | 480 | 87 | 2 |
| MTPOMO | 6.242088 | 7.218438 | 3 | 10 | 74 | 481 | 4 |
| MVMOE | 6.242088 | 7.867767 | 4 | 16 | 64 | 366 | 3 |

### OVRPBLtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.243164`
- Mean-cost leader: `RELD_MTL` (`6.307886`)
- Top1 leader: `RELD_MTL` (`5428`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.243164 | 6.307886 | 1 | 5428 | 3855 | 663 | 1 |
| RELD_MOEL | 6.243164 | 6.349055 | 2 | 4191 | 4841 | 865 | 2 |
| MTPOMO | 6.243164 | 7.210580 | 3 | 210 | 655 | 4919 | 3 |
| MVMOE | 6.243164 | 7.883809 | 4 | 171 | 649 | 3553 | 4 |

### OVRPBLval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `6.246004`
- Mean-cost leader: `RELD_MTL` (`6.310260`)
- Top1 leader: `RELD_MTL` (`556`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 6.246004 | 6.310260 | 1 | 556 | 369 | 63 | 1 |
| RELD_MOEL | 6.246004 | 6.355338 | 2 | 401 | 503 | 85 | 2 |
| MTPOMO | 6.246004 | 7.199121 | 3 | 24 | 68 | 492 | 3 |
| MVMOE | 6.246004 | 7.848755 | 4 | 19 | 60 | 360 | 4 |

## OVRPBLTW

- Datasets: `OVRPBLTWtest, OVRPBLTWtrain, OVRPBLTWval`
- Oracle top1 mean cost: `10.424761`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPBLTWtest

- Instances: `1000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `10.471024`
- Mean-cost leader: `RELD_MTL` (`10.592773`)
- Top1 leader: `RELD_MTL` (`478`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.471024 | 10.592773 | 1 | 478 | 422 | 100 | 1 |
| RELD_MOEL | 10.471024 | 10.627977 | 2 | 447 | 424 | 129 | 2 |
| MVMOE | 10.471024 | 12.040483 | 3 | 75 | 154 | 643 | 3 |
| MTPOMO | 10.471024 | 15.283536 | 4 | 0 | 0 | 128 | 4 |

### OVRPBLTWtrain

- Instances: `10000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `10.421470`
- Mean-cost leader: `RELD_MTL` (`10.547633`)
- Top1 leader: `RELD_MTL` (`4853`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.421470 | 10.547633 | 1 | 4853 | 4021 | 1126 | 1 |
| RELD_MOEL | 10.421470 | 10.580266 | 2 | 4309 | 4482 | 1209 | 2 |
| MVMOE | 10.421470 | 11.890103 | 3 | 838 | 1496 | 6443 | 3 |
| MTPOMO | 10.421470 | 15.166317 | 4 | 0 | 1 | 1222 | 4 |

### OVRPBLTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `10.411414`
- Mean-cost leader: `RELD_MTL` (`10.541174`)
- Top1 leader: `RELD_MTL` (`471`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.411414 | 10.541174 | 1 | 471 | 422 | 107 | 1 |
| RELD_MOEL | 10.411414 | 10.573320 | 2 | 450 | 416 | 134 | 2 |
| MVMOE | 10.411414 | 11.795510 | 3 | 79 | 162 | 641 | 3 |
| MTPOMO | 10.411414 | 15.131536 | 4 | 0 | 0 | 118 | 4 |

## OVRPBTW

- Datasets: `OVRPBTWtest, OVRPBTWtrain, OVRPBTWval`
- Oracle top1 mean cost: `10.455431`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPBTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `10.478424`
- Mean-cost leader: `RELD_MTL` (`10.608986`)
- Top1 leader: `RELD_MTL` (`461`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.478424 | 10.608986 | 1 | 461 | 426 | 113 | 1 |
| RELD_MOEL | 10.478424 | 10.644272 | 2 | 444 | 414 | 142 | 2 |
| MVMOE | 10.478424 | 11.858678 | 3 | 95 | 160 | 634 | 3 |
| MTPOMO | 10.478424 | 15.253699 | 4 | 0 | 0 | 111 | 4 |

### OVRPBTWtrain

- Instances: `10000`; methods: `4`; tie instances: `3`
- Oracle top1 mean cost: `10.460631`
- Mean-cost leader: `RELD_MTL` (`10.589365`)
- Top1 leader: `RELD_MTL` (`4745`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.460631 | 10.589365 | 1 | 4745 | 4069 | 1186 | 1 |
| RELD_MOEL | 10.460631 | 10.617405 | 2 | 4372 | 4364 | 1264 | 2 |
| MVMOE | 10.460631 | 11.851035 | 3 | 883 | 1566 | 6407 | 3 |
| MTPOMO | 10.460631 | 15.208726 | 4 | 0 | 1 | 1143 | 4 |

### OVRPBTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `10.380432`
- Mean-cost leader: `RELD_MTL` (`10.506777`)
- Top1 leader: `RELD_MTL` (`461`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.380432 | 10.506777 | 1 | 461 | 411 | 128 | 1 |
| RELD_MOEL | 10.380432 | 10.536261 | 2 | 453 | 426 | 121 | 2 |
| MVMOE | 10.380432 | 11.804059 | 3 | 86 | 163 | 634 | 3 |
| MTPOMO | 10.380432 | 15.155493 | 4 | 0 | 0 | 117 | 4 |

## OVRPL

- Datasets: `OVRPLtest, OVRPLtrain, OVRPLval`
- Oracle top1 mean cost: `7.524704`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPLtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.531027`
- Mean-cost leader: `RELD_MTL` (`7.592861`)
- Top1 leader: `RELD_MTL` (`494`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.531027 | 7.592861 | 1 | 494 | 343 | 125 | 1 |
| RELD_MOEL | 7.531027 | 7.634092 | 2 | 385 | 396 | 169 | 2 |
| MTPOMO | 7.531027 | 8.299554 | 3 | 85 | 169 | 465 | 3 |
| MVMOE | 7.531027 | 8.902560 | 4 | 36 | 92 | 241 | 4 |

### OVRPLtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.525110`
- Mean-cost leader: `RELD_MTL` (`7.588017`)
- Top1 leader: `RELD_MTL` (`5069`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.525110 | 7.588017 | 1 | 5069 | 3111 | 1490 | 1 |
| RELD_MOEL | 7.525110 | 7.630860 | 2 | 3691 | 4093 | 1745 | 2 |
| MTPOMO | 7.525110 | 8.300787 | 3 | 865 | 1904 | 4591 | 3 |
| MVMOE | 7.525110 | 9.016318 | 4 | 375 | 892 | 2174 | 4 |

### OVRPLval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `7.514318`
- Mean-cost leader: `RELD_MTL` (`7.583246`)
- Top1 leader: `RELD_MTL` (`485`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 7.514318 | 7.583246 | 1 | 485 | 335 | 135 | 1 |
| RELD_MOEL | 7.514318 | 7.619351 | 2 | 386 | 398 | 162 | 2 |
| MTPOMO | 7.514318 | 8.226538 | 3 | 82 | 182 | 459 | 3 |
| MVMOE | 7.514318 | 9.045072 | 4 | 47 | 85 | 244 | 4 |

## OVRPLTW

- Datasets: `OVRPLTWtest, OVRPLTWtrain, OVRPLTWval`
- Oracle top1 mean cost: `10.025455`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPLTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `10.017423`
- Mean-cost leader: `RELD_MTL` (`10.103602`)
- Top1 leader: `RELD_MTL` (`492`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.017423 | 10.103602 | 1 | 492 | 408 | 100 | 1 |
| RELD_MOEL | 10.017423 | 10.127295 | 2 | 430 | 459 | 111 | 2 |
| MVMOE | 10.017423 | 11.312225 | 3 | 78 | 133 | 677 | 3 |
| MTPOMO | 10.017423 | 14.455833 | 4 | 0 | 0 | 112 | 4 |

### OVRPLTWtrain

- Instances: `10000`; methods: `4`; tie instances: `5`
- Oracle top1 mean cost: `10.024799`
- Mean-cost leader: `RELD_MTL` (`10.113588`)
- Top1 leader: `RELD_MTL` (`4772`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.024799 | 10.113588 | 1 | 4772 | 4163 | 1065 | 1 |
| RELD_MOEL | 10.024799 | 10.138449 | 2 | 4462 | 4389 | 1149 | 2 |
| MVMOE | 10.024799 | 11.375585 | 3 | 766 | 1447 | 6527 | 3 |
| MTPOMO | 10.024799 | 14.509354 | 4 | 0 | 1 | 1259 | 4 |

### OVRPLTWval

- Instances: `1000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `10.040047`
- Mean-cost leader: `RELD_MTL` (`10.127205`)
- Top1 leader: `RELD_MTL` (`495`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.040047 | 10.127205 | 1 | 495 | 401 | 104 | 1 |
| RELD_MOEL | 10.040047 | 10.154016 | 2 | 433 | 450 | 117 | 2 |
| MVMOE | 10.040047 | 11.359611 | 3 | 72 | 149 | 651 | 3 |
| MTPOMO | 10.040047 | 14.467725 | 4 | 0 | 0 | 128 | 4 |

## OVRPTW

- Datasets: `OVRPTWtest, OVRPTWtrain, OVRPTWval`
- Oracle top1 mean cost: `10.011344`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### OVRPTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `10.033850`
- Mean-cost leader: `RELD_MTL` (`10.123299`)
- Top1 leader: `RELD_MTL` (`480`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.033850 | 10.123299 | 1 | 480 | 417 | 103 | 1 |
| RELD_MOEL | 10.033850 | 10.143173 | 2 | 429 | 442 | 129 | 2 |
| MVMOE | 10.033850 | 11.403935 | 3 | 91 | 141 | 645 | 3 |
| MTPOMO | 10.033850 | 14.507176 | 4 | 0 | 0 | 123 | 4 |

### OVRPTWtrain

- Instances: `10000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `10.014349`
- Mean-cost leader: `RELD_MTL` (`10.103682`)
- Top1 leader: `RELD_MTL` (`4695`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 10.014349 | 10.103682 | 1 | 4695 | 4259 | 1046 | 1 |
| RELD_MOEL | 10.014349 | 10.120903 | 2 | 4555 | 4311 | 1134 | 2 |
| MVMOE | 10.014349 | 11.301199 | 3 | 750 | 1428 | 6619 | 3 |
| MTPOMO | 10.014349 | 14.459164 | 4 | 0 | 2 | 1201 | 4 |

### OVRPTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.958790`
- Mean-cost leader: `RELD_MTL` (`10.048435`)
- Top1 leader: `RELD_MTL` (`477`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.958790 | 10.048435 | 1 | 477 | 422 | 101 | 1 |
| RELD_MOEL | 9.958790 | 10.066680 | 2 | 457 | 436 | 107 | 2 |
| MVMOE | 9.958790 | 11.380933 | 3 | 66 | 142 | 661 | 3 |
| MTPOMO | 9.958790 | 14.350275 | 4 | 0 | 0 | 131 | 4 |

## TSP

- Datasets: `TSPLIB, TSPtest, TSPtrain, TSPval`
- Oracle top1 mean cost: `8.055567`
- Mean-cost dataset wins: `{'LEHD': 4}`
- Top1 dataset wins: `{'LEHD': 4}`
- Same-method top1+mean wins: `{'LEHD': 4}`

### TSPLIB

- Instances: `49`; methods: `10`; tie instances: `2`
- Oracle top1 mean cost: `8.219978`
- Mean-cost leader: `LEHD` (`8.258462`)
- Top1 leader: `LEHD` (`32`)
- Same method wins both: `LEHD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| LEHD | 8.219978 | 8.258462 | 1 | 32 | 15 | 0 | 1 |
| OMNI | 8.219978 | 8.578038 | 2 | 3 | 16 | 18 | 3 |
| INVIT | 8.219978 | 8.706001 | 3 | 1 | 9 | 9 | 5 |
| GLOP | 8.219978 | 8.933969 | 4 | 1 | 1 | 10 | 6 |
| DIFUSCO | 8.219978 | 9.111516 | 5 | 10 | 4 | 2 | 2 |
| T2T | 8.219978 | 9.415106 | 6 | 0 | 0 | 3 | 7 |
| DACT | 8.219978 | 10.069326 | 7 | 0 | 0 | 0 | 8 |
| UDC | 8.219978 | 13.192162 | 8 | 0 | 0 | 0 | 9 |
| ELG | 8.219978 | 13.974902 | 9 | 2 | 4 | 7 | 4 |
| LIH | 8.219978 | 113.687182 | 10 | 0 | 0 | 0 | 10 |

### TSPtest

- Instances: `1000`; methods: `10`; tie instances: `3`
- Oracle top1 mean cost: `8.093983`
- Mean-cost leader: `LEHD` (`8.157863`)
- Top1 leader: `LEHD` (`706`)
- Same method wins both: `LEHD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| LEHD | 8.093983 | 8.157863 | 1 | 706 | 149 | 85 | 1 |
| OMNI | 8.093983 | 8.318756 | 2 | 210 | 559 | 186 | 2 |
| GLOP | 8.093983 | 8.687625 | 3 | 14 | 95 | 337 | 4 |
| INVIT | 8.093983 | 8.704566 | 4 | 7 | 124 | 279 | 5 |
| DIFUSCO | 8.093983 | 9.235499 | 5 | 56 | 60 | 69 | 3 |
| T2T | 8.093983 | 9.464934 | 6 | 0 | 3 | 17 | 7 |
| DACT | 8.093983 | 10.128731 | 7 | 0 | 0 | 0 | 8 |
| UDC | 8.093983 | 13.084382 | 8 | 0 | 0 | 0 | 9 |
| ELG | 8.093983 | 16.902471 | 9 | 7 | 10 | 27 | 6 |
| LIH | 8.093983 | 116.108027 | 10 | 0 | 0 | 0 | 10 |

### TSPtrain

- Instances: `10000`; methods: `10`; tie instances: `34`
- Oracle top1 mean cost: `8.059538`
- Mean-cost leader: `LEHD` (`8.127289`)
- Top1 leader: `LEHD` (`7043`)
- Same method wins both: `LEHD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| LEHD | 8.059538 | 8.127289 | 1 | 7043 | 1566 | 715 | 1 |
| OMNI | 8.059538 | 8.278670 | 2 | 2039 | 5610 | 1841 | 2 |
| GLOP | 8.059538 | 8.636439 | 3 | 149 | 814 | 3505 | 5 |
| INVIT | 8.059538 | 8.642378 | 4 | 170 | 1223 | 2951 | 4 |
| DIFUSCO | 8.059538 | 9.198087 | 5 | 545 | 587 | 592 | 3 |
| T2T | 8.059538 | 9.445118 | 6 | 2 | 7 | 84 | 7 |
| DACT | 8.059538 | 10.045222 | 7 | 0 | 1 | 4 | 8 |
| UDC | 8.059538 | 13.003409 | 8 | 0 | 0 | 0 | 9 |
| ELG | 8.059538 | 17.074398 | 9 | 52 | 192 | 308 | 6 |
| LIH | 8.059538 | 115.854567 | 10 | 0 | 0 | 0 | 10 |

### TSPval

- Instances: `1000`; methods: `10`; tie instances: `5`
- Oracle top1 mean cost: `7.969377`
- Mean-cost leader: `LEHD` (`8.038186`)
- Top1 leader: `LEHD` (`700`)
- Same method wins both: `LEHD`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| LEHD | 7.969377 | 8.038186 | 1 | 700 | 143 | 79 | 1 |
| OMNI | 7.969377 | 8.182173 | 2 | 197 | 567 | 178 | 2 |
| GLOP | 7.969377 | 8.537977 | 3 | 15 | 85 | 360 | 5 |
| INVIT | 7.969377 | 8.558233 | 4 | 19 | 118 | 275 | 4 |
| DIFUSCO | 7.969377 | 9.098994 | 5 | 59 | 65 | 67 | 3 |
| T2T | 7.969377 | 9.343406 | 6 | 0 | 2 | 6 | 7 |
| DACT | 7.969377 | 9.923410 | 7 | 0 | 0 | 1 | 8 |
| UDC | 7.969377 | 12.925378 | 8 | 0 | 0 | 0 | 9 |
| ELG | 7.969377 | 16.905773 | 9 | 10 | 20 | 34 | 6 |
| LIH | 7.969377 | 115.952039 | 10 | 0 | 0 | 0 | 10 |

## VRPB

- Datasets: `VRPBtest, VRPBtrain, VRPBval`
- Oracle top1 mean cost: `9.242159`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPBtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.217913`
- Mean-cost leader: `RELD_MTL` (`9.295063`)
- Top1 leader: `RELD_MTL` (`483`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.217913 | 9.295063 | 1 | 483 | 396 | 111 | 1 |
| RELD_MOEL | 9.217913 | 9.330275 | 2 | 436 | 406 | 144 | 2 |
| MTPOMO | 9.217913 | 10.701428 | 3 | 22 | 50 | 351 | 4 |
| MVMOE | 9.217913 | 11.584618 | 4 | 59 | 148 | 394 | 3 |

### VRPBtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.241975`
- Mean-cost leader: `RELD_MTL` (`9.316559`)
- Top1 leader: `RELD_MTL` (`5137`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.241975 | 9.316559 | 1 | 5137 | 3724 | 1055 | 1 |
| RELD_MOEL | 9.241975 | 9.367410 | 2 | 4102 | 4273 | 1463 | 2 |
| MTPOMO | 9.241975 | 10.715950 | 3 | 190 | 578 | 3393 | 4 |
| MVMOE | 9.241975 | 11.603011 | 4 | 571 | 1425 | 4089 | 3 |

### VRPBval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.268251`
- Mean-cost leader: `RELD_MTL` (`9.340269`)
- Top1 leader: `RELD_MTL` (`504`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.268251 | 9.340269 | 1 | 504 | 390 | 103 | 1 |
| RELD_MOEL | 9.268251 | 9.396019 | 2 | 413 | 425 | 150 | 2 |
| MTPOMO | 9.268251 | 10.822326 | 3 | 17 | 55 | 332 | 4 |
| MVMOE | 9.268251 | 11.803180 | 4 | 66 | 130 | 415 | 3 |

## VRPBL

- Datasets: `VRPBLtest, VRPBLtrain, VRPBLval`
- Oracle top1 mean cost: `9.241830`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPBLtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.177293`
- Mean-cost leader: `RELD_MTL` (`9.249180`)
- Top1 leader: `RELD_MTL` (`537`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.177293 | 9.249180 | 1 | 537 | 356 | 100 | 1 |
| RELD_MOEL | 9.177293 | 9.309500 | 2 | 402 | 440 | 135 | 2 |
| MTPOMO | 9.177293 | 10.599956 | 3 | 13 | 64 | 352 | 4 |
| MVMOE | 9.177293 | 11.212746 | 4 | 48 | 140 | 413 | 3 |

### VRPBLtrain

- Instances: `10000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `9.244008`
- Mean-cost leader: `RELD_MTL` (`9.320910`)
- Top1 leader: `RELD_MTL` (`5015`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.244008 | 9.320910 | 1 | 5015 | 3851 | 1020 | 1 |
| RELD_MOEL | 9.244008 | 9.375451 | 2 | 4121 | 4171 | 1504 | 2 |
| MTPOMO | 9.244008 | 10.572554 | 3 | 207 | 560 | 3446 | 4 |
| MVMOE | 9.244008 | 11.134370 | 4 | 657 | 1418 | 4030 | 3 |

### VRPBLval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `9.284586`
- Mean-cost leader: `RELD_MTL` (`9.359268`)
- Top1 leader: `RELD_MTL` (`528`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 9.284586 | 9.359268 | 1 | 528 | 355 | 106 | 1 |
| RELD_MOEL | 9.284586 | 9.424117 | 2 | 386 | 440 | 162 | 2 |
| MTPOMO | 9.284586 | 10.705363 | 3 | 21 | 60 | 344 | 4 |
| MVMOE | 9.284586 | 11.321271 | 4 | 65 | 145 | 388 | 3 |

## VRPBLTW

- Datasets: `VRPBLTWtest, VRPBLTWtrain, VRPBLTWval`
- Oracle top1 mean cost: `19.443668`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPBLTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `19.468737`
- Mean-cost leader: `RELD_MTL` (`19.676957`)
- Top1 leader: `RELD_MTL` (`510`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.468737 | 19.676957 | 1 | 510 | 359 | 130 | 1 |
| RELD_MOEL | 19.468737 | 19.799754 | 2 | 382 | 459 | 159 | 2 |
| MVMOE | 19.468737 | 21.506060 | 3 | 107 | 182 | 572 | 3 |
| MTPOMO | 19.468737 | 24.793583 | 4 | 1 | 0 | 139 | 4 |

### VRPBLTWtrain

- Instances: `10000`; methods: `4`; tie instances: `2`
- Oracle top1 mean cost: `19.453659`
- Mean-cost leader: `RELD_MTL` (`19.666086`)
- Top1 leader: `RELD_MTL` (`4984`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.453659 | 19.666086 | 1 | 4984 | 3762 | 1251 | 1 |
| RELD_MOEL | 19.453659 | 19.779709 | 2 | 3943 | 4434 | 1612 | 2 |
| MVMOE | 19.453659 | 21.572153 | 3 | 1070 | 1790 | 5734 | 3 |
| MTPOMO | 19.453659 | 24.899868 | 4 | 3 | 14 | 1403 | 4 |

### VRPBLTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `19.318692`
- Mean-cost leader: `RELD_MTL` (`19.519778`)
- Top1 leader: `RELD_MTL` (`483`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.318692 | 19.519778 | 1 | 483 | 400 | 117 | 1 |
| RELD_MOEL | 19.318692 | 19.667521 | 2 | 408 | 431 | 159 | 2 |
| MVMOE | 19.318692 | 21.470786 | 3 | 109 | 168 | 573 | 3 |
| MTPOMO | 19.318692 | 24.715246 | 4 | 0 | 1 | 151 | 4 |

## VRPBTW

- Datasets: `VRPBTWtest, VRPBTWtrain, VRPBTWval`
- Oracle top1 mean cost: `19.469472`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPBTWtest

- Instances: `1000`; methods: `4`; tie instances: `1`
- Oracle top1 mean cost: `19.317747`
- Mean-cost leader: `RELD_MTL` (`19.526309`)
- Top1 leader: `RELD_MTL` (`488`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.317747 | 19.526309 | 1 | 488 | 368 | 144 | 1 |
| RELD_MOEL | 19.317747 | 19.630883 | 2 | 394 | 465 | 140 | 2 |
| MVMOE | 19.317747 | 21.310201 | 3 | 118 | 165 | 581 | 3 |
| MTPOMO | 19.317747 | 24.685686 | 4 | 0 | 2 | 135 | 4 |

### VRPBTWtrain

- Instances: `10000`; methods: `4`; tie instances: `3`
- Oracle top1 mean cost: `19.485718`
- Mean-cost leader: `RELD_MTL` (`19.692419`)
- Top1 leader: `RELD_MTL` (`4989`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.485718 | 19.692419 | 1 | 4989 | 3750 | 1260 | 1 |
| RELD_MOEL | 19.485718 | 19.809165 | 2 | 3962 | 4494 | 1535 | 2 |
| MVMOE | 19.485718 | 21.489391 | 3 | 1047 | 1748 | 5877 | 3 |
| MTPOMO | 19.485718 | 24.878005 | 4 | 2 | 8 | 1328 | 4 |

### VRPBTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `19.458733`
- Mean-cost leader: `RELD_MTL` (`19.669740`)
- Top1 leader: `RELD_MTL` (`510`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 19.458733 | 19.669740 | 1 | 510 | 377 | 113 | 1 |
| RELD_MOEL | 19.458733 | 19.775302 | 2 | 398 | 457 | 144 | 2 |
| MVMOE | 19.458733 | 21.619582 | 3 | 92 | 165 | 606 | 3 |
| MTPOMO | 19.458733 | 24.888895 | 4 | 0 | 1 | 137 | 4 |

## VRPL

- Datasets: `VRPLtest, VRPLtrain, VRPLval`
- Oracle top1 mean cost: `12.408966`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPLtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `12.371230`
- Mean-cost leader: `RELD_MTL` (`12.456806`)
- Top1 leader: `RELD_MTL` (`454`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 12.371230 | 12.456806 | 1 | 454 | 300 | 178 | 1 |
| RELD_MOEL | 12.371230 | 12.489746 | 2 | 356 | 365 | 202 | 2 |
| MTPOMO | 12.371230 | 13.343418 | 3 | 127 | 214 | 389 | 3 |
| MVMOE | 12.371230 | 14.163921 | 4 | 63 | 121 | 231 | 4 |

### VRPLtrain

- Instances: `10000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `12.411139`
- Mean-cost leader: `RELD_MTL` (`12.487517`)
- Top1 leader: `RELD_MTL` (`4627`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 12.411139 | 12.487517 | 1 | 4627 | 2957 | 1805 | 1 |
| RELD_MOEL | 12.411139 | 12.530213 | 2 | 3500 | 3635 | 2005 | 2 |
| MTPOMO | 12.411139 | 13.385177 | 3 | 1214 | 2182 | 3850 | 3 |
| MVMOE | 12.411139 | 14.169946 | 4 | 659 | 1226 | 2340 | 4 |

### VRPLval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `12.424977`
- Mean-cost leader: `RELD_MTL` (`12.508747`)
- Top1 leader: `RELD_MTL` (`446`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 12.424977 | 12.508747 | 1 | 446 | 297 | 196 | 1 |
| RELD_MOEL | 12.424977 | 12.546506 | 2 | 372 | 350 | 193 | 2 |
| MTPOMO | 12.424977 | 13.454017 | 3 | 121 | 221 | 385 | 3 |
| MVMOE | 12.424977 | 14.360135 | 4 | 61 | 132 | 226 | 4 |

## VRPLTW

- Datasets: `VRPLTWtest, VRPLTWtrain, VRPLTWval`
- Oracle top1 mean cost: `18.576484`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPLTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `18.305706`
- Mean-cost leader: `RELD_MTL` (`18.426862`)
- Top1 leader: `RELD_MTL` (`502`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.305706 | 18.426862 | 1 | 502 | 380 | 118 | 1 |
| RELD_MOEL | 18.305706 | 18.495057 | 2 | 410 | 464 | 126 | 2 |
| MVMOE | 18.305706 | 20.012325 | 3 | 88 | 155 | 622 | 3 |
| MTPOMO | 18.305706 | 23.163571 | 4 | 0 | 1 | 134 | 4 |

### VRPLTWtrain

- Instances: `10000`; methods: `4`; tie instances: `5`
- Oracle top1 mean cost: `18.616738`
- Mean-cost leader: `RELD_MTL` (`18.743242`)
- Top1 leader: `RELD_MTL` (`4955`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.616738 | 18.743242 | 1 | 4955 | 3939 | 1106 | 1 |
| RELD_MOEL | 18.616738 | 18.818920 | 2 | 4096 | 4551 | 1353 | 2 |
| MVMOE | 18.616738 | 20.451790 | 3 | 946 | 1506 | 6145 | 3 |
| MTPOMO | 18.616738 | 23.493168 | 4 | 3 | 4 | 1396 | 4 |

### VRPLTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `18.444721`
- Mean-cost leader: `RELD_MTL` (`18.576974`)
- Top1 leader: `RELD_MTL` (`505`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.444721 | 18.576974 | 1 | 505 | 372 | 123 | 1 |
| RELD_MOEL | 18.444721 | 18.652349 | 2 | 403 | 468 | 128 | 2 |
| MVMOE | 18.444721 | 20.274024 | 3 | 92 | 160 | 604 | 3 |
| MTPOMO | 18.444721 | 23.279750 | 4 | 0 | 0 | 145 | 4 |

## VRPTW

- Datasets: `VRPTWtest, VRPTWtrain, VRPTWval`
- Oracle top1 mean cost: `18.638939`
- Mean-cost dataset wins: `{'RELD_MTL': 3}`
- Top1 dataset wins: `{'RELD_MTL': 3}`
- Same-method top1+mean wins: `{'RELD_MTL': 3}`

### VRPTWtest

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `18.435115`
- Mean-cost leader: `RELD_MTL` (`18.574120`)
- Top1 leader: `RELD_MTL` (`473`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.435115 | 18.574120 | 1 | 473 | 406 | 121 | 1 |
| RELD_MOEL | 18.435115 | 18.627283 | 2 | 428 | 447 | 125 | 2 |
| MVMOE | 18.435115 | 20.243225 | 3 | 99 | 147 | 607 | 3 |
| MTPOMO | 18.435115 | 23.240192 | 4 | 0 | 0 | 147 | 4 |

### VRPTWtrain

- Instances: `10000`; methods: `4`; tie instances: `3`
- Oracle top1 mean cost: `18.652387`
- Mean-cost leader: `RELD_MTL` (`18.781431`)
- Top1 leader: `RELD_MTL` (`5002`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.652387 | 18.781431 | 1 | 5002 | 3919 | 1079 | 1 |
| RELD_MOEL | 18.652387 | 18.852350 | 2 | 4105 | 4577 | 1316 | 2 |
| MVMOE | 18.652387 | 20.504243 | 3 | 893 | 1501 | 6182 | 3 |
| MTPOMO | 18.652387 | 23.544962 | 4 | 0 | 3 | 1423 | 4 |

### VRPTWval

- Instances: `1000`; methods: `4`; tie instances: `0`
- Oracle top1 mean cost: `18.708291`
- Mean-cost leader: `RELD_MTL` (`18.850502`)
- Top1 leader: `RELD_MTL` (`472`)
- Same method wins both: `RELD_MTL`

| method | oracle_top1_mean_cost | mean_cost | mean_rank | top1 | top2 | top3 | top1_rank |
|---|---|---|---|---|---|---|---|
| RELD_MTL | 18.708291 | 18.850502 | 1 | 472 | 416 | 112 | 1 |
| RELD_MOEL | 18.708291 | 18.895425 | 2 | 430 | 435 | 135 | 2 |
| MVMOE | 18.708291 | 20.522358 | 3 | 98 | 147 | 624 | 3 |
| MTPOMO | 18.708291 | 23.643270 | 4 | 0 | 2 | 129 | 4 |
