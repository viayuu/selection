# TRAIN native-label distribution (not model accuracy)

| Problem | Dominant canonical allocation | Unique wins of that method | Tie rate |
| --- | --- | --- | --- |
| MIS | T2T_ER: 5,055/10,000 (50.55%) | 5,055 (50.55%) | 22.42% |
| JSSP | MWKR: 6,042/10,000 (60.42%) | 6,027 (60.27%) | 2.61% |
| FJSP | DAN SD1 20x10: 2,727/10,000 (27.27%) | 2,590 (25.90%) | 8.74% |
| BPP | MBS': 8,646/10,000 (86.46%) | 8,646 (86.46%) | 13.54% |
| MKP | LPDualGreedy: 9,276/10,000 (92.76%) | 9,262 (92.62%) | 0.14% |
| SMTWTP | ATC: 9,202/10,000 (92.02%) | 7,840 (78.40%) | 13.62% |

Canonical: first exact native FP64 best in frozen candidate order; not a unique winner.
Fractional weights: unique=1; each member of an exact k-way best set=1/k; not real winner counts.
Full zero-preserving distributions and solver IDs are in method_distribution.csv.
