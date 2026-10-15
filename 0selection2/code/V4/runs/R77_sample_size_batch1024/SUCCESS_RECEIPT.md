# R77 Success Receipt

Completed UTC: 2026-10-14T20:07:26.030081+00:00

Both arms completed49680 successful updates, unique VAL-best locks, one full TEST and analysis.
Both approved4090s have no compute processes and were observed idle. NSS may be assigned them now.

| Arm | TRAIN rows/problem | Selected update | TEST native Top1 % | CE | Actual regret % | vs FULL TRAIN SBS % |
|---|---:|---:|---:|---:|---:|---:|
| A | 10000 | 16146 | 70.5691 | 0.719735 | 0.412741 | -0.175897 |
| B | 70114 | 34776 | 71.2252 | 0.707470 | 0.394014 | -0.194048 |

finalcomparison.csv includes all saved TRAIN/VAL and locked TEST, all18 tasks, four families, subset counts, common/own TRAIN-fixed SBS and common_updates.
No repeated TEST inference, rule change, GPU reassignment or Git commit was performed.
