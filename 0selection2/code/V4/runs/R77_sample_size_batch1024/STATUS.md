# R77 Root Status

Updated UTC: 2026-10-14T20:11:49.034624+00:00

Phase: complete. Leave healthy training/monitor tmux processes running.

| Arm | Phase | Successful updates | Latest complete VAL | Top1 % | CE | Regret % | Current best | TEST complete |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | complete | 49680 | 49680 | 69.4808 | 0.853683 | 0.539679 | 16146 | True |
| B | complete | 49680 | 49680 | 70.8415 | 0.708262 | 0.410186 | 34776 | True |

## Latest Common Update

Update: 49680

| Arm | Top1 % | CE | Regret % | vs common FULL TRAIN SBS % |
|---|---:|---:|---:|---:|
| A | 69.4808 | 0.853683 | 0.539679 | -0.067623 |
| B | 70.8415 | 0.708262 | 0.410186 | -0.184136 |

## TRAIN Curve

A: two consecutive TRAIN-CE-down / VAL-CE-up intervals suggest overfitting; reporting only.
- Update6210: TRAIN/VAL Top1 69.4594/69.2127%; CE 0.745757/0.754349; regret 0.497902/0.502318%.
- Update12420: TRAIN/VAL Top1 70.5756/70.0927%; CE 0.716422/0.735690; regret 0.418686/0.428340%.
- Update16146: TRAIN/VAL Top1 70.9189/70.1997%; CE 0.693358/0.724285; regret 0.405340/0.418613%.
- Update18630: TRAIN/VAL Top1 71.3444/70.4407%; CE 0.688938/0.740814; regret 0.403593/0.421504%.
- Update24840: TRAIN/VAL Top1 72.4111/70.1469%; CE 0.648730/0.765204; regret 0.377135/0.457060%.
- Update31050: TRAIN/VAL Top1 74.2328/69.9943%; CE 0.604301/0.779675; regret 0.336088/0.491967%.
- Update37260: TRAIN/VAL Top1 75.5294/69.7704%; CE 0.575446/0.827365; regret 0.305277/0.507244%.
- Update43470: TRAIN/VAL Top1 76.1350/69.4608%; CE 0.561502/0.848687; regret 0.296537/0.543615%.
- Update49680: TRAIN/VAL Top1 76.2850/69.4808%; CE 0.558489/0.853683; regret 0.293842/0.539679%.
B: no persistent TRAIN-CE-down / VAL-CE-up evidence in measured intervals.
- Update6210: TRAIN/VAL Top1 69.1893/69.0030%; CE 0.741162/0.744323; regret 0.487952/0.494654%.
- Update12420: TRAIN/VAL Top1 70.2421/69.9900%; CE 0.744089/0.748503; regret 0.424935/0.432169%.
- Update18630: TRAIN/VAL Top1 70.5899/70.4236%; CE 0.715367/0.720400; regret 0.426576/0.429573%.
- Update24840: TRAIN/VAL Top1 70.8159/70.5292%; CE 0.712952/0.720245; regret 0.412097/0.419830%.
- Update31050: TRAIN/VAL Top1 70.9893/70.7502%; CE 0.703014/0.710963; regret 0.406661/0.411797%.
- Update34776: TRAIN/VAL Top1 71.1767/70.8943%; CE 0.704273/0.713642; regret 0.393012/0.401327%.
- Update37260: TRAIN/VAL Top1 71.1458/70.8972%; CE 0.701081/0.710511; regret 0.402064/0.407576%.
- Update43470: TRAIN/VAL Top1 71.1760/70.8815%; CE 0.697944/0.707773; regret 0.399810/0.407944%.
- Update49680: TRAIN/VAL Top1 71.1509/70.8415%; CE 0.698277/0.708262; regret 0.401800/0.410186%.

## Continuation

Read this root STATUS and per-arm status/history/locks. Leave healthy tmux processes running; no new preflight, cache build, capacity scan or solver execution. Do not restart a healthy run.

A resumed its own saved R77 update2484 state; not historical weights and not a Git commit.

Both arms must finish49680 successful updates. TEST follows each immutable own VAL-best lock once; never use TEST for rules/checkpoint selection. No Git commits.

Per-arm launch_receipt.json identifies live GPU process PIDs. Controller and monitor remain in tmux R77_sample_size_batch1024; snapshot receipts and STATUS refresh on a CPU-only reporter.

Once both analyses finish, TEST_ROWS.md / paired_test_comparison.csv include ALL, four families and all18 tasks.
