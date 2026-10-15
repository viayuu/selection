# R78 Round003 Fixed-Gate Final Report

Outcome: finite_replacement_trial_closed_negative. Full18, batch1024, seed2;14904 successful updates per arm.
Original B ep28/u34776 is fallback. CE is descriptive only; no TEST read or coefficient retry.

| Rule | Continuation update | Top1 % | Raw mean cost | Actual regret % | CE | Eligible |
|---|---:|---:|---:|---:|---:|---|
| B | 0 | 70.894308943 | 11.395558343544 | 0.401326745 | 0.713641610 | fallback |
| control | 4968 | 70.921409214 | 11.395716839270 | 0.405973048 | 0.705434735 | False |
| control | 9936 | 70.850092711 | 11.395744982192 | 0.408499816 | 0.708710366 | False |
| control | 14904 | 70.814434460 | 11.395855343252 | 0.410925983 | 0.708355988 | False |
| treatment | 4968 | 70.897161603 | 11.395848830432 | 0.407708383 | 0.700871229 | False |
| treatment | 9936 | 70.832976751 | 11.395886356713 | 0.410676066 | 0.703012548 | False |
| treatment | 14904 | 70.828697761 | 11.395991601929 | 0.413110693 | 0.702879354 | False |

## Selected Whole18 Rules

| Branch | Selected continuation update | Checkpoint |
|---|---:|---|
| control | 0 | /public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2/best.pt |
| treatment | 0 | /public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2/best.pt |

A selected update0 means B fallback, not identical fitted trajectories. All negative endpoint scores/checkpoints are retained.
Treatment-specific positive requires strictly higher selected Top1 than matched control; cost/regret differences against control remain reported.
No automatic control promotion or TEST release; overall research goal remains active.
GPU release is checked after both workers exit, separately from scientific acceptance.

## GPU Release And Workflow Disposition

At 2026-10-14T23:46:24.710960+00:00, read-only NVML on gpu03 confirmed both approved UUIDs at 0% utilization, 422.6875 MiB used memory and no compute processes. R78 PIDs772905/772906 are absent; tmux R78_round003_matched_pair has exited. No other processes were killed and R78 did not cancel Slurm allocation1465.

Release evidence: gpu_release.json. Final receipts confirm14904 successful updates per arm,29808 total; sampled14800 progress counters are not completion counts.

The independent R78 native goal remains blocked as actually returned by get_goal; no native goal status was mutated. File-based workflow remains active, waiting for Astra's single next direction. The legacy main-thread paused goal was neither queried nor changed. Both selected rules retain B fallback; no TEST, negative-rule promotion, new training or additional audit.
