# R45 execution record

- Software reused: easynco, torch 2.5.1+cu124; no package or model backend changes.
- Actual compute: existing user Slurm allocation 1465, gpu03, two RTX 3090
  24 GiB, driver 560.35.05. admin01 is the control node and has no GPU devices.
- CUDA witness: both devices visible and finite matrix multiplication passed.
- Focused acceptance suite: 36 tests passed on the compute node.
- Full-batch preflight: all A/B/C updates passed on TSP/CVRP/ATSP/OVRPTW,
  including 499/500-node batches. Largest allocation was 8.932 GiB.
- Preflight models were discarded; actual training initialization is fresh.
- Common initial parameters match A/B/C. All trainable B/C parameters match;
  only their fixed semantic row ownership differs.
- Train/val reference predictions copied from
  `R43_pairwise_selection/reference_predictions/{train,val}`; no test copied.
  The existing reader checks raw costs, winners, and solver identities.
- Launch time: 2026-10-08 22:11 Asia/Shanghai. B/C started data loading at
  22:12. Actual training runs in tmux session `r45_embeddings`.

## Invocation

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
srun --jobid=1465 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
  bash code/V4/run_v4_r45_campaign.sh
```

No API key or new external request is involved. The original corpus, locked
embedding vectors, and historical experiments are preserved.

## Completion

- Campaign completed successfully on 2026-10-09 at 00:45:06 Asia/Shanghai.
- A: 40 epochs, 56,880 successful updates, validation-best epoch 38.
- B: early stop at 38 epochs, 54,036 successful updates, best epoch 30.
- C: early stop at 36 epochs, 51,192 successful updates, best epoch 28.
- Every problem has exactly 79 successful updates per epoch, including tail
  batches. AMP overflow retries did not advance the data schedule.
- Actual peak training allocations: A 8.885 GiB; B/C 8.928 GiB.
- All three validation-best checkpoint hashes were locked before test.
- All 54 test prediction files contain 1,000 unique instances and FP64 costs.
  Recomputed prediction metrics match exactly; the maximum error is zero.
- Frozen training source hashes remain unchanged. No historical experiment,
  existing solver data, external embedding cache, or API key was modified.
- Final reports: comparison.md/csv, family_comparison.csv,
  decision_changes.csv, size_comparison.csv, arm_distributions.csv.
- Postprocessing adds convergence_zoom.png and observations.csv (108
  model/split/problem rows), plus 18 validation-best solver plots per arm.
  It only reads saved metrics and does not run model inference or select
  new checkpoints. Its source is plot_convergence.py in this directory.
- Main finding: B does not improve on A. Versus C, B has slightly lower
  selection cost but lower Top1; useful source-semantic correspondence was
  not established in this single-seed experiment.
- The owned r45_embeddings tmux session exited normally. Other user jobs
  and the existing Slurm allocation were left untouched.
