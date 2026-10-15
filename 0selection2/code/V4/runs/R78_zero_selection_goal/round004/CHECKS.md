# Necessary Round004 Checks

Only the following three checks were performed. No broad audit, full-gradient
snapshot, stationarity certificate, additional fitting budget or GPU smoke.

1. Synthetic masked FP32 probability /D0 identity /exact tie test: passed.
2. Synthetic original FP64 targets /unit prior /fixed cosine /natural tail-fill
   test: passed. The two unittest cases completed in0.018 seconds.
3. Real first TRAIN/TSP batch1024: loss1.3575456229899254, finite gradient and
   finite updated matrix. This was successful update1 of the declared1380 TSP
   updates, not a discarded additional optimizer step. See first_batch_check.json.

Command used: source /public/home/shiys/miniconda3/etc/profile.d/conda.sh &&
conda activate easynco && python -m unittest code.V4.test_r78_frozen_score_cost_head.

Normal minibatch nonfinite guards do not certify an optimum and never extend
the budget. All18 tasks completed exactly1380 updates; final receipt24840.
