# R40 Protocol Interpretation

The final plan in the user message is executed: scratch dual-stream versus direct classification, seed 2, batch 64, 60 data epochs, identical winner-cost loss. The earlier checkpoint-continuation versus pure-CE proposal is not part of this run.

Each problem's per-epoch 9,600 sample indices are identical to R39A. Every old 640-instance batch is split into ten genuine 64-instance optimizer batches, without accumulation. A and B share exactly the same sample and task schedules.

R40 updates the 18 tasks in balanced 64-instance cycles. This changes the flattened cross-task interleaving from R39A's 640-instance cycles. Therefore R40A versus historical R39A tests the update-density training protocol as a whole, including switching frequency and successful-update LR scheduling; it does not isolate batch size alone. The A/B architecture comparison remains paired under the same schedule.

Each run has 150 updates per task per data epoch, 9,000 updates per task in total, 162,000 successful global updates, and 10,368,000 sample presentations. Validation uses every instance each epoch. Full train evaluation uses eval mode and includes the tail every five epochs.

The measured machine has one RTX 4090. Both jobs run concurrently on that GPU. W&B is offline without login credentials. No test split is loaded.

A fresh same-family reviewer withdrew its initial stronger global-order interpretation after reading the user's exact per-problem sequencing requirement. The raw review and clarification are retained in .aris/traces/experiment-audit/2026-09-30_r40/. Its multi-GPU watcher portability warning does not apply to this measured single-GPU execution.

