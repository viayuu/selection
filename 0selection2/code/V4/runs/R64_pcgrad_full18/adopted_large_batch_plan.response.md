Locked amendment for **restarted R64A/B**. No subprocesses, code execution, or GPU access performed. File contents were unavailable through the permitted read tools, so existing numerical optimizer settings are retained by reference.

1. **Bounded TRAIN-only calibration.** Sol may measure at most **three distinct real per-task batch sizes**, using the actual update path on a triple of the largest-node TRAIN tasks. Start at **256** given unverified local memory requirements. If feasible, try 512, then 1024; if 512 fails, use 384 as the third candidate. If 256 fails, try 128, then 192 only if 128 passes. No further search.

   For each candidate and each arm, run two warmup and three timed complete updates, including native CE, both R-Drop forwards, gradient handling, clipping, and initialized AdamW state. Use original TRAIN rows and labels. Measure complete-update throughput and peak GPU footprint with normal resident allocations. Eligibility requires approximately **15% physical GPU memory headroom**, including existing occupancy. Choose the eligible batch with highest measured PCGrad presentations/second; break indistinguishable ties toward the smaller batch. Both arms use this single global batch. No microbatching or gradient accumulation. If neither 256 nor 128 fits, report the capacity blocker.

2. **One LR and budget policy: retain LR; preserve data coverage.** Keep original R64 LR values, AdamW settings, clipping, objective, architecture, and validation-driven scheduling rules unchanged. **Do not scale LR with batch size.** Preserve existing warmup duration in fractional epochs/data coverage, converting any update-based warmup constant to the new updates-per-epoch. Apply the same mechanical conversion to other update-based schedule durations.

   For selected per-task batch \(b\):
   \[
   C=\lceil10000/b\rceil,\quad U_{\rm epoch}=6C,\quad
   U_{\max}=40(6C),\quad r=10000-b(C-1).
   \]
   Every cycle has six triples covering all 18 tasks once. The last cycle has six updates with **\(r\) real rows per task**; retain tails without dropping, padding, or duplicating rows. Preserve existing loss normalization.

   | Batch | Cycles/epoch | Updates/epoch | Maximum updates | Tail/task |
   |---|---:|---:|---:|---:|
   | 256 | 40 | 240 | 9,600 | 16 |
   | 512 | 20 | 120 | 4,800 | 272 |
   | 1024 | 10 | 60 | 2,400 | 784 |

   Maximum remains **40 epochs and 7.2M presentations per arm**. Aborted batch-42 allowed 57,360 updates; the reduced optimizer-update budget is explicitly part of this amendment.

3. **Pairing and prompt launch.** Discard calibration state. Restart both arms from the same original initialization, with identical TRAIN permutations, triple order, batches, tails, and dropout streams; isolate PCGrad randomness. A alone determines the LR trajectory and realized horizon under existing rules; B replays every LR and exactly that horizon. Select checkpoints by integer validation-correct counts with the existing deterministic tie rule.

   Sol/max launches ordinary paired training immediately after bounded calibration, with no additional review gate. Lock both arms using validation only before **one full TEST evaluation per arm**. Preserve all 18 original 10,000-row tasks and native winners; no R58, solvers, new labels, experts, or TEST tuning. Use only assigned UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` on `gpu03/index1`, within allocation `1465`.

This remains amended R64, comparing mean against PCGrad in the new batch regime. The archived batch-42 run is aborted, with no continuation or matched-comparison claim.