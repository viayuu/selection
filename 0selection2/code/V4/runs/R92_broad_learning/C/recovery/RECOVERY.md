# R92C: Evaluation Recovery

All18 classifiers were already trained for3000 trees/task. No tree was refitted.
Canonical input statistics and recomputed FP32 statistics changed three TSP validation selections
(indices458,1770,3099). Correct count stayed2080, but mean cost differed.
No tolerance was loosened. Known-split evaluation now reuses the canonical357-column input-only feature cache.
The same previously selected800-tree checkpoint and original protocol were retained.
The formal global validation lock preceded TEST. No labels, costs, Oracle or indices enter prediction features.
Raw-input LIB uses the untouched factory and is a separately disclosed numerical path;
it is not claimed bitwise equivalent to cached split features.

VAL/TRAIN were recovered on ADMIN01 CPU. TEST ran CPU-only in the original allocation-owned container
because its existing node cache is under container /tmp. LIB/PDF then ran on ADMIN01 CPU.
The first CPU recovery reached the VAL lock, then failed on that unavailable ADMIN01 TEST-cache path;
the explicit post-lock continuation did not rerun selection or training.
Original failed worker/controller/error records and wrong-path VAL predictions are preserved.
Protected originals unchanged: True; fitted trees during recovery:0.
Authoritative completion: C/recovery_complete.json and C/recovery/recovery_completion_receipt.json.
Original launch failure receipts remain failed; the recovery is not a forged original rc0.
