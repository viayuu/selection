# Necessary Round005 Checks

Two synthetic unittest cases passed in0.003s:

1. Fixed FP32 probability averaging preserves masks and resolves exact
   probability ties by ascending original global solver ID.
2. Descriptive mixture CE uses normalized mixture probabilities.

Native data assertions only:3895 distinct source rows/task, FP32 final logits,
FP64 original costs, source-row alignment, pool IDs/names, native winners,
nodes and valid masks agree. B native row order is preserved for reductions.
The sole candidate includes all70110 rows and uses processing chunks1024;
these are not optimizer minibatches or training updates.

One startup metadata Path-type error was fixed before any VAL scores were
read or candidate evaluated. The original traceback is retained in
preparation_failure.log. The subsequent sole candidate run exited0.
No repeated candidate, scan, model/solver/GPU/API pass, TEST/LIB, benchmark,
precision certificate, review loop or new training was performed.
