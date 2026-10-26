# Resource Entry Attempts

Before the input-only memory witness, two resource-entry invocations failed.
Neither reached model construction, CUDA execution, training, or a solver.

- The platform ai_docker wrapper split an inline bash -lc activation string:
  `conda.sh: line 1: source: filename argument required`. Replaced the inline
  string with the on-disk run_v4_r109_witness.sh entry point.
- The on-disk entry omitted the allocation identifier in the container:
  `Worker must use easynco in the original allocation-owned container`.
  Added explicit SLURM_JOB_ID=1465 and the original user identity, matching
  the established production container launcher. No resource gate was removed.

Both failed before started.json could be written. No formal model process was
started, resumed, stopped, or replaced by these corrections.
