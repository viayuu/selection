# Multidomain selector

Base commit: `04866414831d25330b0d2e575e4b63c68b4196d0`.

Target: extend the complete routing selector to MIS, JSSP, FJSP, BPP, MKP,
and SMTWTP, with real typed inputs and separate deployment actions. Preserve
the frozen routing release and accepted nonrouting labels. Complete missing
nonrouting data in a new version rather than editing historical labels.

Inputs: raw instances, frozen split membership, legal deployment metadata,
and TRAIN supervision. Outputs: a prepared data manifest; runnable model,
objective, training/evaluation entry points; one locked checkpoint per formal
experiment; native-objective metrics, predictions, and downloadable PDFs.

Success: the full declared task pool trains and evaluates through the actual
entry point. Incomplete pools and failed/deferred candidates remain explicit;
an early three-task spine is not a complete 24-task result. Scientific success
is measured against TRAIN-fitted fixed methods and conditional priors on the
same pool, with ties and objective directions retained.

Use the two authorized RTX3090 and two RTX4090 through their existing
allocations. Preserve existing workers. Astra owns literature and scientific
recommendations; Sol owns implementation, protocol, execution and reporting.
