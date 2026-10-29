# Return To The Original 18 Tasks

Base commit: `04866414831d25330b0d2e575e4b63c68b4196d0`.
The active user goal has four requirements: implement the post-R119 scientific
directions, investigate heterogeneous model routing, change real missed-winner
penalties, and build full18 Top-2 refinement. Preserve that full scope.

R120 completes its unchanged four-GPU train/VAL/locked TEST/LIB/report first.
CPU preparation and research can proceed meanwhile. Subsequent versions return
to the frozen new90k routing data and legal pools, not a ReLD-only subproblem.

Planned version ownership: R121 structural families (D1); R122 whole-model hard
decision constraints (D2 and missed-winner loss); R123 heterogeneous model MoE;
R124 candidate-pair-conditioned Top-2 refinement. Each is one principal method,
not a sweep. Exact budgets and scientific implementations remain to be locked
before GPU execution. The existing Astra report is frozen; new scientific work
goes to a new 20261030 directory. Sol owns code and execution.

Completion requires actual version results, one VAL-selected model per version,
TEST/LIB and portable PDF/CSV delivery, with real recovery versus false picks
and full18 cost metrics. A plan, CPU unit test or a live process is not success.
