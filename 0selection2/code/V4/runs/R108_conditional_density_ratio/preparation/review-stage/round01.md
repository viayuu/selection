# R108 Focused Engineering And Math Review, Round 01

2026-10-24 Asia/Shanghai. Bounded runner/model/objective integration review,
not an efficacy audit. Only test_r108_experiment.py and these three review
files are reviewer-owned. No production source or R107 source was edited.

## Findings By Priority

**Open blocking findings: none. Open warnings: none.**

**P2, resolved and retested:** main bound the private engine's consistency
callback to the R108 zero callback at `code/V4/r108_experiment.py:310`.
The same runner contract now verifies zero at updates1/4320/57600, exact
callback identity, and the unchanged shared R84 ramp. Previously only the
passed/logged consistency weight was stale: ConditionalPriorObjective already
ignored it, so there was no demonstrated KL/R-Drop contamination of the loss.

**P3, resolved and retested:** main changed describe to eval256 at
`code/V4/r108_experiment.py:465`. Both arms now match CLI/protocol eval256.
No production source was edited by the reviewer.

## Bounded Evidence

Reviewer command, after activating easynco:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest code.V4.test_r108_experiment -v`

Final rerun after main's two fixes: **all five test methods PASS in2.718s**.
Only the original runner suite was rerun, with no expanded checks. The earlier
2.761s run had four passing methods and three failing subcases in the fifth,
exactly the now-resolved P2/P3 items. No unrelated numerical, initialization
or restore failure occurred.

- The complete original 1,842,953 seed2 trainables match both arms, including
  exact full-state equality after the same geometry and frozen prior buffers.
  Tests explicitly expect active_parameter_matched=False and identical forward
  models; they do not claim identical gradient activation.
- A synthetic trained tensor change survives the checkpoint factory for both
  arms. JSON reading and prior installation are forbidden during restoration.
  Missing trainables/buffers and legacy objective_state are rejected. Tests
  use generated eight-row fixtures, never real arrays or real checkpoints.
- Stateless classification objectives pass exactly one selector call per
  microbatch and one optimizer step per logical1024. Two row-weighted micro512
  calls match one micro1024 call for both arms. Original RDrop loss metadata
  is rejected and path metadata is absent.
- Input-only forwarding is unchanged by poisoned query supervision. TSP/CVRP
  true N follows node_mask, including the actual CVRP depot, not customer-only
  N or padded width. An unseen N uses the task prior. Final original maximin
  and prior-adjusted utility have the same top decision.
- Main reports57 combined model/objective/report/admin CPU tests PASS. This is
  collaborator evidence, not an additional reviewer rerun or real-prior audit.

## Mathematical Boundary And Judgment

For eta_m=P(Y=m|x,g), the population B gradient in t_m is
0.5*((eta_m+pi_m)*sigmoid(t_m)-eta_m). Thus its ideal stationary logit is
log(eta_m/pi_m), and adding log pi gives log eta. The negative expectation
includes the actual true y; removing it changes this objective. A is ordinary
CE on s, not CE on the final maximin diagnostic logits. Both objectives use
one dropout realization, with no pair/risk/RDrop/entropy term.

CE permits a common additive shift and has zero data-gradient on its shared
final scalar bias. The ratio objective identifies its absolute origin.
Identical tensors/forward models therefore do not mean identical active
parameter directions, gradient scales or optimization. There is no new
Bayes decision information and no guaranteed finite-budget cost gain.

**Final engineering review: READY (8/10); no unresolved findings.**
The two actual findings are closed by the unchanged five-test CPU suite.
No further review round, expanded checks or hardware investigation is requested.
Main reports the existing A 4090 micro512/1024 witness PASS, peak allocated
13,367 MiB / reserved 17,366 MiB; reviewer did not run or independently audit
that witness. Caps remain512/1024/eval256, natural logical1024, budget57600.
This is final focused engineering approval, not efficacy certification or job
authorization. Launch configuration remains main-owned. Final confirmation
only reread the two repaired source lines; no additional tests were run.

Process deviation: one initial read-only git diff --stat was mistakenly run
before the explicit no-Git boundary was enforced. It did not modify Git or
workspace state; no further Git commands were used. Initial bare conda
activation also failed; subsequent commands source the shell hook and activate
easynco, and all Python tests ran in that environment. No GPU/job commands,
real data/checkpoint loading, secondary model or new literature were used.
