# R108 Focused Reviewer Handoff

- Own only code/V4/test_r108_experiment.py and the three R108 preparation
  review-stage files. Production runner belongs to main; model/objective/prior
  and admin/report sources belong to other workers. Never edit R107.
- Round01 is a bounded engineering/math review, not a four-round audit,
  efficacy gate, real-prior audit, job authorization or new GPU probe.
- Final focused engineering review READY8/10; no open blocking findings or
  warnings, no efficacy certification. Main fixed both findings: private zero consistency callback
  (r108_experiment.py:310) and describe eval256 (:465). Objective already ignored
  the consistency argument; the earlier mismatch was callback/logging only.
- Final reviewer rerun: original five runner CPU test methods all PASS2.718s,
  no expanded checks. Earlier four-PASS/three-failing-subcase result2.761s is
  superseded by this repair verification. Main reports57 combined model/
  objective/report/admin CPU tests PASS; reviewer has not rerun those suites.
- Round01 complete. No additional review round, new probes or outstanding
  source corrections requested. Launch configuration/resource decisions remain main's.
  Final READY confirmation reread only the two repaired lines; no new tests.
- Passed complete fresh state including frozen priors, synthetic trained
  checkpoint strict restore with prior I/O/installation forbidden, stateless
  single-forward objectives, micro512/1024 row weighting, query-label isolation,
  true N including depot and unknown-N task fallback. Only eight-row synthetic
  fixtures were read, never real TRAIN arrays/checkpoints or real priors.json.
- Both arms preserve1,842,953 original R88B trainables and common forward/init;
  expect active_parameter_matched=False. CE has a common-shift gauge, while
  ratio loss identifies the absolute scalar origin. Never claim equal active
  gradient directions or equal gradient scales.
- B remains exactly0.5*softplus(-t_y)+0.5*sum pi_m*softplus(t_m), including y
  in the negative term. A is raw s CE, not maximin CE. No entropy/pair/risk/
  RDrop term or extra dropout forward; same ideal Bayes argmax, no efficacy
  guarantee. Final maximin and natural logical1024/budget57600 remain fixed.
- Main reports existing A4090 witness512/1024 PASS, allocated13,367MiB,
  reserved17,366MiB. This is reported evidence only. Reviewer ran no GPU probe.
- One accidental initial read-only git diff --stat occurred, no writes and no
  further Git. Bare conda activation initially failed; shell hook fixed this
  before any Python tests. Production sources were not edited.
