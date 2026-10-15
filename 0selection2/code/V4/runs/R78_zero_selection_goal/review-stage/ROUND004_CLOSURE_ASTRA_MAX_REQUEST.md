# Round004 Closure And Cumulative Research Conclusion

One bounded research-only criticism, gpt-6-astra / max. Same-family/provisional.
Return at most650 words. NO new method, fifth variant, optimizer retry, extra
statistics pipeline, experiment proposal, code review or implementation advice.
The user explicitly requests closing this finite route and waiting for the
already-running NSS locked-model TRAIN/VAL handoff. Overall R78 stays active;
old broad app goal remains paused, not complete/blocked/resumed.

## Read/Action Limits

This request is self-contained. If essential, only these exact additional
experiment reports may be read:

- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round004/RESULTS.md
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round004/all18_TRAIN_VAL_metrics.csv
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round004/native_method_counts.csv
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round004/decision_changes.csv
- /public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round004/decision_change_costs_by_category.csv

No project source, arrays, checkpoint files, TEST/LIB, root run results/status,
new paper search, new computations/experiments, GPU/solver/model execution,
Git, app-goal tools or edits. No subagents or nested reviewer. For a report-read
shell command, initialize conda shell hook and activate easynco first. The
caller saves your final response. Do not edit files yourself.

## Completed Round004

Frozen B ep28/u34776, full18,89 task-candidate entries. One independent matrix
per task,513 coefficients total. Inputs p=softmax(frozen B logits), temperature1.
g(p)=1-p+D p, D initialized0. Squared-error target is each method's TRAIN FP64
relative regret r_j divided by fixed task mean a_t, plus ||D||^2/(2N_t).
Choose argmin g, exact ties use original B logits then solver ID. This is not
a new winner probability model; reported CE is unchanged frozen-B CE.

The user-approved CPU fit completed1380 updates/task,24840 total, actual
batch1024, seed2, the fixed AdamW/cosine schedule and unit-prior penalty. All
heads were frozen before one complete VAL candidate. No TEST/LIB read, extra
budget, coefficient sweep or numerical-optimum certification.

| Split/rule | Top1 % | Raw macro mean cost | Actual macro regret % | Zero pairs |
|---|---:|---:|---:|---:|
| TRAIN B |71.176702703|11.365977443942|0.393011700|25|
| TRAIN actual fitted |68.119776364|11.378174190314|0.479358581|21|
| VAL B |70.894308943|11.395558343544|0.401326745|30|
| VAL actual fitted |67.947511054|11.407802464375|0.492726054|23|

All three VAL gates fail: native Top1 strictly up, raw mean cost nonincreased,
actual regret nonincreased, relative to B with original zero tolerance.
Selected whole18 rule remains B. The reports correctly separate B, ACTUAL
FITTED heads and fallback_B; unlike round003's selected-only TRAIN table, the
TRAIN deterioration above genuinely describes this fitted candidate.

Per-task reading: TRAIN Top1 falls18/18; VAL Top1 falls17/18. OVRPL is the only
VAL Top1 improvement (+0.92426pp) but both costs worsen. VAL relative regret
improves only on CVRP, whose raw mean cost and Top1 worsen. VAL raw mean cost
improves on TSP and ATSP, whose Top1 and relative regret worsen. No claim that
every task worsens every metric; no taskwise candidate assembly is permitted.

TRAIN changed182326 predictions: corrected59129, harmed97709, wrong-to-wrong
25488. VAL changed10024: corrected3276, harmed5342, wrong-to-wrong1406.
VAL exact cost-tie changes1; native winner labels remain unchanged.

Cost-category contributions below use the existing raw macro mean-cost metric:
(1/18) sum_tasks [(sum_category DeltaC)/N_task]. They are NOT an unqualified
sum of physical routing costs across different tasks.

- VAL corrected: -0.010819386199; harmed: +0.022790167595.
- VAL wrong-to-wrong: savings -0.000950753017, increases +0.001224092451,
  net +0.000273339434. Together net mean-cost change +0.012244120831.
- TRAIN corrected: -0.010792706643; harmed: +0.022687678936;
  wrong-to-wrong net +0.000301774079. Net +0.012196746372.

Zero-count reduction is again not a repair. TRAIN zero pairs25->21, but
winner mass in those pairs22458->24960. VAL zero pairs30->23, winner mass
2648->1441. Different method-set membership matters; counts are descriptive.

## Prior Research Context, Not New File Reads

R78 round001 natural-CE fixed offsets on old R76 moved Top1 slightly up but
worsened costs; its strict numerical convergence was incomplete. It did not
prove all calibrations impossible. Round002 fixed-scale cost-weighted squared
offsets were strongly bad on actual TRAIN, including14 converged tasks; stop
was practical, no more optimizer/tolerance budget.

R77 independently supplied a much better full-B baseline: old R76 VAL zero-
winner coverage20.3281% became3.7769%, alongside improved Top1 and regret.
Changes involved optimization/batch/LR/update regime and data amount, not a
single-variable batch cause. Do not assign R77 gains to the R78 interventions.

R78 round003 matched late continuation on B compared original objective with
monotone cost-risk replacement; all6 scheduled endpoints failed the native
Top1/raw-cost/relative-regret conjunction. NewRisk costs were worse than control
at all3 points despite lower CE. Its selected TRAIN/VAL rows were B fallback
and did NOT expose endpoint TRAIN fits. First VAL zero pairs30->27 reclassified
660 true winners after only5/6 picks of the3 newly selected pairs; it was not
660 correctly recovered selections. That finite route is already closed.

TSP input-role fact is owner-reported only: no explicit physical node0 start
marker. R57 source-level endpoint roles were not bound to historical deployment
and cost-column semantics. No role model, solver rerun, R58 restart or code
audit is warranted from that unresolved fact.

## Proposed Final Conclusion For Criticism

Close round004 as finite_frozen_score_cost_head_trial_closed_negative, retain B,
and do not manufacture a fifth nearby loss/head/coefficient variant. Actual
TRAIN and VAL both deteriorate, so a purely validation-only overfitting story
is insufficient. However, the reports do not identify whether approximation,
surrogate/decision mismatch, regularization, finite optimization, or score
information is the limiting cause. No missing-information or Bayes-optimal-B
claim, no universal impossibility claim for conditional cost learning.

The four tested interventions produced no accepted improvement under their
respective protocols. Native zero counts, winner coverage, CE or a theoretically
well-defined regression target cannot replace realized decision quality and
the frozen three-part gate. Do not claim the round004 fitting MSE improved:
that metric is not in the supplied reports, and unchanged frozen CE is not
a head-quality result.

Only next research action: wait for the already-planned NSS locked-model
TRAIN/VAL method-coverage handoff, with comparable instances/candidate pool/
relative-cost definitions identified by Sol. Do not request new solver runs,
new diagnostic pipelines or an audit. A change in method counts alone is not
new mechanism evidence; actual net quality and scope are needed, without
assigning any future advantage uniquely to architecture or a start-role feature.
NSS is still running3-4h by user report; no job monitoring here. Idle GPUs do
not create a reason to train. R78 remains active while waiting; no app-goal
operation, completion/blocking, STATUS.json edit or TEST/LIB-based next rule.

Please correct only factual/claim-boundary errors and state a clear final
research disposition. No new fifth experiment, alternatives menu, fresh paper
list, numerical verification requests or code-level prescriptions.
