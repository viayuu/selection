# R72 Fixed Heterogeneous Consensus

The approved plan is `runs/R72_heterogeneous_consensus/approved_plan.md`.
One new B contains exactly complete R60B39, R67B36/global76 and R70B23/global63
on every query. All original states, buffers, constructors and input paths are
preserved in one offline composite checkpoint. All parameters are frozen.

Returned FP32 logits are aligned by ascending global solver ID. Legal doubled
midranks are `2*strictly_higher + equal_including_self + 1`. The integer sum
over all three members is minimized; final ties use the smallest global ID.
`-Q/6` defines reporting-only Borda-score CE, not calibrated probabilities.
Labels never enter member forwards or aggregation. No density NLL applies.

Run the one synthetic check with `bash code/V4/run_v4_r72.sh cpu-check`, then
the one authorized original full18 validation with
`bash code/V4/run_v4_r72.sh validate` in owned tmux `r72_validate`.
The launcher uses allocation1465/gpu03 and only the assigned UUID, threads1,
the existing python_compat bridge and offline selector W&B project.

There are zero new TRAIN epochs, updates, TRAIN forwards and solver executions.
Validation has 18,000 native rows, batch128/104-row tails, 144 query batches,
432 member forwards and 54,000 member-row forwards. No standalone A or component
evaluation is authorized. Official B requires strictly more than8885 correct;
otherwise fixed R60B39 A remains. This phase does not freeze or execute TEST.
Any actual failure is preserved without automatic retry or scientific changes.

Weights, raw aligned scores/NPZ, W&B and caches remain local. Historical training
is separate from zero new training, with shared R67 stage1 counted once. This is
not a matched-capacity/compute comparison or proof of independent member errors.
R66's weak ensemble result remains counterevidence. R58 and eight legacy B
semantic issues remain deferred/unfixed. Goal70 remains active/unachieved;
the original future TEST threshold is12600/18000. Parent owns review and Git.
