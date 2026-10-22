# R97 Launch And Progress

One frozen public TabICL v2 bank, not another gradient-trained selector. Full
81K natural TRAIN context/task; seed2; one preset estimator; all18 frozen pools.
See PLAN.md and protocol_lock.json. Runtime packages are isolated under
runtime/python; easynco's existing packages and all healthy R94/R95 jobs remain
unchanged. Public checkpoint SHA256 is recorded in the protocol.

## Queue

Target only gpu05/job1475 allocation-owned container, slot4090_1:
`GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`. Wait for R94B's complete
train/TEST/timing/LIB/PDF rc0 delivery, then require actual idle GPU/CPU/memory
and the shared GPU mutex. No ordinary-SSH training, new allocation, or sibling
restart. CPU4/practical RAM9GiB; full context stages once to
`/tmp/shiys_R97_tabular_context`. W&B is enabled, explicitly offline.

The queued command (one attempt only, not a command to replay):

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -u -m code.V4.r97_launch queue \
  --spec code/V4/runs/R97_launch/specs/A.json \
  --main-ready-evidence /public/home/shiys/0selection2/code/V4/runs/R97_tabular_in_context/main_ready.json \
  --runner-tests-passed
```

Queue tmux: `r97_queue_A`. Once launched: `r97_A`.
Queue/controller/worker receipts: `code/V4/runs/R97_launch/A/`.
Model phase: `code/V4/runs/R97_tabular_in_context/status.json`.
Per-stage logs: `code/V4/runs/R97_tabular_in_context/A/`.

## Execution

First, full81K TSP context and32 fixed VAL input rows, with NO VAL labels or
accuracy calculation. Passing CPU import/tiny synthetic forward is not this
capacity witness. CPU/disk fallback and math-attention fallback are forbidden;
guard only the new process's memory/time. Preserve a failure without an automatic
retry or smaller context.

Then fixed full18 VAL, one context-bank lock, TEST input provenance binding,
full18 TEST and native TSPLIB/CVRPLIB. Labels are used after inference for
evaluation only. No task-wise model/setting selection or TEST tuning. CPU PDF
reporting follows GPU rc0. Every prediction re-encodes the historical context;
the report discloses cached-statistics timing exclusions rather than claiming
end-to-end query latency.

`r97_goal_watch` observes completion quietly and publishes one curated GitHub
commit after the complete version. It neither launches nor retries training.
Code/config/source hashes are frozen by the launch plan before queuing. Do not
edit r97 runtime sources after readiness. Final PDFs are primary delivery;
weights, context arrays, raw predictions and W&B storage are not committed.
