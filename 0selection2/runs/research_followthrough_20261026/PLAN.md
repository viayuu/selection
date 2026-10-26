# Persistent Research Followthrough, 2026-10-26

Invocation-capability repair is ready for main's controlled restart, NOT
started/signalled by its implementer. The user's whole18 continuous-improvement
goal remains ACTIVE; finite monitor delivery is not completion of that goal.
This fresh continuation is enabled; a stale native goal's paused state is not
consulted. Only a future explicit pause in this monitor's control.json applies.

## Exact CLI

All commands source conda and activate easynco through the wrapper. Main may
start this monitor independently of its CPU-only container font installation.
There is no GPU/LLM probe on start while the jobs are waiting/running. The new
queue's main approval is still required independently; monitor never grants it.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh start --main-approved
bash /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh status
```

The tmux session is research_followthrough_20261026. To run in a main-managed
existing tmux pane, use the same command with `run --main-approved` instead of
`start --main-approved`. Do not run two controllers; flock and process claims
refuse duplicates. No launch was performed for this delivery.

Main has updated queue_handoff.json to the actually launched
preparation/migration_3090_portable_inode_20261026 queue. After main stops the
old controller, restart in main's existing tmux pane with:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh run --main-approved --queue-handoff /public/home/shiys/0selection2/runs/research_followthrough_20261026/queue_handoff.json
```

Do not signal/start it as part of this repair. Existing needs_action repair
and R109 closeout ledgers/attempts are immutable, not reset or retried. R109
publication is handled separately by Kierkegaard; do not duplicate it.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
bash /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh pause
bash /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh resume
```

Pause wakes the controller and stops only its owned model CLI/supervisor, not
Mencius/Turing/main experiments. Partial model work is marked uncertain and
never blindly rerun on resume. Resume re-enables scheduling, not failed action
retries. SIGTERM stops only the controller; a restarted controller adopts live
owned workers. Inspect a stale tmux session/PID before manually replacing it.

## Read-Only Authorities

- R109: existing goal_progress + github_publication, same-lock saved artifacts.
  A real PDF-only error after rc0 train/TEST/timing/LIB permits one CPU report
  recovery call, with immutable old receipts and new linked recovery evidence
  under recoveries/R109. It never blocks queue GPU handover or edits sources.
- R112: already published commit 446ac3c046b8c3dfb5722b529a4356b1ead80a61.
  Consume authentic CPU recovery, full saved delivery and publish_receipt.
  Original failed PDF dispatch remains failed; no duplicate repair/publish.
- R113: ONLY main's current queue_handoff.json authority, now pointing to
  code/V4/runs/R113_executable_process/preparation/migration_3090_portable_inode_20261026/
  HANDOFF.md, controller.json, ready.json and finish_adapter.py, with
  ready.json main_approved=true. Current owner_check_sha256 is
  d3e840cf4945ecfcda330a46acb36c1e891a42f5776134e10c7ade0b1b529781.
  Both the handoff and HANDOFF.md record actual all18 finite training by
  A_scratch+C_transition on the two3090s, not just readiness. Never guess/switch
  paths or reuse superseded approvals. The head/gpu03 st_dev mismatch is
  ALREADY fixed by main, not a new monitor repair task.
  Wait for R109 UUID locks; A fresh, B/C resource resumes from20160 with
  optimizer/RNG retained. Disclose B2286/C1872 unsaved updates replayed.
  Ignore retired four-slot controller, old KeyboardInterrupt and stale status.
  The private adapter links old actual pretrain8640 and failed attempts without
  changing them. All three final VAL locks need full TEST/LIB/PDF + publication.
- R114: exact handed-off scheduler progress path; only an authoritative
  terminal complete/blocked/failed/interrupted result with completed analysis
  and complete_labels OR explicit partial_negative_closeout + publication.
  Native-contract partiality is a legitimate pilot conclusion, not an endless
  solver-repair request. Acquisition of demand/TW responses on 16 tasks x32
  TRAIN bases does NOT test downstream function learning or TSP/ATSP responses.

  Addendum on2026-10-26: v1 CVRP/ICAM failed at a CUDA-to-NumPy host-audit
  boundary, not at the solver's task capability. The other15 problems continue
  unchanged. A separately locked full-CVRP correction is queued on3090_1 after
  R113/B; exact scope is32 bases /96 endpoints /64 pairs /960 candidate labels.
  The old CPU-only publisher485283 was retired without touching GPU workers;
  its handoff is recorded in the tensor-boundary recovery directory. The new
  CPU-only `r114_recovery_finish wait --publish` owns one combined publication.
  `queue_handoff.json:r114_required_finish_dir` prevents synthesis of the old
  partial result before this real correction/combined closeout. The monitor
  accepts the publisher's shared receipt only when its delivery directory
  matches exactly. Original failures and per-problem execution locks remain
  visible; merged labels must not pretend to share a single execution lock.

Never use 4090/job1475/new allocations. No frozen backend/candidate/label edits,
new NN arms, loss/hyperparameter changes, TEST-based selection or fabricated
receipts. The host's configured provider/auth stays untouched.

## Finite Model Stages

The controller reads small receipts once every 300 seconds, with no periodic
status spam, GPU/W&B polls, raw-data audits, Feishu or Slack. events.jsonl and
stdout contain only deduplicated completion/failure/required-action messages.
Runtime state is confined to this directory. Full report-file snapshot hashes
are collected once at the final evidence gate, not every polling cycle.

1. Necessary execution/recovery only: gpt-6.1-sol/max; one call per distinct
   stable failure, at most four automatic recovery calls total. Existing live
   queue, finisher or claimed worker is inspected/adopted, never duplicated.
   Empty recovery_argv alone is not a blocker: one durably claimed, bounded
   bootstrap-wrapper repair is permitted only with authoritative positive proof
   of pre-entry/worker/CUDA failure and untouched model/source/checkpoints.
   New versioned wrapper/approval, immutable old failures, no self-granted
   main approval, no scientific changes or second attempt. This is within the
   existing failure claim, not a new retry of an old monitor action.
2. After all four scientific closeouts/publications, Sol/max authenticates real
   saved reports and writes evidence/evidence_pack.md, checkpoints.csv (7 whole
   models), metrics.csv (280 rows), verification.json, source_snapshot.json.
   Same-version VAL/TEST all18+ALL and both LIBs; native winner/SBS/Oracle/regret
   definitions, equal-problem ALL, negative/partial facts and hypothesis fidelity.
3. ONLY after that pack validates, gpt-6-astra/max researches full 20261025
   reports + fresh results and primary papers, with explicit live web search:
   `codex --search exec -m gpt-6-astra -c model_reasoning_effort="max" ...`.
   It writes REPORT.md/sources.csv/directions.csv in the new dated research
   directory. Literature/conceptual mechanism and broad finite next ideas only;
   no result/code review, code edits, rendering, training or Git.
4. Sol/max renders the Chinese REPORT.pdf and publishes ONE completed research
   version under /public/home/shiys/.git/codex-publication.lock, with exact-path
   staging and write-ahead publication receipt. No unrelated staged changes.

Sol execution/verification/publication explicitly uses `--sandbox
danger-full-access` at the repository cwd, consistent with main's permission.
Astra explicitly uses `--sandbox workspace-write` with BOTH CLI and subprocess
cwd restricted to docs/research/post_major_paradigm_results_20261026, created
before its supervisor starts. It can read repository inputs by absolute path,
not write them. No --add-dir; inherited writable_roots are cleared, /tmp and
$TMPDIR are excluded. Per-invocation sandbox_workspace_write.network_access=true
supports primary-paper access alongside --search. These keys were checked
against [official OpenAI documentation](https://learn.chatgpt.com/docs/config-file/config-reference)
after reading the installed CLI help. No auth/provider/config file is changed.
Astra writes only prose/CSVs, never code/result review/rendering/Git; Sol owns
those execution stages. The CLI, not Astra's shell, saves -o response.json.

Every call has `-c approval_policy="never"`, --json and -o response.json, a
durable claim, and a six-hour timeout. Model unavailable is explicit and halts
further calls, no fallback. Exact requested CLI flags are recorded; a model's
self-reported JSON is NOT independent provider routing proof. Failed, empty,
uncertain or no-progress calls are not automatically retried. An invalid final
evidence gate is latched for main intervention rather than repeatedly audited.
Do not delete action ledgers/claims to "retry" uncertain work.
request.json records the exact requested model/max; codex.json and exit.json
record actual argv/cwd, and provider_route_verified remains false. Exact model
unavailability halts future calls explicitly without a substitute or probe.

## Verification Without Services

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python -B -m unittest code.V4.test_research_completion_monitor -v
bash -n /public/home/shiys/0selection2/code/V4/run_research_completion_monitor.sh
```

Fixtures mock all subprocess/model/GPU/signal/Git actions. No model service
probe, controller launch, experiment run or commit belongs to these checks.

Verification for this capability repair is CPU-only mock fixtures and shell
syntax, with no actual model API/GPU/Git calls, no launch/signals, and no writes
to real queue_handoff/state, experiment code or existing action receipts.
Verified: 40 CPU mock tests passed; wrapper shell syntax passed. Runtime model
availability/provider routing is deliberately unprobed and not claimed verified.
