# R74 Progress Snapshot, 2026-10-13

This publication records unfinished work at the user's explicit request. It is
not a completed R74 experiment, a new performance result, or permission to
resume the paused goal. The latest pause record takes precedence over older
launch-time status files. Process observations are timestamped historical
observations, not a fresh assertion that the trainer has reached a stopped state.

## Included Progress

- Implementation and launch scripts for adaptive incomplete acquisition,
  training, conditional delivery, analysis and in-progress recovery.
- The adopted plan, protocol, input/source identities, execution amendments,
  failure and handoff records, original source snapshots and completed checks.
- Complete feature-generation metadata and the partial successful-update log.
- Review/planning records and the persistent goal/reviewer state.
- `PAUSED.md` and `PAUSE_STATE.json`, including the durable checkpoint boundary.

All180000 TRAIN and18000 VAL feature rows were generated. The last logged
training progress is update91 of9600 planned updates, in epoch1/global41.
The durable local checkpoint and its independent backup contain update60;
updates61-91 are logged but not durably checkpointed. No complete R74 policy
validation, checkpoint selection lock, TEST or result analysis has completed.

The goal is paused. Owned automatic execution/delivery controllers were stopped.
At the last observation, the trainer was in uninterruptible kernel wait with
SIGSTOP pending, not confirmed fully stopped. Its CUDA resources were retained.
Continuation8 edits were interrupted before syntax validation or a resumed
launch. They are preserved as work in progress, not a verified recovery path.

## Assets Retained Locally

Git contains code and experiment evidence, not the complete executable training
state. Keep the following assets together with this snapshot:

- `B_adaptive_seed2/initial.pt`, `B_adaptive_seed2/last.pt` and
  `pause_checkpoint_update000060.pt`, plus frozen selector/probe weights.
- Persistent `feature_cache/`, continuation feature caches and `probe_states/`.
- Raw W&B files, large per-batch generation logs and the full cache-copy
  manifest. These rebuildable or large artifacts are not included in Git.
- The node-local `/tmp/shiys_r74_feature_cache_continuation8` mirror, if still
  present. Its39688996290 bytes are ephemeral; original persistent caches remain
  authoritative. The final `continuation8/cache_mirror.log` entry records its
  completion, while the periodic progress JSON is stale. Training had not been
  switched to the mirror.

No new model check, solver run, training, VAL or TEST was started to publish this
snapshot. Publication checks are limited to commit scope, whitespace, artifact
size and obvious credential patterns. Existing unrelated worktree changes and
staged historical cleanup are left untouched and outside this commit.

R58 and the eight legacy B-semantics discrepancies remain deferred/unfixed.
Explicit user instructions are required before any goal or R74 resumption.
