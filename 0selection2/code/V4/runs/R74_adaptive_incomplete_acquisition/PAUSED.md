# R74 Paused

User requested pause of the active goal and R74 on 2026-10-13.
The goal API now reports `paused`. No automatic continuation is authorized.

## Process State

- Training: gpu03, step1465.118, PID101584, tmux r74_continuation7 / %148.
- GPU: index1, GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d.
- STOP was sent through Slurm and directly to the owned trainer.
- Last observed state: D (uninterruptible kernel wait), shared pending signal
  0x40000 = SIGSTOP. A fully stopped T state has NOT been observed.
- No logged progress after update91 at 12:58:09UTC.
- The in-memory process and its CUDA context are retained, not terminated.
  GPU memory/resources have not been released. Other jobs were not touched.
- The execution-model session and conditional delivery observer were stopped.
  No controller remains authorized to launch training, VAL, TEST or analysis.

## Saved Progress

- Last logged successful update:91, epoch1/global41, of9600 planned updates.
- Native presentations:69888; successful leaf-view rows:373248; AMP retries:0.
- Durable checkpoint: `B_adaptive_seed2/last.pt`, update60 / next_step60.
- Preserved independent backup: `pause_checkpoint_update000060.pt`.
- Saved checkpoint contains model, ScheduleFree optimizer, scaler and RNG;
  averaged-x/eval mode,46080 native rows and245760 leaf-view rows.
- Updates61-91 are logged but NOT in that durable checkpoint. Do not represent
  update91 as a disk-restorable checkpoint or silently replay those31 updates.

All180000 TRAIN and18000 VAL feature rows are complete and retained.
No complete policy VAL epoch, checkpoint selection lock or new TEST has run.
No completed solver was executed to generate new costs or labels.

## Interrupted Runtime Work

The byte-preserving cache mirror completed on gpu03:
`/tmp/shiys_r74_feature_cache_continuation8`.
It contains3405 groups/69780 files/39688996290 bytes; copy took604.083s.
Evidence: final complete=true line in `continuation8/cache_mirror.log`.
The mirror is ephemeral; original persistent feature caches remain authoritative.

Training was NOT switched to this mirror. Continuation8 resume code was being
prepared when interrupted; no syntax receipt, checkpoint cut, resumed launch
or fresh delivery3 observer was completed. Those source edits remain preserved
but must not be treated as a verified recovery path.

## Resume

Await explicit user instructions. Preserve the live process until deciding
whether its in-memory state can safely continue. If it cannot, recover from the
saved update60 only with explicit accounting for the31 later logged updates.
Do not launch the unfinished continuation8 automatically.

Seven prior failed steps (3839s Slurm wall) and the previous unsaved successful
update (768 native rows/3584 leaf views) remain separately recorded.
R58 remains deferred; the native70 objective is unachieved, not completed.
Full machine-readable status: `PAUSE_STATE.json`.
