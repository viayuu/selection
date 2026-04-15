easynco_v3_bridge has been trimmed to keep the current useful pieces easy to find.

Current active files:
- `config.py`: shared bridge configuration.
- `RUNBOOK.md`: high-level workflow notes.
- `scripts/build_label_jobs_v4.py`: build the v4 label job list.
- `scripts/run_label_queue.py`: execute queued label jobs.
- `scripts/generate_mvrp_diverse_v4.py`: generate the diverse MVRP train set.
- `scripts/generate_atsp_mvrp_valtest_v4.py`: generate ATSP/MVRP val and test splits.
- `scripts/export_nss_like_final_full.py`: export the final NSS-like package.
- `scripts/monitor_v4.py`, `scripts/chat_status_v4.py`, `scripts/watchdog_v4.py`: v4 monitoring helpers.
- `manifests/label_jobs_v4.json`: the retained full v4 job manifest.

Working directories:
- `logs/`: empty by default; future run logs can go here.
- `exports/`: empty by default; future exported packages can go here.

Archived historical material:
- `archive/commands/generated_commands/`: auto-generated shell launchers from old runs.
- `archive/logs/run_logs/`: all old per-scale logs, tmux logs, and monitor snapshots.
- `archive/manifests/`: old v3 manifests and split v4 manifests.
- `archive/scripts/`: v3-only or superseded helper scripts.
- `archive/docs/`: stale status documents kept only for reference.

Cleanup rule used here:
- keep the files needed to regenerate the current v4 data/labels/export flow;
- archive bulky intermediate artifacts instead of deleting them.
