# Deployment Status

**Date**: 2026-04-14

## Implemented

The external V3 bridge is now in place:

- [config.py](/public/home/zhoucl/shiys/easynco_v3_bridge/config.py)
- [generate_v3_datasets.py](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/generate_v3_datasets.py)
- [prepare_v3_checkpoints.py](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/prepare_v3_checkpoints.py)
- [build_label_jobs.py](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py)
- [run_label_queue.py](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py)
- [status_v3.py](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/status_v3.py)
- [launch_v3_tmux.sh](/public/home/zhoucl/shiys/easynco_v3_bridge/scripts/launch_v3_tmux.sh)
- [RUNBOOK.md](/public/home/zhoucl/shiys/easynco_v3_bridge/RUNBOOK.md)

## Data Ready

The following datasets have already been prepared under `EasyNCO/data/datasets/offline_init_v3/`:

- copied `NSS` train datasets:
  - [manifest.json](/public/home/zhoucl/shiys/EasyNCO/data/datasets/offline_init_v3/nss_copy/manifest.json)
- full `ATSP` 10k shards:
  - [manifest.json](/public/home/zhoucl/shiys/EasyNCO/data/datasets/offline_init_v3/atsp/manifest.json)
- full non-base `MVRP` variant shards:
  - [manifest.json](/public/home/zhoucl/shiys/EasyNCO/data/datasets/offline_init_v3/mvrp/manifest.json)

## Real Validation Already Passed

`MVRP` init-only label path:

- generated `OVRP` shard via `MVRPGenerator`
- ran `mtpomo` through `eval.py`
- exported:
  - [instance_results.jsonl](/public/home/zhoucl/shiys/EasyNCO/results/label_v3_verify/mtpomo_ovrp_scale_50/instance_results.jsonl)

`ATSP` init-only label path:

- generated `ATSP` shard via `ATSPGenerator`
- ran `matnet` through `eval.py`
- required one extra override for `scale=20`:
  - `settings.env.pomo_size=20`
  - `settings.module.initialization_params.pomo_size=20`
- exported:
  - [instance_results.jsonl](/public/home/zhoucl/shiys/EasyNCO/results/label_v3_verify/matnet_atsp_scale_20_fix/instance_results.jsonl)

## Currently Running

One real `NSS` init-only rerun is already in progress on the copied dataset:

- `dact @ TSPtrain_v3copy`

Active log:

- [eval.log](/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/dact_tsp/TSPtrain_v3copy/eval.log)

Result directory:

- [TSPtrain_v3copy](/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/dact_tsp/TSPtrain_v3copy)

## Generated Command Collections

- [run_nss_labels.sh](/public/home/zhoucl/shiys/easynco_v3_bridge/commands/run_nss_labels.sh)
- [run_atsp_labels.sh](/public/home/zhoucl/shiys/easynco_v3_bridge/commands/run_atsp_labels.sh)
- [run_mvrp_labels.sh](/public/home/zhoucl/shiys/easynco_v3_bridge/commands/run_mvrp_labels.sh)
- [run_all_labels.sh](/public/home/zhoucl/shiys/easynco_v3_bridge/commands/run_all_labels.sh)

## Recommended Next Launch Order

After the current `dact @ TSPtrain_v3copy` run finishes:

1. launch the full `NSS` queue
2. launch the `ATSP` queue
3. launch the `MVRP` queue

Commands:

```bash
source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh
conda activate easynco_zhoucl
export PYTHONPATH=/public/home/zhoucl/shiys

python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups nss
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups atsp
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups mvrp
```
