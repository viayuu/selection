# EasyNCO V3 Bridge

This directory contains the external scripts for the V3 experiment bridge.

## Scope

These scripts follow the V3 plan:

- do not modify `EasyNCO`
- copy `NSS` train datasets for `TSP` and base `CVRP`
- generate `ATSP` and non-base `MVRP` variant datasets by calling existing `EasyNCO` generators
- rerun labels with `EasyNCO`
- keep labels `init-only`

## Directory Layout

- `config.py`: fixed paths, scales, method pools
- `scripts/generate_v3_datasets.py`: copy/generate datasets
- `scripts/prepare_v3_checkpoints.py`: unpack `mtpomo` and `mvmoe` checkpoints
- `scripts/build_label_jobs.py`: render label jobs and shell command collections
- `scripts/run_label_queue.py`: run label jobs sequentially
- `scripts/status_v3.py`: show completion counts
- `scripts/launch_v3_tmux.sh`: start background tmux sessions

## Environment

These commands assume:

- workspace root: `/public/home/zhoucl/shiys`
- `EasyNCO` root: `/public/home/zhoucl/shiys/EasyNCO`
- conda env: `easynco_zhoucl`
- `PYTHONPATH=/public/home/zhoucl/shiys`

## One-Time Setup

```bash
cd /public/home/zhoucl/shiys
source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh
conda activate easynco_zhoucl
export PYTHONPATH=/public/home/zhoucl/shiys
```

## 1. Prepare Checkpoints

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/prepare_v3_checkpoints.py
```

## 2. Prepare Datasets

Copy `NSS` train datasets:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/generate_v3_datasets.py copy-nss
```

Generate `ATSP` shards:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/generate_v3_datasets.py generate-atsp
```

Generate non-base `MVRP` variant shards:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/generate_v3_datasets.py generate-mvrp --mode 1
```

Run all required V3 data preparation in one shot:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/generate_v3_datasets.py all --mode 1
```

## 3. Build Command Collections

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/build_label_jobs.py
```

This writes:

- `easynco_v3_bridge/manifests/label_jobs.json`
- `easynco_v3_bridge/commands/run_nss_labels.sh`
- `easynco_v3_bridge/commands/run_atsp_labels.sh`
- `easynco_v3_bridge/commands/run_mvrp_labels.sh`
- `easynco_v3_bridge/commands/run_all_labels.sh`

## 4. Launch Labels

Run only `TSP/CVRP` labels:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups nss
```

Run only `ATSP` labels:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups atsp
```

Run only `MVRP` labels:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups mvrp
```

Preview commands without running:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --groups atsp --dry-run
```

Run one specific job:

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --name matnet_atsp_scale_50
```

## 5. Background Deployment With tmux

Start dataset generation and `TSP/CVRP` label jobs immediately:

```bash
bash /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/launch_v3_tmux.sh deploy-now
```

Later, after data generation finishes, start:

```bash
bash /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/launch_v3_tmux.sh atsp-labels
bash /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/launch_v3_tmux.sh mvrp-labels
```

Inspect sessions:

```bash
tmux ls
tmux attach -t v3_prepare_data
tmux attach -t v3_nss_labels
```

## 6. Status

```bash
python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/status_v3.py
```

## Notes

- Base `CVRP` is handled by copied `NSS` data, so the `MVRP` generator only covers the 15 non-base variants here.
- `PCTSP` is not part of the first selector release under strict `init-only`, so dataset generation support is optional and no label jobs are included for it by default.
