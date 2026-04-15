#!/bin/bash
# Test DACT on TSPLIB benchmark

# Step 1: Convert TSPLIB to DACT format
# ======================================

# Option 1: Convert a single TSPLIB instance
python convert_tsplib_to_dact.py \
    --input /path/to/tsplib/bier127.tsp \
    --output datasets/tsplib/bier127.pkl \
    --normalize

# Option 2: Convert entire TSPLIB directory (e.g., all instances with <=100 nodes)
python convert_tsplib_to_dact.py \
    --input /path/to/tsplib/ \
    --output datasets/tsplib/all_instances_dact.pkl \
    --normalize \
    --num_samples 100


# Step 2: Run DACT inference on converted data
# ============================================

cd /mnt/d/Study/3/neural-solver-selection/0methods/DACT

# Example 1: Test TSP-100 on TSPLIB (single instance, no data augment)
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem tsp \
    --graph_size 100 \
    --step_method 2_opt \
    --eval_only \
    --init_val_met greedy \
    --load_path 'pretrained/tsp100-epoch-195.pt' \
    --T_max 10000 \
    --val_size 1 \
    --val_dataset 'datasets/tsplib/bier127.pkl' \
    --val_m 1 \
    --no_saving \
    --no_tb

# Example 2: Test TSP-100 on TSPLIB (multiple instances, with 8x data augment)
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem tsp \
    --graph_size 100 \
    --step_method 2_opt \
    --eval_only \
    --init_val_met greedy \
    --load_path 'pretrained/tsp100-epoch-195.pt' \
    --T_max 10000 \
    --val_size 10 \
    --val_dataset 'datasets/tsplib/all_instances_dact.pkl' \
    --val_m 8 \
    --no_saving \
    --no_tb

# Example 3: Test TSP-50 on TSPLIB
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem tsp \
    --graph_size 50 \
    --step_method 2_opt \
    --eval_only \
    --init_val_met greedy \
    --load_path 'pretrained/tsp50-epoch-199.pt' \
    --T_max 10000 \
    --val_size 10 \
    --val_dataset 'datasets/tsplib/tsp50_instances_dact.pkl' \
    --val_m 8 \
    --no_saving \
    --no_tb

# Example 4: Test TSP-20 on TSPLIB (small instances)
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem tsp \
    --graph_size 20 \
    --step_method 2_opt \
    --eval_only \
    --init_val_met greedy \
    --load_path 'pretrained/tsp20-epoch-199.pt' \
    --T_max 10000 \
    --val_size 10 \
    --val_dataset 'datasets/tsplib/tsp20_instances_dact.pkl' \
    --val_m 8 \
    --no_saving \
    --no_tb