#!/bin/bash
# Example script to convert TSPLIB to DIFUSCO format and run evaluation

# Step 1: Convert TSPLIB to DIFUSCO format
# ===========================================

# Option 1: Convert a single TSPLIB file (without optimal tour)
python convert_tsplib_to_difusco.py \
    --input /path/to/tsplib/bier127.tsp \
    --output data/bier127_difusco.txt \
    --normalize

# Option 2: Convert with LKH solver (requires LKH installed)
python convert_tsplib_to_difusco.py \
    --input /path/to/tsplib/bier127.tsp \
    --output data/bier127_difusco_lkh.txt \
    --normalize \
    --solver lkh \
    --lkh_path LKH-3.0.6/LKH

# Option 3: Convert entire directory of TSPLIB files
python convert_tsplib_to_difusco.py \
    --input /path/to/tsplib/ \
    --output data/tsplib_all.txt \
    --normalize

# Option 4: Convert with Concorde solver (requires pyconcorde installed)
python convert_tsplib_to_difusco.py \
    --input /path/to/tsplib/ \
    --output data/tsplib_concorde.txt \
    --normalize \
    --solver concorde


# Step 2: Run DIFUSCO evaluation on converted data
# ================================================

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python difusco/train.py \
  --task tsp \
  --wandb_logger_name "tsplib_eval" \
  --diffusion_type categorical \
  --do_test \
  --storage_path . \
  --training_split data/tsp50_train_concorde.txt \
  --validation_split data/tsp50_test_concorde.txt \
  --test_split data/bier127_difusco.txt \
  --batch_size 1 \
  --inference_diffusion_steps 50 \
  --parallel_sampling 10 \
  --two_opt_iterations 1000 \
  --ckpt_path /path/to/checkpoint.ckpt \
  --resume_weight_only