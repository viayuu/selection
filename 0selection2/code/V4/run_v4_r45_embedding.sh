#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
# No arguments: plan only. --stage all generates and assembles vectors, not training.
python -u -m code.V4.solver_code_embeddings \
  --corpus code/V4/runs/R45_solver_code_embeddings/corpus_atlas/corpus.json \
  --cache-dir code/V4/runs/R45_solver_code_embeddings/embedding_cache_atlas \
  --output code/V4/runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt "$@"
