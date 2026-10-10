# R54 execution

Repository: `/public/home/shiys/0selection2`.
Environment: `easynco`. Seed: 2. Existing allocation: 1465, node gpu03.
Only the previously idle RTX3090 is used:
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` (physical index 1).
The other 3090 and the 4090 are not used or stopped.

After inventory and a user approval for full route replay:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -m unittest code.V4.test_r54 code.V4.test_r53
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G \
  env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  python -m code.V4.r54_routes --stage precheck
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G \
  env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=offline \
  bash code/V4/run_v4_r54.sh routes --allow-route-generation
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G \
  env CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=offline \
  python -m code.V4.r54_experiment --stage all
```

The native route precheck and route replay run in tmux sessions
`r54_routes_precheck` and `r54_routes`. Training and the locked-checkpoint test run
in `r54_train`. Logs are retained locally in this run directory; W&B is offline
under project `selector` and entity
`yjkds-southern-university-of-science-technology`.

No selector from a prior Rxx run initializes R54. The frozen R45A predictions are
reused as the unchanged first-stage ranking. R53A initialization and sample-plan
hashes are compared only for matching, not loaded as trained model parameters.
