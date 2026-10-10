# R51 execution

Only GPU1 on the existing gpu03 allocation is used. The other RTX3090 and the RTX4090 are untouched.
All Python runs use the existing easynco environment. No new data, new complete cost labels, or test reads.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
tmux new-session -d -s r51_behavior_probe \
  'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G env R51_GPU_UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh all'
```

Separate stages supported by the wrapper: `tests`, `prepare`, `cache`, `train`, `timing`, `analysis`.
Preparation and training refuse to overwrite already locked artifacts. Do not rerun `all` on existing results.
CPU analysis and focused unit checks do not require the GPU allocation:

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh tests
bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh analysis
```

The 25% maximum total-latency overhead screen is fixed before feature extraction/training. Both predictive
criteria must also pass on the internal and original validation sets. No extra lengths, seeds or classifier search.

After training, the timing path was changed to stop collecting unused diagnostic counters during the
chosen solver's completion. Original model/environment decoding remains unchanged. Caches, training,
selected checkpoints and predictions were not regenerated. Timing was repeated with this deployment path:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
tmux new-session -d -s r51_timing \
  'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G env R51_GPU_UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh timing'
bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh analysis
bash /public/home/shiys/0selection2/code/V4/run_v4_r51.sh tests
```
