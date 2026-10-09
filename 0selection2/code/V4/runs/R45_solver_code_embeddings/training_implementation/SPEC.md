# R45 training and evaluation

Base commit: `307a9b6dfda6797dc782550bca1017d3e8b01421`.

Use the locked, real MongoDB Atlas `voyage-code-4` embeddings already generated.
Train three seed-2, from-scratch R43A selectors: handcrafted, code, and one fixed
derangement of code ownership. Keep the data, native winners, candidate order,
pairwise objective, maximin decisions, R-Drop, and validation selection rules.

Run B/C concurrently on the user's existing two-3090 Slurm allocation, then run
A on the first available device. Read test only after all three validation-best
checkpoints are locked. Save complete predictions, curves, and paired decisions.
Do not call the external API, alter historical runs, or add model/loss searches.

Acceptance: full-batch CUDA witnesses, focused unit tests, all three completed
training records, locked-checkpoint test results, and replayed prediction metrics.
