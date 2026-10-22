# R100 Pairwise Binary Frozen TabICL

One fixed full18 bank, NEW90k seed2, TRAIN81000/VAL4500/TEST4500 per problem.
No SGD, fine-tuning, TRAIN self-accuracy, epochs or batch1024 interpretation.
Public R97 classifier weights/runtime and immutable TRAIN X/constant-column
drop are reused. No R98 regression modules or results are modified.

For every legal method pair, include every strict original FP64 TRAIN cost
comparison, even if a third method is full-pool winner. Exact ties are excluded
for that pair alone. Labels are 0 first wins, 1 second wins. All TRAIN base IDs
are checked BEFORE pair filtering; any query overlap is rejected. No outside
TRAIN outcomes enter context, and no current query outcomes enter forward.

Each nonconstant pair is fit independently with one binary classifier resident
at a time. Exact one-class contexts alone give a deterministic result; no
strict rows give .5. Record all support/tie counts and shortcuts. Keep the
original n_estimators1/norm-none/batch1/AMP/seed2/kv_cache=False/GPU-only/Flash-only
parameters; binary2 fits the public max_classes10 without changing it.

P[i,j] means P(i beats j conditional on a strict TRAIN relation), with
P[j,i]=1-P[i,j]. Fixed decision: argmax mean opponent win probability
(expected-rank/Borda), ascending stable global-ID exact ties. This is NOT
full-pool winner probability, coupling or a new cost-sensitive claim. No VAL
aggregator grid, calibration or heuristic threshold. Formal log-softmax of
mean-win is for CE/report interoperability only, never decision selection.

CPU prepare binds source/context/strict-label/public-weight identities and VAL
input fingerprints before GPU. Run full18 VAL, lock one unchanged bank, bind
TEST inputs, run full18 TEST, then native LIB. TEST/LIB never select settings.
Common launcher/main owns GPU4090_1 and start/spec. No implicit retry.

CLI: python -m code.V4.r100_experiment prepare|run|report, with --root,
--context-root, --feature-root, --data-root, --prepared-root, --protocol-lock,
--device and --wandb-mode offline. Staged R97 context paths may differ but
must have identical metadata/artifact content hashes. Report is CPU-only.

Primary final output RESULTS.pdf; auxiliary CSVs include all18 macro/SBS/
Oracle/actual-regret, paired selected/native-winner distributions (Top1 titles),
NSS four-dataset comparison, fixed36 true recovery/loss/false cost vs R89B/R84A,
pair supports and timings. Runtime includes K(K-1)/2 contexts with explicit
constant shortcuts. Cached TEST inputs do not measure fresh graph-statistics
latency. These are historical-development-exposed, not wholly blind datasets.
