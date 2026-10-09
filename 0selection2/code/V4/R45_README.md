# R45 Solver Source Embeddings

R45 retains the R43A score-difference/maximin selector and replaces only its
34-dimensional handcrafted solver branch. No R44 adapter is used. This change
implements the experiment; it does not run training or a paid API request.

## Full Training And Evaluation

The locked real Atlas vectors are available as
`runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt`.
No API key or network is needed for training or inference.

On a node exposing the two authorized GPUs, run:

```bash
bash code/V4/run_v4_r45_campaign.sh
```

The campaign checks full batch128 CUDA updates on TSP, CVRP, ATSP and OVRPTW,
then discards these preflight models. B/C start from scratch concurrently;
A starts on the first freed GPU. All three use seed2, the locked common batch
schedule and R43A's objective. Model checkpoints include the fixed embeddings.
After validation selection finishes for all arms, the campaign locks the best
checkpoints, evaluates the full 18-problem test split once, and writes reports.

Outputs include `orchestration.log`, `gpu_metrics.csv`, per-arm `train.log`,
`history.json`, `best.pt`, `last.pt`, prediction files, local curves, and
`comparison.md`/`comparison.csv`. W&B is recorded offline as in R43.
The historical R43A test result is reused only after checking the data manifest.
This launch does not change the source corpus or make another embedding request.

## MongoDB Atlas: Ready To Embed

All 22 solver identities and all four role inputs are prepared in
`runs/R45_solver_code_embeddings/corpus_atlas/`. Missing semantic parameters
use local defaults; unrecoverable historical metadata is omitted. Historical
source revisions and checkpoint hashes identify old runs; they are not inputs
required by the embedding model and do not block this prepared corpus.

The provider is MongoDB Atlas, not the old Voyage dashboard:
`https://ai.mongodb.com/v1/embeddings`, model `voyage-code-4`, 1024 floats,
`input_type=null`, `truncation=false`. The model name was already `voyage-code-4`;
this update changes the gateway, rate limits, and input organization.

The dedicated launcher defaults to a local plan, without API calls or training:

```bash
bash code/V4/run_v4_r45_embedding.sh
```

After creating your API key, run from the project directory:

```bash
read -rsp 'MongoDB Atlas Model API key: ' VOYAGE_API_KEY
export VOYAGE_API_KEY
bash code/V4/run_v4_r45_embedding.sh --stage all --allow-upload --max-tokens 200000
```

Use a **Model API Key created in Atlas**, not an Atlas Administration API key
or a key from the old Voyage dashboard. Do not reuse the previous key merely
because the environment variable has the same name.

This sends source code to Atlas and assembles
`runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt`. It contains the
`[22,4,1024]` reference summaries **and all individual component vectors, masks,
variant/chunk indices and source provenance**. The model uses the complete
component bank, not the legacy reference averages. No training is started.

The current real-source plan has **80 unique complete views / 159923 tokens**.
All 104 deployment-role views fit intact; maximum length is **29754 tokens**.
The 200000 flag is a total uncached-input token ceiling, not a price or TPM cap.
Interrupted calls resume using `embedding_cache_atlas/`; locked bundles are not
overwritten. The old `embedding_cache/` and `corpus_assumed/` remain untouched.

### Semantic Views And Aggregation

Each Encoder / Decision / Inference / Config role is embedded as one input
whenever it fits the **32000-token** model context, including the context header.
Complete classes and functions are kept together; there is no fixed 4096-token
split. Only oversized views are split at class/method/statement boundaries,
with call-related components kept adjacent and selected call links included
as context. An oversized atomic statement raises a specific error rather than
being silently truncated or cut into arbitrary characters.

Every returned embedding remains in the cache and locked bundle, together with
its component symbols, original source ranges/hashes and variant-role mapping.
For genuine multi-chunk views, each role projects all chunks, learns attention
weights, and fuses the attended summary with a channelwise max summary. This
lets training emphasize different components and retain locally strong signals;
it does not guarantee a performance improvement. Single-chunk views bypass
this aggregator. Distinct deployment variants remain equally weighted after
projection; repeated deployments are deduplicated.

Rebuild the prepared corpus after changing the manifest or source:

```bash
conda activate easynco
HF_HUB_OFFLINE=1 python -m code.V4.solver_code_defaults
bash code/V4/run_v4_r45_embedding.sh
```

The vectors, component index map and masks are persistent checkpoint buffers.
Training recomputes projections; inference needs neither the API key nor network.
Future training defaults to the Atlas bundle. Legacy R45 bundles/checkpoints
and the original handcrafted model path remain loadable.

### HTTP 429

The generator now defaults to the user's Atlas limits: **2000 requests/minute**
and **8000000 tokens/minute**. Requests contain at most 16 texts by default and
are bounded by the provider's **320000 total tokens per request**. Per-text
context (32000), per-request tokens (320000), TPM and the total authorized upload
budget are separate limits. HTTP `Retry-After` is honored.
Repeated 429 errors show safe limit/account guidance without exposing error
bodies or your key. Explicit quota/payment failures are not blindly retried.

Check the organization and project Rate Limits for `voyage-code-4` in Atlas
AI Model APIs, plus Billing if the response reports a quota/payment restriction.
Use those RPM/TPM values with `--requests-per-minute` / `--tokens-per-minute` to
avoid unnecessarily slow generation. `--batch-size 1` can help diagnose a
request too large for a low token allowance; it does not increase account quota.
The `--max-tokens 200000` flag caps total uncached input, NOT tokens per minute.
Provider and preprocessing hashes prevent legacy Voyage chunks from being
reused as Atlas whole-view embeddings. No API key is stored in either cache.

Official references:
https://www.mongodb.com/docs/voyageai/api-reference/overview/ and
https://www.mongodb.com/docs/api/doc/atlas-embedding-and-reranking-api/operation/operation-createembedding .

Do not put the key into a source file, manifest or Git. Alternatively supply a
private `--api-key-file` outside the project, as described below. All original
strict-corpus instructions farther below refer to the historical preparation
mode, not a remaining requirement for this prepared default-value corpus.

## Historical Preparation Notes

The following section records the pre-Atlas 4096-token corpus. Use the Atlas
commands above for new embeddings; these old artifacts are retained for provenance.

### Accepted Defaults Update (2026-10-08)

The user's completed manifest remains the historical-evidence source of truth.
Missing representation parameters can now use explicitly assumed local defaults,
without changing recorded values, IDs, gaps or historical binding statuses.
`ready=true` for this experimental corpus means usable with assumptions, NOT
that the historical solver runs were reproduced. Native-version/RNG/checkpoint
history is not invented. GLOP remains the verified insertion-only implementation.

Prepare or inspect this offline alternative:

```bash
conda activate easynco
HF_HUB_OFFLINE=1 python -m code.V4.solver_code_defaults --legacy-chunks \
  --output code/V4/runs/R45_solver_code_embeddings/corpus_assumed
python -m code.V4.solver_code_embeddings --stage plan \
  --provider voyage --cache-dir code/V4/runs/R45_solver_code_embeddings/embedding_cache \
  --corpus code/V4/runs/R45_solver_code_embeddings/corpus_assumed/corpus.json
```

This writes `corpus_assumed/source_manifest.json`, the actual assumed values and
their source hashes in `assumed_defaults.json`/`assumed_defaults.md`, and the
accepted corpus. It does not overwrite the user's manifest or `corpus_reviewed`.
The corpus, future vector bundles, checkpoints and final reports retain the
unverified historical gaps. Rebuilding this derived manifest through the corpus
CLI requires `--allow-assumptions`; the original strict mode is unchanged.

Later, with a key and a separate upload budget authorization, the explicit
generation commands are:

```bash
python -m code.V4.solver_code_embeddings --stage embed --allow-upload \
  --provider voyage --cache-dir code/V4/runs/R45_solver_code_embeddings/embedding_cache \
  --corpus code/V4/runs/R45_solver_code_embeddings/corpus_assumed/corpus.json \
  --max-tokens 200000
python -m code.V4.solver_code_embeddings --stage assemble \
  --provider voyage --cache-dir code/V4/runs/R45_solver_code_embeddings/embedding_cache \
  --corpus code/V4/runs/R45_solver_code_embeddings/corpus_assumed/corpus.json \
  --output code/V4/runs/R45_solver_code_embeddings/solver_code_vectors_assumed.pt
```

The token limit is an example, not a price cap; inspect the new plan first.
Pass that vector path with `run_v4_r45.sh --embeddings ...` for future experiments.
No API generation or training was performed for this update. The previous
sections below describe the original strict corpus and its original snapshot.

## Decisions Fixed Before Implementation

| Decision | Choice | Reason |
|---|---|---|
| Base | commit `307a9b6dfda6797dc782550bca1017d3e8b01421`, R43A | User's baseline; no historical files overwritten. |
| Identity | Current `GLOBAL_SOLVERS`, exact order | Neither candidate IDs nor cost columns are renumbered. |
| Source evidence | Draft corpus allowed; production requires reviewed deployment bindings | Local source availability does not establish historical label provenance. |
| External model | `voyage-code-4`, 1024 floats, input_type=null, truncation=false | Fixed backend; no silent replacement with another model. |
| Tokenization | Official model tokenizer, whole-view first within 32000 tokens | No character-count approximation or silent truncation. |
| Credentials | Environment `VOYAGE_API_KEY`, or private file outside Git | Never store a key in project JSON, checkpoints, logs, or committed scripts. |
| Upload | Explicit `--allow-upload` and `--max-tokens` | No upload or charge during corpus inspection/model inference. |
| Controls | A handcrafted, B code, C fixed global derangement seed4502 | C moves all four roles and masks together, not IDs/labels. |
| Initialization | Copy common parameters from a fresh A; B/C new branches share a separate RNG | Same seed alone is insufficient for different architectures. |
| Scope | Code and offline checks only | No training, validation selection, or test run in this turn. |

## Representation

Four fixed roles: `encoder / decision / inference / config`.
Each 1024-vector passes through its own shared role projection:
`LayerNorm(no affine) -> Linear(1024,d/4) -> GELU`.
Masked role outputs plus four masks are concatenated and fused through
`Linear -> GELU -> Dropout -> Linear`. The final solver token is
`LayerNorm(ID_embedding + fused_code)`. Fixed vectors/masks/row mapping are
persistent buffers; projections and ID remain trainable.

Only external vectors are cached. Projected tokens are recomputed each forward.
The old handcrafted path remains the default and old checkpoints remain valid.

## API Registration

Register/sign in at https://cloud.mongodb.com/ . Open the desired project,
**AI Model APIs -> Model API Keys -> Create model API key**. Use the key whose
displayed endpoint is `ai.mongodb.com` (unscoped cloud/geography).
Official instructions: https://www.mongodb.com/docs/voyageai/management/api-keys/ .
Model reference: https://www.mongodb.com/docs/voyageai/models/ .

In the shell that will generate embeddings:

```bash
conda activate easynco
read -rsp 'MongoDB Atlas Model API key: ' VOYAGE_API_KEY
export VOYAGE_API_KEY
```

The `read` command hides the key and does not put its value in shell history.
An optional alternative is a file at `~/.config/selection/voyage_api_key`, owned
by you with mode600; pass `--api-key-file` when generating embeddings. Keep it
outside the project, in the ignored private configuration directory. Do not
send the key in chat. A `.env` file is not automatically loaded.

## Files

| File | Purpose |
|---|---|
| `solver_source_manifest.json` | Explicit ordered 22-solver source allowlist, roles, verified config values, evidence/gaps. |
| `solver_code_corpus.py` / `solver_code_views.py` | Whole-view extraction, overflow-only AST/call-aware splitting, source snapshots/hashes. |
| `solver_code_embeddings.py` | Atlas calls, rate/token budgets, provider-isolated cache, all-component locked bundles. |
| `solver_code_encoder.py` | Role projections, learned overflow aggregation, masked fusion plus ID, offline loader. |
| `r45_experiment.py` | A/B/C initialization, paired schedule, unchanged R43 loss/update/evaluation, checkpoint locking. |
| `r45_analysis.py` | Curves, full/family/size metrics, prediction replay, corrections/harm/cost changes. |
| `run_v4_r45.sh` | Activates easynco; with no arguments writes a plan only. |
| `run_v4_r45_embedding.sh` | Prepared-corpus plan or embedding generation/assembly only; no training. |
| `test_r45.py` | Offline unit fixtures; these are not real code embeddings or experiment results. |

## Source Coverage and Current Limit

The checked manifest covers all 22 current solver IDs and 62 selected Python files.
All selected core symbols resolve locally. The draft contains 67 unique chunks,
100794 model tokens in total, with at most 3919 tokens per chunk. Counts refer to
the current source/config snapshot, not a cost quote or a fixed future budget.

**This is a real-source draft, not a certified historical deployment corpus.**
R41 did not recover all original recipes. In particular, the NSS-imported TSP
labels lack exact historical entry/checkpoint/budget bindings, T2T500's original
weight was not found, and several bridge deployments still lack effective
dependency/configuration bindings. RELD repeat evidence covers OVRPTW, not all
registered variants. Some Config roles therefore have no verified content yet.
Core source availability must not be confused with proof of its label provenance.

The exact gaps are in
`runs/R45_solver_code_embeddings/corpus/missing_sources.md` and the manifest's
deployment records. Review/fill the actual sources and parsed configurations,
then mark the corresponding deployment `binding_status: confirmed` with its
evidence and remove resolved gaps. Do not simply approve all candidates or infer
the meaning of `500` from a name. Add separate deployments when actual source or
configuration differs; repeated equivalent content is not given extra weight.
Missing optional training background can remain omitted.

Production upload/assembly refuses an unresolved draft. There is no random
fallback, automatic provider substitution, or deletion of unavailable solvers.
Supplying an API key alone does not resolve the source-binding gaps.

## Commands

Run from `/public/home/shiys/0selection2` in `easynco`. `tokenizers==0.22.2` has
been installed (on a fresh environment: `python -m pip install tokenizers==0.22.2`).
The API uses Python's standard HTTP library, not a chat model.
The official `voyageai/voyage-code-4` tokenizer has been cached; no code-model
weights are downloaded. Source text is sent outside the machine only by the
explicit `embed --allow-upload` command.

Inspect without uploading or training:

```bash
conda activate easynco
HF_HUB_OFFLINE=1 python -m code.V4.solver_code_corpus --draft
python -m code.V4.solver_code_embeddings --stage plan
bash code/V4/run_v4_r45.sh
python -m unittest code.V4.test_r45 code.V4.test_r43 code.V4.test_dual_stream code.V4.test_solver_features -v
```

On another machine, download the official tokenizer or supply its local file
using `solver_code_corpus --tokenizer /path/to/tokenizer.json`; no approximate
token counter is used. This machine's Hugging Face download used the HF mirror
because the direct endpoint timed out. The cached official repository revision
is `ac780b405e54ff8b3aff681270e0e03840fed2f3`, and the tokenizer's serialized
SHA256 is `f884026e05f6dfffe68b72d580b588006d98b3bc0a128bb1ac425c76f7fc438c`.
The corpus records the tokenizer/content hashes.

**After resolving source bindings**, generate the real, fixed embedding bundle:

```bash
HF_HUB_OFFLINE=1 python -m code.V4.solver_code_corpus
python -m code.V4.solver_code_embeddings --stage plan
python -m code.V4.solver_code_embeddings --stage embed --allow-upload --max-tokens 120000
python -m code.V4.solver_code_embeddings --stage assemble
```

`120000` is an example explicit input-token budget; read the new plan first and
set an authorized limit. It is not a monetary price cap. Cached valid chunks
are not billed again by the tool. If an interrupted HTTP request was processed
by the provider but no response was received, its billing is provider-dependent.
Source/config/tokenizer/chunking changes invalidate the corresponding cache keys.
Assembly refuses to overwrite an existing locked bundle. The lock includes its
SHA256; API revision/time/usage metadata are kept with the cache.

If requests fail before reaching the server, check the shell's proxy settings.
This machine had an unreachable inherited proxy during dependency installation;
use a working proxy or unset the `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` variables
and their lowercase counterparts for the generation command as appropriate.
The code does not silently change your proxy, model, or credentials.

Future training commands, **not executed in this implementation**:

```bash
bash code/V4/run_v4_r45.sh --stage prepare --device cuda:0
bash code/V4/run_v4_r45.sh --stage train --group A --device cuda:0
bash code/V4/run_v4_r45.sh --stage train --group B --device cuda:0
bash code/V4/run_v4_r45.sh --stage train --group C --device cuda:0
bash code/V4/run_v4_r45.sh --stage test --device cuda:0
bash code/V4/run_v4_r45.sh --stage analysis
```

Preparation checks four actual training problem types and gradients, without an
optimizer step or test access. Run preparation once before launching separate
arms (in tmux, on distinct GPUs if available). It locks one batch schedule and
all common initial parameters. Training reads only the offline bundle; it has
no API dependency. W&B uses the existing project/entity, in offline mode.

An explicit `--stage all` performs prepare, all three trainings, locked test,
and analysis sequentially. The default is deliberately **not** `all`.
Old/partial runs and locked vectors are not overwritten.

## Offline Inference

```python
from code.V4.solver_code_encoder import load_r45_checkpoint

model, checkpoint = load_r45_checkpoint("/absolute/path/to/best.pt", "cuda:0")
output = model(batch_without_costs)
```

Code vectors, role masks, semantic row mapping and solver identity digest are
in the checkpoint. No key, corpus directory, external vector file, or network
is needed for inference. Load only trusted checkpoints.

## Checks Performed

The 17 new offline checks and 22 existing R43/dual-stream/feature compatibility
checks passed. They cover branch/backbone gradients, fixed buffers, absent-role
bias masking, B/C parameter equality and fixed ownership, candidate/node padding,
permutations, label-independent forward, offline checkpoint replay, full AST
tails, bounded chunking, source symbol coverage, credential/upload guards and
selection-metric replay. Synthetic fixtures do not establish embedding quality.
The actual historical R43A best checkpoint also loads strictly and produces
bit-identical CPU outputs through the old/default factory for four synthetic
problem inputs (TSP, CVRP, ATSP, OVRPTW); this is not a new dataset evaluation.

No selector training, paid embedding request, validation selection, or test
evaluation was run. No production vectors, accuracy claim, or new trained
checkpoint is presented as a result of this implementation.
