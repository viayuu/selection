# Result-to-Claim Evidence Snapshot (2026-04-20)

Source: user-provided experimental summary in the Codex session on 2026-04-20.

Use: durable local snapshot for claim drafting. This is not a raw log export; if later raw tables disagree, raw tables take precedence.

## Intended claim under review

"A unified selector matches a strong baseline across 18 routing families, and its exact-top1 errors split into two regimes: benign low-margin ambiguity and a smaller regret-dominant >1% failure subset."

## Experiment scope

- Unified supervised selector over 18 routing problems: TSP / CVRP / ATSP + 15 MVRP composites.
- 1000 train / val / test instances per problem.
- `M_GLOBAL=18` neural solvers with overlapping per-problem pools of 3-10 arms.

## Main result models

- `S5 (baseline)`: `S4_distill_seed2` (4-epoch distillation from 18 specialists) + 1-epoch tail fine-tune on TSP / CVRP / ATSP only, `LR=1e-5`.
- `R18 (main model)`: `S4_distill_seed2` + 3-epoch tail fine-tune on all 18 problems, `LR=1e-5`.
- `R22 (falsification)`: from `R18`, +3-epoch fine-tune with 2x oversampling on CVRP / VRPL / VRPTW / OVRPTW, `LR=1e-5`. Preregistered success bar: `+0.008` val top1.
- `R21 (pairwise reranker)`: Bradley-Terry comparator on top-3 base picks, alpha-blended at inference.

## Test metrics

Total test instances: `18 problems x 1000 = 18,000`.

| Method | macro top1 | macro vs_sbs | macro fuzzy@1% |
|:---|---:|---:|---:|
| S5 | 0.5181 | -0.0150% | 0.7114 |
| R18 | 0.5192 | -0.0175% | 0.7110 |
| R22 | 0.5185 | -0.0173% | 0.7104 |
| R18+R21 `alpha=0.5` | 0.5133 | -0.0000% | val gained, test lost |

## Bootstrap 95% CIs (5000 samples, per-problem resampling)

- `R18 - S5` top1 delta: mean `+0.0012`, CI `[-0.0025, +0.0049]`.
- `R22 - R18` top1 delta: mean `-0.0008`, CI `[-0.0038, +0.0022]`.
- `R22 - S5` top1 delta: mean `+0.0004`, CI `[-0.0039, +0.0049]`.
- All cost-delta CIs also include `0`.

## Margin-bucket decomposition

Bucket variable: SBS-vs-oracle gap.

| Bucket | Mass | Exact-top1 | Fuzzy@1% | Mean regret |
|:---|---:|---:|---:|---:|
| `<0.1%` (SBS approx oracle) | 53.7% | 0.81 | 0.91 | +0.28% |
| `0.1-0.5%` | 8.1% | 0.17 | 0.98 | +0.28% |
| `0.5-1%` | 9.0% | 0.17 | 0.98 | +0.63% |
| `>1%` (clear gap) | 29.2% | 0.17 | 0.18 | +2.69% |

Bucket contribution to total relative regret:

- `<0.1%`: `14-16%`.
- `0.1-0.5%`: `2.2%`.
- `0.5-1%`: `5.6%`.
- `>1%`: `76-78%`.

## Risk-coverage

`R18` vs `R18+R21 blend`:

- Validation mean regret: `1.0005% -> 0.9909%` (helps by `0.0096` points at every coverage level).
- Test mean regret: `1.0072% -> 1.0202%` (hurts by `0.0130` points at nearly every coverage level).
- The validation-to-test sign flip is uniform, not localized to one alpha or threshold.

## Arm distribution

- Zero-pick arms across 82-arm union: `S5=48`, `R18=50`, `R22=50`.
- Zero-pick arms with oracle wins: `S5=33`, `R18=35`, `R22=35`.
- Share of test mass with oracle on a zero-pick arm: `S5=17.2%`, `R18=20.0%`, `R22=20.3%`.
- Top-2 oracle arms account for `67.4%` (CVRP) to `97.6%` (ATSP) of oracle wins per problem; mean about `90%`.

## Negative results

All failed to beat `S5` baseline on test:

- Architecture variants: arm-attention, CoE constraint-expert FFN, FiLM, depth `=6`, `d=192`.
- Loss variants: focal CE, gap-regression + pairwise logistic, winner-margin, Plackett-Luce, entropy diversity.
- Ensemble variants: `mean_logit`, `mean_prob`, `geo_mean`, plurality, Borda, temperature-calibrated, `min_cost_proxy`.
- SWA: `0.449` top1, `+2.55%` vs_sbs.
- Rerankers: shared-features `R12` and raw-token pairwise `R21`; validation gains did not transfer to test.
- `R22` hard-family 2x oversampling did not meet the preregistered success bar.

## Label ceiling analysis

- `94.6%` of test instances have a uniquely-best arm at `0.1%` relative margin.
- Only `57.9%` do at `1%` margin.
- Current best strict top1: `0.5193`, well below the `0.58` ceiling but within the low-margin regime.

## Known caveats

- Validation and test are IID from the same generator, not OOD.
- Labels come from 18 neural solvers; the oracle is best-of-18, not globally optimal.
- Solvers are already close on most instances; median gap is under `1%`.
- Only 1000 train instances per problem may be insufficient for comparator generalization.
