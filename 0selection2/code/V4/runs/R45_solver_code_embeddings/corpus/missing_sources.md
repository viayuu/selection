# R45 source coverage

Ready for upload: False

- BQ / bq: NSS-imported label generation command, exact checkpoint, and beam/greedy budget are not bound to the available bq-nco source.
- BQ / config: No verified content for required role
- DIFUSCO / difusco: Historical NSS recipe is missing; available difusco_tsp100.ckpt is a candidate, not a verified label binding.
- DIFUSCO / config: No verified content for required role
- DIFUSCO500 / difusco: Historical NSS recipe is missing; available difusco_tsp500.ckpt is unbound. The suffix is not interpreted as an inference budget.
- DIFUSCO500 / config: No verified content for required role
- ELG / elg: Confirm per-problem source variant, local-size parameters, checkpoint, and original inference budget.
- ELG / config: No verified content for required role
- GLOP / glop_atsp: ATSP decomposition/local-reconstruction implementation is available; historical checkpoint and actual partition/reconstruction settings require binding.
- GLOP / config: No verified content for required role
- ICAM / icam: Bind CVRP label generation entry and actual adaptation/checkpoint/inference settings.
- ICAM / config: No verified content for required role
- ICAM_ATSP / icam_atsp: Bridge entry and nearest-row/column conversion identified; confirm effective model_params and checkpoint per deployed scale.
- LEHD / lehd: Confirm TSP/CVRP weights and actual reconstruction iterations; NSS TSP labels have no recovered full recipe.
- LEHD / config: No verified content for required role
- MATNET / matnet: Historical ATSP source revision, checkpoint and inference budget need binding.
- MATNET / config: No verified content for required role
- MATPOENET / matpoenet: Do not confuse this candidate with UNICO_MatPOENet; confirm actual label recipe and positional encoding configuration.
- MATPOENET / config: No verified content for required role
- MTPOMO / mtpomo: Eval entry identified, but top-level greedy setting and module default sampling were not resolved to a historical effective decoder configuration; confirm weights and environment mask dependency.
- MVMOE / mvmoe: Resolve historical effective decoding strategy, source environment masks, checkpoint, and CVRP versus multi-task deployment differences.
- MoSES_CaDA / moses_cada: Bridge policy settings verified; bind scale50/100 checkpoints and historical rl4co decoder/environment dependency versions.
- MoSES_RF / moses_rf: Bridge policy settings verified; bind scale50/100 checkpoints and historical rl4co decoder/environment dependency versions.
- OMNI / omni: TSP train has an EasyNCO recipe but val came from NSS; do not assume they used the same effective deployment. CVRP recipe also needs binding.
- OMNI / config: No verified content for required role
- RELD_CVRP / reld_cvrp: Entry and CVRP config identified; bind original parsed model/inference settings and checkpoint hash.
- RELD_CVRP / config: No verified content for required role
- RELD_MOEL / reld_moel: R41 verified OVRPTW repeat costs with this weight; other registered problems and constraint environment dependencies are not yet bound in this source corpus.
- RELD_MTL / reld_mtl: R41 verified OVRPTW repeat costs with this weight; other registered problems and constraint environment dependencies are not yet bound in this source corpus.
- RouteFinder / routefinder: Bridge entry verified; bind scale-dependent checkpoints and inherited rl4co decoder, environment and effective inference configuration.
- T2T / t2t: NSS-imported label recipe is missing; available t2t_tsp100.ckpt is not a proven historical binding.
- T2T / config: No verified content for required role
- T2T500 / t2t: Original checkpoint is missing; t2t_tsp500.ckpt was not found. Shared candidate source is recorded without guessing what 500 means.
- T2T500 / config: No verified content for required role
- UNICO_MatPOENet / unico_matpoenet: UnicoMatpoenetRunner identified; infer and verify effective encoder layers and checkpoint per deployed scale before production corpus approval.
- UNICO_MatPOENet / config: No verified content for required role
