# R45 source coverage

Ready for upload: False

- BQ / bq: NSS generation entry, source revision and historical per-problem checkpoint identities are not recovered.; Beam/greedy choice, beam width, augmentation and input normalization remain unknown; current defaults and expansion recipes are not used.
- BQ / config: No verified content for required role
- DIFUSCO / difusco: Original NSS source/checkpoint binding and invocation are missing.; Diffusion type/steps/schedule, parallel/sequential sampling, sparse graph settings, augmentation and merge/2-opt budgets are unknown.
- DIFUSCO / config: No verified content for required role
- DIFUSCO500 / difusco: Original NSS source/checkpoint/config binding is missing; the suffix is not interpreted as an inference budget.; Effective diffusion, sampling, sparsity, augmentation and postprocessing settings remain unknown.
- DIFUSCO500 / config: No verified content for required role
- ELG / elg: Historical per-problem implementation/checkpoint, local-size and network configuration are unbound.; Decoder, starts, augmentation and original inference budget are unknown; later paperlike settings are not substituted.
- ELG / config: No verified content for required role
- GLOP / glop_atsp_init_only: Historical source/native-extension version and binary digest were not captured by the run.; The executed random_insertion C++ core is not covered by the Python-only corpus; this is not a complete neural GLOP paper deployment.
- ICAM / icam: The historical solver revision, effective wrapper defaults and runtime checkpoint digest were not archived; current source/weight hashes alone do not prove them.
- ICAM_ATSP / icam_atsp: The original bridge/source revision and effective imported configuration snapshot are not archived; local code-derived values cannot certify the historical bytes or RNG/dependency state.
- LEHD / lehd: Per-problem NSS entry, checkpoint/source revision and network settings are missing.; Original reconstruction count, decoder, augmentation and effective inference budget are not recovered.
- LEHD / config: No verified content for required role
- MATNET / matnet: Historical source revision and actual loaded weight digest were not captured; the available implementation and current local hash are not a full runtime certification.
- MATPOENET / matpoenet: Historical source revision and runtime checkpoint/native libtsp.so identity were not captured; native NN core remains outside the Python corpus.
- MTPOMO / mtpomo: Effective greedy/sampling and historical solver revision are unresolved; no decoder value is inserted into semantic config.; Saved environment target uses the old EasyNCO.methods.envs namespace; current MVRPEnv masks/input conversion and runtime checkpoint bytes need historical binding. Current mode1 backhaul code can reduce N configured starts to min(floor(0.8*N), N).
- MVMOE / mvmoe: Original NSS CVRP source, checkpoint, network, normalization and decoder/inference recipe are unknown. The MoEPolicy source is only a candidate; no MVRP configuration or weight is projected onto CVRP.
- MVMOE / config: No verified content for required role
- MVMOE / mvmoe: Effective decoder and historical solver revision are unresolved despite top-level greedy; retain the R41 sampling hypothesis without claiming confirmation.; Old environment namespace, constraint/input implementation and effective backhaul start count need historical source binding; runtime checkpoint digest not captured.
- MoSES_CaDA / moses_cada: Historical rf_easnco_env/rl4co decoder, environment/import revisions and effective decoder/RNG parameters were not saved; do not infer them from today's defaults.; Original per-run loaded checkpoint digests and checkpoint-resolved base network parameters are not fully bound; current weight hashes identify local candidates only.
- MoSES_RF / moses_rf: Historical rl4co decoder/MTVRPEnv dependency revisions and effective decoder/RNG are unbound; current package defaults are not historical evidence.; Historical checkpoint byte identities and checkpoint-resolved base network settings remain incomplete; current bucket hashes are not runtime-captured digests.
- OMNI / omni: Historical source revision, unlogged constructor defaults and runtime loaded-weight digest remain unknown; file equality is not a complete deployment certification.
- OMNI / omni: Original NSS validation command, source, checkpoint and effective model/decoder/augmentation/fine-tuning recipe are missing. Train/test settings must not be copied here.
- OMNI / config: No verified content for required role
- OMNI / omni: CVRP NSS source/checkpoint, effective decoder, normalization, augmentation and adaptation recipe are unknown; TSP recipe is not applicable by name alone.
- OMNI / config: No verified content for required role
- RELD_CVRP / reld_cvrp: Historical bridge/source/dependency revision and runtime checkpoint hash were not captured; recovered current YAML/bridge parameters remain a partial historical binding.
- RELD_MOEL / reld_moel: All label-file ancestries are bound, but historical source/environment versions and effective per-problem runtime metadata outside the R41 OVRPTW verification are not fully archived. No new solver repeat is claimed.
- RELD_MTL / reld_mtl: Historical source/constraint-environment version and complete per-problem runtime metadata remain missing outside the preserved R41 OVRPTW verification. File equality alone does not certify all deployments.
- RouteFinder / routefinder: Historical load_from_checkpoint-resolved network, rl4co decoder/MTVRPEnv versions and effective decoder/RNG remain unbound; do not substitute current defaults.; The historical weight digests were not captured per run; current bucket hashes are local identity evidence only.
- T2T / t2t: Original NSS entry/source/checkpoint binding is missing; local 100-node weights are not interchangeable by filename.; Historical diffusion/sampling, gradient guidance, rewrite steps/ratio, inference schedule, augmentation, normalization and postprocessing budgets are unknown.
- T2T / config: No verified content for required role
- T2T500 / t2t: The original NSS 500 deployment checkpoint/source/config identity remains unbound despite locating a candidate elsewhere; do not substitute tsp100 or infer a budget from 500.; Effective diffusion, gradient search/rewrite, sampling, sparsity, augmentation, normalization and postprocessing settings are unknown.
- T2T500 / config: No verified content for required role
- UNICO_MatPOENet / unico_matpoenet: Historical bridge/source/dependency and libtsp.so versions, resolved test module snapshot, effective batch/RNG history and runtime weight digests were not captured; current key inspection is not a historical replay.
- UNICO_MatPOENet / unico_matpoenet: Historical bridge/source/dependency/native versions, effective batch/RNG history and per-run weight digests are not archived. Current exact/mix availability and test module values do not prove the original on-disk state.
