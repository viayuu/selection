# R45 Explicit Default Assumptions

Ready for embedding: True
Historical bindings verified: False

Recorded values and historical gaps were preserved. No training/API/solver run was performed.

Only missing semantic parameters use the following defaults; revisions, hashes and RNG history are not invented.

## BQ / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "dim_emb": 192,
  "dim_ff": 512,
  "nb_layers_encoder": 9,
  "nb_heads": 12,
  "dropout": 0.0,
  "batchnorm": false,
  "activation_ff": "relu",
  "activation_attention": "softmax",
  "beam_size": 1,
  "knns": -1
}
```

Default sources:
- sources:bq-nco/args.py / add_common_args

Preserved historical gaps:
- NSS generation entry, source revision and historical per-problem checkpoint identities are not recovered.
- Beam/greedy choice, beam width, augmentation and input normalization remain unknown; current defaults and expansion recipes are not used.

## DIFUSCO / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "n_layers": 12,
  "hidden_dim": 256,
  "aggregation": "sum",
  "diffusion_type": "categorical",
  "diffusion_schedule": "linear",
  "diffusion_steps": 1000,
  "sparse_factor": -1,
  "node_feature_only": false,
  "parallel_sampling": 1,
  "sequential_sampling": 1,
  "inference_diffusion_steps": 50,
  "inference_schedule": "cosine",
  "inference_trick": "ddim"
}
```

Default sources:
- easynco:configs_old/tsp_difusco_config.yaml / model

Preserved historical gaps:
- Original NSS source/checkpoint binding and invocation are missing.
- Diffusion type/steps/schedule, parallel/sequential sampling, sparse graph settings, augmentation and merge/2-opt budgets are unknown.

## DIFUSCO500 / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "n_layers": 12,
  "hidden_dim": 256,
  "aggregation": "sum",
  "diffusion_type": "categorical",
  "diffusion_schedule": "linear",
  "diffusion_steps": 1000,
  "sparse_factor": -1,
  "node_feature_only": false,
  "parallel_sampling": 1,
  "sequential_sampling": 1,
  "inference_diffusion_steps": 50,
  "inference_schedule": "cosine",
  "inference_trick": "ddim"
}
```

Default sources:
- easynco:configs_old/tsp_difusco_config.yaml / model

Preserved historical gaps:
- Original NSS source/checkpoint/config binding is missing; the suffix is not interpreted as an inference budget.
- Effective diffusion, sampling, sparsity, augmentation and postprocessing settings remain unknown.

## ELG / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_heads": 8,
  "qkv_dim": 16,
  "num_encoder_layers": 6,
  "normalization": "instance",
  "feedforward_hidden": 512,
  "logit_clipping": 50,
  "use_graph_mean": false,
  "am_mode": false,
  "first_placeholder": false,
  "first_mode": "random",
  "local_enable": false,
  "local_size": 40,
  "xi": 1.0,
  "euclidean": false
}
```

Default sources:
- easynco:settings/elg_settings.yaml / model

Preserved historical gaps:
- Historical per-problem implementation/checkpoint, local-size and network configuration are unbound.
- Decoder, starts, augmentation and original inference budget are unknown; later paperlike settings are not substituted.

## GLOP / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- insertion:__init__.py / ['_tsp_get_parameters']

Preserved historical gaps:
- Historical source/native-extension version and binary digest were not captured by the run.
- The executed random_insertion C++ core is not covered by the Python-only corpus; this is not a complete neural GLOP paper deployment.

## ICAM / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128
}
```

Default sources:
- easynco:neural_solvers/methods/icam/policy.py / CVRPICAMPolicy.__init__

Preserved historical gaps:
- The historical solver revision, effective wrapper defaults and runtime checkpoint digest were not archived; current source/weight hashes alone do not prove them.

## ICAM_ATSP / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- sources:ICAM/ICAM_ATSP/ATSPModel_ICAM.py / ['ATSP_Encoder', 'EncoderLayer', 'EncodingBlock', 'AddAndInstanceNormalization', 'FeedForward', 'adaptation_attention_free_module']

Preserved historical gaps:
- The original bridge/source revision and effective imported configuration snapshot are not archived; local code-derived values cannot certify the historical bytes or RNG/dependency state.

## LEHD / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "layer_num": 6,
  "embed_dim": 128,
  "num_heads": 8,
  "qkv_dim": 16,
  "feedforward_hidden": 512,
  "multihead_bias": false,
  "first_mode": "random"
}
```

Default sources:
- easynco:neural_solvers/methods/lehd/policy.py / LEHDPolicy.__init__

Preserved historical gaps:
- Per-problem NSS entry, checkpoint/source revision and network settings are missing.
- Original reconstruction count, decoder, augmentation and effective inference budget are not recovered.

## MATNET / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- easynco:neural_solvers/methods/matnet/matnet_encoder.py / ['*']

Preserved historical gaps:
- Historical source revision and actual loaded weight digest were not captured; the available implementation and current local hash are not a full runtime certification.

## MATPOENET / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- easynco:neural_solvers/methods/matnet/matnet_encoder.py / ['*']

Preserved historical gaps:
- Historical source revision and runtime checkpoint/native libtsp.so identity were not captured; native NN core remains outside the Python corpus.

## MTPOMO / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "decoder_strategy": "sampling"
}
```

Default sources:
- easynco:phases/rl/ar_reinforce.py / ARREINFORCELightning.__init__

Preserved historical gaps:
- Effective greedy/sampling and historical solver revision are unresolved; no decoder value is inserted into semantic config.
- Saved environment target uses the old EasyNCO.methods.envs namespace; current MVRPEnv masks/input conversion and runtime checkpoint bytes need historical binding. Current mode1 backhaul code can reduce N configured starts to min(floor(0.8*N), N).

## MVMOE / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embedding_dim": 128,
  "ff_hidden_dim": 512,
  "num_heads": 8,
  "qkv_dim": 16,
  "encoder_layer_num": 6,
  "normalization": "instance",
  "norm_loc": "norm_last",
  "logit_clipping": 10,
  "MoE_param": {
    "light": false,
    "expert_loc": [
      "Enc0",
      "Enc1",
      "Enc2",
      "Enc3",
      "Enc4",
      "Enc5",
      "Dec"
    ],
    "num_experts": 4,
    "topk": 2,
    "routing_level": "node",
    "routing_method": "input_choice"
  },
  "decoder_strategy": "sampling"
}
```

Default sources:
- easynco:settings/mvmoe_settings.yaml / model
- easynco:phases/rl/ar_reinforce.py / ARREINFORCELightning.__init__

Preserved historical gaps:
- Original NSS CVRP source, checkpoint, network, normalization and decoder/inference recipe are unknown. The MoEPolicy source is only a candidate; no MVRP configuration or weight is projected onto CVRP.

## MVMOE / deployment 1

Assumed values (empty means only the historical binding is waived):

```json
{
  "decoder_strategy": "sampling"
}
```

Default sources:
- easynco:settings/mvmoe_settings.yaml / model
- easynco:phases/rl/ar_reinforce.py / ARREINFORCELightning.__init__

Preserved historical gaps:
- Effective decoder and historical solver revision are unresolved despite top-level greedy; retain the R41 sampling hypothesis without claiming confirmation.
- Old environment namespace, constraint/input implementation and effective backhaul start count need historical source binding; runtime checkpoint digest not captured.

## MoSES_CaDA / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_encoder_layers": 6,
  "num_heads": 8,
  "feedforward_hidden": 512,
  "use_graph_context": false,
  "linear_bias_decoder": false,
  "mask_inner": true,
  "out_bias_pointer_attn": false
}
```

Default sources:
- sources:moses_vrp/models/policy.py / CadaMultiLoRAPolicy.__init__

Preserved historical gaps:
- Historical rf_easnco_env/rl4co decoder, environment/import revisions and effective decoder/RNG parameters were not saved; do not infer them from today's defaults.
- Original per-run loaded checkpoint digests and checkpoint-resolved base network parameters are not fully bound; current weight hashes identify local candidates only.

## MoSES_RF / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_encoder_layers": 6,
  "num_heads": 8,
  "feedforward_hidden": 512,
  "use_graph_context": false,
  "linear_bias_decoder": false,
  "mask_inner": true,
  "out_bias_pointer_attn": false
}
```

Default sources:
- sources:moses_vrp/models/policy.py / MultiLoRAPolicy.__init__

Preserved historical gaps:
- Historical rl4co decoder/MTVRPEnv dependency revisions and effective decoder/RNG are unbound; current package defaults are not historical evidence.
- Historical checkpoint byte identities and checkpoint-resolved base network settings remain incomplete; current bucket hashes are not runtime-captured digests.

## OMNI / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128
}
```

Default sources:
- easynco:neural_solvers/methods/omni/policy.py / OMNI_POMO_Policy.__init__

Preserved historical gaps:
- Historical source revision, unlogged constructor defaults and runtime loaded-weight digest remain unknown; file equality is not a complete deployment certification.

## OMNI / deployment 1

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_heads": 8,
  "qkv_dim": 16,
  "num_encoder_layers": 6,
  "normalization": "instance",
  "feedforward_hidden": 512,
  "logit_clipping": 10,
  "use_graph_mean": false,
  "am_mode": true,
  "first_placeholder": true,
  "first_mode": "random"
}
```

Default sources:
- easynco:neural_solvers/methods/omni/policy.py / OMNI_POMO_Policy.__init__

Preserved historical gaps:
- Original NSS validation command, source, checkpoint and effective model/decoder/augmentation/fine-tuning recipe are missing. Train/test settings must not be copied here.

## OMNI / deployment 2

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_heads": 8,
  "qkv_dim": 16,
  "num_encoder_layers": 6,
  "normalization": "instance",
  "feedforward_hidden": 512,
  "logit_clipping": 10,
  "use_graph_mean": false,
  "am_mode": true,
  "first_placeholder": true,
  "first_mode": "random"
}
```

Default sources:
- easynco:neural_solvers/methods/omni/policy.py / OMNI_POMO_Policy.__init__

Preserved historical gaps:
- CVRP NSS source/checkpoint, effective decoder, normalization, augmentation and adaptation recipe are unknown; TSP recipe is not applicable by name alone.

## RELD_CVRP / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- reld:CVRP/CVRPModel.py / ['CVRP_Encoder', 'EncoderLayer', 'FeedForward']

Preserved historical gaps:
- Historical bridge/source/dependency revision and runtime checkpoint hash were not captured; recovered current YAML/bridge parameters remain a partial historical binding.

## RELD_MOEL / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- reld:Multi-Task/models/MOEModel_Light.py / ['MTL_Encoder', 'EncoderLayer', 'FeedForward', 'Add_And_Normalization_Module']

Preserved historical gaps:
- All label-file ancestries are bound, but historical source/environment versions and effective per-problem runtime metadata outside the R41 OVRPTW verification are not fully archived. No new solver repeat is claimed.

## RELD_MTL / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- reld:Multi-Task/models/MTLModel.py / ['MTL_Encoder', 'EncoderLayer', 'FeedForward', 'Add_And_Normalization_Module']

Preserved historical gaps:
- Historical source/constraint-environment version and complete per-problem runtime metadata remain missing outside the preserved R41 OVRPTW verification. File equality alone does not certify all deployments.

## RouteFinder / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "embed_dim": 128,
  "num_encoder_layers": 6,
  "num_heads": 8,
  "normalization": "instance",
  "feedforward_hidden": 512,
  "encoder_use_post_layers_norm": false,
  "encoder_use_prenorm": false,
  "use_graph_context": false
}
```

Default sources:
- sources:routefinder/routefinder/models/policy.py / RouteFinderPolicy.__init__

Preserved historical gaps:
- Historical load_from_checkpoint-resolved network, rl4co decoder/MTVRPEnv versions and effective decoder/RNG remain unbound; do not substitute current defaults.
- The historical weight digests were not captured per run; current bucket hashes are local identity evidence only.

## T2T / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "n_layers": 12,
  "hidden_dim": 256,
  "aggregation": "sum",
  "diffusion_type": "categorical",
  "diffusion_schedule": "linear",
  "diffusion_steps": 1000,
  "sparse_factor": -1,
  "node_feature_only": false,
  "parallel_sampling": 1,
  "sequential_sampling": 1,
  "inference_diffusion_steps": 50,
  "inference_schedule": "cosine",
  "inference_trick": "ddim",
  "norm": true,
  "rewrite": true,
  "rewrite_steps": 3,
  "rewrite_ratio": 0.4,
  "rewrite_inference_steps": 10
}
```

Default sources:
- easynco:configs_old/tsp_t2t_config.yaml / model

Preserved historical gaps:
- Original NSS entry/source/checkpoint binding is missing; local 100-node weights are not interchangeable by filename.
- Historical diffusion/sampling, gradient guidance, rewrite steps/ratio, inference schedule, augmentation, normalization and postprocessing budgets are unknown.

## T2T500 / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{
  "n_layers": 12,
  "hidden_dim": 256,
  "aggregation": "sum",
  "diffusion_type": "categorical",
  "diffusion_schedule": "linear",
  "diffusion_steps": 1000,
  "sparse_factor": -1,
  "node_feature_only": false,
  "parallel_sampling": 1,
  "sequential_sampling": 1,
  "inference_diffusion_steps": 50,
  "inference_schedule": "cosine",
  "inference_trick": "ddim",
  "norm": true,
  "rewrite": true,
  "rewrite_steps": 3,
  "rewrite_ratio": 0.4,
  "rewrite_inference_steps": 10
}
```

Default sources:
- easynco:configs_old/tsp_t2t_config.yaml / model

Preserved historical gaps:
- The original NSS 500 deployment checkpoint/source/config identity remains unbound despite locating a candidate elsewhere; do not substitute tsp100 or infer a budget from 500.
- Effective diffusion, gradient search/rewrite, sampling, sparsity, augmentation, normalization and postprocessing settings are unknown.

## UNICO_MatPOENet / deployment 0

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- sources:UniCO/MatPOENet/ATSPModel.py / ['ATSP_Encoder', 'EncoderLayer', 'EncodingBlock']

Preserved historical gaps:
- Historical bridge/source/dependency and libtsp.so versions, resolved test module snapshot, effective batch/RNG history and runtime weight digests were not captured; current key inspection is not a historical replay.

## UNICO_MatPOENet / deployment 1

Assumed values (empty means only the historical binding is waived):

```json
{}
```

Default sources:
- sources:UniCO/MatPOENet/ATSPModel.py / ['ATSP_Encoder', 'EncoderLayer', 'EncodingBlock']

Preserved historical gaps:
- Historical bridge/source/dependency/native versions, effective batch/RNG history and per-run weight digests are not archived. Current exact/mix availability and test module values do not prove the original on-disk state.

