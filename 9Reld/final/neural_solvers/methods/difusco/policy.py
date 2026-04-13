from EasyNCO.neural_solvers.methods.difusco.tsp_difusco import TSPDiffusionPolicy
from EasyNCO.neural_solvers.methods.difusco.mis_difusco import MISDiffusionPolicy

class DIFUSCOPolicy:
    def __init__(self,
                 env_name,
                 n_layers,
                 hidden_dim,
                 aggregation,
                 diffusion_type,
                 diffusion_schedule,
                 diffusion_steps,
                 sparse_factor,
                 use_activation_checkpoint,
                 parallel_sampling,
                 sequential_sampling,
                 inference_diffusion_steps,
                 inference_schedule,
                 inference_trick,
                 node_feature_only=False):
        if env_name == 'tsp':
            self.model = TSPDiffusionPolicy(
                env_name,
                n_layers,
                hidden_dim,
                aggregation,
                diffusion_type,
                diffusion_schedule,
                diffusion_steps,
                sparse_factor,
                use_activation_checkpoint,
                parallel_sampling,
                sequential_sampling,
                inference_diffusion_steps,
                inference_schedule,
                inference_trick,
                node_feature_only=False
            )
        elif env_name == 'mis':
            self.model = MISDiffusionPolicy(
                env_name,
                n_layers,
                hidden_dim,
                aggregation,
                diffusion_type,
                diffusion_schedule,
                diffusion_steps,
                sparse_factor,
                use_activation_checkpoint,
                parallel_sampling,
                sequential_sampling,
                inference_diffusion_steps,
                inference_schedule,
                inference_trick,
                node_feature_only=True
            )
    def _get_model(self):
        return self.model