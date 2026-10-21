"""R83: one fresh model-only source run using the unchanged R80D/R77 engine.

Corpus cleaning and embedding generation belong to the upstream owner. Describe
and status do not load datasets, embeddings or checkpoints; no GPU is allocated.
"""

import argparse
from pathlib import Path

from code.unified_selector.registry import GLOBAL_SOLVERS
from . import r80_experiment as engine
from .solver_code_embeddings import load_bundle as load_r45_bundle, validate_bundle as validate_r45_bundle
from .solver_code_encoder import ROLES


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R83_model_only_source_embedding'
DEFAULT_EMBEDDINGS = DEFAULT_ROOT / 'model_only_vectors.pt'
GROUP = 'model_only_seed2'
ENGINE_ARM = 'D_code_embedding'
MODEL_FACTORY = 'code.V4.r83_experiment.load_checkpoint'
SOURCE_SCOPE = 'solver MODEL nn.Module code only; no config/train/solver inference/environment code'
NO_MODEL_SOLVERS = ['GLOP']


def validate_model_only_bundle(bundle):
    """Keep the standard R45 provenance checks and enforce absent non-model roles."""
    validate_r45_bundle(bundle)
    if (bundle.get('provider') != 'mongodb_atlas'
            or bundle.get('schema') not in ('r45-atlas-v1', 'r45-embeddings-v2')):
        raise ValueError('R83 requires a production R45 Atlas bundle with complete chunk trace')
    if bundle.get('model_only_scope', {}).get('no_neural_model_solvers') != NO_MODEL_SOLVERS:
        raise ValueError('R83 must declare GLOP as the deliberate no-neural-model candidate')
    mask, indices = bundle['role_mask'], bundle['chunk_indices']
    glop = GLOBAL_SOLVERS.index('GLOP')
    neural = [index for index, name in enumerate(GLOBAL_SOLVERS) if name not in NO_MODEL_SOLVERS]
    if (not mask[neural, :2].all() or mask[:, 2:].any() or (indices[:, 2:] != -1).any()
            or mask[glop].any() or (indices[glop] != -1).any()):
        raise ValueError('R83 needs encoder/decision for neural candidates, no inference/config, and GLOP all roles absent')


def load_bundle(path):
    bundle = load_r45_bundle(path)
    validate_model_only_bundle(bundle)
    return bundle


def initialize_shared(initial, geometry, device='cpu', bundle=None):
    validate_model_only_bundle(bundle)
    return engine.initialize_shared(initial, geometry, ENGINE_ARM, device, bundle)


def load_checkpoint(path, device='cpu'):
    """Restore every source/ID/geometry buffer offline, never reopen embeddings."""
    model, state = engine.load_checkpoint(path, device, experiment='R83', arms=(GROUP,),
                                         bundle_validator=validate_model_only_bundle)
    config = state['args']
    if (config.get('engine_arm') != ENGINE_ARM or config.get('seed') != 2
            or config['model_params'].get('solver_representation') != 'code'
            or config.get('source_scope') != SOURCE_SCOPE or config.get('model_factory') != MODEL_FACTORY):
        raise ValueError('Require the unchanged R83 model-only/R80D configuration')
    return model, state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def protocol(args, ready, manifest, geometry, baselines, params, bundle, order, caps):
    validate_model_only_bundle(bundle)
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, bundle, order, caps)
    config.update(engine_arm=ENGINE_ARM, group=GROUP, model_factory=MODEL_FACTORY,
        embeddings_path=str(args.embeddings.resolve()), source_scope=SOURCE_SCOPE,
        model_only_scope=bundle['model_only_scope'],
        source_content_verification='cleaned model-only corpus/manifest supplied upstream; not re-parsed by runner',
        role_policy={role: role in ROLES[:2] for role in ROLES},
        role_policy_exception='GLOP glop_atsp_init_only: random insertion, no neural model; all source roles absent, original trainable ID only',
        source_plan='unchanged R77 seed2 task/source requests, full18 single-RELD, one 1024-row update per request',
        architecture='unchanged R80D SolverCodeEncoder + learned solver ID + complete R77 dual-stream/maximin model',
        objective='unchanged complete R77 RDropObjective; no added or removed loss')
    for name in ('r83_experiment.py', 'r83_reporting.py', 'run_v4_r83.sh',
                 'r80_reporting.py', 'solver_code_encoder.py', 'solver_code_embeddings.py',
                 'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py', 'r77_experiment.py'):
        config['source_sha256'][name] = engine.file_hash(V4 / name)
    return config


def train(args, device):
    return engine.train(args, device, checkpoint_loader=load_checkpoint,
                        bundle_loader=load_bundle, protocol_builder=protocol)


def test(args, device):
    return engine.test(args, device, checkpoint_loader=load_checkpoint)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--group', '--arm', dest='arm', choices=(GROUP,), default=GROUP)
    result.add_argument('--prepared-root', '--r77-root', type=Path, default=engine.DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=engine.DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--embeddings', type=Path, help='Default: ROOT/model_only_vectors.pt')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='3090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R83', engine_arm=ENGINE_ARM)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.embeddings is None:
        args.embeddings = args.root / 'model_only_vectors.pt'
    if args.command == 'status':
        from .r83_reporting import status_snapshot
        return status_snapshot(args.root)
    return engine.run(args, train_fn=train, test_fn=test)


if __name__ == '__main__':
    main()
