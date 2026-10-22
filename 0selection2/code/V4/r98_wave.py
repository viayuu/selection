"""CPU-only exact command specs for the four post-R97 experiment arms."""

import argparse
from pathlib import Path

from . import r98_launch as launch


V4 = launch.REPO/'code/V4'
DATA = '/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
PREPARED = V4/'runs/R84_new90k_attention/prepared'
FEATURES = V4/'runs/R92_broad_learning/features'
CONTEXT = V4/'runs/R97_tabular_in_context/context'
INPUT_SOURCE = V4/'runs/R84_new90k_attention/input_cache'
ARM_SLOTS = dict(R98_A='4090_0', R99_A='3090_0', R99_B='3090_1', R100_A='4090_1')


def make_spec(run_id):
    if run_id not in ARM_SLOTS:
        raise ValueError('Only the four predeclared independent arms are supported')
    version, arm = run_id.split('_')
    directory = dict(R98='R98_performance_context', R99='R99_global_competition_input',
                     R100='R100_pairwise_context')[version]
    root = V4/'runs'/directory
    protocol = root/'protocol_lock.json'
    sources = [protocol, root/'PLAN.md', V4/'r98_launch.py', V4/'r98_wave.py', launch.WRAPPER]
    stages = []
    if version == 'R99':
        run = ['{python}', '-B', '-u', '-m', 'code.V4.r99_experiment']
        options = ['--root', str(root), '--arm', arm, '--prepared-root', str(PREPARED),
            '--data-root', DATA, '--cache-root', '{cache_root}',
            '--feature-root', '/tmp/shiys_R99_'+arm+'_relation_features',
            '--protocol-lock', str(protocol), '--gpu-type', '{gpu_type}',
            '--large-micro-cap', '352', '--eval-batch-size', '256',
            '--device', '{device}', '--wandb-mode', 'offline']
        stages = [dict(name=name, device='gpu', argv=run+[command]+options)
            for name, command in (('input_features', 'prepare'), ('train', 'train'),
                                  ('TEST', 'test'), ('timing', 'timing'))]
        report = ['{python}', '-B', '-u', '-m', 'code.V4.r99_reporting']
        report_options = ['--root', str(root), '--arm', arm, '--device', '{device}']
        stages += [dict(name='LIB', device='gpu', argv=report+['lib']+report_options+['--lib-batch-size', '2']),
                   dict(name='PDF', device='cpu', argv=report+['report']+report_options)]
        source, local, kind = INPUT_SOURCE, '/tmp/shiys_R84_new90k_cache', 'r84_inputs'
        run_dir = root/arm
    else:
        module = 'code.V4.'+version.lower()+'_experiment'
        options = ['--root', str(root), '--context-root', '{cache_root}',
            '--feature-root', str(FEATURES), '--data-root', DATA,
            '--prepared-root', str(PREPARED), '--protocol-lock', str(protocol),
            '--device', '{device}', '--wandb-mode', 'offline']
        if version == 'R98':
            options += ['--target-root', str(root/'targets')]
        run = ['{python}', '-B', '-u', '-m', module]
        stages = [dict(name='VAL_lock_TEST_LIB', device='gpu', argv=run+['run']+options),
                  dict(name='PDF', device='cpu', argv=run+['report']+options)]
        source, local, kind = CONTEXT, '/tmp/shiys_R97_tabular_context', 'r97_context'
        run_dir = root
        runtime = launch.base.read(root/'protocol_lock.json')
        sources += list(Path(runtime['library_root']).rglob('*.py'))
        parameters = runtime['regressor_parameters' if version == 'R98' else 'classifier_parameters']
        sources += [Path(parameters['model_path'])]
        sources += list((root/'runtime').glob('*.json'))
        sources += list((root/'targets').rglob('*.json'))
        sources += list((root/'pairs').rglob('*.json'))
    sources += list(V4.glob(version.lower()+'*.py'))
    sources += list(V4.glob('test_'+version.lower()+'*.py'))
    if any(not Path(path).is_file() for path in sources):
        raise RuntimeError('Wait for the runner/report/protocol/PLAN files to be ready')
    return dict(schema_version=1, run_id=run_id, run_dir=str(run_dir), common_root=str(root),
        science_protocol=str(protocol), data_root=DATA, prepared_root=str(PREPARED),
        cache_source=str(source), cache_root=str(local), cache_kind=kind,
        slots=[ARM_SLOTS[run_id]], cpu_count=4, memory_gib=9, wandb_mode='offline',
        source_files=sorted({str(Path(path)) for path in sources}), stages=stages)


def write_spec(run_id):
    spec = make_spec(run_id)
    path = launch.ROOT/'specs'/(run_id+'.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve existing dispatch spec; no replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('run_id', choices=tuple(ARM_SLOTS))
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    print(write_spec(args.run_id))


if __name__ == '__main__':
    main()
