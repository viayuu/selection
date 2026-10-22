"""CPU-only exact specs for matched two-stage, donor-free R101 students."""

import argparse
from pathlib import Path

from . import r101_launch as launch


V4 = launch.REPO/'code/V4'
DATA = '/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
PREPARED = V4/'runs/R84_new90k_attention/prepared'


def make_spec(arm):
    if arm not in launch.ARM_SLOTS:
        raise ValueError('R101 has only its matched A/B teacher-source comparison')
    root = launch.MODEL_ROOT
    teacher_root = launch.TEACHER_ROOTS[arm]
    protocol = root/'protocol_lock.json'
    run = ['{python}', '-B', '-u', '-m', 'code.V4.r101_experiment']
    options = ['--root', str(root), '--arm', arm, '--prepared-root', str(PREPARED),
        '--data-root', DATA, '--cache-root', '{cache_root}', '--protocol-lock', str(protocol),
        '--gpu-type', '4090', '--large-micro-cap', '352', '--eval-batch-size', '256',
        '--device', '{device}', '--wandb-mode', 'offline']
    stages = [dict(name=name, device='gpu', argv=run+[command]+options+
        (['--teacher-root', str(teacher_root)] if name in ('pretrain', 'train') else []))
        for name, command in (('pretrain', 'pretrain'), ('train', 'train'),
                              ('TEST', 'test'), ('timing', 'timing'))]
    reporter = ['{python}', '-B', '-u', '-m', 'code.V4.r101_reporting']
    reporting = ['--root', str(root), '--arm', arm, '--device', '{device}']
    stages += [dict(name='LIB', device='gpu', argv=reporter+['lib']+reporting+['--lib-batch-size', '2']),
               dict(name='PDF', device='cpu', argv=reporter+['report']+reporting)]
    receipt = root/'preparation'/('A_teacher_export_receipt.json' if arm == 'A'
                                else 'B_teacher_metadata_receipt.json')
    sources = [protocol, root/'PLAN.md', receipt, launch.WRAPPER]
    sources += list(V4.glob('r101_*.py')) + list(V4.glob('test_r101*.py'))
    sources += [PREPARED/name for name in ('prepared.json', 'geometry_stats.json',
                                          'source_cache_manifest.json', 'identicalinit_seed2.pt')]
    if any(not Path(path).is_file() for path in sources):
        raise RuntimeError('Complete runner, metadata, primary preparation and source receipt before readiness')
    spec = dict(schema_version=1, run_id='R101_'+arm, run_dir=str(root/arm),
        common_root=str(root), science_protocol=str(protocol), data_root=DATA,
        prepared_root=str(PREPARED),
        cache_source=str(V4/'runs/R84_new90k_attention/input_cache'),
        cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs',
        slots=[launch.ARM_SLOTS[arm]], cpu_count=4, memory_gib=9,
        wandb_mode='offline', teacher_root=str(teacher_root), teacher_receipt=str(receipt),
        source_files=sorted({str(Path(path)) for path in sources}), stages=stages)
    if arm == 'A':
        spec['teacher_source'] = str(root/'teacher_cache/A')
    return spec


def write_spec(arm):
    spec = make_spec(arm)
    path = launch.ROOT/'specs'/('R101_'+arm+'.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve the original R101 spec; no implicit replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('arm', choices=tuple(launch.ARM_SLOTS))
    return value


def main(argv=None):
    print(write_spec(parser().parse_args(argv).arm))


if __name__ == '__main__':
    main()
