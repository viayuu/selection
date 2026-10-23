"""Build future R102 specs for gpu05 direct-cache reuse only after team ready.

No spec is written on import; this never queues, launches, exports or stages.
"""

import argparse
from pathlib import Path

from . import r102_launch as launch


def make_spec(arm):
    if arm not in launch.ARM_SLOTS:
        raise ValueError('R102 has only A and B')
    root = launch.MODEL_ROOT
    sources = [root / 'protocol_lock.json', root / 'PLAN.md',
        root / 'preparation/LAUNCH_PLAN.md', launch.WRAPPER]
    code = launch.REPO / 'code/V4'
    sources += [code / name for name in ('r102_model.py', 'r102_features.py',
        'r102_experiment.py', 'r102_reporting.py', 'r102_launch.py', 'r102_wave.py', 'r102_goal.py')]
    sources += list(code.glob('r102_*.py')) + list(code.glob('test_r102*.py'))
    sources += [launch.REPO / 'docs/research' / name for name in
        ('pretrained_representation_coverage_tsp_atsp.md', 'post_r100_broad_direction.md')]
    sources += [launch.PREPARED_ROOT / name for name in
        ('prepared.json', 'geometry_stats.json', 'source_cache_manifest.json', 'identicalinit_seed2.pt')]
    if any(not Path(path).is_file() for path in sources):
        raise RuntimeError('Wait for main protocol, donor/features, runner/report and complete R84 preparation')
    spec = dict(schema_version=1, run_id='R102_' + arm, run_dir=str(root / arm), common_root=str(root),
        science_protocol=str(root / 'protocol_lock.json'), data_root=str(launch.DATA_ROOT),
        prepared_root=str(launch.PREPARED_ROOT), cache_source=str(launch.CACHE_SOURCE),
        cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs',
        slots=[launch.ARM_SLOTS[arm]], cpu_count=4, memory_gib=9, wandb_mode='offline',
        feature_root=str(launch.FEATURE_ROOTS[arm]), routing_feature_root=str(launch.ROUTING_ROOT),
        routing_mode='reuse_original_R94B', cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]],
        source_files=sorted({str(Path(path)) for path in sources}), stages=launch.expected_stages(arm))
    launch.locked_protocol(spec['science_protocol'], spec)
    return spec


def write_spec(arm):
    spec = make_spec(arm)
    path = launch.ROOT / 'specs' / ('R102_' + arm + '.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve the original R102 spec; no implicit replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('arm', choices=tuple(launch.ARM_SLOTS))
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    print(write_spec(args.arm))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
