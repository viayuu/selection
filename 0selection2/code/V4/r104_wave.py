"""Write exact future R104 specs only; never ready, queue or launch on import."""

import argparse

from . import r104_launch as launch


def make_spec(arm):
    identifier = launch.run_id(arm)
    root = launch.MODEL_ROOT
    spec = dict(schema_version=1, run_id=identifier, run_dir=str(root / arm), common_root=str(root),
        science_protocol=str(root / 'protocol_lock.json'), data_root=str(launch.DATA_ROOT),
        prepared_root=str(launch.PREPARED_ROOT), cache_source=str(launch.CACHE_SOURCE),
        cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs', wandb_mode='offline',
        slots=[launch.ARM_SLOTS[arm]], cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, stages=launch.expected_stages(arm),
        source_files=sorted({str(p) for p in launch.declared_sources()}))
    launch.locked_protocol(spec['science_protocol'], spec)
    launch.source_files(spec)
    return spec


def write_spec(arm):
    spec = make_spec(arm)
    path = launch.ROOT / 'specs' / (launch.run_id(arm) + '.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve existing R104 spec; no implicit replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=launch.ARMS)
    args = parser.parse_args(argv)
    print(write_spec(args.arm))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
