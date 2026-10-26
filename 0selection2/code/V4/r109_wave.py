"""R109 draft-spec API. Default CLI prints JSON only; main writes final specs.

No readiness, source freeze, allocation or launch is performed here.
"""

import argparse
import json

from . import r109_launch as launch


def make_spec(arm, *, final=False):
    identifier = launch.run_id(arm)
    root = launch.MODEL_ROOT
    paths = dict(data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
                 cache_root=str(launch.CACHE_ROOT))
    science = launch.locked_protocol(root / 'protocol_lock.json', paths) if final else None
    return dict(schema_version=1, run_id=identifier, run_dir=str(root / arm), common_root=str(root),
        spec_draft=not final, main_thread_ready=False, science_protocol=str(root / 'protocol_lock.json'),
        **paths, cache_source=str(launch.CACHE_SOURCE), cache_kind='r84_inputs', wandb_mode='offline',
        slots=[launch.ARM_SLOTS[arm]], cpus=list(launch.SLOT_CPUS[launch.ARM_SLOTS[arm]]),
        cpu_count=4, memory_gib=9, stages=launch.expected_stages(arm, science),
        source_files=sorted({str(p) for p in launch.declared_sources()}))


def write_spec(arm):
    """Explicit main action; never overwrite a different spec or mark it ready."""
    spec = make_spec(arm, final=True)
    path = launch.ROOT / 'specs' / (launch.run_id(arm) + '.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve existing R109 spec; no implicit replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=launch.ARMS)
    parser.add_argument('--write-final', action='store_true', help='Explicit main-only spec write after protocol lock')
    args = parser.parse_args(argv)
    if args.write_final:
        print(write_spec(args.arm))
    else:
        print(json.dumps(make_spec(args.arm), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
