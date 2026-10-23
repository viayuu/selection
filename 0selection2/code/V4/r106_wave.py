"""Write R106 specs only. Drafts never create readiness or launch anything."""

import argparse
import json

from . import r106_launch as launch


def make_spec(arm, *, final=False):
    identifier = launch.run_id(arm)
    root = launch.MODEL_ROOT
    spec = dict(schema_version=1, run_id=identifier, run_dir=str(root / arm), common_root=str(root),
        science_protocol=str(root / 'protocol_lock.json'), data_root=str(launch.DATA_ROOT),
        prepared_root=str(launch.PREPARED_ROOT), cache_source=str(launch.CACHE_SOURCE),
        cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs', wandb_mode='offline',
        slots=[launch.ARM_SLOTS[arm]], cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, spec_draft=not final, main_thread_ready=False,
        source_files=sorted({str(p) for p in launch.declared_sources()}))
    science = launch.locked_protocol(spec['science_protocol'], spec) if final else None
    spec['stages'] = launch.expected_stages(arm, science)
    return spec


def write_spec(arm, *, final=False):
    spec = make_spec(arm, final=final)
    directory = 'specs' if final else 'spec_drafts'
    path = launch.ROOT / directory / (launch.run_id(arm) + '.json')
    if path.exists() and launch.base.read(path) != spec:
        raise RuntimeError('Preserve existing R106 spec; no implicit replacement')
    if not path.exists():
        launch.base.save(path, spec, exclusive=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=launch.ARMS)
    parser.add_argument('--final', action='store_true', help='Require the final main science lock; still not ready')
    parser.add_argument('--print-only', action='store_true')
    args = parser.parse_args(argv)
    if args.print_only:
        print(json.dumps(make_spec(args.arm, final=args.final), indent=2))
    else:
        print(write_spec(args.arm, final=args.final))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
