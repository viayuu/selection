"""Read-only CPU diagnosis of the concrete R102 frozen-TEST header failure."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


REPO = Path(__file__).resolve().parents[2]
CACHE = Path('/tmp/shiys_R84_new90k_cache/test')
MANIFEST = REPO / 'code/V4/runs/R84_new90k_attention/prepared/source_cache_manifest.json'
DEST = REPO / 'code/V4/runs/R102_full_type_pretrained_fusion/preparation/postprocess_recovery'
CONTAINERS = {
    'gpu03': ('1465', 'b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c'),
    'gpu05': ('1475', '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'),
}


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=tuple(CONTAINERS), required=True)
    parser.add_argument('--mode', choices=('headers', 'hashes'), default='headers')
    args = parser.parse_args(argv)
    job, identity = CONTAINERS[args.node]
    if (os.environ.get('SLURM_JOB_ID') != job
            or identity not in Path('/proc/self/cgroup').read_text()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise RuntimeError('Use the original allocation container with CPU-only visibility')
    target = DEST / (args.node + '_' + args.mode + '.json')
    if target.exists():
        raise RuntimeError('Preserve the original one-shot diagnosis')
    reference = json.loads(MANIFEST.read_text())['splits']['test']
    records = {}
    for problem, frozen in reference.items():
        directory = CACHE / problem
        actual = json.loads((directory / 'metadata.json').read_text())
        differences = {key: dict(actual=actual.get(key), reference=frozen.get(key))
                       for key in sorted(actual.keys() | frozen.keys())
                       if key not in ('directory', 'cache_root')
                       and actual.get(key) != frozen.get(key)}
        row = dict(header_differences=differences)
        if args.mode == 'hashes':
            if actual['arrays'] != frozen['arrays']:
                raise ValueError('Array shape/dtype/filename schema differs')
            row['arrays'] = {}
            for key, spec in frozen['arrays'].items():
                path = directory / spec['file']
                if not path.is_file() or path.is_symlink():
                    raise ValueError('Require original regular array files')
                row['arrays'][key] = dict(file=spec['file'], bytes=path.stat().st_size,
                    sha256=file_hash(path))
        records[problem] = row
    value = dict(node=args.node, host=socket.gethostname(), allocation=job,
        container_id=identity, mode=args.mode, CPU_only=True, no_GPU_context=True,
        cache_modified=False, raw_values_interpreted=False,
        arrays_hashed=args.mode == 'hashes', problems=records)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(saved=str(target), node=args.node, mode=args.mode,
        different_header_fields=sorted({key for row in records.values()
                                       for key in row['header_differences']}))), flush=True)


if __name__ == '__main__':
    main()
