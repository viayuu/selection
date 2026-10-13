"""One authorized CPU-only byte-preserving copy of immutable feature groups."""

import json
import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition')
RECORD = ROOT / 'continuation8'
DESTINATION = Path('/tmp/shiys_r74_feature_cache_continuation8')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    if socket.gethostname().split('.')[0] != 'gpu03':
        raise ValueError('The immutable feature mirror belongs only on gpu03')
    manifest = json.loads((ROOT / 'feature_cache_complete.json').read_text())
    if not manifest['complete'] or DESTINATION.exists() or (RECORD / 'mirror_started.json').exists():
        raise ValueError('Require complete generation and one exclusive new mirror copy')
    DESTINATION.mkdir(mode=0o700)
    with (RECORD / 'mirror_started.json').open('x') as stream:
        json.dump(dict(utc=now(), host=socket.gethostname(), pid=os.getpid(),
                       destination=str(DESTINATION), CPU_only=True), stream, indent=2)
    started = time.monotonic()
    directories, files, total = {}, [], 0
    try:
        for task in manifest['per_task'].values():
            for group in task['groups']:
                source = Path(group['directory'])
                relative = source.relative_to(ROOT)
                destination = DESTINATION / relative
                destination.mkdir(parents=True, exist_ok=False)
                directories[str(source)] = str(destination)
                names = [key + '.npy' for key in group['specs']]
                names += ['source_indices.npy', 'metadata.json']
                for name in names:
                    incoming, outgoing = source / name, destination / name
                    before = incoming.stat()
                    with incoming.open('rb') as src, outgoing.open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    after = incoming.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('An original immutable feature file changed during copy: ' + str(incoming))
                    if outgoing.stat().st_size != before.st_size:
                        raise ValueError('Incomplete ordinary file copy: ' + str(outgoing))
                    shutil.copystat(incoming, outgoing)
                    outgoing.chmod(0o444)
                    files.append(dict(source=str(incoming), destination=str(outgoing),
                                      bytes=before.st_size, source_mtime_ns=before.st_mtime_ns))
                    total += before.st_size
                destination.chmod(0o555)
                if len(directories) % 100 == 0:
                    progress = dict(utc=now(), groups=len(directories), files=len(files), bytes=total,
                                    seconds=time.monotonic() - started, complete=False)
                    write(RECORD / 'mirror_progress.json', progress)
                    print(json.dumps(progress), flush=True)
        receipt = dict(completed_utc=now(), complete=True, host=socket.gethostname(), pid=os.getpid(),
                       destination=str(DESTINATION), directories=directories, files=files,
                       groups=len(directories), file_count=len(files), bytes=total,
                       seconds=time.monotonic() - started, copy_buffer_bytes=1024 * 1024,
                       original_manifest=str(ROOT / 'feature_cache_complete.json'),
                       copy_method='ordinary binary file copy, source size/mtime stable, destination size equal',
                       readonly_files=True, hashes_or_arrays_loaded=False, model_forwards=0,
                       new_feature_generation=False, mmap_lru_capacity_unchanged=128)
        write(RECORD / 'mirror_manifest.json', receipt)
        write(DESTINATION / 'mirror_manifest.json', receipt)
        (DESTINATION / 'mirror_manifest.json').chmod(0o444)
        DESTINATION.chmod(0o555)
        print(json.dumps({key: value for key, value in receipt.items() if key not in ('directories', 'files')}), flush=True)
    except BaseException as error:
        import traceback
        write(RECORD / 'mirror_failure.json', dict(utc=now(), error=str(error), traceback=traceback.format_exc(),
              groups=len(directories), files=len(files), bytes=total, seconds=time.monotonic()-started,
              automatic_retry=False, original_training_untouched=True))
        raise


if __name__ == '__main__':
    main()
