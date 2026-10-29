"""Explicit one-attempt R122 TEST recovery; resource metadata only, no reselection.

The frozen runner remains the TEST producer. Only TEST cache metadata accepts
the same six cosmetic transport fields as return18_launch; arrays are not edited.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import sys
import traceback
from types import FunctionType

from . import r122_experiment as runner
from . import return18_launch as transport


COSMETIC_FIELDS = frozenset(('directory', 'cache_root', 'created_utc',
                             'completed_utc', 'seconds', 'test_lock_root'))
ORIGINAL_VALIDATE = runner.validate_cache_copy


def test_cache_copy(directory, expected, *, label_reference):
    """Keep the original typed-array checks; relax only TEST cosmetic metadata."""
    directory, reference = Path(directory), Path(label_reference)
    if expected.get('split') != 'test':
        return ORIGINAL_VALIDATE(directory, expected, label_reference=reference)
    if (transport.PORTABLE_CACHE_FIELDS != COSMETIC_FIELDS
            or directory.parent.name != 'test' or reference.parent.name != 'test'
            or directory.name != expected['problem'] or reference.name != expected['problem']):
        raise ValueError('Recovery permits only the declared TEST task and six cosmetic fields')
    path, source = directory / 'metadata.json', reference / 'metadata.json'
    actual, frozen = runner.read_json(path), runner.read_json(source)
    semantic = lambda value: {k: v for k, v in value.items() if k not in COSMETIC_FIELDS}
    if semantic(actual) != semantic(expected) or semantic(frozen) != semantic(expected):
        raise ValueError('TEST cache semantic source/pool/array identity differs from frozen manifest')
    before = {str(p): runner.file_hash(p) for p in (path, source)}
    proof = transport.portable_cache_identity(path, source)
    # Isolate this single metadata read, not runner.read_json or any model global.
    original_read = ORIGINAL_VALIDATE.__globals__['read_json']
    checked_globals = dict(ORIGINAL_VALIDATE.__globals__, read_json=lambda p:
        expected if Path(p).resolve() == path.resolve() else original_read(p))
    FunctionType(ORIGINAL_VALIDATE.__code__, checked_globals)(
        directory, expected, label_reference=reference)
    if any(runner.file_hash(Path(p)) != digest for p, digest in before.items()):
        raise ValueError('TEST metadata changed during resource validation')
    return dict(problem=expected['problem'], expected_metadata=expected,
                metadata_sha256=before, **proof)


@contextmanager
def resource_adapter(proofs):
    original = runner.validate_cache_copy
    if original is not ORIGINAL_VALIDATE:
        raise RuntimeError('Another cache adapter is already installed; do not stack overrides')
    def validate(directory, expected, *, label_reference):
        proof = test_cache_copy(directory, expected, label_reference=label_reference)
        if proof is not None:
            proofs.append(proof)
    runner.validate_cache_copy = validate
    try:
        yield
    finally:
        runner.validate_cache_copy = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', required=True, help='Parent-assigned cuda:0; no auto-selection')
    parser.add_argument('--cache-root', type=Path, default=runner.LOCAL_CACHE_ROOT)
    parser.add_argument('--state-hash')
    options = parser.parse_args(argv)
    root = options.root.expanduser().resolve()
    run = root / 'model'
    config = runner.read_json(run / 'protocol.json')
    args = runner.parser().parse_args(['test', '--root', str(root), '--arm', 'model',
        '--device', options.device, '--cache-root', str(options.cache_root.expanduser().resolve()),
        '--prepared-root', config['prepared_root'], '--data-root', config['data_root'],
        '--preparation-root', config['preparation_root'],
        '--state-hash', options.state_hash or runner.engine.json_hash(config)])
    for key in ('root', 'cache_root', 'prepared_root', 'data_root', 'preparation_root'):
        setattr(args, key, Path(getattr(args, key)).expanduser().resolve())
    device = runner.torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Recovery delegates formal TEST only on an explicit parent-assigned CUDA device')
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock, checked = runner.authorize_test(args)
        if checked != config:
            raise PermissionError('Original locked protocol changed')
        if any((run / name).exists() for name in ('test_started.json', 'test_results.json', 'test_complete.json')):
            raise PermissionError('TEST has already started/completed; inspect it, never auto-retry')
        error = runner.read_json(run / 'error.json')
        if (error.get('type') != 'ValueError'
                or not error.get('message', '').startswith('Resource cache must retain original metadata/source identity:')
                or '/test/' not in error['message'] or 'validate_cache_copy' not in error.get('traceback', '')):
            raise PermissionError('Recovery is only for the preserved pre-forward TEST metadata failure')
        output = run / 'test_recovery_v1'
        output.mkdir(exist_ok=False)
        protected = [run / name for name in ('protocol.json', 'training_complete.json',
                                              'final_test_lock.json', 'error.json')]
        for name in ('error.json', 'status.json', 'postprocess_status.json'):
            if (run / name).is_file():
                shutil.copy2(run / name, output / ('previous_' + name))
        proofs = []
        record = dict(schema='r122-test-resource-recovery-v1', complete=False, phase='started',
            started_utc=runner.utc(), original_test='code.V4.r122_experiment.test',
            adapter_sha256=runner.file_hash(Path(__file__)),
            transport_sha256=runner.file_hash(Path(transport.__file__)),
            original_source_sha256=config['source_sha256'], checkpoint_lock=lock,
            protected_sha256={str(p): runner.file_hash(p) for p in protected},
            cosmetic_fields=sorted(COSMETIC_FIELDS), cache_proofs=proofs,
            input_proof='typed arrays + exact fixed rows, NOT full input hash',
            automatic_retry=False, selection_changed=False, inference_changed=False)
        receipt = output / 'resource_adapter.json'
        runner.write_json(receipt, record)
        try:
            runner.torch.set_num_threads(2)
            runner.configure_torch()
            with resource_adapter(proofs):
                result = runner.test(args, device)
            if {p['problem'] for p in proofs} != set(runner.PROBLEMS) or len(proofs) != 18:
                raise ValueError('Require exactly18 TEST resource equivalence proofs')
            after, _ = runner.authorize_test(args)
            if after != lock or any(runner.file_hash(Path(p)) != h for p, h in record['protected_sha256'].items()):
                raise PermissionError('Recovery changed a protected original artifact')
            record.update(complete=True, phase='complete', result_identity=result['identity'],
                          test_results_sha256=runner.file_hash(run / 'test_results.json'))
        except BaseException:
            record.update(phase='failed', error=traceback.format_exc())
            raise
        finally:
            record['ended_utc'] = runner.utc()
            runner.write_json(receipt, record)
        print(json.dumps(dict(complete=True, receipt=str(receipt), checkpoint_sha256=lock['checkpoint_sha256'])), flush=True)
        return record


if __name__ == '__main__':
    main(sys.argv[1:])
