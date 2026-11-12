"""Publish only the completed R125 version, leaving unrelated work untouched."""

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from .r76_experiment import file_hash, read_json, utc, write_json
from .r125_experiment import ARM, DEFAULT_ROOT


REPO = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    'r125_data.py', 'r125_experiment.py', 'r125_launch.py', 'r125_reporting.py',
    'r125_publish.py', 'test_r125_data.py',
    'test_r125_experiment.py', 'test_r125_launch.py', 'test_r125_reporting.py',
)


def command(*args):
    return subprocess.run(args, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


def publish(root=DEFAULT_ROOT, push=False):
    root = Path(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError('This publisher owns only R125')
    run = root / ARM
    with (root / 'publication.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finished = read_json(run / 'postprocess_status.json')
        tested = read_json(run / 'test_results.json')
        lock = read_json(run / 'final_test_lock.json')
        if (finished.get('complete') is not True or tested.get('complete') is not True
                or tested['checkpoint_sha256'] != lock['checkpoint_sha256']
                or file_hash(run / 'best.pt') != lock['checkpoint_sha256']):
            raise ValueError('Finish the locked TEST and all reports before publishing')
        result_path = root / 'github_publication.json'
        if result_path.exists():
            previous = read_json(result_path)
            if previous.get('pushed') is True:
                return previous
            if previous.get('commit'):
                if push:
                    command('git', 'push', 'origin', 'HEAD')
                    previous.update(pushed=True, pushed_utc=utc())
                    write_json(result_path, previous)
                return previous
        config = read_json(run / 'protocol.json')
        source_data = Path(config['data_root'])
        data_manifest = root / 'data_release'
        data_manifest.mkdir(exist_ok=True)
        for name in ('scenario_manifest.json', 'data_preparation.json', 'baselines.json'):
            write_json(data_manifest / name, read_json(source_data / name))
        metrics = read_json(run / 'final_metrics.json')
        lines = ['# R125: classic source selector without BQ/LEHD', '',
            'One fresh seed2 model, 18 tasks, original frozen new90k instance splits.',
            'BQ and LEHD are absent from the forward candidate pools and all losses.', '',
            '| Split | Top1 (%) | actual regret (%) | macro mean cost |',
            '| --- | ---: | ---: | ---: |']
        for split in ('train', 'val', 'test'):
            values = metrics[split]['macro']
            lines.append('| {} | {:.4f} | {:.5f} | {:.8f} |'.format(split.upper(),
                values['macro_top1'] * 100, values['macro_actual_regret_pct'], values['macro_mean_cost']))
        lines += ['', 'TRAIN-fitted SBS and FP64 Oracle are recomputed on the reduced pool.',
            'ALL weights tasks equally; cross-task raw mean cost is descriptive only.',
            'Architecture reference: R80D; its older data/full-pool result is NOT a matched causal control.',
            'Deleting candidates changes the task and Oracle. A Top1 increase alone does not prove family imbalance.',
            'Primary results: code_seed2/full_model_statistics/MODEL_STATISTICS.pdf.',
            'Distribution plots: code_seed2/full_model_statistics/selection_distributions.pdf.',
            'The same VAL-selected checkpoint is used for TEST and both LIBs; no TEST retuning.',
            'Checkpoint SHA256: ' + lock['checkpoint_sha256'], '']
        (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        files = [REPO / 'code/V4' / name for name in SOURCE_NAMES]
        files.append(REPO / 'run_v4_r125.sh')
        files += [path for path in root.rglob('*') if path.is_file() and not path.is_symlink()
            and path.suffix in ('.md', '.json', '.csv', '.pdf', '.png')
            and not any(part in ('wandb', 'mplconfig', '__pycache__', 'probes') for part in path.parts)
            and path.name != 'github_publication.json']
        files = sorted(set(path.resolve() for path in files if path.is_file()))
        if any(path.stat().st_size > 40 * 1024 * 1024 for path in files):
            raise ValueError('No checkpoints, input arrays, or large artifacts may enter this version commit')
        relative = [str(path.relative_to(REPO)) for path in files]
        # --only protects previously staged, unrelated user work as well.
        command('git', 'add', '--', *relative)
        command('git', 'commit', '--only', '-m', 'research: R125 source selector without BQ and LEHD', '--', *relative)
        sha = command('git', 'rev-parse', 'HEAD')
        record = dict(experiment='R125', commit=sha, checkpoint_sha256=lock['checkpoint_sha256'],
            files=relative, pushed=False, completed_utc=utc())
        write_json(result_path, record)
        if push:
            command('git', 'push', 'origin', 'HEAD')
            record.update(pushed=True, pushed_utc=utc())
            write_json(result_path, record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.push), indent=2))
