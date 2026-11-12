"""Curated one-version publication for a completed R126 run only."""

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from .r76_experiment import file_hash, read_json, utc, write_json
from .r126_experiment import ARM, DEFAULT_ROOT


REPO = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ('r126_data.py', 'r126_experiment.py', 'r126_launch.py',
    'r126_reporting.py', 'r126_publish.py', 'test_r126_data.py',
    'test_r126_experiment.py', 'test_r126_launch.py', 'test_r126_reporting.py')
READ_ONLY_SHARED_SOURCES = ('r125_data.py', 'r125_launch.py', 'r125_reporting.py')


def command(*args):
    return subprocess.run(args, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


def publish(root=DEFAULT_ROOT, push=False):
    root = Path(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError('This publisher owns R126 only, never R125 or unrelated changes')
    run = root / ARM
    with (root / 'publication.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finished, tested = read_json(run / 'postprocess_status.json'), read_json(run / 'test_results.json')
        lock = read_json(run / 'final_test_lock.json')
        if (finished.get('complete') is not True or tested.get('complete') is not True
                or tested['checkpoint_sha256'] != lock['checkpoint_sha256']
                or file_hash(run / 'best.pt') != lock['checkpoint_sha256']):
            raise ValueError('Complete locked TEST/LIB and PDFs before publishing')
        result_path = root / 'github_publication.json'
        if result_path.exists():
            previous = read_json(result_path)
            if previous.get('commit'):
                if push and not previous.get('pushed'):
                    command('git', 'push', 'origin', 'HEAD')
                    previous.update(pushed=True, pushed_utc=utc())
                    write_json(result_path, previous)
                return previous
        config, metrics = read_json(run / 'protocol.json'), read_json(run / 'final_metrics.json')
        data = Path(config['data_root'])
        directory = root / 'data_release'
        directory.mkdir(exist_ok=True)
        for name in ('scenario_manifest.json', 'data_preparation.json', 'baselines.json'):
            write_json(directory / name, read_json(data / name))
        for name in ('candidate_filter_summary.csv', 'winner_migration.csv'):
            if (data / name).exists():
                (directory / name).write_bytes((data / name).read_bytes())
        lines = ['# R126: R83 model-only source selector without MoSES_RF', '',
            'One fresh seed2 full18 model. BQ and LEHD are retained. Original frozen new90k splits.', '',
            '| Split | Top1 (%) | actual regret (%) | macro mean cost |',
            '| --- | ---: | ---: | ---: |']
        for split in ('train', 'val', 'test'):
            values = metrics[split]['macro']
            lines.append('| {} | {:.4f} | {:.5f} | {:.8f} |'.format(split.upper(),
                values['macro_top1'] * 100, values['macro_actual_regret_pct'], values['macro_mean_cost']))
        lines += ['', 'Only encoder/decision MODEL code is embedded; inference/config/train code is excluded.',
            'SBS identities are fitted on TRAIN after candidate filtering; native winner uses FP64 costs.',
            'ALL weights problems equally; cross-problem raw mean cost is descriptive only.',
            'Historical R83 used older data; R125 uses a different source scope and candidate pool.',
            'Neither is a matched single-factor control. Removing a candidate also changes the Oracle.',
            'The same VAL-selected checkpoint is used for TEST and both LIBs; no TEST/LIB retuning.',
            'Primary: model_only_seed2/full_model_statistics/MODEL_STATISTICS.pdf.',
            'Checkpoint SHA256: ' + lock['checkpoint_sha256'], '']
        (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        files = [REPO / 'code/V4' / name for name in (*SOURCE_NAMES, *READ_ONLY_SHARED_SOURCES)]
        files.append(REPO / 'run_v4_r126.sh')
        files += [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
            and p.suffix in ('.md', '.json', '.csv', '.pdf', '.png')
            and not any(part in ('wandb', 'mplconfig', '__pycache__', 'probes') for part in p.parts)
            and p.name != 'github_publication.json']
        files = sorted({p.resolve() for p in files if p.is_file()})
        if any(p.stat().st_size > 40 * 1024 * 1024 for p in files):
            raise ValueError('Do not commit checkpoints or large data arrays')
        paths = [str(p.relative_to(REPO)) for p in files]
        command('git', 'add', '--', *paths)
        command('git', 'commit', '--only', '-m', 'research: R126 R83 model-only selector without MoSES_RF', '--', *paths)
        record = dict(experiment='R126', commit=command('git', 'rev-parse', 'HEAD'),
            checkpoint_sha256=lock['checkpoint_sha256'], files=paths, pushed=False, completed_utc=utc())
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
