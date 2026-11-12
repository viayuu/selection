"""One curated R127 commit, only after locked TEST/LIB and PDFs are complete."""

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from .r76_experiment import file_hash, read_json, utc, write_json
from .r127_experiment import ARM, DEFAULT_ROOT


REPO = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ('r127_data.py', 'r127_experiment.py', 'r127_launch.py', 'r127_reporting.py',
    'r127_publish.py', 'test_r127_data.py', 'test_r127_experiment.py',
    'test_r127_launch.py', 'test_r127_reporting.py')
READ_ONLY_SHARED_SOURCES = ('r125_data.py', 'r125_launch.py', 'r125_reporting.py',
    'r126_data.py', 'r126_experiment.py', 'r126_launch.py', 'r126_reporting.py')


def command(*args):
    return subprocess.run(args, cwd=REPO, check=True, text=True, capture_output=True).stdout.strip()


def publish(root=DEFAULT_ROOT, push=False):
    root = Path(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError('This publisher owns R127 only')
    run = root / ARM
    with (root / 'publication.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finished, tested = read_json(run / 'postprocess_status.json'), read_json(run / 'test_results.json')
        lock = read_json(run / 'final_test_lock.json')
        if (finished.get('complete') is not True or tested.get('complete') is not True
                or tested['checkpoint_sha256'] != lock['checkpoint_sha256']
                or file_hash(run / 'best.pt') != lock['checkpoint_sha256']):
            raise ValueError('Complete locked TEST/LIB and PDFs before publishing')
        receipt = root / 'github_publication.json'
        if receipt.exists():
            previous = read_json(receipt)
            if previous.get('commit'):
                if push and not previous.get('pushed'):
                    command('git', 'push', 'origin', 'HEAD')
                    previous.update(pushed=True, pushed_utc=utc())
                    write_json(receipt, previous)
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
        lines = ['# R127: R83 model-only source selector without MVMOE', '',
            'One fresh seed2 full18 model. BQ/LEHD/MoSES_RF retained wherever originally legal.',
            'Original new90k memberships; only MVMOE is additionally removed.', '',
            '| Split | Top1 (%) | actual regret (%) | macro mean cost |',
            '| --- | ---: | ---: | ---: |']
        for split in ('train', 'val', 'test'):
            values = metrics[split]['macro']
            lines.append('| {} | {:.4f} | {:.5f} | {:.8f} |'.format(split.upper(),
                values['macro_top1'] * 100, values['macro_actual_regret_pct'], values['macro_mean_cost']))
        lines += ['', 'R83 model-only source roles, original complete dual stream and original R77 loss.',
            'SBS fitted on TRAIN after filtering; FP64 native winner recomputed in each legal pool.',
            'ALL weights tasks equally; cross-problem raw mean cost is descriptive only.',
            'Candidate deletion changes the Oracle and classification difficulty; Top1 alone is not a causal test of imbalance.',
            'One VAL-selected checkpoint for TEST and LIB; no TEST/LIB retuning.',
            'Primary: model_only_seed2/full_model_statistics/MODEL_STATISTICS.pdf.',
            'Checkpoint SHA256: ' + lock['checkpoint_sha256'], '']
        (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        paths = [REPO / 'code/V4' / name for name in (*SOURCE_NAMES, *READ_ONLY_SHARED_SOURCES)]
        paths.append(REPO / 'run_v4_r127.sh')
        paths += [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
            and p.suffix in ('.md', '.json', '.csv', '.pdf', '.png')
            and not any(part in ('wandb', 'mplconfig', '__pycache__', 'probes') for part in p.parts)
            and p.name != 'github_publication.json']
        files = sorted({p.resolve() for p in paths if p.is_file()})
        if any(p.stat().st_size > 40 * 1024 * 1024 for p in files):
            raise ValueError('Do not commit checkpoints or large arrays')
        relative = [str(p.relative_to(REPO)) for p in files]
        command('git', 'add', '--', *relative)
        command('git', 'commit', '--only', '-m', 'research: R127 R83 source selector without MVMOE', '--', *relative)
        record = dict(experiment='R127', commit=command('git', 'rev-parse', 'HEAD'),
            checkpoint_sha256=lock['checkpoint_sha256'], files=relative, pushed=False, completed_utc=utc())
        write_json(receipt, record)
        if push:
            command('git', 'push', 'origin', 'HEAD')
            record.update(pushed=True, pushed_utc=utc())
            write_json(receipt, record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.push), indent=2))
