"""Finalize an explicitly cancelled R40A from its last saved checkpoint."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .direct_selector import make_r40_model
from .multitask_probe import dump, file_hash
from .r40_experiment import ROOT, RUN_NAMES, evaluate, prepare_data
from .train import configure_torch


def finalize(args):
    directory = args.root / RUN_NAMES['A']
    history = json.loads((directory / 'history.json').read_text())
    config = json.loads((directory / 'args.json').read_text())
    checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
    final = history[-1]
    if checkpoint['epoch'] + 1 != final['epoch'] or checkpoint['successful_updates'] != final['updates']:
        raise ValueError('The saved checkpoint and completed validation history do not match')
    if (directory / 'result.json').exists():
        raise ValueError('Do not overwrite an already finalized result')
    shutil.copy2(directory / 'history.json', directory / 'history_before_user_stop.json')
    stopped = dict(group='A', reason='explicit user request: validation has deteriorated; stop A and summarize',
                   recorded_at=datetime.now(timezone.utc).isoformat(), requested_epochs=config['epochs'],
                   checkpoint_epoch=final['epoch'], checkpoint_updates=final['updates'],
                   checkpoint_hash=file_hash(directory / 'last.pt'),
                   incomplete_epoch_updates='not part of the checkpoint or final metrics', test_read=False)
    dump(args.root / 'user_stop.json', stopped)
    protocol = json.loads((args.root / 'protocol.json').read_text())
    changes = {}
    for name in ('r40_analysis.py', 'r40_verify.py'):
        original = file_hash(args.root / 'source_launch' / name)
        if original != protocol['source_hashes'][name]:
            raise ValueError('The original launch source snapshot changed')
        changes[name] = dict(launch_hash=original, current_hash=file_hash(Path(__file__).parent / name))
    dump(args.root / 'postprocess_source_changes.json',
         dict(reason='explicit user stop; reporting and verification accept the saved A endpoint, not a completed 60/60 pair',
              files=changes, finalizer_hash=file_hash(Path(__file__)), training_code_changed=False))
    loaders, raw, _ = prepare_data(args)
    model = make_r40_model(config['model_params']).to(args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    train = evaluate(model, loaders['train'], raw['train'], args.eval_batch_size,
                     directory / 'predictions' / 'train_final')
    dump(directory / f"train_eval_epoch{final['epoch']:03d}.json", train)
    final['train_eval'] = train['macro']
    final['train_eval_after_user_stop'] = True
    dump(directory / 'history.json', history)
    result = dict(group='A', seed=2, epochs=final['epoch'], updates=final['updates'],
                  updates_by_problem=final['updates_by_problem'], skipped_attempts=final['skipped_attempts_total'],
                  best=min(history, key=lambda r: r['val']['macro_vs_sbs_pct']), final=final,
                  elapsed_seconds=final['elapsed_seconds'],
                  last5={k: float(np.mean([r['val'][k] for r in history[-5:]])) for k in final['val']},
                  train_final=train['macro'], encoder_initial_hash=config['encoder_initial_hash'],
                  schedule_hash=config['schedule_hash'], source_hashes=config['source_hashes'],
                  stopped_by_user=True, completed_budget=False, stop_metadata='user_stop.json', test_read=False)
    dump(directory / 'result.json', result)
    print(f"[finalized] A stopped by user; epoch={final['epoch']} saved_updates={final['updates']}; no test", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(1)
    finalize(args)
