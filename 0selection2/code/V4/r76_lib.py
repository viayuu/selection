"""Frozen R76 library evaluation on the existing NSS-comparison datasets."""

import argparse
import csv
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from .nss_comparison import OUTPUT as NSS_ROOT, infer, read_costs
from .performance_evaluation import decision_metrics
from .r41_frozen_evaluation import attach_geometry
from .r76_experiment import file_hash, fixed_sbs_metrics, load_r76_checkpoint, read_json, utc, write_json
from .r76_old_test import verify_checkpoint
from .train import configure_torch, make_loader


RUN = Path('code/V4/runs/R76_R45A_single_reld_mt')
SCENARIO = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
DATASETS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}


@torch.inference_mode()
def evaluate(args):
    checkpoint_path, lock = verify_checkpoint(args.run_root)
    result_path = args.root / f'{args.dataset}_results.json'
    if result_path.exists():
        raise RuntimeError('This LIB evaluation is already complete; use --stage analysis without reinference')
    configure_torch()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, _ = load_r76_checkpoint(checkpoint_path, device)
    model.eval()
    problem = DATASETS[args.dataset]
    raw = read_costs(problem, args.dataset)
    scenario = read_json(SCENARIO / 'scenario_manifest.json')
    record = scenario['problems'][problem]
    if record['methods'] != raw['pool'] or record['global_solver_ids'] != raw['pool_ids']:
        raise ValueError('The library cost columns differ from R76 candidate identities')
    directory = Path(raw['directory'])
    sources = dict(dataset=str(directory / 'dataset.pkl'), labels=str(directory / 'raw_label.pkl'),
                   dataset_sha256=file_hash(directory / 'dataset.pkl'),
                   labels_sha256=file_hash(directory / 'raw_label.pkl'))
    protocol = dict(dataset=args.dataset, problem=problem, checkpoint=str(checkpoint_path),
        checkpoint_sha256=lock['checkpoint_sha256'], checkpoint_epoch=lock['epoch'],
        candidate_pool=raw['pool'], pool_ids=raw['pool_ids'], sources=sources,
        normalization='R76 TRAIN statistics in checkpoint; no LIB refitting',
        decision='single forward, greedy maximin-logit argmax; no solver execution or augmentation',
        sbs='LIB post-hoc minimum mean-cost fixed solver, for historical NSS comparison',
        auxiliary_sbs='R76 TRAIN-selected fixed solver evaluated on LIB',
        top1_primary='native raw_label ind', top1_nss_comparison='FP32 cost argmin, NSS historical convention',
        batch_size=args.batch_size, checkpoint_reselection=False, created_utc=utc())
    write_json(args.root / f'{args.dataset}_protocol.json', protocol)
    loader = make_loader(problem, args.dataset, args.batch_size, 0, shuffle=False, cache_device=device)
    if loader.size != len(raw['winner']):
        raise ValueError('Library inputs and original cost rows have different lengths')
    np.testing.assert_array_equal(loader.batch['pool_ids'].cpu().numpy(), raw['pool_ids'])
    np.testing.assert_array_equal(loader.batch['ind'].cpu().numpy(), raw['winner'])
    started = time.monotonic()
    attach_geometry(loader, device)
    path = args.root / 'predictions' / f'{args.dataset}.npz'
    nss_metrics = infer(model, loader, args.batch_size, raw, path)
    with np.load(path, allow_pickle=False) as saved:
        native, pred = decision_metrics(saved['scores'], raw)
        np.testing.assert_array_equal(pred, saved['pred'])
        train_sbs, _ = fixed_sbs_metrics(saved['scores'], raw, record['train_sbs'])
    native['correct'] = int(np.count_nonzero(pred == raw['winner']))
    result = dict(native=native, nss_comparable=nss_metrics, train_sbs_reference=train_sbs,
        protocol=protocol, host=socket.gethostname(),
        device=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        seconds=time.monotonic() - started, completed_utc=utc())
    write_json(result_path, result)
    print(f"[R76 {args.dataset}] n={native['n']} native Top1={native['top1']*100:.4f}% "
          f"NSS-convention Top1={nss_metrics['top1']*100:.4f}% mean_cost={native['mean_cost']:.6f}", flush=True)


def analyze(args):
    _, lock = verify_checkpoint(args.run_root)
    rows, results = [], {}
    with (NSS_ROOT / 'summary.csv').open(newline='') as stream:
        historical = {row['dataset']: row for row in csv.DictReader(stream)
                      if row['model_version'] == 'NSS' and row['dataset'] in DATASETS}
    for dataset in DATASETS:
        result = results[dataset] = read_json(args.root / f'{dataset}_results.json')
        if result['protocol']['checkpoint_sha256'] != lock['checkpoint_sha256']:
            raise ValueError('Library results belong to a different checkpoint')
        native, nss = result['native'], result['nss_comparable']
        baseline = historical[dataset]
        if (int(baseline['n']) != native['n']
                or abs(float(baseline['sbs_cost']) - native['sbs_cost']) > 1e-6
                or abs(float(baseline['oracle_cost']) - native['oracle_cost']) > 1e-6):
            raise ValueError('Historical NSS and R76 do not share library reference costs')
        rows.append(dict(model='R76', dataset=dataset, n=native['n'],
            top1_pct=native['top1']*100, top2_pct=native['top2']*100, top3_pct=native['top3']*100,
            nss_convention_top1_pct=nss['top1']*100, mean_cost=native['mean_cost'],
            sbs_method=native['sbs_name'], sbs_cost=native['sbs_cost'], vs_sbs_pct=native['vs_sbs_pct'],
            oracle_cost=native['oracle_cost'], vs_oracle_pct=native['vs_oracle_pct'],
            actual_regret_pct=native['actual_regret_pct'],
            nss_top1_pct=float(baseline['top1'])*100,
            nss_mean_cost=float(baseline['mean_cost']),
            cost_difference_vs_nss=native['mean_cost']-float(baseline['mean_cost'])))
    with (args.root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.root / 'results.json', results)
    lines = ['# R76 TSPLIB / CVRPLIB', '',
        f"Frozen validation-selected checkpoint: epoch {lock['epoch']}; `{lock['checkpoint_sha256']}`.",
        'No retraining, solver runs, input augmentation, or selection using LIB results.',
        'The original NSS comparison instances, preprocessing, pools, and FP64 costs are unchanged.', '',
        '| Dataset | Instances | Top1 native | Top2 native | Top3 native | Mean cost | vs SBS | vs Oracle | Actual regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['n']} | {row['top1_pct']:.4f}% | {row['top2_pct']:.4f}% | "
            f"{row['top3_pct']:.4f}% | {row['mean_cost']:.6f} | {row['vs_sbs_pct']:+.4f}% | "
            f"{row['vs_oracle_pct']:+.4f}% | {row['actual_regret_pct']:.4f}% |")
    lines += ['', '## Historical NSS Comparison',
        'Top1 in this table uses the historical NSS FP32-argmin convention for both models.',
        'TSPLIB has one row where that winner differs from native ind; CVRPLIB has no differences.',
        'NSS numbers are reused, not rerun. Cost metrics use common raw FP64 SBS/Oracle references.', '',
        '| Dataset | R76 Top1 NSS convention | NSS Top1 | R76 mean cost | NSS mean cost | R76 minus NSS cost |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['nss_convention_top1_pct']:.4f}% | {row['nss_top1_pct']:.4f}% | "
            f"{row['mean_cost']:.6f} | {row['nss_mean_cost']:.6f} | {row['cost_difference_vs_nss']:+.6f} |")
    lines += ['', 'SBS above is each LIB dataset post-hoc best fixed method, matching previous NSS reports.',
        'The separate train_sbs_reference object records R76 TRAIN-fixed SBS on the same LIB costs.',
        'These are cross-scenario evaluations of the new-data-trained R76 model, not matched-training-budget NSS experiments.',
        'Frozen R76 TRAIN normalization is retained. Predictions retain native winners, FP64 costs, and all candidate scores.', '',
        '## Command', '```bash', 'bash code/V4/run_v4_r76_lib.sh',
        'bash code/V4/run_v4_r76_lib.sh analysis  # saved predictions only', '```']
    (args.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R76_LIB', mode='offline', dir=str(args.root), config={'checkpoint': lock})
    run.log({row['dataset'] + '/' + key: value for row in rows for key, value in row.items()
             if isinstance(value, (int, float))})
    run.finish()
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('evaluate', 'analysis'), required=True)
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.root = (args.root or args.run_root / 'lib_evaluation').resolve()
    if args.stage == 'evaluate' and (args.dataset is None or args.batch_size < 1):
        parser.error('Evaluation requires a LIB dataset and positive batch size')
    args.root.mkdir(parents=True, exist_ok=True)
    (evaluate if args.stage == 'evaluate' else analyze)(args)


if __name__ == '__main__':
    main()
