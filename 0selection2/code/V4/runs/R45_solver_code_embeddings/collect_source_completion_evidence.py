"""Read-only local provenance checks; never import or run a solver or call an API."""

import argparse
import ast
import csv
import datetime
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

from omegaconf import OmegaConf


WORKSPACE = Path('/public/home/shiys')
PROJECT = WORKSPACE / '0selection2'
EASY = WORKSPACE / 'EasyNCO'
EXPORTS = WORKSPACE / 'easynco_v3_bridge/exports'
OUTPUT = Path(__file__).resolve().parent
SOURCES = WORKSPACE / 'methods\u8bba\u6587_mineru_output/source_code'
PROBLEMS = ['TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
            'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW',
            'OVRPLTW', 'VRPBLTW', 'OVRPBLTW']
SPLITS = ('train', 'val', 'test')
NSS_NAMES = {'BQ': 'bq', 'OMNI': 'Omni', 'MVMOE': 'MVMoE'}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def vector_fingerprint(values):
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError('Non-finite archived score in provenance comparison')
        digest.update(struct.pack('!qd', index, value))
    return {'count': len(values), 'id_score_float64_sha256': digest.hexdigest()}


def csv_vector(path):
    values = []
    with path.open(newline='') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if int(row[0]) != len(values):
                raise ValueError(f'Non-contiguous archived CSV identifiers: {path}')
            values.append(float(row[1]))
    return values


def jsonl_vector(paths, field, local_indices=False):
    scores = {}
    offset = 0
    for path in paths:
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row.get('global_index', count))
                if local_indices:
                    if index != count:
                        raise ValueError(f'Unexpected local shard ordering: {path}')
                    index += offset
                if index in scores:
                    raise ValueError(f'Duplicate archived JSONL identifiers: {path}')
                scores[index] = float(row[field])
                count += 1
        offset += count
    if sorted(scores) != list(range(len(scores))):
        raise ValueError('Non-contiguous archived JSONL identifiers')
    return [scores[index] for index in range(len(scores))]


def archived_shards(problem, split):
    data = EASY / 'data/datasets'
    if problem == 'ATSP':
        path = (data / 'offline_init_v3/atsp/manifest.json' if split == 'train' else
                data / f'offline_init_v4/nss_style_splits_v1/atsp/ATSP{split}/manifest.json')
        return read_json(path)['shards']
    path = (data / 'offline_init_v4/mvrp_diverse_v1/manifest.json' if split == 'train' else
            data / f'offline_init_v4/nss_style_splits_v1/mvrp_diverse/{split}/manifest.json')
    return read_json(path)['variants'][problem]['shards']


def ancestor_candidates(problem, split, method):
    dataset = problem + split
    candidates = []
    if problem in ('TSP', 'CVRP'):
        nss = WORKSPACE / f'0selection/neural-solver-selection/datasets/{dataset}/results'
        candidates.append(('nss_import', 'csv', [nss / f'result_{NSS_NAMES.get(method, method)}.txt'], None, False))
        run = EASY / f'results/nss_eval/{method.lower()}_{problem.lower()}/{dataset}'
        for field in ('no_aug_score', 'aug_score'):
            candidates.append(('easynco_nss_eval', 'jsonl', [run / 'instance_results.jsonl'], field, False))
    if problem == 'ATSP' and method in ('GLOP', 'MATNET', 'MATPOENET'):
        run = (EASY / f'results/label_v3/{method.lower()}_atsp' if split == 'train' else
               EASY / f'results/label_v4/atsp/{split}/{method.lower()}')
        paths = [run / f"scale_{shard['scale']}/instance_results.jsonl" for shard in archived_shards(problem, split)]
        candidates.append(('easynco_atsp_shards', 'jsonl', paths, 'no_aug_score', True))
    if problem not in ('TSP', 'CVRP', 'ATSP') and method in ('MTPOMO', 'MVMOE'):
        run = EASY / f'results/label_v4/mvrp/{split}/{method.lower()}_{problem.lower()}'
        paths = [run / f"scale_{shard['scale']}/instance_results.jsonl" for shard in archived_shards(problem, split)]
        candidates.append(('easynco_mvrp_shards', 'jsonl', paths, 'no_aug_score', True))
    paper = {'RouteFinder': 'routefinder', 'MoSES_RF': 'moses_rf', 'MoSES_CaDA': 'moses_cada',
             'ICAM_ATSP': 'icam_atsp', 'UNICO_MatPOENet': 'unico_matpoenet'}
    if method in paper:
        paths = sorted((EXPORTS / f'paper_source_labels_v1/_shards/{paper[method]}/{dataset}').glob('*.jsonl'))
        candidates.append(('paper_source_shards', 'jsonl', paths, 'cost', False))
    reld = {'RELD_CVRP': ('reld_cvrp_single_no_aug_v1', 'RELD_CVRP'),
            'RELD_MOEL': ('reld_nss_like_no_aug_v1', 'RELD_MOEL'),
            'RELD_MTL': ('reld_nss_like_no_aug_v1', 'RELD_MTL')}
    if method in reld:
        directory, filename = reld[method]
        path = EXPORTS / f'{directory}/{dataset}/results/result_{filename}.txt'
        candidates.append(('reld_export', 'csv', [path], None, False))
    return candidates


def label_bindings():
    output = []
    for problem in PROBLEMS:
        for split in SPLITS:
            directory = PROJECT / f'data/{problem}{split}/results'
            for target in sorted(directory.glob('result_*.txt')):
                method = target.stem.removeprefix('result_')
                expected = vector_fingerprint(csv_vector(target))
                record = {'problem': problem, 'split': split, 'method': method,
                          'target': str(target), 'target_sha256': file_hash(target), **expected, 'ancestors': []}
                for origin, kind, paths, field, local in ancestor_candidates(problem, split, method):
                    item = {'origin': origin, 'paths': [str(path) for path in paths], 'field': field,
                            'local_indices_with_manifest_offsets': local}
                    if not paths or not all(path.is_file() for path in paths):
                        item['status'] = 'missing'
                    else:
                        try:
                            values = csv_vector(paths[0]) if kind == 'csv' else jsonl_vector(paths, field, local)
                            fingerprint = vector_fingerprint(values)
                            item.update(fingerprint)
                            item['status'] = 'match' if fingerprint == expected else 'different'
                            item['file_sha256'] = {str(path): file_hash(path) for path in paths}
                        except (ValueError, KeyError) as error:
                            item.update(status='invalid_or_unavailable_field', error=str(error))
                    record['ancestors'].append(item)
                output.append(record)
    return output


def log_config(path):
    fields, locators, command = {}, {}, None
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number > 100:
                break
            if line_number == 2 and line.startswith('/'):
                command = line.strip()
            if '[INFO] - ' not in line:
                continue
            payload = line.split('[INFO] - ', 1)[1].strip()
            key, separator, value = payload.partition(': ')
            if not separator or ' ' in key:
                continue
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            if key.startswith('settings.'):
                fields.setdefault('settings', {})[key.split('.', 1)[1]] = value
            else:
                fields[key] = value
            locators[key] = line_number
    if not fields:
        return None
    try:
        resolved = OmegaConf.to_container(OmegaConf.create(fields), resolve=True)
        resolution = 'resolved_logged_values_only'
    except Exception as error:
        resolved, resolution = fields, f'unresolved: {type(error).__name__}'
    return {'file': str(path), 'sha256': file_hash(path), 'command': command,
            'locators': locators, 'resolution': resolution, 'config': resolved}


def configs():
    records = []
    for method, problem in [('elg', 'cvrp'), ('icam', 'cvrp'), ('lehd', 'cvrp'),
                            ('omni', 'cvrp'), ('omni', 'tsp')]:
        for split in SPLITS:
            path = EASY / f'results/nss_eval/{method}_{problem}/{problem.upper()}{split}/eval.log'
            if path.is_file():
                record = log_config(path)
                if record:
                    records.append(record)
    for directory in [EASY / 'results/label_v3', EASY / 'results/label_v4']:
        for path in sorted(directory.glob('**/.hydra/config.yaml')):
            raw = OmegaConf.load(path)
            resolved = OmegaConf.to_container(raw, resolve=True)
            records.append({'file': str(path), 'sha256': file_hash(path),
                            'resolution': 'resolved_archived_hydra_config', 'config': resolved})
    return records


def checkpoints():
    references = [
        'sources:bq-nco/pretrained_models/tsp.best', 'sources:bq-nco/pretrained_models/cvrp.best',
        'easynco:pretrained/DIFUSCO_pretrain/difusco_tsp100.ckpt',
        'easynco:pretrained/DIFUSCO_pretrain/difusco_tsp500.ckpt',
        'sources:ELG/TSP/weights/ELG.pt', 'easynco:pretrained/ELG/elg_cvrp100.ckpt',
        'easynco:pretrained/icam_pretrain/icam_cvrp_best.ckpt',
        'sources:ICAM/pretrained/icam_atsp.pt', 'easynco:pretrained/lehd/lehd_tsp100.ckpt',
        'easynco:pretrained/lehd/lehd_cvrp100.ckpt', 'easynco:pretrained/omni/omni_tsp_maml_fomaml.ckpt',
        'easynco:pretrained/omni/omni_cvrp_maml_fomaml.ckpt',
        'easynco:pretrained/glop/glop_policy_atsp.pt', 'easynco:pretrained/matnet_pretrain/matnet_atsp100.ckpt',
        'easynco:pretrained/matpoenet/MatNet-POE_mix.pt',
        'easynco:pretrained/mtpomo/mtpomo/mtpomo_mvrp_100.ckpt',
        'easynco:pretrained/mvmoe/mvmoe/mvmoe_mvrp100.ckpt',
        'reld:CVRP/weights/ReLD/model_epoch_90.pt',
        'reld:Multi-Task/pretrained/reld_moe_light/epoch-5000.pt',
        'reld:Multi-Task/pretrained/reld_mtl/epoch-5000.pt',
        'easynco:pretrained/T2T_pretrain/t2t_tsp100.ckpt',
        'easynco:pretrained/T2T_pretrain/t2t_tsp500.ckpt',
        'sources:T2TCO/ckpts/tsp100_categorical.ckpt', 'sources:T2TCO/ckpts/tsp500_categorical.ckpt',
    ]
    for bucket in (50, 100):
        references.extend([
            f'sources:routefinder/checkpoints/{bucket}/rf-transformer.ckpt',
            f'sources:moses_vrp/pretrained_moses_model/rf/{bucket}/multilora_denseroute_softplus.ckpt',
            f'sources:moses_vrp/pretrained_moses_model/cada/{bucket}/multilora_denseroute_sigmoid.ckpt',
        ])
    for scale in (20, 50, 100, 'mix'):
        references.append(f'sources:UniCO/ckpts/_matpoenet_extract/ckpts/MatNet-POE_{scale}.pt')
    roots = {'easynco': EASY, 'sources': SOURCES, 'reld': WORKSPACE / 'reld-nco-main'}
    records = []
    for reference in references:
        root, relative = reference.split(':', 1)
        path = roots[root] / relative
        record = {'reference': reference, 'path': str(path), 'exists': path.is_file(),
                  'hash_scope': 'current_local_bytes_not_a_historical_runtime_digest'}
        if path.is_file():
            record.update(sha256=file_hash(path), size_bytes=path.stat().st_size)
        records.append(record)
    return records


def checkpoint_shapes(records):
    import torch

    for record in records:
        if not record['exists'] or 'UniCO/ckpts/_matpoenet_extract/' not in record['reference']:
            continue
        path = Path(record['path'])
        try:
            with path.open('rb') as handle:
                compressed = handle.read(2) == b'\x1f\x8b'
            opener = gzip.open if compressed else open
            with opener(path, 'rb') as handle, torch.serialization.safe_globals([set]):
                checkpoint = torch.load(handle, map_location='cpu', weights_only=True)
            state = checkpoint['model_state_dict']
            layer_ids = {int(key.split('.')[2]) for key in state
                         if key.startswith('encoder.layers.') and key.split('.')[2].isdigit()}
            record['encoder_layer_num_from_state_keys'] = max(layer_ids) + 1 if layer_ids else None
            record['embedding_shapes'] = {key: list(value.shape) for key, value in state.items()
                                          if key in ('encoder.W_row.weight', 'encoder.W_col.weight',
                                                     'encoder.W_row.bias', 'encoder.W_col.bias')}
            record.pop('metadata_inspection_error', None)
            del checkpoint, state
        except Exception as error:
            record['metadata_inspection_error'] = f'{type(error).__name__}: {str(error)[:400]}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT / '\u6765\u6e90\u8865\u5168\u6838\u5bf9\u8bc1\u636e.json')
    parser.add_argument('--reuse-evidence', action='store_true')
    args = parser.parse_args()
    if args.reuse_evidence:
        record = read_json(args.output)
        checkpoint_shapes(record['checkpoints'])
        args.output.write_text(json.dumps(record, ensure_ascii=True, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'checkpoint_shapes_refreshed': True}))
        return
    baseline = OUTPUT / 'solver_source_manifest.before_completion.json'
    weights = checkpoints()
    checkpoint_shapes(weights)
    bindings = label_bindings()
    archived_configs = configs()
    record = {'schema': 'r45-source-completion-evidence-v1',
              'checked_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'baseline_manifest_sha256': file_hash(baseline),
              'safety': {'solver_execution': False, 'gpu_execution': False, 'training': False,
                         'network_or_api_calls': False, 'datasets_or_labels_modified': False,
                         'observed_scores_in_output': False},
              'hash_scope_note': 'Local current file hashes do not prove historical binary or source revision identity.',
              'label_bindings': bindings, 'archived_configs': archived_configs, 'checkpoints': weights}
    args.output.write_text(json.dumps(record, ensure_ascii=True, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'label_groups_checked': len(bindings),
                      'label_groups_with_matching_ancestor': sum(any(item['status'] == 'match' for item in row['ancestors'])
                                                               for row in bindings),
                      'archived_configs': len(archived_configs), 'checkpoints': len(weights)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
