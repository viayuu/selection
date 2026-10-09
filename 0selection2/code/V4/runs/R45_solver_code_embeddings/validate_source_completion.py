"""Offline checks for the metadata handoff, without solver execution or API calls."""

import datetime
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
from code.V4.solver_code_corpus import check_sensitive, digest, semantic_config, source_path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    manifest_path = PROJECT / 'code/V4/solver_source_manifest.json'
    manifest = read_json(manifest_path)
    before = read_json(ROOT / 'solver_source_manifest.before_completion.json')
    evidence = read_json(ROOT / '\u6765\u6e90\u8865\u5168\u6838\u5bf9\u8bc1\u636e.json')
    corpus = read_json(ROOT / 'corpus_reviewed/corpus.json')
    checks = []

    def require(name, condition, detail):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    identity = [(row['solver_id'], row['solver_name']) for row in manifest['solvers']]
    require('solver_identity_and_order', identity == list(enumerate(GLOBAL_SOLVERS)) ==
            [(row['solver_id'], row['solver_name']) for row in before['solvers']], identity)
    preserved = []
    for old, new in zip(before['solvers'], manifest['solvers']):
        for old_deployment in old['deployments']:
            old_config = old_deployment.get('resolved_config', {})
            candidates = [row for row in new['deployments'] if
                          all(row['resolved_config'].get(key) == value for key, value in old_config.items())]
            preserved.append(bool(candidates))
            for key in ('checkpoint', 'checkpoint_sha256'):
                if key in old_deployment:
                    preserved.append(any(row.get(key) == old_deployment[key] for row in candidates))
    require('confirmed_information_preserved', all(preserved),
            'All original nonempty config fields and RELD checkpoint identities retained in an applicable deployment.')
    require('original_candidate_implementations_retained',
            set(before['implementations']) <= set(manifest['implementations']),
            'The paper GLOP implementation remains as an explicitly unused candidate.')

    unchanged = []
    changed = []
    with zipfile.ZipFile(ROOT / 'R45_source_handoff.zip') as archive:
        for name in archive.namelist():
            if name.endswith('/') or name == 'code/V4/solver_source_manifest.json':
                continue
            path = PROJECT / name
            if path.is_file() and hashlib.sha256(archive.read(name)).hexdigest() == file_hash(path):
                unchanged.append(name)
            else:
                changed.append(name)
        original_manifest_hash = hashlib.sha256(archive.read('code/V4/solver_source_manifest.json')).hexdigest()
    require('handoff_and_original_corpus_unchanged', not changed,
            {'unchanged_files': len(unchanged), 'changed_or_missing_files': changed})
    require('backup_matches_original', original_manifest_hash ==
            file_hash(ROOT / 'solver_source_manifest.before_completion.json') == evidence['baseline_manifest_sha256'],
            'Backup, original ZIP manifest and collector baseline hash agree.')

    expected = {(problem, method, split) for problem in PROBLEMS for method in POOLS[problem]
                for split in ('train', 'val', 'test')}
    bindings = evidence['label_bindings']
    observed = {(row['problem'], row['method'], row['split']) for row in bindings}
    require('all_384_label_ancestries_matched', len(bindings) == len(expected) == 384 and observed == expected and
            all(any(ancestor['status'] == 'match' for ancestor in row['ancestors']) for row in bindings),
            {'groups': len(bindings), 'missing': sorted(expected - observed)})
    altered_labels = [row['target'] for row in bindings if file_hash(Path(row['target'])) != row['target_sha256']]
    require('current_label_files_unchanged_since_collection', not altered_labels, altered_labels)

    deployment_coverage = set()
    bad_scopes = []
    for solver in manifest['solvers']:
        for deployment in solver['deployments']:
            semantic_config(deployment['resolved_config'])
            for problem in deployment['problems']:
                if solver['solver_name'] not in POOLS.get(problem, []):
                    bad_scopes.append([solver['solver_name'], problem])
                for split in deployment['splits']:
                    deployment_coverage.add((problem, solver['solver_name'], split))
            for item in deployment.get('evidence', []):
                if not Path(item['file']).is_file():
                    bad_scopes.append(['missing evidence', item['file']])
    require('deployment_scope_and_evidence', deployment_coverage == expected and not bad_scopes,
            {'covered_groups': len(deployment_coverage), 'invalid_scopes_or_paths': bad_scopes})
    require('unknown_nss_configs_not_invented', all(not deployment['resolved_config']
            for solver in manifest['solvers'] for deployment in solver['deployments']
            if deployment['label_binding']['origin'] == 'nss_import'),
            'NSS-only deployments retain empty semantic configs and explicit gaps.')
    require('historical_gaps_not_silenced', all(deployment['gaps'] and
            deployment['binding_status'] in ('partial', 'unverified')
            for solver in manifest['solvers'] for deployment in solver['deployments']) and not corpus['ready'],
            'No bulk confirmation, no cleared unresolved bindings and no production-ready claim.')

    old_corpus = read_json(ROOT / 'corpus/corpus.json')
    changed_sources = [reference for reference, expected_hash in old_corpus['source_hashes'].items()
                       if file_hash(source_path(before, reference)) != expected_hash]
    require('original_solver_source_bytes_unchanged', not changed_sources, changed_sources)
    require('reviewed_corpus_matches_manifest_and_sources', corpus['manifest_sha256'] == digest(manifest) and
            all(file_hash(source_path(manifest, reference)) == expected_hash
                for reference, expected_hash in corpus['source_hashes'].items()) and
            all(len(variant['roles']['encoder']) and len(variant['roles']['decision'])
                for solver in corpus['solvers'] for variant in solver['variants']),
            {'selected_source_files': len(corpus['source_hashes']), 'chunks': len(corpus['chunks'])})
    for chunk in corpus['chunks']:
        check_sensitive(chunk['text'])
        semantic_config(chunk['config_context'])
    require('semantic_config_and_sensitive_content_guards', True,
            'All configs and generated chunks pass existing R45 result/path/secret guards.')

    inventory = {row['reference']: row for row in evidence['checkpoints']}
    checkpoint_errors = []
    checked = 0
    for solver in manifest['solvers']:
        for deployment in solver['deployments']:
            refs = list(deployment.get('checkpoint_candidates', [])) + list(deployment.get('checkpoints', []))
            if 'checkpoint' in deployment:
                refs.append({'checkpoint': deployment['checkpoint'], 'sha256': deployment['checkpoint_sha256']})
            for ref in refs:
                checked += 1
                candidate = inventory.get(ref['checkpoint'])
                if candidate is None or candidate['exists'] != ref.get('exists', True):
                    checkpoint_errors.append(ref['checkpoint'])
                elif candidate['exists'] and candidate['sha256'] != ref.get('sha256'):
                    checkpoint_errors.append(ref['checkpoint'])
    require('checkpoint_inventory_consistent', not checkpoint_errors,
            {'deployment_references': checked, 'inventory_paths': len(inventory),
             'existing_inventory_paths': sum(row['exists'] for row in inventory.values()), 'errors': checkpoint_errors,
             'hash_scope': 'current local bytes; historical runtime identity not inferred'})

    archived = evidence['archived_configs']
    uniform = []
    for method, expected_count, expected_aug in [('glop', 243, None), ('matnet', 243, 128),
                                                 ('matpoenet', 243, 8), ('mtpomo', 2295, 1), ('mvmoe', 2295, 1)]:
        if method in ('mtpomo', 'mvmoe'):
            rows = [row for row in archived if '/results/label_v4/mvrp/' in row['file'] and
                    f'/{method}_' in row['file'] and row['file'].endswith('/.hydra/config.yaml')]
        else:
            rows = [row for row in archived if
                    (f'/results/label_v3/{method}_atsp/' in row['file'] or
                     '/results/label_v4/atsp/' in row['file'] and f'/{method}/' in row['file']) and
                    row['file'].endswith('/.hydra/config.yaml')]
        passed = len(rows) == expected_count
        for row in rows:
            cfg = row['config']
            settings = cfg['settings']
            passed = passed and settings['iteration']['_target_'].endswith('.NoIteration')
            if expected_aug is not None:
                passed = passed and settings['env']['aug_factor'] == expected_aug
                passed = passed and settings['env']['pomo_size'] == cfg['scale']
            if method in ('mtpomo', 'mvmoe'):
                passed = passed and 'decoder_strategy' not in settings['module']
                passed = passed and settings['initialization']['_target_'].endswith('.POMOInitialization')
                passed = passed and cfg['batch_size'] == 64 and cfg['seed'] == 1234
            else:
                passed = passed and settings['module']['decoder_strategy'] == 'greedy'
            if method == 'glop':
                passed = passed and cfg['batch_size'] == 1
        uniform.append({'method': method, 'configs': len(rows), 'passed': bool(passed),
                        'model_variants': len({json.dumps(row['config']['settings']['model'], sort_keys=True)
                                               for row in rows})})
    require('archived_configuration_claims', all(row['passed'] for row in uniform), uniform)
    by_name = {row['solver_name']: row for row in manifest['solvers']}
    require('corrected_actual_call_chains',
            by_name['GLOP']['deployments'][0]['implementation'] == 'glop_atsp_init_only' and
            not by_name['GLOP']['deployments'][0]['resolved_config']['learned_policy_used'] and
            any(source['file'] == 'bridge:run_reld_cvrp_single_bridge.py' and 'run_no_aug_scores' in source['symbols']
                for source in manifest['implementations']['reld_cvrp']['views']['inference']['sources']) and
            by_name['RELD_CVRP']['deployments'][0]['resolved_config']['pomo_size'] == 'min(number of customers, 100)',
            'GLOP initializer replaces the unused paper tester; RELD_CVRP uses the single-task bridge.')
    unico = by_name['UNICO_MatPOENet']['deployments']
    def applicable(deployment, scale):
        return (scale in deployment.get('scales', range(deployment.get('scale_range', [scale, scale])[0],
                                                       deployment.get('scale_range', [scale, scale])[1] + 1)) and
                scale not in deployment.get('excluded_scales', []))
    require('unico_scale_variants_are_disjoint_and_complete', all(sum(applicable(row, scale) for row in unico) == 1
            for scale in range(20, 101)) and
            all(row['resolved_config']['encoder_layer_num'] == (8 if scale in (20, 50) else 5)
                for scale in range(20, 101) for row in unico if applicable(row, scale)),
            'Exact20/50: eight layers; all other registered scales: five layers; no overlapping deployment averaging.')

    passed = all(row['passed'] for row in checks)
    output = {'checked_at': datetime.datetime.now().astimezone().isoformat(),
              'passed': passed, 'checks': checks, 'manifest_sha256': file_hash(manifest_path),
              'reviewed_corpus_sha256': corpus['sha256'],
              'deployment_statuses': dict(Counter(row['binding_status'] for solver in manifest['solvers']
                                                for row in solver['deployments'])),
              'unit_test_run': {'command': 'python -m unittest code.V4.test_r45 -v',
                                'reference': 'See the separate offline validation log, not a solver benchmark.'},
              'safety': evidence['safety']}
    output_path = ROOT / '\u8865\u5168\u9a8c\u6536.json'
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': passed, 'checks': len(checks), 'output': str(output_path),
                      'failed': [row for row in checks if not row['passed']]}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
