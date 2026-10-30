"""CPU-only completed-version publication: read-only by default, explicit --apply.

No finish/inference/training commands are invoked. Git is only run with --apply.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


V4 = Path(__file__).resolve().parent
REPO, GIT_ROOT = V4.parents[1], Path('/public/home/shiys')
RETURN18 = V4 / 'runs/Return18_post_R120'
CODE = {
    'R121': 'r121_experiment.py r121_model.py r121_structural_data.py r121_collection.py '
            'r121_finish.py test_r121_finish.py r121_followup.py test_r121_followup.py return18_results.py '
            'r121_numeric_collection.py test_r121_numeric_collection.py '
            'r121_parallel_collection_v3.py test_r121_parallel_collection_v3.py '
            'r121_followup_numeric.py test_r121_followup_numeric.py '
            'return18_moe_support.py test_return18_moe_support.py '
            'r121_collection_recovery.py r121_native_output_diagnostic.py r121_input_boundary.py '
            'r121_universal_collection.py r121_parallel_collection.py r121_parallel_collection_v2.py '
            'test_r121_experiment.py test_r121_model.py test_r121_structural_data.py test_r121_collection.py '
            'test_r121_collection_recovery.py test_r121_native_output_diagnostic.py test_r121_input_boundary.py '
            'test_r121_universal_collection.py test_r121_parallel_collection.py',
    'R122': 'r122_experiment.py r122_objective.py r122_finish.py r122_test_recovery.py '
            'test_r122_experiment.py test_r122_objective.py test_r122_finish.py test_r122_test_recovery.py',
    'R123': 'r123_experiment.py r123_model.py r123_meta_data.py r123_prediction_cache.py r123_finish.py '
            'test_r123_experiment.py test_r123_model.py test_r123_meta_data.py test_r123_finish.py',
    'R124': 'r124_experiment.py r124_model.py r124_finish.py r123_meta_data.py r123_prediction_cache.py '
            'test_r124_experiment.py test_r124_model.py test_r124_finish.py',
}
COMMON = 'return18_publish.py test_return18_publish.py return18_launch.py test_return18_launch.py run_v4_return18.sh r120_closeout.py'
SUFFIXES = {'.py', '.sh', '.md', '.pdf', '.csv', '.json'}
R121_REPORTS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
    'training_and_method_diagnostics.pdf', 'structural_objectives.pdf', 'metrics_by_problem.csv',
    'selection_distribution.csv', 'single_method_comparison.csv', 'nss_comparison.csv', 'protocol.json')
R121_SUPPLEMENT = ('R121_MECHANISM.pdf', 'natural_metrics.csv', 'natural_fixed36_error_budgets.csv',
    'family_diagnostics.csv', 'target_recovery.csv', 'target_recovery_summary.csv', 'paired_changes.csv',
    'historical_references.csv', 'protocol.json')
R121_STAGES = ('full89_export', 'family_cpu_features', 'principal_gpu_stages', 'saved_only_finish', 'four_route_analysis')
R121_SCIENCE = ('REPORT.pdf', 'REPORT.md', 'sources.csv', 'ROUTING.json', 'render_notes.json', 'render_pdf.py')
R121_COLLECTIONS = {
    'collection_explicit_cpu_v2': ('universal', 'r121_universal_explicit_cpu_collection_v2_20261030',
                                  'followup_principal_v1', 'immutable_source_metadata_sha256'),
    'collection_numeric_boundary_v3': ('numeric', 'r121_numeric_boundary_collection_v3_20261030',
                                      'followup_principal_v2', 'immutable_metadata_sha256'),
}
R121_NUMERIC_METADATA = ('numeric_API.md', 'route_limit_failure_DIAG.md',
    'collection_numeric_boundary_v3/numeric.json', 'collection_numeric_boundary_v3/protocol.json',
    'route_limit_diagnostic_v1/diagnostic.json', 'collection_persistent_v1/unavailable/VRPBL__reld_mt.json')
MOE_REPORTS = ('RESULTS.pdf', 'support_summary.csv', 'fixed36_support.csv', 'method_support.csv',
               'verified_bank_decomposition.csv')
R124_CSV_FIELDS = {
    'metrics.csv': 'top1_pct mean_cost actual_regret_pct',
    'r77b_comparison.csv': 'r124_top1_pct r124_actual_regret_pct first_stage_checkpoint_sha256',
    'reranking.csv': 'final_top1_pct actual_regret_pct corrected_count damaged_count',
    'candidate_coverage.csv': 'method global_id native_winner_count',
    'selection_distribution.csv': 'method global_id model_selected_count native_winner_count',
    'inference_cost.csv': 'seconds full_stage0_plus_reranker',
    'nss_four_datasets.csv': 'dataset n r124_checkpoint_sha256 r77b_checkpoint_sha256 nss_status',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def r121_collection_gate(root, config, structural):
    collection = Path(config['family_labels']['native_receipt_root'])
    if collection.name not in R121_COLLECTIONS or collection != root.parent / collection.name:
        raise ValueError('R121 requires an explicitly supported single-root collection')
    layer, version, _, immutable_key = R121_COLLECTIONS[collection.name]
    layer_key = layer + '_sha256'
    path = collection / 'family_labels.receipt.json'
    metadata_path = collection / (layer + '.json')
    export, protocol, metadata = read(path), read(collection / 'protocol.json'), read(metadata_path)
    structural.verify_seal(protocol, 'collection_sha256')
    structural.verify_seal(metadata, layer_key)
    groups = {p + '/' + name for p, pool in config['candidate_pools'].items() for name in pool}
    if (export != config['family_labels'] or export.get('complete') is not True
            or export.get('full18') is not True or export.get('candidate_pool_unchanged') is not True
            or export.get('historical_labels_replaced') is not False
            or export.get('labels') != 113920 or export.get('endpoints') != 23040
            or len(groups) != 89 or set(protocol['group_order']) != groups or len(protocol['group_order']) != 89
            or protocol['planned_labels'] != 113920 or protocol['lanes'] != 4
            or export['native_receipt_root'] != str(collection) or export['path'] != str(collection / 'family_labels.jsonl')
            or export[layer_key] != metadata[layer_key]
            or export[layer + '_version'] != version or export[layer + '_version'] != metadata['version']
            or export['collection_sha256'] != protocol['collection_sha256']
            or export['collection_sha256'] != metadata['native_collection_sha256']
            or export['manifest_sha256'] != protocol['manifest_sha256']
            or export['family_input_manifest'] != protocol['input_manifest']
            or sha(export['path']) != export['sha256']):
        raise ValueError('R121 requires the complete single-root full18/89/113920/23040 export')
    if layer == 'numeric' and (export['independent_numeric_version'] != metadata['checker_version']
            or export['source_policy'] != metadata['source_root_by_deployment']):
        raise ValueError('R121 numeric export differs from its sealed checker/fixed-source policy')
    for pins in (protocol['source_sha256'], metadata['source_sha256'], metadata[immutable_key]):
        if any(sha(p) != digest for p, digest in pins.items()):
            raise ValueError('Pinned R121 collection source/protocol changed')
    for lane in range(4):
        marker = read(collection / 'bootstrap' / ('lane' + str(lane) + '.json'))
        structural.verify_seal(marker, 'bootstrap_sha256')
        if (marker['status'] != 'bootstrap_complete' or marker['lane'] != lane
                or marker['collection_sha256'] != export['collection_sha256']
                or marker[layer_key] != export[layer_key]):
            raise ValueError('R121 single-root bootstrap is incomplete/mismatched')
    for group in groups:
        name = group.replace('/', '__') + '.json'
        state = read(collection / 'progress' / name)
        if (state.get('completed') != 1280 or state.get('next_index') != 1280 or state.get('status') != 'complete'
                or state.get('collection_sha256') != export['collection_sha256']
                or not (collection / 'preflight' / name).is_file() or (collection / 'unavailable' / name).exists()):
            raise ValueError('R121 deployment collection is incomplete: ' + group)
    return [path, collection / 'protocol.json', metadata_path]


def r121_finish_files(run, config, lock, test, base_delivery):
    api = importlib.import_module('code.V4.r121_finish')
    output = run / 'learner_finish'
    path = output / 'reports/receipt.json'
    identity = api.identity_for(lock, test['identity']['model_state_sha256'])
    saved = read(path)
    if set(saved.get('artifacts', {})) != set(R121_SUPPLEMENT) or set(api.REPORT_FILES) != set(R121_SUPPLEMENT):
        raise ValueError('R121 supplemental report artifact set differs from the frozen API')
    saved = api.completed_receipt(path, identity)
    if (saved.get('phase') != 'complete_post_TEST_learner_finish' or saved.get('all18') is not True
            or saved.get('train_complete') is not True or saved.get('train_rows_per_problem') != 81000
            or saved['train_eligible_rows_per_problem'] != config['natural_ce_eligible_rows_per_problem']
            or saved.get('family_diagnostics_complete') is not True or saved.get('fixed36_count') != 36
            or saved['family_baselines'] != ['zero_potential', 'family_train_mean_potential']
            or saved['VAL'] != base_delivery['VAL'] or saved['TEST'] != base_delivery['TEST']
            or any(saved.get(k) is not False for k in ('model_inference', 'GPU_execution', 'solver_execution',
                                                      'training', 'checkpoint_reselection', 'research_goal_complete'))):
        raise ValueError('R121 full TRAIN/family saved-only finish is incomplete or changed')
    if (saved['source_sha256'].get('r121_finish.py') != sha(V4 / 'r121_finish.py')
            or any(sha(V4 / name) != digest for name, digest in saved['source_sha256'].items())):
        raise ValueError('R121 supplemental producer source changed')
    links = {'base_delivery_manifest': run / 'delivery_manifest.json',
             'capture_train_receipt': output / 'capture_train_receipt.json',
             'family_capture_receipt': output / 'family_capture_receipt.json'}
    for key, source in links.items():
        if api.artifact(source) != saved[key]:
            raise ValueError('R121 finish receipt link changed: ' + key)
    captures = {}
    for key in ('capture_train_receipt', 'family_capture_receipt'):
        record = read(links[key])
        api.check_identity(record, identity)
        if (record.get('complete') is not True or record.get('all18') is not True
                or any(saved['input_files_sha256'].get(p) != digest for p, digest in record['input_files_sha256'].items())
                or any(api.artifact(item['path']) != item for item in record['artifacts'].values())):
            raise ValueError('R121 capture inputs/artifacts differ from the validated final receipt')
        captures[key] = record
    train, family = captures['capture_train_receipt'], captures['family_capture_receipt']
    if (train.get('phase') != 'locked_full_train_capture' or train.get('train_complete') is not True
            or train.get('train_rows_per_problem') != 81000 or train['TRAIN'] != saved['TRAIN']
            or train['train_eligible_rows_per_problem'] != config['natural_ce_eligible_rows_per_problem']
            or family.get('phase') != 'locked_family_capture' or family.get('endpoints') != 23040
            or family['family_baselines'] != saved['family_baselines']):
        raise ValueError('R121 requires full81000/task and complete same-model family captures')
    files = [path, links['capture_train_receipt'], links['family_capture_receipt']]
    for name, item in saved['artifacts'].items():
        expected = output / 'reports' / name
        if Path(item['path']).resolve() != expected.resolve():
            raise ValueError('R121 supplemental artifact escaped its report directory')
        files.append(expected)
    return files


def r121_followup_files(root, config, lock):
    collection = Path(config['family_labels']['native_receipt_root'])
    followup = root.parent / R121_COLLECTIONS[collection.name][2]
    contract, launch = read(followup / 'contract.json'), read(followup / 'learner_launch.json')
    protocol = root / 'principal/protocol.json'
    if (contract.get('main_approved') is not True or contract['root'] != str(followup)
            or contract['learner_root'] != str(root) or contract['collection_root'] != str(collection)
            or contract['planned_labels'] != 113920 or contract['endpoints'] != 23040
            or (REPO / launch['protocol']['path']).resolve() != protocol.resolve()
            or launch['protocol']['sha256'] != sha(protocol) or launch['protocol']['state_hash'] != lock['state_hash']
            or launch['plan'] != contract['resource_plan'] or launch['plan_sha256'] != sha(launch['plan'])
            or any(sha(REPO / name) != digest for name, digest in contract['source_sha256'].items())
            or any(sha(item['path']) != item['sha256'] for item in contract['immutable_inputs'])):
        raise ValueError('R121 followup contract/launch differs from the frozen principal protocol')
    files = [followup / name for name in ('README.md', 'contract.json', 'learner_launch.json')]
    for name in R121_STAGES:
        path = followup / (name + '.json')
        stage = read(path)
        if stage.get('stage') != name or stage.get('state') != 'complete' or stage.get('exit_status') != 0:
            raise ValueError('R121 prepublication followup stage is incomplete: ' + name)
        files.append(path)
    result_api = importlib.import_module('code.V4.return18_results')
    rows, current = result_api.collect(root)
    output = RETURN18 / 'final_results'
    evidence = read(output / 'evidence.json')
    if (not current.get('all_four_complete') or current['checkpoint_sha256']['R121'] != lock['checkpoint_sha256']
            or {k: v for k, v in evidence.items() if k != 'artifacts'} != current
            or set(evidence['artifacts']) != {'RESULTS.pdf', 'comparison.csv'}):
        raise ValueError('Return18 final analysis is stale or not the four completed locked versions')
    for name, item in evidence['artifacts'].items():
        path = output / name
        if item['path'] != str(path) or item['sha256'] != sha(path):
            raise ValueError('Return18 final analysis output changed: ' + name)
        files.append(path)
    with (output / 'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        if list(csv.DictReader(stream)) != [{k: str(v) for k, v in row.items()} for row in rows]:
            raise ValueError('Return18 comparison differs from the current saved metrics')
    return [*files, output / 'evidence.json']


def r121_moe_files(config):
    output = RETURN18 / 'moe_support'
    evidence = read(output / 'evidence.json')
    r123_lock = read(V4 / 'runs/R123_heterogeneous_moe/model/final_test_lock.json')
    if (evidence.get('complete') is not True or evidence['checkpoint'] != r123_lock
            or evidence['data_root'] != config['data_root']
            or evidence['source_sha256'] != sha(V4 / 'return18_moe_support.py')
            or any(evidence.get(k) is not True for k in ('no_nn_inference', 'no_gpu', 'no_solver', 'no_training'))
            or evidence['interpretation'].get('deployable') is not False
            or evidence['interpretation'].get('model_accuracy') is not False):
        raise ValueError('MoE support must remain a completed offline non-deployable frozen-bank diagnosis')
    files = [output / 'evidence.json']
    for name in MOE_REPORTS:
        path = output / name
        if sha(path) != evidence['artifacts_sha256'][name]:
            raise ValueError('MoE support report artifact changed: ' + name)
        files.append(path)
    for name in ('CHECKS.json', 'RESULTS.md'):
        if (output / name).is_file():
            files.append(output / name)
    return files


def r121_completion_gate(root):
    if root != V4 / 'runs/R121_structural_families/preparation/learner_v1':
        raise ValueError('Use the literal R121 preparation/learner_v1 version root, not its principal arm')
    path = root / 'principal/delivery_manifest.json'
    receipt = read(path)
    if (receipt.get('complete') is not True or receipt.get('experiment') != 'R121'
            or receipt.get('arm') != 'principal' or receipt.get('all18') is not True
            or set(receipt.get('native_libs', [])) != {'TSPLIB', 'CVRPLIB'}
            or set(receipt.get('artifacts', {})) != set(R121_REPORTS)):
        raise ValueError('R121 saved-only finish/delivery is incomplete')
    learner = importlib.import_module('code.V4.r121_experiment')
    common = importlib.import_module('code.V4.r80_reporting')
    reports = importlib.import_module('code.V4.r84_reporting')
    original = learner.locked_training(SimpleNamespace(root=root))
    run, config, lock, test, val = common.locked_inputs(root, 'principal', experiment='R121', arms={'principal': 14400})
    if original != (run, config, lock) or any(sha(V4 / name) != digest for name, digest in config['source_sha256'].items()):
        raise ValueError('R121 frozen principal source/lock changed')
    training = read(run / 'training_complete.json')
    if training.get('complete') is not True or training.get('selected_best') != lock or training.get('exposure') != config['exposures']:
        raise ValueError('R121 principal training is not complete at its locked exposure')
    common.check_identity(receipt, lock, 'R121 delivery', require_epoch=False)
    metadata = ('family_population', 'loss', 'historical_control') + (('train_diagnostic',) if 'train_diagnostic' in config else ())
    if (receipt['VAL'] != val['macro'] or receipt['TEST'] != test['macro']
            or receipt['model_factory'] != 'code.V4.r121_experiment.load_checkpoint'
            or config['family_population']['family_endpoints'] != 23040
            or receipt['family_export'] != config['family_labels']
            or any(receipt[key] != config[key] for key in metadata)
            or receipt['exposure'] != config['exposures']):
        raise ValueError('R121 delivery differs from locked metrics/family objective/exposures')
    extras = r121_collection_gate(root, config, learner.structural)
    model_hash = test['identity']['model_state_sha256']
    for split, result in (('val', val), ('test', test)):
        archive = common.selected_val_path(run, lock) if split == 'val' else run / 'evaluation/test/selected_best/rank000.npz'
        identity_path = archive.parent / 'identity.json'
        identity = read(identity_path)
        if (identity != result['identity'] or identity['experiment'] != 'R121' or identity['arm'] != 'principal'
                or identity['split'] != split or identity['state_hash'] != lock['state_hash']
                or identity['successful_updates'] != lock['selected_update'] or identity['model_state_sha256'] != model_hash
                or split == 'test' and identity['checkpoint_sha256'] != lock['checkpoint_sha256']
                or not archive.is_file() or any(row['n'] != 4500 for row in result['per_problem'].values())):
            raise ValueError('R121 full18 VAL/TEST capture identity differs from one checkpoint')
        extras.append(identity_path)
    completion = read(run / 'lib_stage_complete.json')
    if (completion.get('complete') is not True or completion.get('experiment') != 'R121'
            or completion.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or set(completion.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('R121 requires both completed same-checkpoint native LIBs')
    libs = reports.saved_libs(run, config, lock, test, {}, [], [])
    if set(libs) != {'TSPLIB', 'CVRPLIB'} or any(
            value['protocol']['model_factory'] != config['model_factory']
            or value['protocol']['locked_inference_rule'] != lock['inference_rule'] for value in libs.values()):
        raise ValueError('R121 native LIB factory/inference rule differs from principal')
    output = run / 'full_model_statistics'
    for name, item in receipt['artifacts'].items():
        artifact = output / name
        if (item['path'] != str(artifact) or artifact.is_symlink() or not 0 < item['bytes'] < 8 * 1024**2
                or artifact.stat().st_size != item['bytes'] or sha(artifact) != item['sha256']):
            raise ValueError('R121 delivery artifact changed: ' + name)
    protocol = read(output / 'protocol.json')
    if (protocol.get('complete') is not True or protocol['checkpoint_lock'] != lock
            or protocol['experiment'] != 'R121' or protocol['arm'] != 'principal'
            or any(protocol[key] != config[key] for key in (*metadata, 'model_factory', 'candidate_pools', 'data_root'))
            or protocol['exposure'] != config['exposures']):
        raise ValueError('R121 report protocol differs from principal/family lock')
    extras.extend(run / 'lib_evaluation' / (dataset + '_results.json') for dataset in ('TSPLIB', 'CVRPLIB'))
    extras.extend(r121_finish_files(run, config, lock, test, receipt))
    extras.extend(r121_followup_files(root, config, lock))
    extras.extend(r121_moe_files(config))
    return lock, [path, *(output / name for name in R121_REPORTS), run / 'lib_stage_complete.json', *extras]


def r124_csvs(output, config, identity):
    """Check only fixed report tables; these checks do NOT retroactively sign them."""
    pools, ids = config['candidate_pools'], config['candidate_pool_ids']
    populations = {(s, p): 4500 for s in ('VAL', 'TEST') for p in pools}
    populations.update({('TSPLIB', 'TSP'): 49, ('CVRPLIB', 'CVRP'): 100})
    populations.update({(s, 'ALL'): 4500 * len(pools) for s in ('VAL', 'TEST')})
    datasets = {'TSP TEST': ('TEST', 'TSP'), 'CVRP TEST': ('TEST', 'CVRP'),
                'TSPLIB': ('TSPLIB', 'TSP'), 'CVRPLIB': ('CVRPLIB', 'CVRP')}
    hashes = {}
    for name, fields in R124_CSV_FIELDS.items():
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size >= 8 * 1024**2:
            raise ValueError('Missing/unsafe R124 report CSV: ' + str(path))
        hashes[path] = sha(path)
        nss, methods = name == 'nss_four_datasets.csv', 'method' in fields.split()
        required = set(fields.split()) | set(('winner_rule data_version' if nss else 'split problem n winner_rule data_version').split())
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not required <= set(reader.fieldnames or ()) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('R124 CSV headers differ: ' + name)
            rows = list(reader)
        expected = set(datasets) if nss else {k for k in populations if not methods or k[1] != 'ALL'}
        if methods:
            expected = {(s, p, m) for s, p in expected for m in pools[p]}
        observed = []
        for row in rows:
            s, p = datasets[row['dataset']] if nss else (row['split'], row['problem'])
            observed.append(row['dataset'] if nss else (s, p, row['method']) if methods else (s, p))
            version = Path(config['data_root']).name if s in ('VAL', 'TEST') else 'original native ' + (s if nss else 'LIB')
            if (None in row or (s, p) not in populations or int(row['n']) != populations[s, p]
                    or p != 'ALL' and (row['winner_rule'] != identity['winner_rule'] or row['data_version'] != version)):
                raise ValueError('R124 CSV population/native/data identity differs: ' + name)
            if methods and int(row['global_id']) != ids[p][pools[p].index(row['method'])]:
                raise ValueError('R124 CSV candidate identity differs: ' + name)
            if nss and (row['r124_checkpoint_sha256'] != identity['checkpoint_sha256']
                        or row['r77b_checkpoint_sha256'] != identity['first_stage_checkpoint_sha256']
                        or row['nss_status'] != 'complete'):
                raise ValueError('R124 NSS CSV checkpoint identity differs')
            if name == 'r77b_comparison.csv' and p != 'ALL' and row['first_stage_checkpoint_sha256'] != identity['first_stage_checkpoint_sha256']:
                raise ValueError('R124 CSV first-stage identity differs')
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError('R124 CSV incomplete/duplicate dataset rows: ' + name)
    return hashes


def completion_gate(root, experiment, *, late_hashes=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if experiment == 'R121':
        return r121_completion_gate(root)
    finish = importlib.import_module('code.V4.' + experiment.lower() + '_finish')
    run, extras = root / 'model', []
    if experiment == 'R122':
        model = finish.locked_inputs(root, {})
        finish.load_libs(model, {})
        lock, output, path = model['lock'], run / 'full_model_statistics', run / 'report_complete.json'
        receipt, protocol = read(path), read(output / 'protocol.json')
        finish.same_identity(receipt, lock)
        if (receipt['report_protocol_sha256'] != sha(output / 'protocol.json')
                or protocol['checkpoint_lock'] != lock or protocol['outputs'] != receipt['outputs']):
            raise ValueError('R122 report protocol/unique VAL lock changed')
        if any(sha(V4 / name) != digest for name, digest in protocol['source_sha256'].items()):
            raise ValueError('R122 report producer source changed')
        artifacts = {output / name: digest for name, digest in receipt['outputs'].items()}
        extras = [output / 'protocol.json', run / 'lib_stage_complete.json']
        recovery = run / 'test_recovery_v1/resource_adapter.json'
        if recovery.exists():
            if read(recovery).get('complete') is not True or read(recovery)['checkpoint_lock'] != lock:
                raise ValueError('R122 resource recovery is not complete for this lock')
            extras.append(recovery)
    elif experiment == 'R123':
        model = finish.locked(SimpleNamespace(root=root))
        lock, output = model['lock'], root / 'finish/reports'
        for relative, kind in (('captures/train', 'metaTRAIN_cached_chosen_scores'),
                               ('lib', 'live_sparse_LIB'), ('timing', 'live_sparse_TEST_fixed_prefix')):
            stage = root / 'finish' / relative / 'receipt.json'
            finish.require_identity(stage, model, kind)
            extras.append(stage)
        path = output / 'receipt.json'
        receipt = finish.require_identity(path, model, 'complete_post_TEST_report')
        if receipt['input_files_sha256'].get(str(Path(finish.__file__).resolve())) != sha(finish.__file__):
            raise ValueError('R123 report producer source changed')
        artifacts = {output / name: digest for name, digest in receipt['artifacts'].items()}
    else:
        run, config, lock, _, identity = finish.locked_inputs(SimpleNamespace(root=root))
        output, path = run / 'finish/reports', run / 'finish/reports/finish_receipt.json'
        receipt, lib = read(path), read(run / 'finish/lib_complete.json')
        if receipt['identity'] != identity or lib.get('complete') is not True or lib['identity'] != identity:
            raise ValueError('R124 report/LIB differs from the unique VAL/TEST identity')
        for dataset in ('TSPLIB', 'CVRPLIB'):
            value = lib['datasets'][dataset]
            if (not value['complete'] or any(value['protocol'][k] != v for k, v in identity.items())
                    or sha(run / 'finish/lib_evaluation/predictions' / (dataset + '.npz')) != value['predictions_sha256']):
                raise ValueError('R124 locked LIB capture changed')
        if any(sha(v['path']) != v['sha256'] for v in receipt['inputs'].values()):
            raise ValueError('R124 report prediction inputs changed')
        artifacts = {Path(value): receipt['output_sha256'][key] for key, value in receipt['outputs'].items()}
        unsigned = {p: h for p, h in r124_csvs(output, config, identity).items() if p not in artifacts}
        artifacts.update(unsigned)
        if unsigned and late_hashes is not None:
            late_hashes.update(kind='LATE_PUBLICATION_HASH', original_finish_receipt_signed_these_csvs=False,
                recorded_utc=datetime.now(timezone.utc).isoformat(), finish_receipt_sha256=sha(path),
                checkpoint_identity=identity, csv_sha256={str(p): h for p, h in unsigned.items()},
                limitation='Current files/headers/populations/identities checked; NOT historical CSV producer attribution')
        extras = [run / 'finish/lib_complete.json', run / 'finish/identity.json']
    if receipt.get('complete') is not True or not artifacts:
        raise ValueError('Final report is incomplete; never start/retry missing stages here')
    if not {'.pdf', '.csv'} <= {p.suffix for p in artifacts}:
        raise ValueError('Completed report requires both PDF and CSV')
    for artifact, digest in artifacts.items():
        if artifact.resolve().parent != output.resolve() or sha(artifact) != digest:
            raise ValueError('Report artifact path/hash changed: ' + str(artifact))
    return lock, [path, *artifacts, *extras]


def curated_paths(root, experiment, reports):
    files = [*reports, *(V4 / name for name in (CODE[experiment] + ' ' + COMMON).split())]
    small_manifests = ({root / 'preparation/manifest.json'} if experiment == 'R123' else
                       {root / 'principal/delivery_manifest.json'} if experiment == 'R121' else set())
    files.extend(small_manifests)
    for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json', 'test_results.json',
                 'test_complete.json', 'final_metrics.json', 'history.json', 'val_history.json'):
        files.extend((root / ('principal' if experiment == 'R121' else 'model')).glob(name))
    if experiment == 'R121':
        for name in ('README.md', 'learner_assumptions.md', 'learner_commands.md', 'learner_natural_isolation_v1.json',
                     'learner_protocol.json', 'learner_finish_API.md', 'collection_INPUT_BOUNDARY.md'):
            files.extend(root.parent.glob(name))
        for name in R121_NUMERIC_METADATA:
            files.extend(root.parent.glob(name))
        science = REPO / 'docs/research/frozen_selector_mixture_information_20261030'
        files.extend(science / name for name in R121_SCIENCE)
    for directory in (root / 'preparation', root / 'preparation/v2'):
        files.extend(directory.glob('*.md'))
        for name in ('prepared.json', 'constraint_reference.json', 'method_config.json', 'frozen_contract.json',
                     'experts.json', 'learner_protocol.json', 'CPU_CHECKS.json', 'FINISH_CPU_CHECKS.json'):
            files.extend(directory.glob(name))
    for name in ('README.md', 'ASSUMPTIONS.md', 'CACHE_PORTABILITY_V2.md', 'PUBLISHING.md',
                 'plan.schema.json', 'CPU_CHECKS.json', 'CPU_CHECKS_v2.json'):
        files.extend((RETURN18 / 'dispatch').glob(name))
    for name in ('SPEC.md', 'ASSUMPTIONS.md', 'BUILD_NOTE.md'):
        files.extend((RETURN18 / 'implement-stage').glob(name))
    files.extend(p for p in (RETURN18 / 'R120_closeout').glob('*') if p.suffix in {'.md', '.pdf', '.csv'})
    files.extend((RETURN18 / 'dispatch').glob('main_approved_' + experiment.lower() + '_*.json'))
    files.extend(RETURN18 / 'R120_closeout' / name for name in ('CHECKS.json', 'evidence.json'))
    science = REPO / 'docs/research/return18_major_routes_20261030'
    files.extend(science / name for name in ('REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv',
        'ROUTING.json', 'routing_receipt.json', 'astra_followup.md', 'astra_followup_sources.csv'))
    result = {}
    for path in files:
        relative = path.resolve().relative_to(GIT_ROOT)
        if (path.is_symlink() or path.suffix not in SUFFIXES or path.stat().st_size >= 8 * 1024**2
                or {'wandb', 'input_cache', 'prediction_cache', 'rawinputs', 'caches'} & set(relative.parts)
                or (path.name.endswith('manifest.json') and path not in small_manifests)
                or any(word in path.name for word in ('raw_label', 'raw_input'))):
            raise ValueError('Outside the small curated publication allowlist: ' + str(path))
        result[str(relative)] = sha(path)
    return dict(sorted(result.items()))


def git(*args):
    return subprocess.run(['git', *args], cwd=GIT_ROOT, check=True, text=True, capture_output=True,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT='0')).stdout.strip()


def publish(root, experiment, *, apply=False):
    root = root.resolve()
    root.relative_to(V4 / 'runs')
    late_hashes = {}
    lock, reports = completion_gate(root, experiment, late_hashes=late_hashes)
    files = curated_paths(root, experiment, reports)
    identity = dict(experiment=experiment, root=str(root), checkpoint_sha256=lock['checkpoint_sha256'])
    if not apply:
        return dict(identity, ready=True, applied=False, files=files, late_publication_hashes=late_hashes)
    path = root / 'github_publication.json'
    def save(value):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    with (GIT_ROOT / '.git/codex-publication.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if Path(git('rev-parse', '--show-toplevel')) != GIT_ROOT or git('branch', '--show-current') != 'main':
            raise RuntimeError('Publication requires the declared Git root and main branch')
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Existing staged changes preserved; nothing added')
        record = read(path) if path.exists() else {}
        if record and any(record.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Existing publication belongs to another identity')
        if record.get('complete') is True:
            return record
        if record and not record.get('commit'):
            raise RuntimeError('Prior staging/commit attempt preserved; manual inspection required, no extra auto-commit')
        if record and git('rev-parse', 'HEAD') != record['commit']:
            raise RuntimeError('Failed-push commit is no longer HEAD; preserve it for inspection')
        record = record or dict(identity, complete=False, publisher_sha256=sha(__file__), files=files,
                               late_publication_hashes=late_hashes)
        try:
            if not record.get('commit'):
                record.update(phase='staging', before=git('rev-parse', 'HEAD'))
                if record['before'] != git('rev-parse', 'refs/remotes/origin/main'):
                    raise RuntimeError('Local unpublished history exists; do not push unrelated commits')
                save(record)
                git('add', '-f', '--', *files)
                staged = set(git('diff', '--cached', '--name-only').splitlines())
                if not staged or not staged <= files.keys() or git('rev-parse', 'HEAD') != record['before']:
                    raise RuntimeError('Concurrent/empty index changed; preserve it, never include unrelated files')
                if any(sha(GIT_ROOT / name) != files[name] for name in staged):
                    raise RuntimeError('Curated source/report changed during staging')
                record.update(phase='committing', staged=sorted(staged))
                save(record)
                git('commit', '-m', experiment.lower() + ': deliver one locked full18 model and final reports')
                record.update(commit=git('rev-parse', 'HEAD'), phase='committed', commits_for_this_version=1)
                save(record)
            record.update(phase='pushing')
            save(record)
            git('push', 'origin', 'HEAD:refs/heads/main')
            record.update(complete=True, phase='complete', push_returncode=0,
                          published_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            record.setdefault('failures', []).append(dict(phase=record.get('phase'), error=str(error),
                                                        returncode=getattr(error, 'returncode', None)))
            record.update(complete=False, phase='failed')
            raise
        finally:
            save(record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', choices=tuple(CODE), required=True)
    parser.add_argument('--apply', action='store_true', help='Commit/push once; omitted means no Git calls or writes')
    args = parser.parse_args(argv)
    value = publish(args.root, args.experiment, apply=args.apply)
    print(json.dumps(value, indent=2), flush=True)
    return value


if __name__ == '__main__':
    main()
