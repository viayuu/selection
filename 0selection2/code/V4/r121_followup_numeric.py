"""Isolated numeric-v3 continuation of the SHA-pinned CPU-only observer.

Compile the unchanged observer privately with exactly three guarded string
substitutions (export module, immutable filename, contract schema), then replace
only this private namespace's roots/hooks. Never edit/import-patch the canonical
observer, collector, learner, finisher, transport, model, or labels. Main alone
creates/runs the production controller and owns the old observer interruption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


REPO = Path('/public/home/shiys/0selection2')
PREPARATION = REPO / 'code/V4/runs/R121_structural_families/preparation'
ROOT = PREPARATION / 'followup_principal_v2'
COLLECTION = PREPARATION / 'collection_numeric_boundary_v3'
LEARNER = PREPARATION / 'learner_v1'
LEGACY_SOURCE = REPO / 'code/V4/r121_followup.py'
LEGACY_SHA = '0da941d9ab693eb91fcd17be182d6ce5cf277c3129ed9061ab0fc0c106af32e2'
LEGACY_CONTRACT = PREPARATION / 'followup_principal_v1/contract.json'
REPLACEMENT = PREPARATION / 'followup_principal_v1/main_external_replacement_numeric_v3.json'
PLAN_PATH = REPO / 'code/V4/runs/Return18_post_R120/dispatch/main_approved_r121_principal_v1.json'
SELF_SOURCE = 'code/V4/r121_followup_numeric.py'
EXTRA_PINS = {
    'code/V4/r121_numeric_collection.py': 'a9c638f48cbb8164171ef95bef77ae3b159a254968eddc78a71676b7dc3df50e',
    'code/V4/r121_parallel_collection_v3.py': '6c8632e5fba4f45c736313773cd0540c0de9e3c75fabb7a92422d37feabd14b0',
}
SCHEMA = 'r121-principal-followup-numeric-v2'
NUMERIC_VERSION = 'r121_numeric_boundary_collection_v3_20261030'
POLICY = 'fixed five InputV1; remaining lane1 V2; all other Original; no longest-prefix/mincost selection'
SUBSTITUTIONS = (
    ("'universal.json'", "'numeric.json'"),
    ("'code.V4.r121_universal_collection'", "'code.V4.r121_numeric_collection'"),
    ("'r121-principal-followup-v1'", repr(SCHEMA)),
)


def _root(value):
    result = Path(value).resolve()
    if result != ROOT:
        raise ValueError('Only the new literal followup_principal_v2 root is writable')
    return result


def _unstarted():
    if PLAN_PATH.exists() or any((LEARNER / 'principal' / name).exists()
                                for name in ('last.pt', 'best.pt', 'training_complete.json')):
        raise ValueError('Preserve existing principal_v1 plan/weights; never create, restart or resume another NN attempt')


def _policy(numeric, config):
    groups = config['group_order']
    special = set(numeric['input_boundary_keys'])
    if (numeric.get('version') != NUMERIC_VERSION or numeric.get('source_policy') != POLICY
            or len(special) != 5 or not special <= set(groups)
            or set(numeric['source_root_by_deployment']) != set(groups)
            or numeric.get('candidate_pool_unchanged') is not True
            or numeric.get('historical_labels_changed') is not False
            or numeric.get('solver_model_weights_seeds_budget_unchanged') is not True):
        raise ValueError('Require the original sealed numeric-v3 fixed-source collection')
    for index, key in enumerate(groups):
        name = ('collection_input_boundary_v1' if key in special else
                'collection_explicit_cpu_v2' if index % 4 == 1 else 'collection_persistent_v1')
        if Path(numeric['source_root_by_deployment'][key]).resolve() != PREPARATION / name:
            raise ValueError('Fixed InputV1/lane1V2/Original source policy changed: ' + key)


def _private():
    body = LEGACY_SOURCE.read_bytes()
    if hashlib.sha256(body).hexdigest() != LEGACY_SHA:
        raise ValueError('Preserve the frozen original observer; its SHA changed')
    source = body.decode('utf-8')
    for old, new in SUBSTITUTIONS:
        if source.count(old) != 1:
            raise ValueError('Pinned observer exact-literal adapter no longer applies: ' + old)
        source = source.replace(old, new, 1)
    ns = dict(__name__='code.V4._r121_followup_numeric_private', __package__='code.V4', __file__=str(LEGACY_SOURCE))
    exec(compile(source, str(LEGACY_SOURCE) + '[private-numeric-v2]', 'exec'), ns)
    old_sources = tuple(ns['SOURCES'])
    old_create, old_run, old_write = ns['create'], ns['run'], ns['write']
    old_progress, old_freeze = ns['label_progress'], ns['freeze_protocol']
    ns.update(ROOT=ROOT, COLLECTION=COLLECTION, LEARNER=LEARNER, PLAN_PATH=PLAN_PATH,
              SOURCES=(*old_sources, *EXTRA_PINS, SELF_SOURCE))

    def source_hashes():
        previous = ns['read'](LEGACY_CONTRACT)
        pins = dict(previous['source_sha256'])
        if set(pins) != set(old_sources) or pins.get('code/V4/r121_followup.py') != LEGACY_SHA:
            raise ValueError('Keep ALL original observer source pins, not a new snapshot of changed sources')
        ns['verify_sources'](previous)
        for name, digest in EXTRA_PINS.items():
            if ns['sha'](REPO / name) != digest:
                raise ValueError('Frozen numeric API/v3 wrapper changed: ' + name)
            pins[name] = digest
        pins[SELF_SOURCE] = ns['sha'](REPO / SELF_SOURCE)
        return pins

    def write(path, data, exclusive=False):
        path = Path(path).resolve()
        if path == ROOT / 'contract.json':
            previous = ns['read'](LEGACY_CONTRACT)
            config = ns['read'](COLLECTION / 'protocol.json')
            numeric = ns['read'](COLLECTION / 'numeric.json')
            _policy(numeric, config)
            # Preserve old immutable anchors too; only progress/cursors stay mutable.
            inputs = {}
            for item in (*previous['immutable_inputs'], *data['immutable_inputs'],
                         dict(path=str(LEGACY_CONTRACT), sha256=ns['sha'](LEGACY_CONTRACT)),
                         dict(path=str(REPLACEMENT), sha256=ns['sha'](REPLACEMENT))):
                if item['path'] in inputs and inputs[item['path']]['sha256'] != item['sha256']:
                    raise ValueError('An original immutable input changed')
                inputs[item['path']] = item
            data.update(immutable_inputs=list(inputs.values()),
                progress_source_roots=[str(PREPARATION / name) for name in
                    ('collection_persistent_v1', 'collection_input_boundary_v1', 'collection_explicit_cpu_v2')],
                progress_source_roots_role='informational only; numeric root alone gates full89',
                first_NN_attempt='principal_v1', old_followup_source_sha256=LEGACY_SHA,
                legacy_followup_root=str(LEGACY_CONTRACT.parent),
                main_external_replacement=dict(path=str(REPLACEMENT), sha256=ns['sha'](REPLACEMENT)),
                source_policy=numeric['source_policy'], source_root_by_deployment=numeric['source_root_by_deployment'],
                numeric_version=numeric['version'], numeric_sha256=numeric['numeric_sha256'],
                adaptation='private compile of pinned old observer; three exact strings plus isolated globals/hooks',
                original_controller_modified=False, NN_or_native_jobs_interrupted_by_this_controller=False)
        if path != PLAN_PATH and not path.is_relative_to(ROOT):
            raise ValueError('Observer writes are limited to the new root and the original principal_v1 plan')
        return old_write(path, data, exclusive=exclusive)

    def create(root):
        _root(root)
        _unstarted()
        # Main owns termination; record its external receipt without touching PID1202384.
        ns['read'](REPLACEMENT)
        source_hashes()
        return old_create(root)

    def verify_contract(contract):
        expected = dict(schema=SCHEMA, root=str(ROOT), collection_root=str(COLLECTION), learner_root=str(LEARNER),
            resource_plan=str(PLAN_PATH), first_NN_attempt='principal_v1', planned_labels=113920,
            endpoints=23040, seed=2, logical_batch=1024, successful_updates=14400, micro_cap=352,
            eval_batch_size=256, lib_batch_size=1, wandb_mode='offline', automatic_retry=False, automatic_resume=False)
        if (any(contract.get(k) != v for k, v in expected.items()) or len(contract['deployment_counts']) != 89
                or contract['deployment_counts'] != ns['read'](LEGACY_CONTRACT)['deployment_counts']):
            raise ValueError('Require the unchanged complete principal_v1 protocol, not a partial/new NN attempt')
        if contract['source_sha256'] != source_hashes():
            raise ValueError('Numeric continuation source pins changed')
        ns['verify_sources'](contract)

    def run(root, poll_seconds):
        _root(root)
        _unstarted()
        verify_contract(ns['read'](ROOT / 'contract.json'))
        return old_run(root, poll_seconds)

    def freeze_protocol(contract):
        _unstarted()
        return old_freeze(contract)

    def label_progress(contract):
        primary = old_progress(dict(contract, progress_source_roots=[]))
        try:
            observed = old_progress(contract)
            primary['collected_unique_labels'] = observed['collected_unique_labels']
            for key in primary['deployments']:
                primary['deployments'][key]['collected_unique'] = observed['deployments'][key]['collected_unique']
            primary['observation_uncertainty'].extend(observed['observation_uncertainty'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            primary['historical_observation_error'] = f'{type(error).__name__}: {error}'
        primary['historical_progress_gates_training'] = False
        return primary

    ns.update(source_hashes=source_hashes, write=write, create=create, run=run,
              freeze_protocol=freeze_protocol, label_progress=label_progress)
    return ns


def create(root=ROOT):
    return _private()['create'](root)


def run(root=ROOT, poll_seconds=60.):
    if poll_seconds < 10:
        raise ValueError('Use at least10 seconds between observations')
    return _private()['run'](root, poll_seconds)


def main(argv=None):
    return _private()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
